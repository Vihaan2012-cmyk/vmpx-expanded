//! Where the language handler's seed slot comes from.
//!
//! The null-read gate needs to know whether the continuation seed at
//! `CONTEXT.Rsp + 0xC0` is zero: a zero seed unwinds to the VM's caller, a
//! non-zero one becomes a jump target. `CONTEXT.Rsp` is the native rsp AT THE
//! FAULT, so the slot's address is `rsp_at_fault + 0xC0`, and this walks the
//! faulting handler's own trace tracking the rsp delta to find out whether any
//! instruction in it writes that address -- and with what.
//!
//! Measuring this before building it, because the last two conclusions I drew
//! about this mechanism from reading code were both wrong.
//!
//! usage: seedtrack <mapped.bin> <handler-rva-hex> [seed-offset-hex]
use std::{env, fs, process::ExitCode};

use iced_x86::{Mnemonic, OpKind, Register};
use vmpx_disasm::{DirectJumpTraceLimits, trace_direct_jump_fragments};
use vmpx_pe::{PeImage, Rva};

/// How `rsp` moves across one instruction, or `None` if it moves unknowably.
fn rsp_delta(instruction: &iced_x86::Instruction) -> Option<i64> {
    match instruction.mnemonic() {
        Mnemonic::Push => Some(-8),
        Mnemonic::Pop => Some(8),
        Mnemonic::Pushfq => Some(-8),
        Mnemonic::Popfq => Some(8),
        Mnemonic::Sub | Mnemonic::Add
            if instruction.op0_kind() == OpKind::Register
                && instruction.op0_register() == Register::RSP =>
        {
            let immediate = match instruction.op1_kind() {
                OpKind::Immediate8to64 => Some(instruction.immediate8to64()),
                OpKind::Immediate32to64 => Some(instruction.immediate32to64()),
                OpKind::Immediate8 => Some(i64::from(instruction.immediate8())),
                OpKind::Immediate32 => Some(i64::from(instruction.immediate32())),
                _ => None,
            }?;
            Some(if instruction.mnemonic() == Mnemonic::Sub {
                -immediate
            } else {
                immediate
            })
        }
        // Any other write to rsp loses the tracking.
        _ => {
            let writes_rsp = instruction.op0_kind() == OpKind::Register
                && instruction.op0_register().full_register() == Register::RSP;
            if writes_rsp { None } else { Some(0) }
        }
    }
}

fn main() -> ExitCode {
    let arguments = env::args().collect::<Vec<_>>();
    let (path, rva, seed) = match arguments.as_slice() {
        [_, path, rva] => (path, rva, 0xC0_i64),
        [_, path, rva, seed] => {
            let text = seed.strip_prefix("0x").unwrap_or(seed);
            (path, rva, i64::from_str_radix(text, 16).unwrap_or(0xC0))
        }
        _ => {
            eprintln!("usage: seedtrack <mapped.bin> <handler-rva-hex> [seed-offset-hex]");
            return ExitCode::FAILURE;
        }
    };
    let text = rva.strip_prefix("0x").or_else(|| rva.strip_prefix("0X")).unwrap_or(rva);
    let Ok(handler) = u32::from_str_radix(text, 16) else {
        eprintln!("invalid hexadecimal RVA");
        return ExitCode::FAILURE;
    };
    let Ok(bytes) = fs::read(path) else {
        eprintln!("failed to read input");
        return ExitCode::FAILURE;
    };
    let Ok(pe) = PeImage::parse(&bytes) else {
        eprintln!("failed to parse the image");
        return ExitCode::FAILURE;
    };
    let Ok(trace) =
        trace_direct_jump_fragments(&pe, Rva(handler), DirectJumpTraceLimits::default())
    else {
        eprintln!("handler trace failed");
        return ExitCode::FAILURE;
    };

    // Pass 1: rsp delta from handler entry at each instruction, and the
    // candidate fault site (an external read, as the walker picks it).
    let instructions: Vec<_> = trace.instructions().collect();
    let mut delta = 0_i64;
    let mut tracked = true;
    let mut deltas: Vec<Option<i64>> = Vec::with_capacity(instructions.len());
    for decoded in &instructions {
        deltas.push(if tracked { Some(delta) } else { None });
        match rsp_delta(&decoded.instruction) {
            Some(step) => delta += step,
            None => tracked = false,
        }
    }

    let fault = instructions.iter().enumerate().find(|(_, decoded)| {
        let instruction = &decoded.instruction;
        instruction.mnemonic() != Mnemonic::Lea
            && (0..instruction.op_count())
                .any(|operand| instruction.op_kind(operand) == OpKind::Memory)
            && !matches!(
                instruction.memory_base().full_register(),
                Register::RSP | Register::RBP | Register::None
            )
    });
    let Some((fault_index, fault_decoded)) = fault else {
        println!("no external read in the trace");
        return ExitCode::SUCCESS;
    };
    let Some(fault_delta) = deltas[fault_index] else {
        println!(
            "rsp tracking was lost before the fault at {} -- cannot place the seed slot",
            fault_decoded.rva
        );
        return ExitCode::SUCCESS;
    };
    // slot = rsp_at_fault + seed = (entry + fault_delta) + seed
    let slot_from_entry = fault_delta + seed;
    println!(
        "fault at {} (instruction {fault_index}), rsp delta {fault_delta:+}",
        fault_decoded.rva
    );
    println!(
        "seed slot = rsp_at_fault{seed:+#x} = rsp_at_entry{slot_from_entry:+#x} ({})",
        if slot_from_entry >= 0 { "ABOVE entry rsp: the VM's own frame" } else { "below entry rsp" }
    );

    // Pass 2: any instruction in the trace that WRITES that address.
    let mut writers = 0_usize;
    for (index, decoded) in instructions.iter().enumerate() {
        let instruction = &decoded.instruction;
        if instruction.op0_kind() != OpKind::Memory {
            continue;
        }
        let Some(at) = deltas[index] else { continue };
        let base = instruction.memory_base().full_register();
        if base != Register::RSP {
            continue;
        }
        let address_from_entry = at + instruction.memory_displacement64() as i64;
        if address_from_entry == slot_from_entry {
            writers += 1;
            println!(
                "  WRITER at {}: {:?} displacement {:#x} -> slot",
                decoded.rva,
                instruction.mnemonic(),
                instruction.memory_displacement64()
            );
        }
    }
    // A push also writes a stack slot.
    for (index, decoded) in instructions.iter().enumerate() {
        if decoded.instruction.mnemonic() != Mnemonic::Push {
            continue;
        }
        let Some(at) = deltas[index] else { continue };
        if at - 8 == slot_from_entry {
            writers += 1;
            println!("  WRITER at {}: push -> slot", decoded.rva);
        }
    }
    if writers == 0 {
        println!("  no instruction in this handler's trace writes the seed slot");
    }
    ExitCode::SUCCESS
}
