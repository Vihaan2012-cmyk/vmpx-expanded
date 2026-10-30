// Decodes a VM entry's linear stream, lifts it to native IR/bytes, and prints the
// result or the first thing that stops the lift.
//
// usage: liftdiag <pe> <entry-rva> [<entry-rva>...] [-v]
use std::fs;

use vmpx_pe::{PeImage, Rva};
use vmpx_recompile::CodeAddress;
use vmpx_vm::{
    EntryProfileRecoveryLimits, Expr, SymbolicExecutionLimits, VmDiscoveryLimits,
    X64NativeRegister, analyze_vm_entry_at, recover_entry_stream,
};
use vmpx_vmlift::{
    DecodedLinearInput, SemanticLowering, first_unliftable, lift_decoded_linear,
    recover_entry_frame, semantic_lowering,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let verbose = args.iter().any(|arg| arg == "-v");
    let next_rva = vmpx_rebuild::next_section_rva(&bytes).expect("next section rva");
    let address = CodeAddress(pe.optional.image_base.0 + u64::from(next_rva.0));
    for target in args[2..].iter().filter(|arg| arg.starts_with("0x")) {
        let target = u32::from_str_radix(target.trim_start_matches("0x"), 16).unwrap();
        println!("=== entry {target:#x}");
        let Some(candidate) = analyze_vm_entry_at(&pe, Rva(target), VmDiscoveryLimits::default())
        else {
            println!("  no VM entry candidate");
            continue;
        };
        let limits = EntryProfileRecoveryLimits {
            instruction_limit: 100_000,
            ..EntryProfileRecoveryLimits::default()
        };
        let stream = match recover_entry_stream(&pe, &candidate, limits) {
            Ok(stream) => stream,
            Err(error) => {
                println!("  stream recovery failed: {error}");
                continue;
            }
        };
        let decode = &stream.decode.decode;
        println!(
            "  decoded {} instructions, stop={:?}",
            decode.instructions.len(),
            decode.stop_reason
        );
        let mut counts = std::collections::BTreeMap::<String, (usize, String)>::new();
        for instruction in &decode.instructions {
            let entry = counts
                .entry(format!("{:?}", instruction.semantic))
                .or_insert((0, format!("{:?}", semantic_lowering(instruction.semantic))));
            entry.0 += 1;
        }
        for (name, (count, lowering)) in &counts {
            println!("    {name} x{count}: {lowering}");
        }
        if let Some((index, semantic, reason)) = first_unliftable(&decode.instructions) {
            println!("  STOP: instruction {index} {semantic:?} not liftable: {reason}");
            continue;
        }
        let exit = stream
            .decode
            .handler_recoveries
            .get(&decode.final_state.handler)
            .and_then(|recovery| recovery.vm_exit.as_ref());
        let Some(exit) = exit else {
            println!("  STOP: stream does not end at a recovered VM-exit handler");
            continue;
        };
        let (frame, top_offset) = match recover_entry_frame(
            &pe,
            candidate.rva,
            stream.profile.effect_roles.vsp,
            limits.trace,
        ) {
            Ok(result) => result,
            Err(error) => {
                println!("  STOP: entry frame not recovered: {error}");
                continue;
            }
        };
        println!("  entry frame: {} cells", frame.len());
        let input = DecodedLinearInput {
            instructions: &decode.instructions,
            frame: &frame,
            top_offset,
            exit,
            memory: Some(&pe),
            limits: SymbolicExecutionLimits::default(),
        };
        match lift_decoded_linear(&input, address) {
            Ok(lifted) => {
                let replacement = &lifted.replacement;
                println!(
                    "  LIFTED {} VM instructions -> {} IR operations, {} native bytes",
                    lifted.instructions_lifted,
                    replacement.function.operations.len(),
                    replacement.compiled.bytes.len()
                );
                println!(
                    "  assumptions: {:?}; returns via add rsp,{} ; ret",
                    lifted.assumptions, lifted.exit_state.return_pop
                );
                println!(
                    "  passthroughs: {:?}, environment boundaries: {:?}",
                    replacement.passthroughs, replacement.environment_boundaries
                );
                let identity = X64NativeRegister::ALL
                    .iter()
                    .filter(|register| {
                        lifted.exit_state.registers.get(register)
                            == Some(&Expr::register(
                                vmpx_vm::BitWidth::W64,
                                register.symbolic_name(),
                            ))
                    })
                    .count();
                println!("  exit registers unchanged from entry: {identity}/15");
                for (register, value) in &lifted.exit_state.registers {
                    if *value
                        != Expr::register(vmpx_vm::BitWidth::W64, register.symbolic_name())
                    {
                        println!("    {register:?} = {value:?}");
                    }
                }
                if verbose {
                    for (index, operation) in replacement.function.operations.iter().enumerate() {
                        println!("    {index:4} {operation:?}");
                    }
                }
            }
            Err(error) => println!("  STOP: {error}"),
        }
        let _ = SemanticLowering::BlockExit;
    }
}
