use std::{env, fs, process::ExitCode};

use vmpx_disasm::{DirectJumpTraceLimits, trace_direct_jump_fragments};
use vmpx_pe::{PeImage, Rva};

fn main() -> ExitCode {
    let arguments = env::args().collect::<Vec<_>>();
    let [_, path, rva] = arguments.as_slice() else {
        eprintln!("usage: cargo run -p vmpx-vm --example trace -- <pe> <rva-hex>");
        return ExitCode::FAILURE;
    };
    let rva = rva
        .strip_prefix("0x")
        .or_else(|| rva.strip_prefix("0X"))
        .unwrap_or(rva);
    let Ok(rva) = u32::from_str_radix(rva, 16) else {
        eprintln!("invalid hexadecimal RVA");
        return ExitCode::FAILURE;
    };
    let Ok(bytes) = fs::read(path) else {
        eprintln!("failed to read input");
        return ExitCode::FAILURE;
    };
    let Ok(pe) = PeImage::parse(&bytes) else {
        eprintln!("input is not a supported PE");
        return ExitCode::FAILURE;
    };
    let Ok(trace) = trace_direct_jump_fragments(
        &pe,
        Rva(rva),
        DirectJumpTraceLimits {
            max_fragments: 4_096,
            max_instructions: 65_536,
            max_instructions_per_fragment: 4_096,
        },
    ) else {
        eprintln!("trace failed");
        return ExitCode::FAILURE;
    };
    for instruction in trace.instructions() {
        println!("{}  {}", instruction.rva, instruction.instruction);
    }
    eprintln!("stop: {:?} at {}", trace.stop.kind, trace.stop.at);
    ExitCode::SUCCESS
}
