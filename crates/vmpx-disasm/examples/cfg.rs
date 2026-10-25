use std::{env, fs, process::ExitCode};

use vmpx_disasm::{CfgLimits, recover_cfg};
use vmpx_pe::{PeImage, Rva};

fn main() -> ExitCode {
    let arguments = env::args().collect::<Vec<_>>();
    let [_, path, rva] = arguments.as_slice() else {
        eprintln!("usage: cargo run -p vmpx-disasm --example cfg -- <pe> <rva-hex>");
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
    let Ok(cfg) = recover_cfg(
        &pe,
        Rva(rva),
        CfgLimits {
            max_blocks: 8_192,
            max_instructions: 131_072,
            max_instructions_per_block: 8_192,
        },
    ) else {
        eprintln!("CFG recovery failed");
        return ExitCode::FAILURE;
    };
    for block in &cfg.blocks {
        println!("block {} -> {:?}", block.start, block.successors);
        for instruction in &block.instructions {
            println!("  {}  {}", instruction.rva, instruction.text);
        }
        println!(
            "  terminal={} calls={:?}",
            block.terminal_flow, block.direct_calls
        );
    }
    eprintln!(
        "status={:?} blocks={} errors={:?}",
        cfg.status,
        cfg.blocks.len(),
        cfg.errors
    );
    ExitCode::SUCCESS
}
