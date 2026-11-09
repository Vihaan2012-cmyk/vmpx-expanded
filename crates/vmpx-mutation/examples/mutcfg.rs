// Recovers the native CFG of a MUTATED native function living in a protector
// section, and reports whether it is complete enough to re-emit.
//
// Two of the sixteen protector targets reached from native code on systems.xpl
// are not VM entries at all: `analyze_vm_entry_at` scores them 0 with no seed
// because there is no bootstrap. Their disassembly is a normal function body
// with dead instructions interleaved and blocks split by `jmp +0`. They need
// the mutation path, not the devirtualizer.
//
// Usage: mutcfg IMAGE RVA [RVA ...]

use vmpx_mutation::{CfgOptions, Terminator, recover_native_cfg_pe_with};
use vmpx_pe::{PeImage, Rva};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let base = pe.optional.image_base.get();

    for raw in &args[2..] {
        let rva = u32::from_str_radix(raw.trim_start_matches("0x"), 16).expect("rva");
        println!("=== {raw} (va {:#x})", base + u64::from(rva));
        let options = CfgOptions {
            model_calls: true,
            assume_stack_disjoint: std::env::var_os("MUTCFG_STACK_DISJOINT").is_some(),
            entry_args_disjoint: std::env::var_os("MUTCFG_STACK_DISJOINT").is_some(),
            ..CfgOptions::PROVEN
        };
        match recover_native_cfg_pe_with(&pe, Rva(rva), options) {
            Err(error) => println!("  REFUSED: {error}"),
            Ok(cfg) => {
                let unsupported: Vec<_> = cfg
                    .blocks
                    .values()
                    .filter_map(|block| match &block.terminator {
                        Terminator::Unsupported(why) => Some((block.start, why.clone())),
                        _ => None,
                    })
                    .collect();
                let lifted = cfg
                    .blocks
                    .values()
                    .filter(|block| block.semantics.is_some())
                    .count();
                println!(
                    "  blocks {} ({lifted} lifted), edges {}, complete {}, stack-disjoint {}",
                    cfg.blocks.len(),
                    cfg.edges.len(),
                    cfg.is_complete(),
                    cfg.assumes_stack_disjoint
                );
                for block in cfg.blocks.values() {
                    println!(
                        "    {:#x}..{:#x} {} {:?}",
                        block.start,
                        block.end,
                        if block.semantics.is_some() { "lift" } else { "NOLIFT" },
                        block.terminator
                    );
                }
                for (at, why) in &unsupported {
                    println!("    unsupported at {at:#x}: {why}");
                }
            }
        }
    }
}
