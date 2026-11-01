use vmpx_disasm::FlagState;
use vmpx_mutation::cfg_recovery::{CfgOptions, recover_native_cfg_pe_with};
use vmpx_pe::{PeImage, Rva};
use vmpx_verify::differential::pretty;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let rva = pe
        .report()
        .exports
        .unwrap()
        .symbols
        .into_iter()
        .find(|s| s.name.as_deref() == Some(args[2].as_str()))
        .unwrap()
        .rva;
    let cfg = recover_native_cfg_pe_with(&pe, Rva(rva.0), CfgOptions::PROVEN).unwrap();
    println!("entry {:#x} complete={}", cfg.entry, cfg.is_complete());
    for (a, b) in &cfg.blocks {
        println!("block {a:#x}..{:#x} term={:?}", b.end, b.terminator);
        if let Some(s) = &b.semantics {
            for g in &s.written {
                println!("  {} = {}", g.name(), pretty(&s.regs[g].simplify(), 12));
            }
            for (f, st) in &s.flags {
                match st {
                    FlagState::Value(e) => println!("  flag {} = {}", f.name(), pretty(&e.simplify(), 8)),
                    FlagState::Clobbered => println!("  flag {} = clobbered", f.name()),
                }
            }
            for w in &s.writes {
                println!("  store {:?} [{}] = {}", w.width, pretty(&w.address, 6), pretty(&w.value, 8));
            }
        }
    }
}
