// Throwaway: full-stack detonation + in-memory image dump.
use std::collections::BTreeMap;
use vmpx_emulate::{detonate, DetonateOptions, SysDllDb};
use vmpx_pe::PeImage;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dtest <pe>");
    let data = std::fs::read(&path).expect("read");
    let pe = PeImage::parse(&data).expect("parse");
    let db = SysDllDb::new();
    let mut export_reference = BTreeMap::new();
    for m in ["ntdll.dll", "kernel32.dll", "kernelbase.dll", "user32.dll"] {
        if let Some(ex) = db.exports(m) {
            export_reference.insert(m.to_string(), ex);
        }
    }
    let opts = DetonateOptions {
        max_steps: 80_000_000,
        dump: true,
        export_reference,
        ..Default::default()
    };
    match detonate(&pe, &opts) {
        Ok(r) => {
            println!(
                "stop={:?} steps={} oep={:?} imports={} globals={}",
                r.stop, r.steps, r.observed.oep, r.observed.imports.len(), r.observed.globals.len()
            );
            for imp in r.observed.imports.iter().take(30) {
                println!("  import {}!{:?}", imp.module, imp.symbol);
            }
            match &r.dump {
                Some(d) => {
                    println!(
                        "DUMP image_base={:#x} regions={} total={:#x} bytes",
                        d.image_base,
                        d.regions.len(),
                        d.total_bytes()
                    );
                    for reg in d.regions.iter().take(16) {
                        println!("  region va={:#x} perm={:?} size={:#x}", reg.va, reg.perm, reg.bytes.len());
                    }
                }
                None => println!("DUMP: none"),
            }
        }
        Err(e) => println!("detonate error: {e:?}"),
    }
}
