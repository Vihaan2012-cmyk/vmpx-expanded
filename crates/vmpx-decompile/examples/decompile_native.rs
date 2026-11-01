use vmpx_mutation::cfg_recovery::{CfgOptions, recover_native_cfg_pe_with};
use vmpx_pe::{PeImage, Rva};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let names: Vec<String> = if args.len() > 2 {
        args[2..].to_vec()
    } else {
        pe.report()
            .exports
            .unwrap()
            .symbols
            .into_iter()
            .filter_map(|s| s.name)
            .collect()
    };
    for name in names {
        let rva = pe
            .report()
            .exports
            .unwrap()
            .symbols
            .into_iter()
            .find(|s| s.name.as_deref() == Some(name.as_str()))
            .unwrap()
            .rva;
        let cfg = match recover_native_cfg_pe_with(&pe, Rva(rva.0), CfgOptions::PROVEN) {
            Ok(c) => c,
            Err(e) => {
                println!("// {name}: cfg recovery failed: {e}\n");
                continue;
            }
        };
        match vmpx_decompile::native::lower(&cfg, &name) {
            Err(e) => println!("// {name}: lowering refused: {e}\n"),
            Ok(f) => match vmpx_decompile::decompile(&f) {
                Err(e) => println!("// {name}: decompile failed: {e}\n"),
                Ok(d) => {
                    print!("{}", d.text);
                    println!(
                        "// structured={} loops={} ifs={} gotos={} fallback={:?} roundtrip={:?}\n",
                        d.structured, d.loops, d.ifs, d.gotos, d.fallback_reason, d.check
                    );
                }
            },
        }
    }
}
