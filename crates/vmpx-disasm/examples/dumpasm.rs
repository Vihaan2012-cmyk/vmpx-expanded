use std::fs;
use vmpx_disasm::decode_one;
use vmpx_pe::{PeImage, Rva};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let mut rva = Rva(u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap());
    let count: usize = args[3].parse().unwrap();
    let base = pe.optional.image_base.0;
    let mut factory = iced_x86::InstructionInfoFactory::new();
    for _ in 0..count {
        let Ok(ins) = decode_one(&pe, rva) else {
            println!("{:#x}: <decode error>", rva.0);
            break;
        };
        let s = ins.summary(&mut factory);
        println!(
            "{:#x} (va {:#x}) {:20} {}",
            rva.0,
            base + u64::from(rva.0),
            s.bytes
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(""),
            s.text
        );
        let Some(next) = ins.next_rva() else { break };
        rva = next;
    }
}
