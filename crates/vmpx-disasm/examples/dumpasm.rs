// Linear disassembly from an RVA, for diagnosing VM bootstraps.
// usage: dumpasm <file> <rva-hex> <count>
use std::fs;
use vmpx_disasm::decode_one;
use vmpx_pe::{PeImage, Rva};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let start = u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap();
    let count: usize = args[3].parse().unwrap();
    let mut rva = Rva(start);
    for _ in 0..count {
        match decode_one(&pe, rva) {
            Ok(d) => {
                let hex: String = d.bytes.iter().map(|b| format!("{b:02x}")).collect();
                println!("{:#010x}  {:<24}  {}", d.rva.0, hex, d.instruction);
                match d.next_rva() {
                    Some(n) => rva = n,
                    None => break,
                }
            }
            Err(e) => {
                println!("{:#010x}  <decode error: {e}>", rva.0);
                break;
            }
        }
    }
}
