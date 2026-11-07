//! Disassemble a byte range of an RVA-mapped analysis image.
//! usage: disasm_at <mapped.bin> <rva_hex> <count>
use iced_x86::{Decoder, DecoderOptions, Formatter, NasmFormatter};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, path, rva, count] = &args[..] else {
        eprintln!("usage: disasm_at <mapped.bin> <rva_hex> <count>");
        std::process::exit(2);
    };
    let rva = u64::from_str_radix(rva.trim_start_matches("0x"), 16).expect("rva");
    let count: usize = count.parse().expect("count");
    let bytes = std::fs::read(path).expect("read");
    let start = usize::try_from(rva).expect("rva fits");
    let window = &bytes[start..(start + count * 16).min(bytes.len())];

    let mut decoder = Decoder::with_ip(64, window, rva, DecoderOptions::NONE);
    let mut formatter = NasmFormatter::new();
    let mut text = String::new();
    for _ in 0..count {
        if !decoder.can_decode() {
            break;
        }
        let instruction = decoder.decode();
        text.clear();
        formatter.format(&instruction, &mut text);
        println!("{:#010x}  {text}", instruction.ip());
    }
}
