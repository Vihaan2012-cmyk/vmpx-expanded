// Disassembles a VM handler so a `has no recovered specification` refusal can be
// classified: a genuine unimplemented semantic, or a fail-closed refusal over a
// sequence that does not actually touch the dispatch roles.
//
// Usage: hdis IMAGE ADDR [COUNT]
//   ADDR is a VA (0x377...) or an RVA (0xc66...); both are accepted and the
//   image base decides which. COUNT defaults to 40 instructions.

use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};
use vmpx_pe::{PeImage, Rva, Va};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let raw = u64::from_str_radix(args[2].trim_start_matches("0x"), 16).expect("addr");
    let count: usize = args.get(3).and_then(|c| c.parse().ok()).unwrap_or(40);

    let base = pe.optional.image_base;
    // A value at or above the image base is a VA; anything smaller is an RVA.
    let (rva, va) = match base.checked_rva_of(Va(raw)) {
        Some(rva) => (rva, raw),
        None => (Rva(raw as u32), base.get().saturating_add(raw)),
    };
    println!("image_base {:#x}  rva {:#x}  va {va:#x}", base.get(), rva.0);

    let offset = pe.rva_to_file_offset(rva).expect("mapped").0 as usize;
    let end = (offset + count * 16).min(bytes.len());
    let bits = if pe.optional.kind == vmpx_pe::PeKind::Pe32Plus { 64 } else { 32 };
    let mut decoder = Decoder::with_ip(bits, &bytes[offset..end], va, DecoderOptions::NONE);
    let mut formatter = IntelFormatter::new();
    let mut text = String::new();
    for _ in 0..count {
        if !decoder.can_decode() {
            break;
        }
        let instruction = decoder.decode();
        text.clear();
        formatter.format(&instruction, &mut text);
        println!("{:#018x}  {text}", instruction.ip());
        if instruction.is_invalid() {
            println!("  (invalid; stopping)");
            break;
        }
    }
}
