// Closes the emitted entry stub's two tail transfers into the protector.
//
// `clean --experimental` emits a DllMain replacement for ONE reason and leaves
// both ends open:
//
//     0xf1f3414  cmp rdx,1          ; DLL_PROCESS_ATTACH?
//     0xf1f3418  je  0xf1f3000      ; yes -> the emitted body
//     0xf1f341e  jmp 0xe0f038b      ; no  -> the PROTECTOR's entry
//     ...
//     0xf1f340f  jmp 0xefd3ec9      ; body tail -> a .vmp3 continuation
//
// Both are `jmp rel32`, five bytes, which is exactly `xor eax,eax; inc eax;
// ret` -- so each can become "return TRUE" in place, with no layout change.
//
// THE ASSUMPTION, stated: that a DllMain returning TRUE is sufficient. The
// evidence for it on this sample is that the application's initialization is
// not in DllMain at all -- the four TLS callbacks the strip PRESERVED are
// MinGW's `_CRT_INIT`/`__dyn_tls` pair, each switching on `edx` (the reason
// argument) and holding the only `.text` references to the `.CRT` init arrays,
// and the plugin's real work hangs off its XPlugin* exports. The loader calls
// TLS callbacks on its own, so that path is intact without an entry point.
// It is an assumption nonetheless, and it is why this is a separate opt-in
// pass rather than part of the strip.
//
// Usage: entrystub IMAGE OUTPUT [--apply]

use iced_x86::{Decoder, DecoderOptions, FlowControl};
use vmpx_pe::{PeImage, Rva};
use vmpx_rebuild::validate_for_loader;

const EXEC: u32 = 0x2000_0000;
/// `xor eax,eax; inc eax; ret` -- RAX = 1 with no upper-bit residue.
const RETURN_TRUE: [u8; 5] = [0x31, 0xC0, 0xFF, 0xC0, 0xC3];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let apply = args.iter().any(|a| a == "--apply");
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let base = pe.optional.image_base.get();
    let mapped = |rva: u32| pe.sections.iter().any(|s| s.contains_mapped_rva(Rva(rva)));

    // Only sections the exception directory does not describe: the emitted ones.
    let described = |start: u32, size: u32| {
        pe.report()
            .exception_functions
            .iter()
            .any(|f| f.end.0 > start && f.begin.0 < start + size)
    };

    let mut patches: Vec<(u32, u64, String)> = Vec::new();
    for section in pe.sections.iter().filter(|s| s.characteristics & EXEC != 0) {
        let start = section.virtual_address.0;
        let size = section.virtual_size.max(section.size_of_raw_data);
        if described(start, size) {
            continue;
        }
        let Ok(offset) = pe.rva_to_file_offset(Rva(start)) else {
            continue;
        };
        let from = offset.0 as usize;
        let to = from.saturating_add(size as usize).min(bytes.len());
        let Some(code) = bytes.get(from..to) else { continue };
        let mut decoder = Decoder::with_ip(64, code, base + u64::from(start), DecoderOptions::NONE);
        while decoder.can_decode() {
            let at = decoder.position();
            let instruction = decoder.decode();
            if instruction.is_invalid() {
                break;
            }
            // An unconditional tail transfer out of the stub into removed space.
            if instruction.flow_control() != FlowControl::UnconditionalBranch
                || instruction.near_branch64() == 0
                || instruction.len() != RETURN_TRUE.len()
            {
                continue;
            }
            let Some(target) = instruction
                .near_branch64()
                .checked_sub(base)
                .and_then(|v| u32::try_from(v).ok())
            else {
                continue;
            };
            if target >= pe.optional.size_of_image || mapped(target) {
                continue;
            }
            let rva = start + u32::try_from(at).unwrap_or(0);
            patches.push((
                rva,
                offset.0 + u64::try_from(at).unwrap_or(0),
                format!("{rva:#x} -> {target:#x}  {instruction}"),
            ));
        }
    }

    println!(
        "{}: {} tail transfer(s) into removed space in emitted section(s)",
        args[1],
        patches.len()
    );
    for (_, _, what) in &patches {
        println!("  {what}");
    }
    if !apply {
        println!("\n(dry run; pass --apply to rewrite each as `return TRUE`)");
        return;
    }

    let mut image = bytes.clone();
    for (rva, offset, _) in &patches {
        let at = usize::try_from(*offset).expect("offset");
        let Some(slot) = image.get_mut(at..at + RETURN_TRUE.len()) else {
            println!("  {rva:#x}: outside the file");
            continue;
        };
        slot.copy_from_slice(&RETURN_TRUE);
        println!("  {rva:#x}: rewritten as `xor eax,eax; inc eax; ret`");
    }
    // The entry point must still land on executable, file-backed bytes.
    match PeImage::parse(&image) {
        Ok(reparsed) => println!(
            "reparse ok; entry {:?} still executable: {}",
            reparsed.optional.address_of_entry_point,
            reparsed.sections.iter().any(|s| s.characteristics & EXEC != 0
                && s.contains_mapped_rva(reparsed.optional.address_of_entry_point))
        ),
        Err(error) => {
            println!("REPARSE FAILED: {error}");
            return;
        }
    }
    match validate_for_loader(&image) {
        Ok(report) => println!(
            "loader validation passed: {} sections, {} imports, {} tls callback(s)",
            report.sections, report.import_symbols, report.tls_callbacks
        ),
        Err(error) => println!("LOADER VALIDATION FAILED: {error}"),
    }
    std::fs::write(&args[2], &image).expect("write");
    println!("wrote {}", args[2]);
}
