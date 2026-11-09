// Recovers the ORIGINAL module entry point of a protected DLL, statically.
//
// The usual methods do not work on this sample: the emulator never reaches the
// hand-off (byte-identical state at 39.1B and 120B steps), `vmpx oep` reports
// the native entry unresolved with 2 of 4 paths having no unique continuation,
// and "no incoming direct call" is no filter at all -- 37406 of 66688 `.text`
// functions are roots in a C++ plugin.
//
// What does work is the CRT's own shape. A DLL's real entry is the toolchain's
// `DllEntryPoint`/`DllMainCRTStartup`, and it is recognisable without
// signatures because of what it must DO: it is called only by the loader, it
// takes the Win64 triple (rcx=hinstDLL, edx=fdwReason, r8=lpvReserved), it
// branches on the reason, and it FORWARDS all three unchanged to the user's
// DllMain. That conjunction is what this looks for.
//
// On systems.xpl the answer is cross-checked by intact DWARF: `.debug_str`
// opens with `hDllHandle`, `_pei386_runtime_relocator`, `DllEntryPoint` -- the
// symbols of MinGW's `dllcrt2.c` -- and `.debug_aranges` puts that compilation
// unit at rva 0x1000..0x137f, which contains the function this finds.
//
// Usage: findoep IMAGE [OUTPUT]   (with OUTPUT, writes the image with the
//                                 recovered entry point installed)

use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{Code, Decoder, DecoderOptions, FlowControl, Register};
use vmpx_pe::{PeImage, Rva};
use vmpx_rebuild::{set_entry_point, validate_for_loader};

const EXEC: u32 = 0x2000_0000;

/// Does this function look like a CRT DLL entry: branch on the reason, then
/// forward the Win64 triple unchanged to another function?
fn is_crt_dll_entry(code: &[u8], va: u64) -> bool {
    let mut decoder = Decoder::with_ip(64, code, va, DecoderOptions::NONE);
    // Which register currently holds a copy of each incoming argument.
    let mut holds_arg1: BTreeSet<Register> = [Register::RCX].into();
    let mut holds_arg2: BTreeSet<Register> = [Register::EDX, Register::RDX].into();
    let mut holds_arg3: BTreeSet<Register> = [Register::R8].into();
    let (mut branched_on_reason, mut forwarded) = (false, false);
    let mut staged = (false, false, false);
    while decoder.can_decode() {
        let instruction = decoder.decode();
        if instruction.is_invalid() {
            break;
        }
        // `mov reg, reg` propagates an argument copy.
        if matches!(
            instruction.code(),
            Code::Mov_r64_rm64 | Code::Mov_r32_rm32 | Code::Mov_rm64_r64 | Code::Mov_rm32_r32
        ) && instruction.op0_kind() == iced_x86::OpKind::Register
            && instruction.op1_kind() == iced_x86::OpKind::Register
        {
            let (dst, src) = (instruction.op0_register(), instruction.op1_register());
            let full = |r: Register| r.full_register();
            for set in [&mut holds_arg1, &mut holds_arg2, &mut holds_arg3] {
                if set.iter().any(|r| full(*r) == full(src)) {
                    set.insert(dst);
                }
            }
        }
        // A test/cmp of the reason argument, then a conditional branch.
        if matches!(instruction.mnemonic(), iced_x86::Mnemonic::Test | iced_x86::Mnemonic::Cmp)
            && holds_arg2
                .iter()
                .any(|r| r.full_register() == instruction.op0_register().full_register())
        {
            branched_on_reason = true;
        }
        // Staging the triple back into rcx/edx/r8 right before a direct call.
        if matches!(
            instruction.code(),
            Code::Mov_r64_rm64 | Code::Mov_r32_rm32 | Code::Mov_rm64_r64 | Code::Mov_rm32_r32
        ) && instruction.op0_kind() == iced_x86::OpKind::Register
            && instruction.op1_kind() == iced_x86::OpKind::Register
        {
            let (dst, src) = (instruction.op0_register(), instruction.op1_register());
            let from = |set: &BTreeSet<Register>| {
                set.iter().any(|r| r.full_register() == src.full_register())
            };
            match dst.full_register() {
                Register::RCX if from(&holds_arg1) => staged.0 = true,
                Register::RDX if from(&holds_arg2) => staged.1 = true,
                Register::R8 if from(&holds_arg3) => staged.2 = true,
                _ => {}
            }
        }
        if instruction.flow_control() == FlowControl::Call && instruction.near_branch64() != 0 {
            if staged == (true, true, true) {
                forwarded = true;
            }
            staged = (false, false, false);
        }
        if branched_on_reason && forwarded {
            return true;
        }
    }
    false
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let base = pe.optional.image_base.get();

    // Incoming direct calls, so a loader-only root can be told apart.
    let mut incoming: BTreeMap<u32, usize> = BTreeMap::new();
    for section in pe.sections.iter().filter(|s| s.characteristics & EXEC != 0) {
        let Ok(offset) = pe.rva_to_file_offset(section.virtual_address) else {
            continue;
        };
        let from = offset.0 as usize;
        let to = from
            .saturating_add(section.size_of_raw_data as usize)
            .min(bytes.len());
        let Some(blob) = bytes.get(from..to) else { continue };
        for at in 0..blob.len().saturating_sub(5) {
            if blob[at] != 0xE8 {
                continue;
            }
            let disp = i32::from_le_bytes(blob[at + 1..at + 5].try_into().expect("4"));
            let here = section.virtual_address.0 + u32::try_from(at).unwrap_or(0);
            let target = i64::from(here) + 5 + i64::from(disp);
            if let Ok(rva) = u32::try_from(target) {
                *incoming.entry(rva).or_default() += 1;
            }
        }
    }

    let mut found: Vec<(u32, u32)> = Vec::new();
    for function in &pe.report().exception_functions {
        let (begin, end) = (function.begin.0, function.end.0);
        // The CRT entry is small and called only by the loader.
        if end <= begin || end - begin > 0x800 || incoming.get(&begin).copied().unwrap_or(0) != 0 {
            continue;
        }
        let Ok(offset) = pe.rva_to_file_offset(Rva(begin)) else {
            continue;
        };
        let start = offset.0 as usize;
        let stop = start.saturating_add((end - begin) as usize).min(bytes.len());
        let Some(code) = bytes.get(start..stop) else { continue };
        // A routine inside a protector section is not the original entry, and
        // the protector has look-alikes: its own dispatchers forward the same
        // Win64 triple, which is how an earlier version of this rule picked
        // 0xc40ea54 out of `.vmp2` while skipping the real one.
        let protector = pe
            .sections
            .iter()
            .find(|s| s.contains_mapped_rva(Rva(begin)))
            .is_some_and(|s| s.name.as_str().starts_with(".vmp"));
        if !protector && is_crt_dll_entry(code, base + u64::from(begin)) {
            found.push((begin, end - begin));
        }
    }

    println!("{}: {} candidate entry point(s)", args[1], found.len());
    // The shape is necessary, not sufficient -- 120 functions on systems.xpl
    // branch on a register and then forward the Win64 triple. The tie-break is
    // LINK ORDER: the toolchain's startup object (`dllcrt2.o`, `crt0.o`) is
    // linked first, so the real entry sits at the very start of `.text`. That
    // is a ranking, not a proof, and it is reported as one.
    found.sort_unstable();
    for (rva, len) in found.iter().take(5) {
        println!("  rva {rva:#x} ({len} bytes)");
    }
    let Some(&(oep, _)) = found.first() else {
        println!("no candidate; refusing to install one");
        return;
    };
    println!("best candidate by link order: {oep:#x}");
    // Independent confirmation when the binary kept its DWARF: the first
    // compilation unit's address range should contain the entry. Absent debug
    // info this is simply skipped -- the ranking above stands on its own.
    if let Some(section) = pe.sections.iter().find(|s| s.name.as_str().starts_with(".debug_")) {
        println!("  (debug info present in {}; see .debug_aranges CU 0 to confirm)", section.name.as_str());
    } else {
        println!("  (no debug info; the ranking is the only evidence)");
    }
    if args.len() < 3 {
        println!("(pass an OUTPUT path to install it)");
        return;
    }
    let image = set_entry_point(&bytes, Rva(oep)).expect("set entry");
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
