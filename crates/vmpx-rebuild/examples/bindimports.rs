// Binds the recovered import table to the application's EXISTING `.idata`
// slots, so the loader fills the 19030 call sites that already read them.
//
// `.idata` is encrypted in the file and the protector's init is what decrypted
// it. A post-init snapshot names 1016 of the 1017 slots the application calls
// through (`vmpx-emulate/examples/idata.rs` writes them as a TSV). This turns
// those bindings into a real import directory.
//
// WHY IN PLACE. The loader walks a descriptor's ILT and writes each resolved
// address into the parallel `FirstThunk` slot, stopping at the ILT's null. So
// pointing `FirstThunk` at the existing `.idata` run makes the loader overwrite
// exactly the slots the code reads -- the encrypted bytes there never matter.
// The alternative, a fresh IAT, would mean repointing 19030 call sites.
//
// `serialize_import_table` lays out descriptors, ILTs, its OWN IAT, then the
// name blobs, and writes its internal IAT RVA into each descriptor's
// `FirstThunk` (descriptor + 16). Only that one field per descriptor is
// redirected here; the serializer is otherwise used unchanged, and its internal
// IAT is left as dead padding rather than risk re-laying out a tested
// serializer.
//
// One descriptor covers one CONTIGUOUS run, so a module whose slots are split
// gets several descriptors -- legal, and on systems.xpl 3 of 23 modules need it.
//
// Usage: bindimports IMAGE BINDINGS.tsv OUTPUT

use std::collections::BTreeMap;

use vmpx_pe::PeImage;
use vmpx_rebuild::{
    AppendSection, ImportModuleSpec, ImportTarget, append_section, next_section_rva,
    serialize_import_table, validate_for_loader,
};

const RW: u32 = 0xC000_0040; // INITIALIZED_DATA | READ | WRITE
const DESCRIPTOR_SIZE: usize = 20;
const FIRST_THUNK_FIELD: usize = 16;
const IMPORT_DIRECTORY: usize = 1;
const IAT_DIRECTORY: usize = 12;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let text = std::fs::read_to_string(&args[2]).expect("bindings");
    let pe = PeImage::parse(&bytes).expect("parse");

    // (module, slot, symbol), ordered by slot so runs are discoverable.
    let mut rows: Vec<(String, u32, String)> = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let mut field = line.split('\t');
        let (Some(slot), Some(module), Some(symbol)) = (field.next(), field.next(), field.next())
        else {
            continue;
        };
        let Ok(slot) = u32::from_str_radix(slot.trim().trim_start_matches("0x"), 16) else {
            continue;
        };
        rows.push((module.to_owned(), slot, symbol.to_owned()));
    }
    rows.sort_by_key(|(module, slot, _)| (module.clone(), *slot));
    println!("{} binding(s) read", rows.len());

    // Split each module into maximal contiguous runs of 8-byte slots.
    let mut runs: Vec<(String, u32, Vec<String>)> = Vec::new();
    for (module, slot, symbol) in rows {
        match runs.last_mut() {
            Some((last_module, start, symbols))
                if *last_module == module
                    && *start + 8 * u32::try_from(symbols.len()).unwrap_or(u32::MAX) == slot =>
            {
                symbols.push(symbol);
            }
            _ => runs.push((module, slot, vec![symbol])),
        }
    }
    let mut per_module: BTreeMap<&str, usize> = BTreeMap::new();
    for (module, _, _) in &runs {
        *per_module.entry(module.as_str()).or_default() += 1;
    }
    println!(
        "{} module(s) in {} contiguous run(s) -> {} descriptor(s)",
        per_module.len(),
        runs.len(),
        runs.len()
    );
    for (module, count) in per_module.iter().filter(|(_, c)| **c > 1) {
        println!("  {module} is split across {count} runs");
    }

    let specifications: Vec<ImportModuleSpec> = runs
        .iter()
        .map(|(module, _, symbols)| ImportModuleSpec {
            name: module.clone(),
            symbols: symbols
                .iter()
                .map(|name| ImportTarget::Name {
                    name: name.clone(),
                    hint: 0,
                })
                .collect(),
        })
        .collect();

    let section_rva = next_section_rva(&bytes).expect("next rva");
    let table = serialize_import_table(pe.optional.kind, section_rva, &specifications)
        .expect("serialize");
    println!(
        "serialized {} byte(s) at {section_rva:?}; import dir {:?} size {:#x}",
        table.bytes.len(),
        table.import_directory.rva,
        table.import_directory.size
    );

    // Redirect each descriptor's FirstThunk at its `.idata` run.
    let mut payload = table.bytes.clone();
    for (index, (module, start, symbols)) in runs.iter().enumerate() {
        let at = index * DESCRIPTOR_SIZE + FIRST_THUNK_FIELD;
        let Some(slot) = payload.get_mut(at..at + 4) else {
            println!("  descriptor {index} is outside the payload");
            return;
        };
        slot.copy_from_slice(&start.to_le_bytes());
        if index < 3 {
            println!(
                "  descriptor {index}: {module} FirstThunk -> {start:#x} ({} symbols)",
                symbols.len()
            );
        }
    }

    let rebuilt = append_section(
        &bytes,
        AppendSection {
            name: *b".vmpximp",
            data: &payload,
            characteristics: RW,
        },
    )
    .expect("append");
    assert_eq!(rebuilt.section.rva.0, section_rva.0, "section moved");
    let mut image = rebuilt.bytes;

    // Install directory 1 ourselves. `install_import_directories` reparses and
    // demands the parsed table match the serializer's own layout exactly, which
    // the FirstThunk redirect deliberately breaks -- so the check below is
    // substituted for it: the reparsed table must name every binding and bind
    // it to the `.idata` slot it came from.
    let pe_offset = u32::from_le_bytes(image[0x3c..0x40].try_into().expect("4")) as usize;
    let directories = pe_offset + 24 + 112;
    let put = |image: &mut Vec<u8>, index: usize, rva: u32, size: u32| {
        let at = directories + index * 8;
        image[at..at + 4].copy_from_slice(&rva.to_le_bytes());
        image[at + 4..at + 8].copy_from_slice(&size.to_le_bytes());
    };
    put(
        &mut image,
        IMPORT_DIRECTORY,
        table.import_directory.rva.0,
        table.import_directory.size,
    );
    // The IAT directory is advisory; describe the `.idata` span the loader now
    // writes into, rather than the serializer's dead internal table.
    let low = runs.iter().map(|(_, start, _)| *start).min().unwrap_or(0);
    let high = runs
        .iter()
        .map(|(_, start, symbols)| start + 8 * u32::try_from(symbols.len()).unwrap_or(0))
        .max()
        .unwrap_or(0);
    put(&mut image, IAT_DIRECTORY, low, high.saturating_sub(low));
    println!("IAT directory -> {low:#x}..{high:#x}");

    // Verify by reparse: every binding present, bound to its original slot.
    let reparsed = PeImage::parse(&image).expect("reparse");
    let report = reparsed.report();
    let mut bound: BTreeMap<u32, (String, String)> = BTreeMap::new();
    for module in &report.imports {
        for symbol in &module.entries {
            if let Some(name) = &symbol.name {
                bound.insert(symbol.iat_rva.0, (module.name.clone(), name.clone()));
            }
        }
    }
    let mut ok = 0_usize;
    let mut wrong: Vec<String> = Vec::new();
    for (module, start, symbols) in &runs {
        for (index, symbol) in symbols.iter().enumerate() {
            let slot = start + 8 * u32::try_from(index).unwrap_or(0);
            match bound.get(&slot) {
                Some((m, s)) if m == module && s == symbol => ok += 1,
                other => wrong.push(format!("{slot:#x}: expected {module}!{symbol}, got {other:?}")),
            }
        }
    }
    println!(
        "reparse: {} module(s), {} symbol(s); bound to the original slot: {ok}, wrong: {}",
        report.imports.len(),
        report.imports.iter().map(|m| m.entries.len()).sum::<usize>(),
        wrong.len()
    );
    for entry in wrong.iter().take(6) {
        println!("  {entry}");
    }
    match validate_for_loader(&image) {
        Ok(report) => println!(
            "loader validation passed: {} sections, {} imports, {} modules",
            report.sections, report.import_symbols, report.import_modules
        ),
        Err(error) => println!("LOADER VALIDATION FAILED: {error}"),
    }
    std::fs::write(&args[3], &image).expect("write");
    println!("wrote {}", args[3]);
}
