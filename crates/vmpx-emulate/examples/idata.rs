// Is the application's import table recoverable from a post-init snapshot?
//
// `.idata` is ENCRYPTED in the file, and the application calls its imports
// through it -- 18272 `call [rip+slot]` sites over 1018 distinct slots. The
// protector's init decrypts and fills them, so the file alone cannot say what
// any slot means. A post-init snapshot can.
//
// Under the emulator every imported symbol gets a sentinel at
// `STUB_REGION + index * 8`, and the index is the symbol's position in the
// image's own flattened import list. So a filled slot holding a sentinel names
// its import exactly, with no address-to-export guessing and no dependence on
// the host's module layout.
//
// Usage: idata IMAGE SNAPSHOT

use vmpx_emulate::snapshot;
use vmpx_pe::{PeImage, Rva};

const STUB_REGION: u64 = 0x0000_0020_0000_0000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let base = pe.optional.image_base.get();
    let report = pe.report();

    // The flattened import order the loader assigns sentinels in.
    let mut names: Vec<(String, String)> = Vec::new();
    for module in &report.imports {
        for sym in &module.entries {
            let func = sym
                .name
                .clone()
                .unwrap_or_else(|| format!("#{}", sym.ordinal.unwrap_or(0)));
            names.push((module.name.clone(), func));
        }
    }
    println!("image declares {} imported symbol(s)", names.len());

    let (cpu, mem) = snapshot::load(std::path::Path::new(&args[2])).expect("snapshot");
    // Where had the protector's init reached when this was captured? If RIP is
    // in application code, the snapshot has already passed the hand-off and the
    // real module entry is observable rather than merely unresolved.
    let where_is = |va: u64| -> String {
        va.checked_sub(base)
            .and_then(|v| u32::try_from(v).ok())
            .and_then(|rva| {
                pe.sections
                    .iter()
                    .find(|s| s.contains_mapped_rva(Rva(rva)))
                    .map(|s| format!("{rva:#x} ({})", s.name.as_str()))
            })
            .unwrap_or_else(|| format!("{va:#x} (outside the image)"))
    };
    println!("snapshot rip: {}", where_is(cpu.rip));
    for (name, value) in [("rsp", cpu.regs[4]), ("rcx", cpu.regs[1]), ("rdx", cpu.regs[2])] {
        println!("  {name} = {}", where_is(value));
    }

    // Which .idata slots does the application actually CALL through? Byte-scan
    // every executable section for `call/jmp [rip+disp32]`; the answer is what
    // a rebuilt import table has to bind, and anything resolved beyond it is
    // slack rather than coverage.
    let mut called: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut call_sites = 0_usize;
    for section in pe.sections.iter().filter(|s| s.characteristics & 0x2000_0000 != 0) {
        let Ok(offset) = pe.rva_to_file_offset(section.virtual_address) else {
            continue;
        };
        let from = offset.0 as usize;
        let to = from
            .saturating_add(section.size_of_raw_data as usize)
            .min(bytes.len());
        let Some(blob) = bytes.get(from..to) else { continue };
        for at in 0..blob.len().saturating_sub(6) {
            if blob[at] != 0xFF || !matches!(blob[at + 1], 0x15 | 0x25) {
                continue;
            }
            let disp = i32::from_le_bytes(blob[at + 2..at + 6].try_into().expect("4"));
            let here = section.virtual_address.0 + u32::try_from(at).unwrap_or(0);
            let target = i64::from(here) + 6 + i64::from(disp);
            // A real IAT slot is 8-byte aligned; an unaligned hit is an `FF 15`
            // byte pair inside a longer instruction, not a call.
            if let Ok(rva) = u32::try_from(target)
                && rva % 8 == 0
            {
                called.insert(rva);
                call_sites += 1;
            }
        }
    }

    for section in &pe.sections {
        if section.name.as_str() != ".idata" {
            continue;
        }
        let start = section.virtual_address.0;
        let size = section.virtual_size.max(section.size_of_raw_data);
        println!(".idata rva {start:#x} size {size:#x} ({} slots)", size / 8);

        let (mut stub, mut zero, mut in_image, mut other, mut unmapped) = (0, 0, 0, 0, 0);
        let mut samples: Vec<String> = Vec::new();
        let mut resolved: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for offset in (0..size).step_by(8) {
            let rva = start + offset;
            match mem.read_uint(base + u64::from(rva), 8) {
                Err(_) => unmapped += 1,
                Ok(0) => zero += 1,
                Ok(value) => {
                    let index = value
                        .checked_sub(STUB_REGION)
                        .filter(|d| d % 8 == 0)
                        .map(|d| (d / 8) as usize)
                        .filter(|i| *i < names.len());
                    if let Some(index) = index {
                        stub += 1;
                        resolved.insert(rva);
                        if samples.len() < 12 {
                            let (module, func) = &names[index];
                            samples.push(format!("{rva:#x} -> {module}!{func}"));
                        }
                    } else if value
                        .checked_sub(base)
                        .and_then(|v| u32::try_from(v).ok())
                        .is_some_and(|r| pe.sections.iter().any(|s| s.contains_mapped_rva(Rva(r))))
                    {
                        in_image += 1;
                    } else {
                        other += 1;
                        if samples.len() < 24 {
                            samples.push(format!("{rva:#x} -> {value:#x} (unclassified)"));
                        }
                    }
                }
            }
        }
        println!("  resolved to a named import stub : {stub}");
        println!("  zero                            : {zero}");
        println!("  points inside the image         : {in_image}");
        println!("  other                           : {other}");
        println!("  not mapped in the snapshot      : {unmapped}");
        for sample in &samples {
            println!("    {sample}");
        }
        // Can a real import directory bind these slots in place? The loader
        // fills a run of slots starting at a descriptor's FirstThunk and ending
        // at a null, so each module's slots must be CONTIGUOUS. `.idata` is
        // encrypted in the file, so the run boundaries are not readable there --
        // this infers them from the resolved slots instead.
        let mut by_module: std::collections::BTreeMap<String, Vec<u32>> =
            std::collections::BTreeMap::new();
        for offset in (0..size).step_by(8) {
            let rva = start + offset;
            if let Ok(value) = mem.read_uint(base + u64::from(rva), 8)
                && let Some(index) = value
                    .checked_sub(STUB_REGION)
                    .filter(|d| d % 8 == 0)
                    .map(|d| (d / 8) as usize)
                    .filter(|i| *i < names.len())
            {
                by_module.entry(names[index].0.clone()).or_default().push(rva);
            }
        }
        let mut contiguous = 0_usize;
        let mut broken: Vec<String> = Vec::new();
        for (module, slots) in &by_module {
            let spans_only_itself = slots
                .windows(2)
                .all(|pair| pair[1] == pair[0] + 8);
            if spans_only_itself {
                contiguous += 1;
            } else {
                let runs = 1 + slots.windows(2).filter(|p| p[1] != p[0] + 8).count();
                broken.push(format!("{module}: {} slots in {runs} runs", slots.len()));
            }
        }
        println!();
        println!("  modules: {}", by_module.len());
        println!("  with all slots in ONE contiguous run : {contiguous}");
        println!("  split across several runs            : {}", broken.len());
        for entry in broken.iter().take(8) {
            println!("    {entry}");
        }

        // Bank the recovered table. Every downstream step -- in-place binding
        // or a fresh IAT plus repointing -- needs exactly these triples, and
        // they cost a 27-billion-step snapshot to obtain, so they are written
        // out rather than recomputed.
        if let Some(path) = std::env::args().nth(3) {
            let mut out = String::from("# slot_rva	module	symbol
");
            for (module, slots) in &by_module {
                for slot in slots {
                    let index = mem
                        .read_uint(base + u64::from(*slot), 8)
                        .ok()
                        .and_then(|v| v.checked_sub(STUB_REGION))
                        .map(|d| (d / 8) as usize);
                    let symbol = index
                        .and_then(|i| names.get(i))
                        .map_or("?", |(_, func)| func.as_str());
                    out.push_str(&format!("{slot:#x}	{module}	{symbol}
"));
                }
            }
            match std::fs::write(&path, out.as_bytes()) {
                Ok(()) => println!("  wrote {} binding(s) to {path}", by_module.values().map(Vec::len).sum::<usize>()),
                Err(error) => println!("  could not write {path}: {error}"),
            }
        }

        let in_section = |r: &u32| *r >= start && *r < start + size;
        let want: std::collections::BTreeSet<u32> =
            called.iter().copied().filter(in_section).collect();
        let covered = want.intersection(&resolved).count();
        println!();
        println!("  .idata slots the application calls through : {}", want.len());
        println!("  of those, resolved by the snapshot         : {covered}");
        println!("  call sites those slots serve               : {call_sites}");
        let missing: Vec<String> = want
            .difference(&resolved)
            .take(10)
            .map(|r| format!("{r:#x}"))
            .collect();
        if missing.is_empty() {
            println!("  EVERY called slot is named: the table is fully rebuildable");
        } else {
            println!("  called but unresolved: {}", missing.join(" "));
        }
    }
}
