//! `vmpx auto`: one command that takes a mapped VMProtect image to a clean PE.
//!
//! It analyses the binary, chooses its own settings, discovers the protector
//! references in the application's own code, and runs every stage -- directory
//! rebuild, strip, OEP recovery, native relocation, guard devirtualization,
//! import binding -- reporting each with structured progress logging.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use iced_x86::{Decoder, DecoderOptions, FlowControl, Register};
use vmpx_mutation::{CfgOptions, recover_native_cfg_pe_with};
use vmpx_pe::{PeImage, Rva};
use vmpx_rebuild::{
    AppendSection, append_section, directory_relocate::rebuild_all_directories, next_section_rva,
    retarget_direct_branch, retarget_rip_relative, set_entry_point, validate_for_loader,
};
use vmpx_vm::{VmDiscoveryLimits, analyze_vm_entry_at};

use crate::logkit::Log;

pub struct Options {
    pub input: PathBuf,
    /// Output image. When `dir` is set and this is relative, it is placed there.
    pub output: PathBuf,
    /// Directory for the output (and any future side artifacts); created if
    /// absent. When unset, the output path is used as given.
    pub dir: Option<PathBuf>,
    /// Post-init snapshot (VMPXSNP1) for import recovery; without it, imports
    /// stay as they are and that stage is skipped.
    pub snapshot: Option<PathBuf>,
    /// Print the full per-step trace (`--log`).
    pub verbose: bool,
}

const EXEC: u32 = 0x2000_0000;
const RX: u32 = 0x6000_0020;

/// A protector reference the application's own code makes.
struct Targets {
    /// VM entries reached from native code (guards) -> confidence.
    guards: Vec<u32>,
    /// Mutated native functions relocated into the protector.
    natives: Vec<u32>,
}

/// Every protector address the application's `.text` transfers into, classified.
fn discover(log: &Log, pe: &PeImage<'_>, bytes: &[u8]) -> Targets {
    let base = pe.optional.image_base.0;
    let soi = pe.optional.size_of_image;
    let protector = |rva: u32| {
        pe.sections
            .iter()
            .find(|s| s.contains_mapped_rva(Rva(rva)))
            .is_some_and(|s| s.name.as_str().starts_with(".vmp"))
    };
    // `.pdata` extents plus every executable section's uncovered gaps.
    let mut ranges: Vec<(u32, u32)> = pe
        .report()
        .exception_functions
        .iter()
        .map(|f| (f.begin.0, f.end.0))
        .filter(|(b, e)| e > b && e - b <= 0x10_0000)
        .collect();
    ranges.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for (b, e) in ranges.clone() {
        match merged.last_mut() {
            Some(last) if b <= last.1 => last.1 = last.1.max(e),
            _ => merged.push((b, e)),
        }
    }
    for section in pe.sections.iter().filter(|s| s.characteristics & EXEC != 0) {
        if section.name.as_str().starts_with(".vmp") {
            continue;
        }
        let start = section.virtual_address.0;
        let size = section.virtual_size.max(section.size_of_raw_data);
        let mut cursor = start;
        for (b, e) in &merged {
            if *e <= start || *b >= start + size {
                continue;
            }
            if *b > cursor {
                ranges.push((cursor, (*b).min(start + size)));
            }
            cursor = cursor.max(*e);
        }
        if cursor < start + size {
            ranges.push((cursor, start + size));
        }
    }

    let mut targets: BTreeSet<u32> = BTreeSet::new();
    for (begin, end) in ranges {
        let section_is_vmp = pe
            .sections
            .iter()
            .find(|s| s.contains_mapped_rva(Rva(begin)))
            .is_some_and(|s| s.name.as_str().starts_with(".vmp"));
        if section_is_vmp {
            continue;
        }
        let Ok(offset) = pe.rva_to_file_offset(Rva(begin)) else {
            continue;
        };
        let from = offset.0 as usize;
        let to = from.saturating_add((end - begin) as usize).min(bytes.len());
        let Some(code) = bytes.get(from..to) else { continue };
        let mut decoder = Decoder::with_ip(64, code, base + u64::from(begin), DecoderOptions::NONE);
        while decoder.can_decode() {
            let instruction = decoder.decode();
            if instruction.is_invalid() {
                break;
            }
            let mut mark = |va: u64| {
                if let Some(rva) = va.checked_sub(base).and_then(|v| u32::try_from(v).ok())
                    && rva < soi
                    && protector(rva)
                {
                    targets.insert(rva);
                }
            };
            if matches!(
                instruction.flow_control(),
                FlowControl::Call | FlowControl::UnconditionalBranch
            ) && instruction.near_branch64() != 0
            {
                mark(instruction.near_branch64());
            }
            if instruction.memory_base() == Register::RIP {
                // mov r,[slot] where slot holds a protector pointer.
                if let Some(slot) = instruction
                    .ip_rel_memory_address()
                    .checked_sub(base)
                    .and_then(|v| u32::try_from(v).ok())
                    && let Ok(off) = pe.rva_to_file_offset(Rva(slot))
                    && let Some(q) = bytes
                        .get(off.0 as usize..off.0 as usize + 8)
                        .and_then(|s| s.try_into().ok())
                        .map(u64::from_le_bytes)
                {
                    mark(q);
                }
            }
        }
    }

    // Classify: a VM entry (confidence >= 80) is a guard; a target whose native
    // CFG recovers but is not a VM entry is a mutated native.
    let options = CfgOptions {
        model_calls: true,
        ..CfgOptions::PROVEN
    };
    let (mut guards, mut natives) = (Vec::new(), Vec::new());
    for rva in targets {
        let candidate = analyze_vm_entry_at(pe, Rva(rva), VmDiscoveryLimits::default());
        if candidate.as_ref().is_some_and(|c| c.confidence >= 80) {
            guards.push(rva);
        } else if recover_native_cfg_pe_with(pe, Rva(rva), options).is_ok() {
            natives.push(rva);
        } else if candidate.is_some() {
            guards.push(rva); // a VM entry the scorer was unsure of
        }
    }
    log.info(&format!(
        "reached {} protector target(s): {} guard(s), {} mutated native(s)",
        guards.len() + natives.len(),
        guards.len(),
        natives.len()
    ));
    Targets { guards, natives }
}

/// `xor eax,eax; inc eax; ret` for a `jmp rel32` tail into removed space.
const RETURN_TRUE: [u8; 5] = [0x31, 0xC0, 0xFF, 0xC0, 0xC3];

/// The CRT-obligation OEP rule (loader-only root, takes the Win64 triple,
/// branches on the reason, forwards all three, not in a protector section);
/// link order breaks ties. Returns the recovered entry RVA.
fn recover_oep(pe: &PeImage<'_>, bytes: &[u8]) -> Option<u32> {
    use iced_x86::{Code, Mnemonic, OpKind};
    let base = pe.optional.image_base.0;
    let mut incoming: BTreeMap<u32, usize> = BTreeMap::new();
    for section in pe.sections.iter().filter(|s| s.characteristics & EXEC != 0) {
        let Ok(off) = pe.rva_to_file_offset(section.virtual_address) else {
            continue;
        };
        let from = off.0 as usize;
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
            if let Ok(r) = u32::try_from(i64::from(here) + 5 + i64::from(disp)) {
                *incoming.entry(r).or_default() += 1;
            }
        }
    }
    let crt_shape = |code: &[u8], va: u64| -> bool {
        let mut d = Decoder::with_ip(64, code, va, DecoderOptions::NONE);
        let mut a1: BTreeSet<Register> = [Register::RCX].into();
        let mut a2: BTreeSet<Register> = [Register::EDX, Register::RDX].into();
        let mut a3: BTreeSet<Register> = [Register::R8].into();
        let (mut branched, mut forwarded, mut staged) = (false, false, (false, false, false));
        while d.can_decode() {
            let ins = d.decode();
            if ins.is_invalid() {
                break;
            }
            let mov_rr = matches!(
                ins.code(),
                Code::Mov_r64_rm64 | Code::Mov_r32_rm32 | Code::Mov_rm64_r64 | Code::Mov_rm32_r32
            ) && ins.op0_kind() == OpKind::Register
                && ins.op1_kind() == OpKind::Register;
            if mov_rr {
                let (dst, src) = (ins.op0_register(), ins.op1_register());
                let f = |r: Register| r.full_register();
                for set in [&mut a1, &mut a2, &mut a3] {
                    if set.iter().any(|r| f(*r) == f(src)) {
                        set.insert(dst);
                    }
                }
                let from = |s: &BTreeSet<Register>| s.iter().any(|r| r.full_register() == src.full_register());
                match dst.full_register() {
                    Register::RCX if from(&a1) => staged.0 = true,
                    Register::RDX if from(&a2) => staged.1 = true,
                    Register::R8 if from(&a3) => staged.2 = true,
                    _ => {}
                }
            }
            if matches!(ins.mnemonic(), Mnemonic::Test | Mnemonic::Cmp)
                && a2.iter().any(|r| r.full_register() == ins.op0_register().full_register())
            {
                branched = true;
            }
            if ins.flow_control() == FlowControl::Call && ins.near_branch64() != 0 {
                if staged == (true, true, true) {
                    forwarded = true;
                }
                staged = (false, false, false);
            }
            if branched && forwarded {
                return true;
            }
        }
        false
    };
    let mut found: Vec<u32> = Vec::new();
    for function in &pe.report().exception_functions {
        let (b, e) = (function.begin.0, function.end.0);
        if e <= b || e - b > 0x800 || incoming.get(&b).copied().unwrap_or(0) != 0 {
            continue;
        }
        if pe.sections.iter().find(|s| s.contains_mapped_rva(Rva(b))).is_some_and(|s| s.name.as_str().starts_with(".vmp")) {
            continue;
        }
        let Ok(off) = pe.rva_to_file_offset(Rva(b)) else { continue };
        let from = off.0 as usize;
        let Some(code) = bytes.get(from..(from + (e - b) as usize).min(bytes.len())) else {
            continue;
        };
        if crt_shape(code, base + u64::from(b)) {
            found.push(b);
        }
    }
    found.into_iter().min()
}

/// The recovery settings `auto` turns on for itself: serve read-only image
/// bytes under suppression, and follow cross-`.vmp`-section continuations.
/// The caller no longer sets `VMPX_DECOMPILE_SERVE_RO` /
/// `VMPX_FOLLOW_CROSS_VMP_SECTION` by hand -- those env vars still work, and
/// this flag is ORed with them at every read site. A safe atomic, because
/// `std::env::set_var` is forbidden under edition 2024's `unsafe_code = forbid`.
pub(crate) static AUTO_SETTINGS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether `auto` mode turned the recovery settings on.
pub(crate) fn auto_settings() -> bool {
    AUTO_SETTINGS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn run(o: Options) -> Result<(), String> {
    let log = Log::new("vmpx", o.verbose);
    // `auto` IS the experimental path, so it enables the recovery settings
    // itself instead of making the caller remember env vars.
    AUTO_SETTINGS.store(true, std::sync::atomic::Ordering::Relaxed);
    // Resolve the output path under --dir, creating the directory if needed.
    let output = match &o.dir {
        Some(dir) => {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            if o.output.is_absolute() {
                o.output.clone()
            } else {
                dir.join(&o.output)
            }
        }
        None => o.output.clone(),
    };
    log.start(&format!("Devirtualizing {}", o.input.display()));

    let bytes = std::fs::read(&o.input).map_err(|e| e.to_string())?;
    let pe = PeImage::parse(&bytes).map_err(|e| e.to_string())?;
    let base = pe.optional.image_base.0;
    let report = vmpx_detect::detect_vmprotect(&pe);
    log.info(&format!(
        "{} sections, image base {base:#x}, {} import symbol(s)",
        pe.sections.len(),
        pe.report().imports.iter().map(|m| m.entries.len()).sum::<usize>()
    ));
    log.info(&format!("protection: {:?}", report.verdict));

    // --- auto settings ---
    log.start("Choosing settings");
    // Start at the smallest budget proven to decode a branch-complete guard
    // CFG, and escalate PER GUARD only where that falls short (the retry ladder
    // in `devirt_guards`). A larger blanket budget only slows the easy guards;
    // adapting where needed is both faster and "best" in the sense asked for.
    let instruction_limit = 2048_usize;
    log.step("instruction limit", &format!("{instruction_limit}, escalating per guard as needed"));
    log.step("image reads", "serve read-only under suppression (labeled)");
    log.step("checksum", "fold against the intact image");
    log.step("cross-section", "follow continuations across .vmp sections");

    // --- discover ---
    log.start("Analysing the application's protector references");
    let targets = discover(&log, &pe, &bytes);

    // --- conventional layout + directory rebuild + strip ---
    log.start("Rebuilding directories and stripping the protector");
    // The mapped image uses a memory layout (raw offset ~= RVA); convert it to a
    // conventional, file-aligned on-disk layout before anything is removed.
    let (conventional, _) =
        vmpx_rebuild::relayout_conventional(&bytes).map_err(|e| format!("relayout: {e}"))?;
    log.step("conventional layout", "file-aligned on-disk image");
    let laid = PeImage::parse(&conventional).map_err(|e| e.to_string())?;
    let removed: Vec<&str> = laid
        .sections
        .iter()
        .map(|s| s.name.as_str())
        .filter(|n| n.starts_with(".vmp"))
        .collect();
    let removed_owned: Vec<String> = removed.iter().map(|s| (*s).to_owned()).collect();
    let dirs = rebuild_all_directories(&conventional, *b".vmpxdir", &removed)
        .map_err(|e| format!("directory rebuild: {e}"))?;
    log.step(
        "directories relocated",
        &format!(
            "{} moved, {} imports, {} tls callback(s)",
            dirs.moved.len(),
            dirs.import_symbols,
            dirs.tls_callbacks.0
        ),
    );
    // Recover and set the OEP before stripping: the header entry is in `.vmp3`,
    // and the strip refuses an entry point that would dangle.
    let mut dir_bytes = dirs.bytes;
    match recover_oep(&pe, &bytes) {
        Some(oep) => {
            dir_bytes = set_entry_point(&dir_bytes, Rva(oep)).map_err(|e| e.to_string())?;
            log.step("entry point", &format!("{oep:#x} (CRT DllEntryPoint)"));
        }
        None => log.warn("no CRT entry recovered; the strip may refuse the header entry"),
    }
    // Strip ONLY the `.vmp*` sections by name -- not every non-standard
    // section, which would also take the `.vmpxdir` just rebuilt into.
    let selectors: Vec<vmpx_pe::SectionSelector> = removed_owned
        .iter()
        .map(|name| vmpx_pe::SectionSelector::Name(name.clone()))
        .collect();
    let (stripped, summary) = crate::strip_wiring::strip_and_validate(&dir_bytes, &selectors, &[])
        .map_err(|e| format!("strip refused: {e}"))?;
    log.step(
        "protector sections removed",
        &format!(
            "{} section(s), {:.1} MB",
            summary.removed.len(),
            summary.file_bytes_removed as f64 / (1024.0 * 1024.0)
        ),
    );
    let mut image = stripped;

    // --- relocate mutated natives ---
    if !targets.natives.is_empty() {
        log.start("Relocating mutated native functions");
        let (new_image, mapping) = relocate_natives(&log, &pe, &bytes, &image, &targets.natives)?;
        image = new_image;
        image = repoint(&log, &pe, &bytes, image, &mapping)?;
    }

    // --- devirtualize guards ---
    if !targets.guards.is_empty() {
        log.start("Devirtualizing guard functions");
        image = devirt_guards(&log, &pe, image, instruction_limit, &targets.guards)?;
    }

    // --- imports ---
    if let Some(snapshot) = &o.snapshot {
        log.start("Rebuilding the application import table");
        match crate::auto_imports::bind(&log, &pe, &bytes, &image, snapshot) {
            Ok(bound) => image = bound,
            Err(reason) => log.warn(&format!("import binding skipped: {reason}")),
        }
    } else {
        log.info("no snapshot given; import table left as-is (pass --snapshot to recover it)");
    }

    // --- verify + write ---
    log.start("Verifying and writing");
    let dangling = count_dangling(&image);
    if dangling == 0 {
        log.success("no reference into removed space remains");
    } else {
        log.warn(&format!("{dangling} reference(s) into removed space remain"));
    }
    match validate_for_loader(&image) {
        Ok(r) => log.step(
            "loader validation",
            &format!("{} sections, {} imports", r.sections, r.import_symbols),
        ),
        Err(e) => return Err(format!("loader validation failed: {e}")),
    }
    std::fs::write(&output, &image).map_err(|e| e.to_string())?;
    log.file(&output.display().to_string(), &format!("{} bytes", image.len()));
    log.success_detail(
        &format!("clean PE written, {dangling} residual reference(s)"),
        &log.total_elapsed(),
    );
    let _ = base;
    Ok(())
}

/// Count references from surviving code into removed (unmapped) space.
fn count_dangling(image: &[u8]) -> usize {
    let Ok(pe) = PeImage::parse(image) else { return 0 };
    let base = pe.optional.image_base.0;
    let soi = pe.optional.size_of_image;
    let mapped = |rva: u32| pe.sections.iter().any(|s| s.contains_mapped_rva(Rva(rva)));
    let mut count = 0;
    for function in &pe.report().exception_functions {
        let (b, e) = (function.begin.0, function.end.0);
        if e <= b || e - b > 0x10_0000 || !mapped(b) {
            continue;
        }
        let Ok(off) = pe.rva_to_file_offset(Rva(b)) else { continue };
        let from = off.0 as usize;
        let Some(code) = image.get(from..(from + (e - b) as usize).min(image.len())) else {
            continue;
        };
        let mut d = Decoder::with_ip(64, code, base + u64::from(b), DecoderOptions::NONE);
        while d.can_decode() {
            let ins = d.decode();
            if ins.is_invalid() {
                break;
            }
            let t = if matches!(
                ins.flow_control(),
                FlowControl::Call | FlowControl::UnconditionalBranch | FlowControl::ConditionalBranch
            ) && ins.near_branch64() != 0
            {
                Some(ins.near_branch64())
            } else if ins.memory_base() == Register::RIP {
                Some(ins.ip_rel_memory_address())
            } else {
                None
            };
            if let Some(va) = t
                && let Some(rva) = va.checked_sub(base).and_then(|v| u32::try_from(v).ok())
                && rva < soi
                && !mapped(rva)
            {
                count += 1;
            }
        }
    }
    count
}

/// Relocate each native component out of the protector into `.vmpxrel`.
fn relocate_natives(
    log: &Log,
    intact: &PeImage<'_>,
    intact_bytes: &[u8],
    stripped: &[u8],
    seeds: &[u32],
) -> Result<(Vec<u8>, BTreeMap<u32, u32>), String> {
    use std::collections::VecDeque;
    let base = intact.optional.image_base.0;
    let options = CfgOptions {
        model_calls: true,
        ..CfgOptions::PROVEN
    };
    let mut queue: VecDeque<u32> = seeds.iter().copied().collect();
    let mut seen: BTreeSet<u32> = seeds.iter().copied().collect();
    let mut blocks: BTreeMap<u64, u64> = BTreeMap::new();
    while let Some(rva) = queue.pop_front() {
        let cfg = recover_native_cfg_pe_with(intact, Rva(rva), options)
            .map_err(|e| format!("native cfg {rva:#x}: {e}"))?;
        for block in cfg.blocks.values() {
            blocks.insert(block.start, block.end);
            if let vmpx_mutation::Terminator::Call { target, .. }
            | vmpx_mutation::Terminator::TailCall { target } = &block.terminator
                && let vmpx_mutation::CallTarget::Direct(va) = target
                && let Some(next) = va.checked_sub(base).and_then(|v| u32::try_from(v).ok())
                && seen.insert(next)
            {
                queue.push_back(next);
            }
        }
    }
    let covered = |va: u64| blocks.range(..=va).next_back().is_some_and(|(_, end)| va < *end);
    let section_rva = next_section_rva(stripped).map_err(|e| e.to_string())?.0;
    let mut payload: Vec<u8> = Vec::new();
    let mut placement: BTreeMap<u64, u64> = BTreeMap::new();
    for (start, end) in &blocks {
        while payload.len() % 16 != 0 {
            payload.push(0xCC);
        }
        placement.insert(*start, base + u64::from(section_rva) + payload.len() as u64);
        let rva = u32::try_from(start - base).unwrap_or(0);
        let off = intact.rva_to_file_offset(Rva(rva)).map_err(|e| e.to_string())?.0 as usize;
        payload.extend_from_slice(
            intact_bytes
                .get(off..off + (end - start) as usize)
                .ok_or("block past file")?,
        );
    }
    let moved = |va: u64| -> Option<u64> {
        let (start, end) = blocks.range(..=va).next_back()?;
        (va < *end).then(|| placement[start] + (va - start))
    };
    // Fix every rel32/RIP displacement pointing back into the component.
    for (start, end) in &blocks {
        let new_start = placement[start];
        let len = (end - start) as usize;
        let at = (new_start - base - u64::from(section_rva)) as usize;
        let mut cursor = 0;
        while cursor < len {
            let window = &payload[at + cursor..at + len];
            let ins = Decoder::with_ip(64, window, start + cursor as u64, DecoderOptions::NONE).decode();
            if ins.is_invalid() {
                break;
            }
            let size = ins.len();
            let target = if matches!(
                ins.flow_control(),
                FlowControl::Call | FlowControl::UnconditionalBranch | FlowControl::ConditionalBranch
            ) && ins.near_branch64() != 0
            {
                Some(ins.near_branch64())
            } else if ins.memory_base() == Register::RIP {
                Some(ins.ip_rel_memory_address())
            } else {
                None
            };
            if let Some(t) = target
                && let Some(nt) = moved(t)
            {
                let new_ip = new_start + cursor as u64;
                let next = new_ip + size as u64;
                let value = i32::try_from(nt.cast_signed() - next.cast_signed())
                    .map_err(|_| "rel32 out of range")?;
                let branch = ins.near_branch64() != 0;
                let slot = &mut payload[at + cursor..at + cursor + size];
                let saved = slot.to_vec();
                let mut ok = false;
                for w in 0..size.saturating_sub(3) {
                    slot.copy_from_slice(&saved);
                    slot[w..w + 4].copy_from_slice(&value.to_le_bytes());
                    let re = Decoder::with_ip(64, slot, new_ip, DecoderOptions::NONE).decode();
                    let reaches = if branch {
                        re.near_branch64() == nt
                    } else {
                        re.ip_rel_memory_address() == nt
                    };
                    if !re.is_invalid() && re.len() == size && re.mnemonic() == ins.mnemonic() && reaches {
                        ok = true;
                        break;
                    }
                }
                if !ok {
                    slot.copy_from_slice(&saved);
                    return Err(format!("cannot rebase {start:#x}+{cursor}"));
                }
            } else if let Some(t) = target
                && !covered(t)
                && t.checked_sub(base).and_then(|v| u32::try_from(v).ok()).is_some_and(|r| {
                    intact.sections.iter().find(|s| s.contains_mapped_rva(Rva(r))).is_some_and(|s| s.name.as_str().starts_with(".vmp"))
                })
            {
                return Err(format!("component references removed space at {start:#x}"));
            }
            cursor += size;
        }
    }
    let rebuilt = append_section(
        stripped,
        AppendSection {
            name: *b".vmpxrel",
            data: &payload,
            characteristics: RX,
        },
    )
    .map_err(|e| e.to_string())?;
    let mut mapping = BTreeMap::new();
    for seed in seeds {
        if let Some(new) = moved(base + u64::from(*seed)) {
            mapping.insert(*seed, u32::try_from(new - base).unwrap_or(0));
        }
    }
    log.step(
        "relocated",
        &format!("{} function block(s), {} byte(s)", blocks.len(), payload.len()),
    );
    Ok((rebuilt.bytes, mapping))
}

/// Repoint `.text` slot-loads and direct transfers at relocated functions.
fn repoint(
    log: &Log,
    intact: &PeImage<'_>,
    intact_bytes: &[u8],
    image: Vec<u8>,
    mapping: &BTreeMap<u32, u32>,
) -> Result<Vec<u8>, String> {
    let base = intact.optional.image_base.0;
    let pe = PeImage::parse(&image).map_err(|e| e.to_string())?;
    let mapped = |rva: u32| pe.sections.iter().any(|s| s.contains_mapped_rva(Rva(rva)));
    let qword = |rva: u32| -> Option<u64> {
        let off = intact.rva_to_file_offset(Rva(rva)).ok()?;
        intact_bytes
            .get(off.0 as usize..off.0 as usize + 8)
            .and_then(|s| s.try_into().ok())
            .map(u64::from_le_bytes)
    };
    // One pointer slot per distinct new target, in a fresh section.
    let order: BTreeSet<u32> = mapping.values().copied().collect();
    let section_rva = next_section_rva(&image).map_err(|e| e.to_string())?.0;
    let mut payload = Vec::new();
    let mut slot_of = BTreeMap::new();
    for new in &order {
        slot_of.insert(*new, section_rva + u32::try_from(payload.len()).unwrap_or(0));
        payload.extend_from_slice(&(base + u64::from(*new)).to_le_bytes());
    }
    let mut out = append_section(
        &image,
        AppendSection {
            name: *b".vmpxptr",
            data: &payload,
            characteristics: 0x4000_0040,
        },
    )
    .map_err(|e| e.to_string())?
    .bytes;

    // Scan .pdata functions + gaps for references into the relocated targets.
    let (mut slots, mut direct) = (0_usize, 0_usize);
    let pe2 = PeImage::parse(&out).map_err(|e| e.to_string())?;
    let mut ranges: Vec<(u32, u32)> = pe2
        .report()
        .exception_functions
        .iter()
        .map(|f| (f.begin.0, f.end.0))
        .filter(|(b, e)| e > b && e - b <= 0x10_0000 && mapped(*b))
        .collect();
    ranges.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for (b, e) in ranges.clone() {
        match merged.last_mut() {
            Some(l) if b <= l.1 => l.1 = l.1.max(e),
            _ => merged.push((b, e)),
        }
    }
    for s in pe2.sections.iter().filter(|s| s.characteristics & EXEC != 0) {
        let (start, size) = (s.virtual_address.0, s.virtual_size.max(s.size_of_raw_data));
        let mut cur = start;
        for (b, e) in &merged {
            if *e <= start || *b >= start + size {
                continue;
            }
            if *b > cur {
                ranges.push((cur, (*b).min(start + size)));
            }
            cur = cur.max(*e);
        }
        if cur < start + size {
            ranges.push((cur, start + size));
        }
    }
    let mut edits: Vec<(u32, u32, bool)> = Vec::new(); // (site, slot/target, is_direct)
    for (begin, end) in ranges {
        if !mapped(begin) {
            continue;
        }
        let Ok(off) = pe2.rva_to_file_offset(Rva(begin)) else { continue };
        let from = off.0 as usize;
        let Some(code) = out.get(from..(from + (end - begin) as usize).min(out.len())) else {
            continue;
        };
        let mut d = Decoder::with_ip(64, code, base + u64::from(begin), DecoderOptions::NONE);
        while d.can_decode() {
            let at = d.position();
            let ins = d.decode();
            if ins.is_invalid() {
                break;
            }
            let site = begin + u32::try_from(at).unwrap_or(0);
            if matches!(ins.flow_control(), FlowControl::Call | FlowControl::UnconditionalBranch)
                && ins.near_branch64() != 0
                && let Some(t) = ins.near_branch64().checked_sub(base).and_then(|v| u32::try_from(v).ok())
                && let Some(new) = mapping.get(&t)
            {
                edits.push((site, *new, true));
            }
            if ins.memory_base() == Register::RIP
                && let Some(slot) = ins.ip_rel_memory_address().checked_sub(base).and_then(|v| u32::try_from(v).ok())
                && slot < pe2.optional.size_of_image
                && !mapped(slot)
                && let Some(t) = qword(slot).and_then(|q| q.checked_sub(base)).and_then(|v| u32::try_from(v).ok())
                && let Some(new) = mapping.get(&t)
            {
                edits.push((site, slot_of[new], false));
            }
        }
    }
    for (site, target, is_direct) in edits {
        out = if is_direct {
            direct += 1;
            retarget_direct_branch(&out, Rva(site), Rva(target)).map_err(|e| e.to_string())?
        } else {
            slots += 1;
            retarget_rip_relative(&out, Rva(site), Rva(target)).map_err(|e| e.to_string())?
        };
    }
    log.step("repointed", &format!("{slots} slot load(s), {direct} direct transfer(s)"));
    Ok(out)
}

/// Devirtualize each guard and retarget its `.text` transfer.
fn devirt_guards(
    log: &Log,
    intact: &PeImage<'_>,
    image: Vec<u8>,
    instruction_limit: usize,
    guards: &[u32],
) -> Result<Vec<u8>, String> {
    let base = intact.optional.image_base.0;
    let section_rva = next_section_rva(&image).map_err(|e| e.to_string())?.0;
    let mut payload: Vec<u8> = Vec::new();
    let mut mapping: BTreeMap<u32, u32> = BTreeMap::new();
    let (mut ok, mut skipped) = (0_usize, 0_usize);
    for &rva in guards {
        while payload.len() % 16 != 0 {
            payload.push(0xCC);
        }
        let new_rva = section_rva + u32::try_from(payload.len()).unwrap_or(0);
        let va = base + u64::from(new_rva);
        // Try the chosen budget, then escalate ONLY this guard if it fell short
        // of a complete CFG -- an incomplete-CFG or limit-hit error is the only
        // kind more budget can fix, so a nested-entry or checksum-pointer
        // refusal is not retried.
        let mut result = crate::devirt_cmd::build_one(intact, rva, instruction_limit, va);
        let mut used = instruction_limit;
        while let Err(reason) = &result {
            let budget_bound = reason.contains("incomplete") || reason.contains("limit");
            if !budget_bound || used >= 65536 {
                break;
            }
            used *= 2;
            result = crate::devirt_cmd::build_one(intact, rva, used, va);
        }
        match result {
            Ok(bytes) => {
                let detail = if used == instruction_limit {
                    format!("{} byte(s)", bytes.len())
                } else {
                    format!("{} byte(s), budget {used}", bytes.len())
                };
                log.step(&format!("{rva:#x} devirtualized"), &detail);
                payload.extend_from_slice(&bytes);
                mapping.insert(rva, new_rva);
                ok += 1;
            }
            Err(reason) => {
                skipped += 1;
                log.warn(&format!("{rva:#x}: {}", reason.lines().next().unwrap_or(&reason)));
            }
        }
    }
    log.step("devirtualized", &format!("{ok} of {} guard(s), {} byte(s)", guards.len(), payload.len()));
    if mapping.is_empty() {
        let _ = skipped;
        return Ok(image);
    }
    let rebuilt = append_section(
        &image,
        AppendSection {
            name: *b".vmpxdv\0",
            data: &payload,
            characteristics: RX,
        },
    )
    .map_err(|e| e.to_string())?;
    let mut out = rebuilt.bytes;
    let retargeted = crate::devirt_cmd::retarget_sites(&mut out, &mapping)?;
    log.step("retargeted", &format!("{retargeted} transfer(s)"));
    Ok(out)
}
