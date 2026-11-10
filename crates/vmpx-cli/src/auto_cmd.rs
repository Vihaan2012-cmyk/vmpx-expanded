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
use vmpx_unpack::{UnpackLimits, locate_packer_table, unpack_to_mapped_image};
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
    /// Print the full per-step trace on the terminal (`--log`). The file
    /// transcript is always full regardless.
    pub verbose: bool,
    /// Where to write the full transcript; defaults to the output with a `.log`
    /// extension.
    pub log_file: Option<PathBuf>,
    /// Treat the input as a packed on-disk `.dll`/`.exe`: unpack it to a mapped
    /// analysis image first.
    pub unpack: bool,
    /// Never prompt; assume "no" for every optional step (for scripts).
    pub yes_to_all_defaults: bool,
    /// Detonate to recover imports when no snapshot is given, running up to this
    /// many instructions. `None` leaves detonation to the interactive prompt.
    pub detonate_steps: Option<u64>,
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
        let Some(code) = bytes.get(from..to) else {
            continue;
        };
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
            log.trace(
                &format!("{rva:#x}"),
                &format!(
                    "guard (VM entry, confidence {})",
                    candidate.as_ref().map_or(0, |c| c.confidence)
                ),
            );
            guards.push(rva);
        } else if recover_native_cfg_pe_with(pe, Rva(rva), options).is_ok() {
            log.trace(&format!("{rva:#x}"), "mutated native");
            natives.push(rva);
        } else if candidate.is_some() {
            log.trace(&format!("{rva:#x}"), "guard (VM entry, low confidence)");
            guards.push(rva); // a VM entry the scorer was unsure of
        } else {
            log.trace(
                &format!("{rva:#x}"),
                "unclassified protector target (left as-is)",
            );
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

/// One residual guard thunk neutralized to its pass path.
pub(crate) struct GuardNeutralized {
    pub guard_rva: u32,
    pub thunk_begin: u32,
    pub thunk_bytes: u32,
    /// How it was neutralized: a whole `.pdata` thunk body ("thunk"), or a single
    /// direct branch instruction in an uncovered gap ("site").
    pub via: &'static str,
}

/// Neutralize residual guard functions devirtualization could not lift. Each is
/// a `.text` thunk whose body tail-transfers into the removed protector to run
/// an anti-tamper / integrity / timing guard, leaving a dangling reference. The
/// thunk's own function body is overwritten with `mov eax,1; ret` (then `int3`
/// filler), so it returns the guard's pass value to its caller with a pristine
/// stack -- no prologue has executed yet, so no teardown is needed. This both
/// removes the dangling transfer and forces the guard's clean path.
///
/// It is deliberately NON-equivalent (it changes behavior under tampering), so
/// every neutralized site is returned for the caller to log. A guard reached
/// only through a transfer `.pdata` does not describe (e.g. the thunk table) is
/// not found here and is left untouched.
pub(crate) fn neutralize_residual_guards(
    image: &mut [u8],
    guard_rvas: &[u32],
) -> Result<Vec<GuardNeutralized>, String> {
    // (guard_rva, function begin rva, begin file offset, END file offset through
    // the tail transfer). Only the bytes from the function entry through the
    // transfer are overwritten -- the entry runs before any prologue, so a
    // `ret` there needs no teardown, and the transfer is the last byte that must
    // be erased. Bytes after it (which `.pdata` may fold into this function but
    // the tail transfer never reaches) are left intact.
    let mut hits: Vec<(u32, u32, usize, usize)> = Vec::new();
    {
        let pe = PeImage::parse(image).map_err(|e| e.to_string())?;
        let base = pe.optional.image_base.0;
        let targets: BTreeSet<u64> = guard_rvas.iter().map(|r| base + u64::from(*r)).collect();
        for function in &pe.report().exception_functions {
            let (b, e) = (function.begin.0, function.end.0);
            if e <= b || e - b > 0x10_0000 {
                continue;
            }
            let Ok(off) = pe.rva_to_file_offset(Rva(b)) else {
                continue;
            };
            let from = off.0 as usize;
            let to = (from + (e - b) as usize).min(image.len());
            let Some(code) = image.get(from..to) else {
                continue;
            };
            let mut decoder = Decoder::with_ip(64, code, base + u64::from(b), DecoderOptions::NONE);
            while decoder.can_decode() {
                let instruction = decoder.decode();
                if instruction.is_invalid() {
                    break;
                }
                if matches!(
                    instruction.flow_control(),
                    FlowControl::Call | FlowControl::UnconditionalBranch
                ) && instruction.near_branch64() != 0
                    && targets.contains(&instruction.near_branch64())
                {
                    let guard_rva = u32::try_from(instruction.near_branch64() - base).unwrap_or(0);
                    let transfer_end = from
                        + usize::try_from(instruction.next_ip() - (base + u64::from(b)))
                            .unwrap_or(to - from);
                    hits.push((guard_rva, b, from, transfer_end.min(to)));
                    break;
                }
            }
        }
    }
    Ok(apply_guard_neutralizations(image, hits))
}

/// Self-contained neutralization for a reintegrated image: find every `.text`
/// thunk that tail-transfers into removed/unmapped space (a residual guard whose
/// body could not be lifted) and force its pass path, exactly as
/// [`neutralize_residual_guards`] does, but without needing the guard RVA list.
/// Used by the `clean --experimental` output, which does not carry that list.
pub(crate) fn neutralize_dangling_guard_thunks(
    image: &mut [u8],
) -> Result<Vec<GuardNeutralized>, String> {
    let mut hits: Vec<(u32, u32, usize, usize)> = Vec::new();
    {
        let pe = PeImage::parse(image).map_err(|e| e.to_string())?;
        let base = pe.optional.image_base.0;
        let soi = pe.optional.size_of_image;
        let mapped = |rva: u32| pe.sections.iter().any(|s| s.contains_mapped_rva(Rva(rva)));
        for function in &pe.report().exception_functions {
            let (b, e) = (function.begin.0, function.end.0);
            if e <= b || e - b > 0x10_0000 {
                continue;
            }
            let Ok(off) = pe.rva_to_file_offset(Rva(b)) else {
                continue;
            };
            let from = off.0 as usize;
            let to = (from + (e - b) as usize).min(image.len());
            let Some(code) = image.get(from..to) else {
                continue;
            };
            let mut decoder = Decoder::with_ip(64, code, base + u64::from(b), DecoderOptions::NONE);
            while decoder.can_decode() {
                let instruction = decoder.decode();
                if instruction.is_invalid() {
                    break;
                }
                if matches!(
                    instruction.flow_control(),
                    FlowControl::Call | FlowControl::UnconditionalBranch
                ) && instruction.near_branch64() != 0
                    && let Some(rva) = instruction
                        .near_branch64()
                        .checked_sub(base)
                        .and_then(|v| u32::try_from(v).ok())
                    && rva < soi
                    && !mapped(rva)
                {
                    let transfer_end = from
                        + usize::try_from(instruction.next_ip() - (base + u64::from(b)))
                            .unwrap_or(to - from);
                    hits.push((rva, b, from, transfer_end.min(to)));
                    break;
                }
            }
        }
    }
    Ok(apply_guard_neutralizations(image, hits))
}

/// Overwrite each hit's `[entry .. transfer_end)` with `mov eax,1; ret` plus
/// `int3` filler, so the thunk returns "pass" with a pristine stack and its
/// dangling transfer is erased. Shared by both neutralization scanners.
fn apply_guard_neutralizations(
    image: &mut [u8],
    hits: Vec<(u32, u32, usize, usize)>,
) -> Vec<GuardNeutralized> {
    let mut done: Vec<GuardNeutralized> = Vec::new();
    let mut overwritten: BTreeSet<u32> = BTreeSet::new();
    for (guard_rva, begin, from, end) in hits {
        if !overwritten.insert(begin) {
            continue;
        }
        if from + RETURN_TRUE.len() > end || end > image.len() {
            continue;
        }
        image[from..from + RETURN_TRUE.len()].copy_from_slice(&RETURN_TRUE);
        for byte in &mut image[from + RETURN_TRUE.len()..end] {
            *byte = 0xCC;
        }
        done.push(GuardNeutralized {
            guard_rva,
            thunk_begin: begin,
            thunk_bytes: u32::try_from(end - from).unwrap_or(0),
            via: "thunk",
        });
    }
    done
}

/// `mov eax, 1` -- the guard's pass value, written over a `call` site in place.
const MOV_EAX_1: [u8; 5] = [0xB8, 0x01, 0x00, 0x00, 0x00];

/// Neutralize a residual guard reached by a direct branch that `.pdata` does not
/// describe -- a branch sitting in an executable *gap*, which
/// [`neutralize_residual_guards`] (which scans only `.pdata` function bodies)
/// cannot see, and which `count_dangling` likewise misses. The guard VA now
/// points into removed space, so the branch is a live dangling reference; patch
/// the 5-byte branch instruction in place so the clean (pass) path is taken
/// without entering the removed guard:
///   - a tail `jmp`  -> `xor eax,eax; inc eax; ret` (return 1 to the original caller)
///   - a `call`      -> `mov eax, 1`                (set the pass value, fall through)
/// Both are exactly 5 bytes, the width of a near rel32 branch, so only the branch
/// is overwritten. Behaviour-changing (forces the anti-tamper pass), same as the
/// thunk path; every site is returned for the caller to log.
pub(crate) fn neutralize_residual_guard_sites(
    image: &mut [u8],
    guard_rvas: &[u32],
) -> Result<Vec<GuardNeutralized>, String> {
    if guard_rvas.is_empty() {
        return Ok(Vec::new());
    }
    // (guard_rva, site rva, branch file offset, is the branch a `call`?)
    let mut hits: Vec<(u32, u32, usize, bool)> = Vec::new();
    {
        let pe = PeImage::parse(image).map_err(|e| e.to_string())?;
        let base = pe.optional.image_base.0;
        let mapped = |rva: u32| pe.sections.iter().any(|s| s.contains_mapped_rva(Rva(rva)));
        let targets: BTreeSet<u64> = guard_rvas.iter().map(|r| base + u64::from(*r)).collect();

        // `.pdata` extents merged with each executable section's uncovered gaps --
        // the same coverage `retarget_sites` walks, so a branch in a gap is seen.
        for (begin, end) in executable_scan_ranges(&pe) {
            if !mapped(begin) {
                continue;
            }
            let Ok(off) = pe.rva_to_file_offset(Rva(begin)) else {
                continue;
            };
            let from = off.0 as usize;
            let to = from.saturating_add((end - begin) as usize).min(image.len());
            let Some(code) = image.get(from..to) else {
                continue;
            };
            let mut decoder = Decoder::with_ip(64, code, base + u64::from(begin), DecoderOptions::NONE);
            while decoder.can_decode() {
                let at = decoder.position();
                let instruction = decoder.decode();
                if instruction.is_invalid() {
                    break;
                }
                // Only a 5-byte near rel32 branch (E8/E9) can be patched in place;
                // the guards sit far away, so every real site is exactly that.
                if matches!(
                    instruction.flow_control(),
                    FlowControl::Call | FlowControl::UnconditionalBranch
                ) && instruction.near_branch64() != 0
                    && targets.contains(&instruction.near_branch64())
                    && instruction.len() == 5
                {
                    let guard_rva = u32::try_from(instruction.near_branch64() - base).unwrap_or(0);
                    let site_rva = begin + u32::try_from(at).unwrap_or(0);
                    let is_call = instruction.flow_control() == FlowControl::Call;
                    hits.push((guard_rva, site_rva, from + at, is_call));
                }
            }
        }
    }

    let mut done: Vec<GuardNeutralized> = Vec::new();
    for (guard_rva, site_rva, from, is_call) in hits {
        if from + 5 > image.len() {
            continue;
        }
        let patch = if is_call { MOV_EAX_1 } else { RETURN_TRUE };
        image[from..from + 5].copy_from_slice(&patch);
        done.push(GuardNeutralized {
            guard_rva,
            thunk_begin: site_rva,
            thunk_bytes: 5,
            via: "site",
        });
    }
    Ok(done)
}

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
        let Some(blob) = bytes.get(from..to) else {
            continue;
        };
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
                let from = |s: &BTreeSet<Register>| {
                    s.iter().any(|r| r.full_register() == src.full_register())
                };
                match dst.full_register() {
                    Register::RCX if from(&a1) => staged.0 = true,
                    Register::RDX if from(&a2) => staged.1 = true,
                    Register::R8 if from(&a3) => staged.2 = true,
                    _ => {}
                }
            }
            if matches!(ins.mnemonic(), Mnemonic::Test | Mnemonic::Cmp)
                && a2
                    .iter()
                    .any(|r| r.full_register() == ins.op0_register().full_register())
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
        if pe
            .sections
            .iter()
            .find(|s| s.contains_mapped_rva(Rva(b)))
            .is_some_and(|s| s.name.as_str().starts_with(".vmp"))
        {
            continue;
        }
        let Ok(off) = pe.rva_to_file_offset(Rva(b)) else {
            continue;
        };
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

/// Ask a yes/no question. Returns `default` without prompting when
/// `no_prompt` is set or stdin is not a terminal (so scripts are deterministic).
fn confirm(log: &Log, question: &str, default: bool, no_prompt: bool) -> bool {
    use std::io::{IsTerminal, Write};
    if no_prompt || !std::io::stdin().is_terminal() {
        log.info(&format!(
            "{question} [{}] (assumed)",
            if default { "Y/n" } else { "y/N" }
        ));
        return default;
    }
    let hint = if default { "[Y/n]" } else { "[y/N]" };
    print!("    ? {question} {hint} ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return default;
    }
    match line.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => true,
        "n" | "no" => false,
        _ => default,
    }
}

/// Whether two paths denote the same file, tolerating one that does not exist
/// yet (the log path): it is resolved through its parent directory. Fails closed
/// to "not the same" only when neither resolution is possible.
fn same_file(a: &std::path::Path, b: &std::path::Path) -> bool {
    let resolve = |path: &std::path::Path| -> Option<PathBuf> {
        if let Ok(canonical) = std::fs::canonicalize(path) {
            return Some(canonical);
        }
        let name = path.file_name()?;
        let parent = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        };
        std::fs::canonicalize(parent).ok().map(|dir| dir.join(name))
    };
    match (resolve(a), resolve(b)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

pub fn run(o: Options) -> Result<(), String> {
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
    // The file transcript always gets the FULL step-by-step trace, even when the
    // terminal is concise; `--log` additionally shows every step on screen.
    let log_path = o
        .log_file
        .clone()
        .unwrap_or_else(|| output.with_extension("log"));
    // Creating the transcript truncates its path, and that happens before the
    // input is read -- so a `--log-file` aimed at the input would destroy it.
    // Refuse any collision with the input or the output image.
    if same_file(&log_path, &o.input) {
        return Err(
            "--log-file resolves to the input image; it would be truncated before being read"
                .to_owned(),
        );
    }
    if same_file(&log_path, &output) {
        return Err(
            "--log-file resolves to the output image; choose a different log path".to_owned(),
        );
    }
    let log = Log::new("vmpx", o.verbose).with_file(&log_path);
    log.start(&format!("Devirtualizing {}", o.input.display()));

    let raw = std::fs::read(&o.input).map_err(|e| e.to_string())?;
    // A packed on-disk `.dll`/`.exe` must be unpacked to a mapped analysis image
    // first. `--unpack` forces it; otherwise a packer table is detected and the
    // user is asked.
    let bytes = {
        let packed = PeImage::parse(&raw)
            .ok()
            .and_then(|pe| locate_packer_table(&pe, UnpackLimits::default()).ok())
            .is_some();
        let do_unpack = o.unpack
            || (packed
                && confirm(
                    &log,
                    "input looks packed; unpack it to a mapped image first?",
                    true,
                    o.yes_to_all_defaults,
                ));
        if do_unpack {
            log.start("Unpacking to a mapped analysis image");
            let result = unpack_to_mapped_image(&raw, UnpackLimits::default())
                .map_err(|e| format!("unpack: {e}"))?;
            log.step(
                "unpacked",
                &format!(
                    "{:?}, {} byte(s) mapped",
                    result.report.packer_table.kind,
                    result.image.len()
                ),
            );
            if let Some(dir) = &o.dir {
                let mapped_path = dir.join("mapped.bin");
                std::fs::write(&mapped_path, &result.image).map_err(|e| e.to_string())?;
                log.file(
                    &mapped_path.display().to_string(),
                    &format!("{} bytes", result.image.len()),
                );
            }
            result.image
        } else {
            raw
        }
    };
    let pe = PeImage::parse(&bytes).map_err(|e| e.to_string())?;
    let base = pe.optional.image_base.0;
    let report = vmpx_detect::detect_vmprotect(&pe);
    log.info(&format!(
        "{} sections, image base {base:#x}, {} import symbol(s)",
        pe.sections.len(),
        pe.report()
            .imports
            .iter()
            .map(|m| m.entries.len())
            .sum::<usize>()
    ));
    log.info(&format!("protection: {:?}", report.verdict));

    // --- auto settings ---
    log.start("Choosing settings");
    // Start at the smallest budget proven to decode a branch-complete guard
    // CFG, and escalate PER GUARD only where that falls short (the retry ladder
    // in `devirt_guards`). A larger blanket budget only slows the easy guards;
    // adapting where needed is both faster and "best" in the sense asked for.
    let instruction_limit = 2048_usize;
    log.step(
        "instruction limit",
        &format!("{instruction_limit}, escalating per guard as needed"),
    );
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
    log.trace("removing sections", &removed.join(", "));
    let dirs = rebuild_all_directories(&conventional, *b".vmpxdir", &removed)
        .map_err(|e| format!("directory rebuild: {e}"))?;
    const DIR_NAMES: [&str; 13] = [
        "export",
        "import",
        "resource",
        "exception",
        "security",
        "basereloc",
        "debug",
        "architecture",
        "globalptr",
        "tls",
        "load_config",
        "bound_import",
        "iat",
    ];
    for moved in &dirs.moved {
        let name = DIR_NAMES.get(moved.index).copied().unwrap_or("directory");
        log.trace(
            &format!("  relocate {name}"),
            &format!(
                "{:#x} -> {:#x} ({} bytes)",
                moved.from.0, moved.to.0, moved.size
            ),
        );
    }
    log.trace(
        "  exception table",
        &format!("{} entries", dirs.exception_entries.0),
    );
    log.trace(
        "  import directory",
        &format!(
            "{} symbols / {} modules",
            dirs.import_symbols, dirs.import_modules
        ),
    );
    log.trace("  tls callbacks", &format!("{}", dirs.tls_callbacks.0));
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
    let mut residual_guards: Vec<u32> = Vec::new();
    if !targets.guards.is_empty() {
        log.start("Devirtualizing guard functions");
        let (new_image, residual) =
            devirt_guards(&log, &pe, image, instruction_limit, &targets.guards)?;
        image = new_image;
        residual_guards = residual;
    }

    // --- neutralize residual guards ---
    // A guard whose virtualized body could not be lifted (e.g. a non-terminating
    // VM-dispatch loop) still leaves a dangling transfer into the removed
    // protector, which would fault if reached. Neutralize each to its clean path:
    // force the integrity/timing check to pass. This CHANGES behaviour under
    // tampering, so it is a named stage and every site is logged.
    if !residual_guards.is_empty() {
        log.start("Neutralizing residual guards");
        log.info("forces the anti-tamper pass path -- behaviour-changing, not a faithful lift");
        match neutralize_residual_guards(&mut image, &residual_guards) {
            Ok(mut done) => {
                // A guard reached only by a direct branch `.pdata` does not
                // describe (a branch in an uncovered gap) is invisible to the
                // thunk scan above -- and to `count_dangling` -- so patch those
                // branch sites in place before declaring anything left as-is.
                let by_thunk: BTreeSet<u32> = done.iter().map(|a| a.guard_rva).collect();
                let still: Vec<u32> = residual_guards
                    .iter()
                    .copied()
                    .filter(|rva| !by_thunk.contains(rva))
                    .collect();
                match neutralize_residual_guard_sites(&mut image, &still) {
                    Ok(site_done) => done.extend(site_done),
                    Err(reason) => log.warn(&format!("site neutralization failed: {reason}")),
                }
                for action in &done {
                    log.step(
                        &format!("{:#x} neutralized", action.guard_rva),
                        &format!(
                            "{} {:#x}, {} byte(s) -> return 1 (pass)",
                            action.via, action.thunk_begin, action.thunk_bytes
                        ),
                    );
                }
                let reached: BTreeSet<u32> = done.iter().map(|a| a.guard_rva).collect();
                let missed: Vec<String> = residual_guards
                    .iter()
                    .filter(|rva| !reached.contains(rva))
                    .map(|rva| format!("{rva:#x}"))
                    .collect();
                log.step(
                    "neutralized",
                    &format!(
                        "{} of {} residual guard(s)",
                        reached.len(),
                        residual_guards.len()
                    ),
                );
                if !missed.is_empty() {
                    log.warn(&format!(
                        "{} residual guard(s) not reached by any thunk or direct branch, left as-is: {}",
                        missed.len(),
                        missed.join(", ")
                    ));
                }
            }
            Err(reason) => log.warn(&format!("neutralization failed: {reason}")),
        }
    }

    // --- imports ---
    // A given snapshot is used directly. With none, look for one beside the
    // input (`<input>.snap`), and if that is absent the application's imports
    // cannot be recovered statically (`.idata` is encrypted); ask whether to
    // proceed without them.
    let snapshot = o.snapshot.clone().or_else(|| {
        let beside = o.input.with_extension("snap");
        beside.exists().then_some(beside)
    });
    // Whether the application's imports were actually rebound. A written image
    // whose imports are still the protector's encrypted `.idata` will not run,
    // so the final line must not claim a clean result when this stays false.
    let mut imports_bound = false;
    if let Some(snapshot) = &snapshot {
        log.start("Rebuilding the application import table");
        log.trace("snapshot", &snapshot.display().to_string());
        match crate::auto_imports::bind(&log, &pe, &bytes, &image, snapshot) {
            Ok(bound) => {
                image = bound;
                imports_bound = true;
            }
            Err(reason) => log.warn(&format!("import binding skipped: {reason}")),
        }
    } else {
        log.start("Application import table");
        log.warn("no snapshot: the application's imports are in encrypted .idata");
        // Detonation can recover them live -- it runs the protector's own init
        // in a contained emulator and records which import each .idata slot
        // resolves to. It is minutes of CPU (the init runs billions of steps),
        // so it is gated.
        let steps = o.detonate_steps.unwrap_or(20_000_000_000);
        let detonate = o.detonate_steps.is_some()
            || confirm(
                &log,
                &format!("detonate to recover imports now (~minutes, up to {steps} steps)?"),
                false,
                o.yes_to_all_defaults,
            );
        if detonate {
            match detonate_imports(&log, &o.input, &pe, steps) {
                Ok(slots) if !slots.is_empty() => {
                    match crate::auto_imports::bind_slots(&log, &pe, &bytes, &image, slots) {
                        Ok(bound) => {
                            image = bound;
                            imports_bound = true;
                        }
                        Err(reason) => log.warn(&format!("import binding failed: {reason}")),
                    }
                }
                Ok(_) => log.warn("detonation resolved no import slots"),
                Err(reason) => log.warn(&format!("detonation failed: {reason}")),
            }
        } else {
            log.info("skipping imports; the written image will not run until they are bound");
            if !confirm(&log, "write the image anyway?", true, o.yes_to_all_defaults) {
                return Err("aborted: run with --snapshot or allow detonation".to_owned());
            }
        }
    }

    // --- verify + write ---
    log.start("Verifying and writing");
    let dangling = count_dangling(&image);
    if dangling == 0 {
        log.success("no reference into removed space remains");
    } else {
        log.warn(&format!(
            "{dangling} reference(s) into removed space remain"
        ));
    }
    match validate_for_loader(&image) {
        Ok(r) => log.step(
            "loader validation",
            &format!("{} sections, {} imports", r.sections, r.import_symbols),
        ),
        Err(e) => return Err(format!("loader validation failed: {e}")),
    }
    std::fs::write(&output, &image).map_err(|e| e.to_string())?;
    log.file(
        &output.display().to_string(),
        &format!("{} bytes", image.len()),
    );
    // Only a result with no residual references AND rebound imports is clean and
    // runnable. Otherwise name the gaps instead of claiming success: the image
    // is written (it is still useful for inspection) but will likely not run.
    if imports_bound && dangling == 0 {
        log.success_detail("clean PE written", &log.total_elapsed());
    } else {
        let mut gaps = Vec::new();
        if !imports_bound {
            gaps.push("imports not bound".to_owned());
        }
        if dangling > 0 {
            gaps.push(format!("{dangling} reference(s) into removed space"));
        }
        log.warn(&format!(
            "PE written in {} with unresolved gaps ({}) -- it may not load or run",
            log.total_elapsed(),
            gaps.join("; "),
        ));
    }
    let _ = base;
    Ok(())
}

/// Detonate the mapped image in the contained emulator and return each
/// `.idata` slot's `(rva, module, symbol)` from the imports the run observed.
/// XPLM is enabled automatically when the image imports it, so an X-Plane
/// plugin's SDK calls do not stop the run; a generic DLL runs without it.
fn detonate_imports(
    log: &Log,
    input_path: &std::path::Path,
    pe: &PeImage<'_>,
    max_steps: u64,
) -> Result<Vec<(u32, String, String)>, String> {
    let base = pe.optional.image_base.0;
    let xplm = pe.report().imports.iter().any(|m| {
        let n = m.name.to_ascii_uppercase();
        n.starts_with("XPLM") || n.starts_with("XPWIDGETS")
    });
    let mut warnings = Vec::new();
    let export_reference =
        crate::clean::dynamic_export_reference(&input_path.to_string_lossy(), &mut warnings);
    for warning in &warnings {
        log.trace("detonate", warning);
    }
    log.step(
        "detonating",
        &format!(
            "up to {max_steps} steps, xplm={xplm}, {} reference module(s)",
            export_reference.len()
        ),
    );
    let options = vmpx_emulate::DetonateOptions {
        export_reference,
        max_steps,
        xplm,
        ..vmpx_emulate::DetonateOptions::default()
    };
    let result = vmpx_emulate::detonate(pe, &options).map_err(|e| format!("{e:?}"))?;
    log.step(
        "detonated",
        &format!(
            "{} step(s), stop {:?}, {} import(s) observed",
            result.steps,
            result.stop,
            result.observed.imports.len()
        ),
    );
    let slots = result
        .observed
        .imports
        .iter()
        .filter_map(|import| {
            let rva = u32::try_from(import.stub_va_or_slot.checked_sub(base)?).ok()?;
            let symbol = match &import.symbol {
                vmpx_emulate::SymbolOrOrdinal::Name(name) => name.clone(),
                vmpx_emulate::SymbolOrOrdinal::Ordinal(ordinal) => format!("#{ordinal}"),
            };
            Some((rva, import.module.clone(), symbol))
        })
        .collect();
    Ok(slots)
}

/// Count references from surviving code into removed (unmapped) space.
/// Executable code ranges to scan for control flow: every `.pdata` function body
/// merged with the uncovered gaps of each executable section. A branch living in
/// a gap -- code outside any `.pdata` record -- is therefore still decoded. This
/// is the coverage the native rewriters (`retarget_sites`,
/// `neutralize_residual_guard_sites`) walk, so the dangling-ref checker can never
/// be weaker than the passes that rewrite into that same space.
fn executable_scan_ranges(pe: &PeImage<'_>) -> Vec<(u32, u32)> {
    let mapped = |rva: u32| pe.sections.iter().any(|s| s.contains_mapped_rva(Rva(rva)));
    let mut ranges: Vec<(u32, u32)> = pe
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
            Some(last) if b <= last.1 => last.1 = last.1.max(e),
            _ => merged.push((b, e)),
        }
    }
    for section in pe.sections.iter().filter(|s| s.characteristics & EXEC != 0) {
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
    ranges
}

fn count_dangling(image: &[u8]) -> usize {
    let Ok(pe) = PeImage::parse(image) else {
        return 0;
    };
    let base = pe.optional.image_base.0;
    let soi = pe.optional.size_of_image;
    let mapped = |rva: u32| pe.sections.iter().any(|s| s.contains_mapped_rva(Rva(rva)));
    let mut count = 0;
    // `.pdata` bodies AND executable gaps -- a branch into removed space that
    // lives in a gap (outside any `.pdata` record) must be counted, or the
    // checker would be blind exactly where the rewriters can still reach.
    for (b, e) in executable_scan_ranges(&pe) {
        if !mapped(b) {
            continue;
        }
        let Ok(off) = pe.rva_to_file_offset(Rva(b)) else {
            continue;
        };
        let from = off.0 as usize;
        let to = from.saturating_add((e - b) as usize).min(image.len());
        let Some(code) = image.get(from..to) else {
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
                FlowControl::Call
                    | FlowControl::UnconditionalBranch
                    | FlowControl::ConditionalBranch
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
    let mut skipped = 0usize;
    while let Some(rva) = queue.pop_front() {
        let cfg = recover_native_cfg_pe_with(intact, Rva(rva), options)
            .map_err(|e| format!("native cfg {rva:#x}: {e}"))?;
        // An incomplete CFG has unresolved edges and can carry spurious blocks
        // spawned off a bad edge (e.g. a short branch that decoded into data --
        // observed as a `jge` into a "popo" address). Relocating such a function
        // verbatim is unsound, so skip it: its blocks are not placed and its
        // edges are not followed. The `.text` reference that named it then
        // dangles and is caught, fail-closed, by the dangling-ref check before
        // any image is written -- never silently relocated as garbage.
        if !cfg.is_complete() {
            log.trace(
                &format!("  skip {rva:#x}"),
                &format!("{} block(s), incomplete CFG -- not relocated", cfg.blocks.len()),
            );
            skipped += 1;
            continue;
        }
        log.trace(
            &format!("  recover {rva:#x}"),
            &format!("{} block(s)", cfg.blocks.len()),
        );
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
    let section_rva = next_section_rva(stripped).map_err(|e| e.to_string())?.0;
    let mut payload: Vec<u8> = Vec::new();
    let mut placement: BTreeMap<u64, u64> = BTreeMap::new();
    for (start, end) in &blocks {
        while payload.len() % 16 != 0 {
            payload.push(0xCC);
        }
        placement.insert(*start, base + u64::from(section_rva) + payload.len() as u64);
        let rva = u32::try_from(start - base).unwrap_or(0);
        let off = intact
            .rva_to_file_offset(Rva(rva))
            .map_err(|e| e.to_string())?
            .0 as usize;
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
            let ins =
                Decoder::with_ip(64, window, start + cursor as u64, DecoderOptions::NONE).decode();
            if ins.is_invalid() {
                break;
            }
            let size = ins.len();
            let target = if matches!(
                ins.flow_control(),
                FlowControl::Call
                    | FlowControl::UnconditionalBranch
                    | FlowControl::ConditionalBranch
            ) && ins.near_branch64() != 0
            {
                Some(ins.near_branch64())
            } else if ins.memory_base() == Register::RIP {
                Some(ins.ip_rel_memory_address())
            } else {
                None
            };
            // The absolute address the reference must still reach after the
            // move. A target that moved with the component follows it. A target
            // in a RETAINED section stays put -- but a rel32/RIP displacement is
            // measured from the instruction, which DID move, so it must be
            // recomputed to the same address or it would shift by the move
            // delta. Only a reference into removed `.vmp` space is unrecoverable.
            let destination = match target {
                Some(t) if moved(t).is_some() => moved(t),
                Some(t) => {
                    let section = t
                        .checked_sub(base)
                        .and_then(|v| u32::try_from(v).ok())
                        .and_then(|r| {
                            intact
                                .sections
                                .iter()
                                .find(|s| s.contains_mapped_rva(Rva(r)))
                        });
                    match section {
                        // Not covered (moved was None), so a `.vmp` target is
                        // genuinely in removed space.
                        Some(s) if s.name.as_str().starts_with(".vmp") => {
                            return Err(format!(
                                "component references removed space at {start:#x}"
                            ));
                        }
                        // Retained section: rebase the displacement to the same
                        // address from the instruction's new location.
                        Some(_) => Some(t),
                        // Unmapped immediate/scratch: nothing to rebase.
                        None => None,
                    }
                }
                None => None,
            };
            if let Some(nt) = destination {
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
                    if !re.is_invalid()
                        && re.len() == size
                        && re.mnemonic() == ins.mnemonic()
                        && reaches
                    {
                        ok = true;
                        break;
                    }
                }
                if !ok {
                    slot.copy_from_slice(&saved);
                    return Err(format!("cannot rebase {start:#x}+{cursor}"));
                }
            }
            cursor += size;
        }
    }
    if payload.is_empty() {
        // Every candidate native was skipped (all incomplete): nothing to
        // relocate, so append no section. Their `.text` references then dangle
        // and the dangling-ref check refuses the write, fail-closed.
        return Ok((stripped.to_vec(), BTreeMap::new()));
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
        &format!(
            "{} function block(s), {} byte(s){}",
            blocks.len(),
            payload.len(),
            if skipped == 0 {
                String::new()
            } else {
                format!(" ({skipped} incomplete native(s) skipped)")
            }
        ),
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
        slot_of.insert(
            *new,
            section_rva + u32::try_from(payload.len()).unwrap_or(0),
        );
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
    for s in pe2
        .sections
        .iter()
        .filter(|s| s.characteristics & EXEC != 0)
    {
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
        let Ok(off) = pe2.rva_to_file_offset(Rva(begin)) else {
            continue;
        };
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
            if matches!(
                ins.flow_control(),
                FlowControl::Call | FlowControl::UnconditionalBranch
            ) && ins.near_branch64() != 0
                && let Some(t) = ins
                    .near_branch64()
                    .checked_sub(base)
                    .and_then(|v| u32::try_from(v).ok())
                && let Some(new) = mapping.get(&t)
            {
                edits.push((site, *new, true));
            }
            if ins.memory_base() == Register::RIP
                && let Some(slot) = ins
                    .ip_rel_memory_address()
                    .checked_sub(base)
                    .and_then(|v| u32::try_from(v).ok())
                && slot < pe2.optional.size_of_image
                && !mapped(slot)
                && let Some(t) = qword(slot)
                    .and_then(|q| q.checked_sub(base))
                    .and_then(|v| u32::try_from(v).ok())
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
    log.step(
        "repointed",
        &format!("{slots} slot load(s), {direct} direct transfer(s)"),
    );
    Ok(out)
}

/// Devirtualize each guard and retarget its `.text` transfer.
/// Returns the devirtualized image and the RVAs of guards that could not be
/// lifted (the residuals a later neutralization stage handles).
fn devirt_guards(
    log: &Log,
    intact: &PeImage<'_>,
    image: Vec<u8>,
    instruction_limit: usize,
    guards: &[u32],
) -> Result<(Vec<u8>, Vec<u32>), String> {
    let base = intact.optional.image_base.0;
    let section_rva = next_section_rva(&image).map_err(|e| e.to_string())?.0;
    let mut payload: Vec<u8> = Vec::new();
    let mut mapping: BTreeMap<u32, u32> = BTreeMap::new();
    let mut ok = 0_usize;
    let mut residual: Vec<u32> = Vec::new();
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
                residual.push(rva);
                log.warn(&format!(
                    "{rva:#x}: {}",
                    reason.lines().next().unwrap_or(&reason)
                ));
            }
        }
    }
    log.step(
        "devirtualized",
        &format!(
            "{ok} of {} guard(s), {} byte(s)",
            guards.len(),
            payload.len()
        ),
    );
    if mapping.is_empty() {
        return Ok((image, residual));
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
    Ok((out, residual))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x1_4000_0000;
    const GUARD_RVA: u32 = 0xc6f2115; // far enough to force a rel32 branch

    /// Minimal PE64 with one executable `.text` (RVA 0x1000, file 0x200) holding
    /// `code`, and no `.pdata` -- so `.text` is an uncovered gap, exactly the case
    /// the thunk scanner misses.
    fn pe_with_text(code: &[u8]) -> Vec<u8> {
        let mut d = vec![0u8; 0x400];
        d[0..2].copy_from_slice(&0x5A4D_u16.to_le_bytes());
        d[0x3C..0x40].copy_from_slice(&0x80_u32.to_le_bytes());
        d[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
        let coff = 0x84;
        d[coff..coff + 2].copy_from_slice(&0x8664_u16.to_le_bytes());
        d[coff + 2..coff + 4].copy_from_slice(&1_u16.to_le_bytes());
        d[coff + 16..coff + 18].copy_from_slice(&0xF0_u16.to_le_bytes());
        d[coff + 18..coff + 20].copy_from_slice(&0x2022_u16.to_le_bytes());
        let o = coff + 20;
        d[o..o + 2].copy_from_slice(&0x020B_u16.to_le_bytes());
        d[o + 16..o + 20].copy_from_slice(&0x1000_u32.to_le_bytes());
        d[o + 20..o + 24].copy_from_slice(&0x1000_u32.to_le_bytes());
        d[o + 24..o + 32].copy_from_slice(&BASE.to_le_bytes());
        d[o + 32..o + 36].copy_from_slice(&0x1000_u32.to_le_bytes());
        d[o + 36..o + 40].copy_from_slice(&0x200_u32.to_le_bytes());
        d[o + 56..o + 60].copy_from_slice(&0x2000_u32.to_le_bytes());
        d[o + 60..o + 64].copy_from_slice(&0x200_u32.to_le_bytes());
        d[o + 68..o + 70].copy_from_slice(&3_u16.to_le_bytes());
        d[o + 72..o + 80].copy_from_slice(&0x10_0000_u64.to_le_bytes());
        d[o + 80..o + 88].copy_from_slice(&0x1000_u64.to_le_bytes());
        d[o + 88..o + 96].copy_from_slice(&0x10_0000_u64.to_le_bytes());
        d[o + 96..o + 104].copy_from_slice(&0x1000_u64.to_le_bytes());
        d[o + 108..o + 112].copy_from_slice(&16_u32.to_le_bytes());
        let s = o + 0xF0;
        d[s..s + 5].copy_from_slice(b".text");
        d[s + 8..s + 12].copy_from_slice(&0x180_u32.to_le_bytes());
        d[s + 12..s + 16].copy_from_slice(&0x1000_u32.to_le_bytes());
        d[s + 16..s + 20].copy_from_slice(&0x200_u32.to_le_bytes());
        d[s + 20..s + 24].copy_from_slice(&0x200_u32.to_le_bytes());
        d[s + 36..s + 40].copy_from_slice(&0x6000_0020_u32.to_le_bytes());
        d[0x200..0x200 + code.len()].copy_from_slice(code);
        d
    }

    /// A near rel32 branch (`op`) at RVA 0x1000 whose target is the guard VA.
    fn branch_to_guard(op: u8) -> Vec<u8> {
        let site_va = BASE + 0x1000;
        let rel = (BASE + u64::from(GUARD_RVA)) as i64 - (site_va as i64 + 5);
        let mut code = vec![op];
        code.extend_from_slice(&(rel as i32).to_le_bytes());
        code.push(0xC3); // a trailing ret, never reached
        code
    }

    #[test]
    fn tail_jmp_to_residual_guard_becomes_return_true() {
        let mut image = pe_with_text(&branch_to_guard(0xE9));
        let done = neutralize_residual_guard_sites(&mut image, &[GUARD_RVA]).unwrap();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].guard_rva, GUARD_RVA);
        assert_eq!(done[0].via, "site");
        assert_eq!(done[0].thunk_bytes, 5);
        assert_eq!(&image[0x200..0x205], &RETURN_TRUE);
    }

    #[test]
    fn call_to_residual_guard_becomes_mov_eax_1() {
        let mut image = pe_with_text(&branch_to_guard(0xE8));
        let done = neutralize_residual_guard_sites(&mut image, &[GUARD_RVA]).unwrap();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].via, "site");
        assert_eq!(&image[0x200..0x205], &MOV_EAX_1);
    }

    #[test]
    fn no_branch_to_guard_patches_nothing() {
        let mut image = pe_with_text(&[0xC3]); // just `ret`
        let before = image.clone();
        let done = neutralize_residual_guard_sites(&mut image, &[GUARD_RVA]).unwrap();
        assert!(done.is_empty());
        assert_eq!(image, before);
    }

    #[test]
    fn empty_guard_list_is_a_noop() {
        let mut image = pe_with_text(&branch_to_guard(0xE9));
        let before = image.clone();
        let done = neutralize_residual_guard_sites(&mut image, &[]).unwrap();
        assert!(done.is_empty());
        assert_eq!(image, before);
    }

    /// A near rel32 branch (`op`) at RVA 0x1000 whose target is `target_rva`.
    fn branch_at_text(op: u8, target_rva: u32) -> Vec<u8> {
        let site_va = BASE + 0x1000;
        let rel = (BASE + u64::from(target_rva)) as i64 - (site_va as i64 + 5);
        let mut code = vec![op];
        code.extend_from_slice(&(rel as i32).to_le_bytes());
        code.push(0xC3); // trailing ret
        code
    }

    #[test]
    fn gap_branch_into_removed_space_is_counted() {
        // A `jmp` in `.text` -- an uncovered gap, no `.pdata` -- to RVA 0x1900,
        // which is within SizeOfImage (0x2000) but mapped by no section: removed
        // space. The old `.pdata`-only checker decoded nothing here and saw 0;
        // the gap-aware scan must count it.
        let image = pe_with_text(&branch_at_text(0xE9, 0x1900));
        assert_eq!(count_dangling(&image), 1);
    }

    #[test]
    fn gap_branch_into_live_space_is_not_counted() {
        // Same gap, but the branch targets live `.text` (RVA 0x1010) -- a mapped
        // address, so it is not a dangling reference. Broadening coverage must
        // not turn a legitimate in-image branch into a false positive.
        let image = pe_with_text(&branch_at_text(0xE9, 0x1010));
        assert_eq!(count_dangling(&image), 0);
    }

    #[test]
    fn complete_native_is_relocated() {
        // A lone `ret` is a complete one-block function: it must be relocated,
        // so the completeness gate never drops a valid native.
        let image = pe_with_text(&[0xC3]);
        let pe = PeImage::parse(&image).unwrap();
        let log = Log::new("t", false);
        let (_out, mapping) = relocate_natives(&log, &pe, &image, &image, &[0x1000]).unwrap();
        assert!(
            mapping.contains_key(&0x1000),
            "a complete native must be relocated"
        );
    }

    #[test]
    fn incomplete_native_is_skipped_complete_one_kept() {
        // Two seeds: a complete `ret` at 0x1000, and at 0x1010 a `jmp rel32` to a
        // target outside the image -- a `Terminator::Unsupported`, so that CFG is
        // incomplete. The incomplete one is skipped (fail-closed on the
        // component) rather than relocated as garbage -- the fix for
        // `cannot rebase 0x706f706f` on the CL650 -- while the complete one is
        // still relocated.
        let mut code = vec![0xCCu8; 0x20];
        code[0] = 0xC3; // 0x1000: a complete `ret`
        let site = BASE + 0x1010;
        let rel = (BASE + u64::from(GUARD_RVA)) as i64 - (site as i64 + 5);
        code[0x10] = 0xE9; // 0x1010: jmp rel32 to an out-of-image target
        code[0x11..0x15].copy_from_slice(&(rel as i32).to_le_bytes());
        let image = pe_with_text(&code);
        let pe = PeImage::parse(&image).unwrap();
        let log = Log::new("t", false);
        let (_out, mapping) =
            relocate_natives(&log, &pe, &image, &image, &[0x1000, 0x1010]).unwrap();
        assert!(
            mapping.contains_key(&0x1000),
            "the complete native must still be relocated"
        );
        assert!(
            !mapping.contains_key(&0x1010),
            "the incomplete native must be skipped, not relocated"
        );
    }
}
