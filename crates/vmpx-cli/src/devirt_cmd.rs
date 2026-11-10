//! `vmpx devirt`: devirtualize guard functions to native code and reintegrate.
//!
//! The guard references that survive stripping are tail calls into virtualized
//! functions whose returned value is `f(args, checksum(.vmp2 region))`. Native
//! emission needs the function's exit register state; `recover_entry_paths`
//! sources it from the linear chain walker, which stops at the guards'
//! conditional branches. This sources it from the UNROLLED full graph instead
//! (`unroll_entry_cfg`, the same the decompiler uses), keeps the legit
//! callee-return path, folds the checksum to a constant against the intact
//! image (`concretize_with_immutable_image` -- the region is in the removed
//! `.vmp2`, so the constant is forced, not a choice), lowers to native, appends
//! it to the stripped image, and retargets the `.text` transfer.

use std::collections::BTreeMap;
use std::path::PathBuf;

use vmpx_pe::{PeImage, Rva};
use vmpx_rebuild::{
    AppendSection, append_section, next_section_rva, retarget_direct_branch, validate_for_loader,
};
use vmpx_recompile::{CodeAddress, PhysicalRegister};
use vmpx_vm::{
    Expr, ImageConcretizationLimits, X64NativeRegister, concretize_with_immutable_image,
};
use vmpx_vmlift::{EffectStore, RExit, RNode, RStore, ReplacementConfig, build_replacement};

pub struct Options {
    /// Intact image (the checksum regions in `.vmp2` must still be present).
    pub intact: PathBuf,
    /// Stripped image to emit into and retarget.
    pub stripped: PathBuf,
    pub output: Option<PathBuf>,
    pub rvas: Vec<u32>,
    pub instruction_limit: usize,
}

const RX: u32 = 0x6000_0020; // CODE | READ | EXECUTE
const EXEC: u32 = 0x2000_0000;

fn physical_for(register: X64NativeRegister) -> PhysicalRegister {
    use PhysicalRegister as P;
    use X64NativeRegister as R;
    match register {
        R::Rax => P::Rax,
        R::Rbx => P::Rbx,
        R::Rcx => P::Rcx,
        R::Rdx => P::Rdx,
        R::Rbp => P::Rbp,
        R::Rsi => P::Rsi,
        R::Rdi => P::Rdi,
        R::R8 => P::R8,
        R::R9 => P::R9,
        R::R10 => P::R10,
        R::R11 => P::R11,
        R::R12 => P::R12,
        R::R13 => P::R13,
        R::R14 => P::R14,
        R::R15 => P::R15,
    }
}

fn replacement_inputs() -> Vec<(String, PhysicalRegister)> {
    let mut inputs: Vec<(String, PhysicalRegister)> = X64NativeRegister::ALL
        .iter()
        .map(|r| (r.symbolic_name().to_owned(), physical_for(*r)))
        .collect();
    inputs.push(("vsp_initial".to_owned(), PhysicalRegister::Rbp));
    inputs.push((
        "native_return_address_initial".to_owned(),
        PhysicalRegister::Rbp,
    ));
    inputs.sort();
    inputs.dedup();
    inputs
}

/// One legit callee-return path: the ordered observable stores leading to it,
/// and its exit register state.
struct LegitPath<'a> {
    stores: Vec<&'a RStore>,
    exit: &'a RExit,
}

/// Walks the rolled tree collecting one `LegitPath` per legit callee-return
/// exit, carrying the stores seen along the way. Tamper arms (region exits),
/// dead arms and back-edges are dropped -- the same arms the decompiler drops
/// to render C. A guard whose environment gates leave exactly one legit path
/// needs no branch in its replacement.
fn collect_legit_paths<'a>(
    node: &'a RNode,
    stores: &mut Vec<&'a RStore>,
    has_loop: &mut bool,
    out: &mut Vec<LegitPath<'a>>,
) {
    match node {
        // Both callee returns and region exits (a computed continuation) are
        // kept: the region case emits a `Computed` continuation. Only dead and
        // back-edge nodes drop.
        RNode::Exit(exit) => out.push(LegitPath {
            stores: stores.clone(),
            exit,
        }),
        RNode::BackEdge { .. } | RNode::Dead(_) => {}
        RNode::Stores { stores: here, next } => {
            let added = here.len();
            stores.extend(here.iter());
            collect_legit_paths(next, stores, has_loop, out);
            stores.truncate(stores.len() - added);
        }
        RNode::Fork { arms, .. } => {
            for (_, arm) in arms {
                collect_legit_paths(arm, stores, has_loop, out);
            }
        }
        RNode::Loop { body, .. } => {
            *has_loop = true;
            collect_legit_paths(body, stores, has_loop, out);
        }
    }
}

/// Native bytes for one devirtualized guard, built at its final VA.
pub(crate) fn build_one(
    intact: &PeImage<'_>,
    rva: u32,
    instruction_limit: usize,
    va: u64,
) -> Result<Vec<u8>, String> {
    let rolled =
        crate::decoded_splice::with_entry_input(intact, Rva(rva), instruction_limit, |input| {
            let resolver = vmpx_vmlift::PeNestedEntries {
                pe: intact,
                limits: vmpx_vm::EntryProfileRecoveryLimits {
                    instruction_limit,
                    ..vmpx_vm::EntryProfileRecoveryLimits::default()
                },
            };
            vmpx_vmlift::roll_decoded_cfg_with_nested(input, Some(&resolver), 200_000)
        })??;
    let mut has_loop = false;
    let mut paths = Vec::new();
    collect_legit_paths(&rolled.root, &mut Vec::new(), &mut has_loop, &mut paths);
    if has_loop {
        return Err("has a loop; flat-exit emission does not cover it".to_owned());
    }
    if paths.is_empty() {
        return Err("no exit path (all nested, dead, or loop)".to_owned());
    }

    // Fold the checksum and resolve image reads against the intact image. The
    // byte cap covers the largest checksummed region a guard weaves in (one is
    // ~48 MiB); folding reads the region once and bakes the constant.
    let limits = ImageConcretizationLimits {
        max_expression_nodes: 400_000,
        max_checksum_bytes: 64 * 1024 * 1024,
    };
    let modulo_limits = vmpx_vm::ModuloImageTargetLimits {
        concretization: limits,
        ..vmpx_vm::ModuloImageTargetLimits::default()
    };
    let mut folded = 0_usize;
    let mut unresolved = std::collections::BTreeSet::new();
    let fold = |expr: &Expr,
                folded: &mut usize,
                unresolved: &mut std::collections::BTreeSet<String>|
     -> Result<Expr, String> {
        // A guard reads `.vmp` at an rdtsc-remainder-driven address and feeds it
        // into a checksum pointer/length. Fold those loads to constants first --
        // provably counter-independent when every residue reads the same image
        // bytes -- so the checksum below can then pin its region and fold.
        let timing = vmpx_vm::fold_timing_invariant_loads(intact, expr, modulo_limits)
            .map_err(|error| format!("timing fold: {error}"))?;
        *folded += timing.folded;
        let report =
            concretize_with_immutable_image(intact, &timing.expression, &BTreeMap::new(), limits)
                .map_err(|error| format!("concretize: {error}"))?;
        *folded += report.resolved_intrinsics;
        // A `memory_load_epoch_N(addr)` concretize could not fold is a runtime
        // read of a location no store on this path touched -- e.g. the caller's
        // `*(u64*)(a15+0xb8)`. That is a legitimate Load, not something to bake,
        // so rewrite the intrinsic to a plain `Load(addr)` the backend lowers.
        // The `_N_version` form (byte-versioned) is the same shape.
        let rewritten = report
            .expression
            .rewrite_intrinsics(&mut |width, name, args| {
                (name.starts_with("memory_load_epoch_") && args.len() == 1)
                    .then(|| Expr::load(width, args[0].clone()))
            })
            .map_err(|error| format!("rewrite memory_load_epoch: {error}"))?;
        let _ = unresolved; // build_replacement is the judge of remaining intrinsics
        Ok(rewritten)
    };

    // Concretize every legit path. The forks here are checksum-tamper gates:
    // the "corrupted" arm weaves the WRONG checksum into its return, but after
    // baking the CORRECT checksum (from the intact image) both arms compute the
    // same value, so the paths collapse to one. If after folding the paths are
    // NOT identical, the branch is genuinely input-dependent and this flat
    // emission cannot represent it -- reported, not guessed.
    struct FoldedPath {
        registers: BTreeMap<X64NativeRegister, Expr>,
        effects: Vec<EffectStore>,
        continuation: Option<Expr>,
        region: bool,
    }
    let mut folded_paths: Vec<FoldedPath> = Vec::new();
    // Win64 callee-saved registers. A function returning to a normal caller
    // must preserve them, so their exit value is their entry value regardless of
    // what the VM's internal bookkeeping tracked (the decompiler's
    // "callee-saved rNN not restored verbatim" note is this same mismatch: the
    // exit expr equals the input semantically but not syntactically).
    const NONVOLATILE: [X64NativeRegister; 8] = [
        X64NativeRegister::Rbx,
        X64NativeRegister::Rbp,
        X64NativeRegister::Rdi,
        X64NativeRegister::Rsi,
        X64NativeRegister::R12,
        X64NativeRegister::R13,
        X64NativeRegister::R14,
        X64NativeRegister::R15,
    ];
    for LegitPath { stores, exit } in &paths {
        let mut registers = BTreeMap::new();
        for (register, expr) in &exit.registers {
            registers.insert(*register, fold(expr, &mut folded, &mut unresolved)?);
        }
        for register in NONVOLATILE {
            registers.insert(
                register,
                Expr::register(vmpx_vm::BitWidth::W64, register.symbolic_name()),
            );
        }
        let mut effects = Vec::new();
        for store in stores {
            effects.push(EffectStore {
                address: fold(&store.address, &mut folded, &mut unresolved)?,
                value: fold(&store.value, &mut folded, &mut unresolved)?,
                width: store.width,
            });
        }
        // A region exit transfers to a computed continuation rather than
        // returning; fold its target too.
        let continuation = match (exit.region, &exit.continuation) {
            (true, Some(target)) => Some(fold(target, &mut folded, &mut unresolved)?),
            (true, None) => return Err("region exit with no recovered continuation".to_owned()),
            (false, _) => None,
        };
        folded_paths.push(FoldedPath {
            registers,
            effects,
            continuation,
            region: exit.region,
        });
    }
    let _ = (&unresolved, folded);
    // When a guard has both callee-return and region-exit arms, the region arms
    // are the tamper responses the decompiler drops; the callee returns are the
    // legit behaviour. Only when EVERY arm is a region exit is the function a
    // genuine trampoline, emitted with a computed continuation.
    let region_only = folded_paths.iter().all(|path| path.region);
    let chosen: Vec<&FoldedPath> = folded_paths
        .iter()
        .filter(|path| path.region == region_only)
        .collect();
    let first = chosen[0];
    if chosen.iter().any(|path| {
        path.registers != first.registers
            || path.effects != first.effects
            || path.continuation != first.continuation
    }) {
        return Err(format!(
            "{} {} path(s) differ after folding the checksum -- genuinely input-dependent branch",
            chosen.len(),
            if region_only { "region" } else { "callee" }
        ));
    }
    let (registers, effects) = (first.registers.clone(), first.effects.clone());
    // A region exit emits a `Computed` continuation: the replacement stages the
    // target and transfers through it, so the destination need not be known.
    let continuation = first
        .continuation
        .clone()
        .map(vmpx_vmlift::ReplacementContinuation::Computed);

    let config = ReplacementConfig {
        inputs: replacement_inputs(),
        max_expression_nodes: 400_000,
        max_spill_slots: 128,
        import_calls: BTreeMap::new(),
        continuation,
        effects,
        effect_calls: Vec::new(),
        rflags: None,
        symbol_rewrites: BTreeMap::new(),
    };
    let replacement = build_replacement(&registers, &config, CodeAddress(va))
        .map_err(|error| format!("build: {error}"))?;
    Ok(replacement.compiled.bytes)
}

/// Retarget every direct `.text` transfer into `old` so it reaches `new`,
/// including transfers from executable bytes `.pdata` does not describe (the
/// thunk table). Returns the number retargeted.
pub(crate) fn retarget_sites(
    image: &mut Vec<u8>,
    mapping: &BTreeMap<u32, u32>,
) -> Result<usize, String> {
    use iced_x86::{Decoder, DecoderOptions, FlowControl};

    let pe = PeImage::parse(image).map_err(|e| e.to_string())?;
    let base = pe.optional.image_base.0;
    let mapped = |rva: u32| pe.sections.iter().any(|s| s.contains_mapped_rva(Rva(rva)));

    // `.pdata` extents plus each executable section's uncovered gaps.
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

    let mut sites: Vec<(u32, u32)> = Vec::new();
    for (begin, end) in ranges {
        if !mapped(begin) {
            continue;
        }
        let Ok(offset) = pe.rva_to_file_offset(Rva(begin)) else {
            continue;
        };
        let from = offset.0 as usize;
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
            if matches!(
                instruction.flow_control(),
                FlowControl::Call | FlowControl::UnconditionalBranch
            ) && instruction.near_branch64() != 0
                && let Some(target) = instruction
                    .near_branch64()
                    .checked_sub(base)
                    .and_then(|v| u32::try_from(v).ok())
                && let Some(new) = mapping.get(&target)
            {
                sites.push((begin + u32::try_from(at).unwrap_or(0), *new));
            }
        }
    }

    let mut done = 0;
    for (site, new) in sites {
        *image = retarget_direct_branch(image, Rva(site), Rva(new))
            .map_err(|error| format!("retarget {site:#x}: {error}"))?;
        done += 1;
    }
    Ok(done)
}

pub fn run(o: Options) -> Result<(), String> {
    let intact_bytes = std::fs::read(&o.intact).map_err(|e| e.to_string())?;
    let intact = PeImage::parse(&intact_bytes).map_err(|e| e.to_string())?;
    let stripped_bytes = std::fs::read(&o.stripped).map_err(|e| e.to_string())?;
    let base = intact.optional.image_base.0;
    if base
        != PeImage::parse(&stripped_bytes)
            .map_err(|e| e.to_string())?
            .optional
            .image_base
            .0
    {
        return Err("intact and stripped image bases differ".to_owned());
    }
    let section_rva = next_section_rva(&stripped_bytes)
        .map_err(|e| e.to_string())?
        .0;

    // Emit each guard sequentially; x64 body size is position-independent, so
    // building at the running VA places each one correctly in one pass.
    let mut payload: Vec<u8> = Vec::new();
    let mut mapping: BTreeMap<u32, u32> = BTreeMap::new();
    let mut failed: Vec<(u32, String)> = Vec::new();
    for &rva in &o.rvas {
        while payload.len() % 16 != 0 {
            payload.push(0xCC);
        }
        let new_rva = section_rva + u32::try_from(payload.len()).map_err(|_| "offset overflow")?;
        let va = base + u64::from(new_rva);
        match build_one(&intact, rva, o.instruction_limit, va) {
            Ok(bytes) => {
                eprintln!("  {rva:#x}: {} native byte(s)", bytes.len());
                payload.extend_from_slice(&bytes);
                mapping.insert(rva, new_rva);
            }
            Err(reason) => {
                eprintln!("  {rva:#x}: SKIPPED -- {reason}");
                failed.push((rva, reason));
            }
        }
    }
    println!(
        "devirtualized {} of {} guard(s); {} byte(s) of native code",
        mapping.len(),
        o.rvas.len(),
        payload.len()
    );
    if mapping.is_empty() {
        return Err("no guard was devirtualized".to_owned());
    }

    let rebuilt = append_section(
        &stripped_bytes,
        AppendSection {
            name: *b".vmpxdv\0",
            data: &payload,
            characteristics: RX,
        },
    )
    .map_err(|e| e.to_string())?;
    if rebuilt.section.rva.0 != section_rva {
        return Err("appended section moved from the planned RVA".to_owned());
    }
    let mut image = rebuilt.bytes;
    let retargeted = retarget_sites(&mut image, &mapping)?;
    println!("retargeted {retargeted} .text transfer(s) into the devirtualized bodies");

    match validate_for_loader(&image) {
        Ok(report) => println!(
            "loader validation passed: {} sections, {} imports",
            report.sections, report.import_symbols
        ),
        Err(error) => return Err(format!("loader validation failed: {error}")),
    }
    if let Some(output) = &o.output {
        std::fs::write(output, &image).map_err(|e| e.to_string())?;
        println!("wrote {}", output.display());
    }
    Ok(())
}
