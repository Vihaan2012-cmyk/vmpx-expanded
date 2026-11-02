//! `vmpx report`: thin orchestration over existing stage APIs that prints the
//! three-number headline (N1 CFG-complete, N2 PROVEN, N3 proven-equivalent) with
//! experimental tiers kept separate. Anything not run or not wired says so.

use std::{
    collections::BTreeMap,
    panic::{self, AssertUnwindSafe},
};

use anyhow::{Context, Result};
use serde::Serialize;
use vmpx_detect::{ImageLayout, PackerKind, detect_packer, detect_vmprotect};
use vmpx_pe::PeImage;
use vmpx_unpack::{UnpackLimits, unpack_to_mapped_image};
use vmpx_vm::{
    EntryProfileRecoveryLimits, ObservedRun, VmCfg, VmCfgAssumption, VmCfgLimits,
    VmDiscoveryLimits, VmEntryCandidate, discover_vm_entries, recover_vm_cfg,
    recover_vm_cfg_with_observed,
};

const DEFAULT_SAMPLE: usize = 50;

#[derive(Clone, Copy, Debug)]
pub struct ReportOptions {
    /// Entries to sweep; 0 sweeps all discovered entries.
    pub sample: usize,
    pub threads: usize,
    pub instruction_limit: usize,
    pub dynamic: bool,
    pub serve_ro: bool,
    /// OBSERVED tier: serve withheld (volatile) read-only image bytes from the
    /// post-init snapshot. Needs `--volatility-snapshot`; labeled, never proven.
    pub serve_observed_ro: bool,
}

/// Why incomplete entries stopped (limit hit, or `stall`), diagnostic only.
static LIMIT_HITS: std::sync::Mutex<BTreeMap<String, usize>> = std::sync::Mutex::new(BTreeMap::new());

/// Snapshot bytes of the volatile, non-writable image ranges (observed seed).
static SNAPSHOT_SEED: std::sync::OnceLock<vmpx_vm::ObservedMemory> = std::sync::OnceLock::new();

impl ReportOptions {
    pub fn new(sample: Option<usize>, all: bool, threads: usize, dynamic: bool) -> Self {
        Self {
            sample: if all {
                0
            } else {
                sample.unwrap_or(DEFAULT_SAMPLE)
            },
            threads: threads.max(1),
            instruction_limit: 512,
            dynamic,
            serve_ro: false,
            serve_observed_ro: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Proven,
    BenignLabeled,
    UnprovenAssumption,
    Observed,
}

#[must_use]
pub fn tier_of<'a>(assumptions: impl IntoIterator<Item = &'a VmCfgAssumption>) -> Tier {
    let mut tier = Tier::Proven;
    for a in assumptions {
        let this = match a {
            VmCfgAssumption::ObservedGlobal
            | VmCfgAssumption::ObservedRecursionDepth
            | VmCfgAssumption::ObservedBranchTarget
            | VmCfgAssumption::AssumesDynamicBranchTargetsComplete => Tier::Observed,
            VmCfgAssumption::AssumesExternalStoresDisjointFromVmStack
            | VmCfgAssumption::AssumesImageProtectionStable
            | VmCfgAssumption::AssumesCrossVmSectionContinuation => Tier::UnprovenAssumption,
            VmCfgAssumption::PreferredImageBase
            | VmCfgAssumption::OverApproximatedEnvironmentGate => Tier::BenignLabeled,
        };
        tier = tier.max(this);
    }
    tier
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct StaticSweep {
    pub discovered: usize,
    pub swept: usize,
    pub sampled: bool,
    /// N1: every path reaches a VM exit, under any labeled assumption.
    pub n1_cfg_complete: usize,
    /// N2: N1 with an EMPTY assumption set.
    pub n2_proven: usize,
    pub benign_labeled: usize,
    pub unproven_assumption: usize,
    pub incomplete: usize,
    pub errors: usize,
    pub panics: usize,
    pub assumption_sets: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DynamicStage {
    pub detonation_steps: usize,
    pub detonation_stop: String,
    pub observed_imports: usize,
    pub observed_globals: usize,
    /// Entries incomplete statically that complete only when seeded by this one run.
    pub observed_tier_completions: usize,
    pub observed_tier_rerun_entries: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Stage {
    pub status: &'static str,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub path: String,
    pub packer: String,
    pub vmprotect_verdict: String,
    pub protection_mode: String,
    pub packing: Stage,
    pub memory_protection: Stage,
    pub imports_parsed: String,
    pub import_resolution: Stage,
    pub anti_analysis: Stage,
    pub sweep: StaticSweep,
    pub n3_proven_equivalent: Stage,
    pub serve_ro_completions: Stage,
    pub observed: Stage,
    pub dynamic: Option<DynamicStage>,
    pub warnings: Vec<String>,
}

enum Outcome {
    Complete(Vec<VmCfgAssumption>),
    Incomplete,
    Error,
    Panic,
}

fn cfg_limits(instruction_limit: usize, serve_ro: bool, serve_observed: bool) -> VmCfgLimits {
    VmCfgLimits {
        entry: EntryProfileRecoveryLimits {
            instruction_limit,
            symbolic: vmpx_vm::SymbolicExecutionLimits {
                serve_read_only_when_suppressed: serve_ro,
                serve_observed_when_suppressed: serve_observed,
                ..vmpx_vm::SymbolicExecutionLimits::default()
            },
            ..EntryProfileRecoveryLimits::default()
        },
        // Post-transition exploration caps scale by VMPX_TRANSITION_BUDGET (integer
        // multiplier, default 1). A larger cap only lets the walk explore further;
        // completion still requires no stall and no limit hit.
        max_visits: VmCfgLimits::default().max_visits.saturating_mul(env_scale("VMPX_VISITS")),
        max_blocks: VmCfgLimits::default().max_blocks.saturating_mul(env_scale("VMPX_VISITS")),
        max_total_instructions: VmCfgLimits::default()
            .max_total_instructions
            .saturating_mul(env_scale("VMPX_TOTAL_INSTRUCTIONS")),
        max_transition_visits: VmCfgLimits::default().max_transition_visits.saturating_mul(transition_scale()),
        max_transition_instructions: VmCfgLimits::default()
            .max_transition_instructions
            .saturating_mul(transition_scale()),
        follow_cross_vm_section: std::env::var_os("VMPX_FOLLOW_CROSS_VMP_SECTION").is_some(),
        // A budget-pass sweep bounds each rerun entry by VMPX_PASS_WALL_SECS (a lower
        // cap only stops an entry sooner, as WallClock, i.e. incomplete).
        max_wall_time: if SCALE_OVERRIDE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_some() {
            std::env::var("VMPX_PASS_WALL_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map(std::time::Duration::from_secs)
                .or(VmCfgLimits::default().max_wall_time)
        } else {
            VmCfgLimits::default().max_wall_time
        },
        ..VmCfgLimits::default()
    }
}

/// Per-pass multiplier override `[transition budget, visits/blocks, total
/// instructions, per-block instruction limit]`, set by the budget-pass sweep.
static SCALE_OVERRIDE: std::sync::Mutex<Option<[usize; 4]>> = std::sync::Mutex::new(None);

fn env_scale(name: &str) -> usize {
    if let Some(a) = *SCALE_OVERRIDE.lock().unwrap_or_else(std::sync::PoisonError::into_inner) {
        return match name {
            "VMPX_TRANSITION_BUDGET" => a[0],
            "VMPX_VISITS" => a[1],
            "VMPX_TOTAL_INSTRUCTIONS" => a[2],
            "VMPX_INSTRUCTION_LIMIT" => a[3],
            _ => 1,
        };
    }
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|k| *k >= 1)
        .unwrap_or(1)
}

fn transition_scale() -> usize {
    env_scale("VMPX_TRANSITION_BUDGET")
}

/// Diagnostic: with `VMPX_STALL_DUMP=<file>` set, appends one JSON line per
/// incomplete entry (rva, limit hit, stalls in discovery order) while a budget
/// pass is active.
fn dump_stalls(entry: &VmEntryCandidate, result: &std::thread::Result<Result<VmCfg, vmpx_vm::CfgError>>) {
    use std::io::Write;
    let Some(path) = std::env::var_os("VMPX_STALL_DUMP") else { return };
    if SCALE_OVERRIDE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_none() {
        return;
    }
    let Ok(Ok(cfg)) = result else { return };
    if cfg.all_paths_reach_vm_exit() {
        return;
    }
    let stalls: Vec<serde_json::Value> = cfg
        .stalls
        .iter()
        .take(12)
        .map(|st| serde_json::json!({"block": format!("{:#x}", st.block.get()), "vip": format!("{:#x}", st.vip.get()), "reason": st.reason}))
        .collect();
    let line = serde_json::json!({
        "rva": format!("{:#x}", entry.rva.0),
        "limit": cfg.limit_hit.map(|h| format!("{h:?}")),
        "n_stalls": cfg.stalls.len(),
        "stalls": stalls,
    });
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}

fn classify(result: std::thread::Result<Result<VmCfg, vmpx_vm::CfgError>>) -> Outcome {
    match result {
        Err(_) => Outcome::Panic,
        Ok(Err(_)) => Outcome::Error,
        Ok(Ok(cfg)) if cfg.all_paths_reach_vm_exit() => {
            Outcome::Complete(cfg.assumptions.iter().copied().collect())
        }
        Ok(Ok(cfg)) => {
            let label = cfg.limit_hit.map_or_else(
                || "stall".to_owned(),
                |hit| format!("{hit:?}"),
            );
            *LIMIT_HITS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .entry(label)
                .or_insert(0) += 1;
            Outcome::Incomplete
        }
    }
}

fn recover_all(
    pe: &PeImage<'_>,
    entries: &[VmEntryCandidate],
    opts: &ReportOptions,
    observed: Option<&ObservedRun>,
) -> Vec<Outcome> {
    if entries.is_empty() {
        return Vec::new();
    }
    let chunk = entries.len().div_ceil(opts.threads).max(1);
    let instruction_limit = opts.instruction_limit.saturating_mul(env_scale("VMPX_INSTRUCTION_LIMIT"));
    let limits = cfg_limits(instruction_limit, opts.serve_ro, opts.serve_observed_ro);
    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let out = std::thread::scope(|scope| {
        let handles: Vec<_> = entries
            .chunks(chunk)
            .map(|slice| {
                scope.spawn(move || {
                    slice
                        .iter()
                        .map(|entry| {
                            let result = panic::catch_unwind(AssertUnwindSafe(|| match observed {
                                Some(run) => recover_vm_cfg_with_observed(pe, entry, limits, run),
                                None => recover_vm_cfg(pe, entry, limits),
                            }));
                            dump_stalls(entry, &result);
                            classify(result)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_default())
            .collect::<Vec<_>>()
    });
    panic::set_hook(prev_hook);
    out
}

fn tally(outcomes: &[Outcome], discovered: usize, sampled: bool) -> StaticSweep {
    let mut sweep = StaticSweep {
        discovered,
        swept: outcomes.len(),
        sampled,
        ..StaticSweep::default()
    };
    for outcome in outcomes {
        match outcome {
            Outcome::Complete(assumptions) => {
                sweep.n1_cfg_complete += 1;
                match tier_of(assumptions) {
                    Tier::Proven => sweep.n2_proven += 1,
                    Tier::BenignLabeled => sweep.benign_labeled += 1,
                    Tier::UnprovenAssumption => sweep.unproven_assumption += 1,
                    Tier::Observed => {}
                }
                *sweep
                    .assumption_sets
                    .entry(format!("{assumptions:?}"))
                    .or_insert(0) += 1;
            }
            Outcome::Incomplete => sweep.incomplete += 1,
            Outcome::Error => sweep.errors += 1,
            Outcome::Panic => sweep.panics += 1,
        }
    }
    sweep
}

fn run_dynamic(
    input_path: &str,
    input_pe: &PeImage<'_>,
    pe: &PeImage<'_>,
    entries: &[VmEntryCandidate],
    static_outcomes: &[Outcome],
    opts: &ReportOptions,
    warnings: &mut Vec<String>,
) -> Option<DynamicStage> {
    let export_reference = crate::clean::dynamic_export_reference(input_path, warnings);
    let base = vmpx_emulate::DetonateOptions::default();
    let max_steps = std::env::var("VMPX_DYNAMIC_MAXSTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(base.max_steps);
    let det = vmpx_emulate::DetonateOptions {
        export_reference,
        max_steps,
        ..base
    };
    let result = match vmpx_emulate::detonate(input_pe, &det) {
        Ok(r) => r,
        Err(error) => {
            warnings.push(format!("detonation could not start: {error:?}"));
            return None;
        }
    };
    let mut run = ObservedRun::new(
        vmpx_vm::ObservedMemory::from_slots(
            result
                .observed
                .globals
                .iter()
                .map(|g| (g.va, g.bytes.clone())),
        ),
        vmpx_vm::ObservedLoopDepths::from_depths(
            result
                .observed
                .loop_depths
                .iter()
                .map(|l| (l.loop_head, l.depth)),
        ),
    );
    let mut grouped: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for bt in &result.observed.branch_targets {
        let targets = grouped.entry(bt.site_vip).or_default();
        if !targets.contains(&bt.target_vip) {
            targets.push(bt.target_vip);
        }
    }
    run.branch_targets = vmpx_vm::ObservedBranchTargets::from_sites(grouped);
    let pending: Vec<VmEntryCandidate> = entries
        .iter()
        .zip(static_outcomes)
        .filter(|(_, o)| !matches!(o, Outcome::Complete(_)))
        .map(|(e, _)| e.clone())
        .collect();
    let rerun = recover_all(pe, &pending, opts, Some(&run));
    let completions = rerun
        .iter()
        .filter(|o| matches!(o, Outcome::Complete(a) if tier_of(a) == Tier::Observed))
        .count();
    Some(DynamicStage {
        detonation_steps: result.steps,
        detonation_stop: format!("{:?}", result.stop),
        observed_imports: result.observed.imports.len(),
        observed_globals: result.observed.globals.len(),
        observed_tier_completions: completions,
        observed_tier_rerun_entries: pending.len(),
    })
}

/// Installs the dynamic-evidence VOLATILITY VETO: every byte of the (already
/// mapped) analysis image that differs from the same address in a post-init
/// emulator snapshot is marked volatile, so the symbolic executor never serves it
/// as a constant from the file. The snapshot is observed-tier and is used ONLY to
/// veto a byte, never to supply a value. Returns `(volatile ranges, volatile bytes)`.
///
/// # Errors
///
/// Fails when the image is not a PE or the snapshot cannot be read.
pub fn install_volatility_veto(
    image: &[u8],
    snapshot: &std::path::Path,
    capture_seed: bool,
) -> Result<(usize, u64)> {
    use vmpx_pe::Va;
    use vmpx_vm::SymbolicMemorySource;
    let pe = PeImage::parse(image).context("volatility veto: image is not a PE")?;
    let (_, memory) = vmpx_emulate::snapshot::load(snapshot)
        .with_context(|| format!("read snapshot {}", snapshot.display()))?;
    let lo = pe.optional.image_base.get();
    let hi = lo.saturating_add(u64::from(pe.optional.size_of_image));
    let mut pages: Vec<(u64, &[u8])> = memory
        .pages()
        .filter(|(base, _, _)| *base >= lo && *base < hi)
        .map(|(base, _, data)| (base, data))
        .collect();
    pages.sort_by_key(|(base, _)| *base);
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    // Observed seed: the snapshot byte of each volatile byte that lies in a
    // readable, NON-writable section (never a writable section or the stack).
    let read_only = vmpx_vm::ReadOnlyImage(&pe);
    let mut seed: Vec<(u64, Vec<u8>)> = Vec::new();
    for (base, data) in pages {
        for (offset, &runtime) in data.iter().enumerate() {
            let address = base + offset as u64;
            if SymbolicMemorySource::read_byte(&pe, Va(address)).is_some_and(|file| file != runtime)
            {
                match ranges.last_mut() {
                    Some((_, end)) if *end + 1 == address => *end = address,
                    _ => ranges.push((address, address)),
                }
                if capture_seed && SymbolicMemorySource::is_read_only_address(&read_only, Va(address)) {
                    match seed.last_mut() {
                        Some((start, bytes)) if *start + bytes.len() as u64 == address => bytes.push(runtime),
                        _ => seed.push((address, vec![runtime])),
                    }
                }
            }
        }
    }
    if capture_seed {
        eprintln!(
            "[report] observed seed: {} byte(s) in {} read-only range(s) from the snapshot",
            seed.iter().map(|(_, b)| b.len()).sum::<usize>(),
            seed.len()
        );
        let _ = SNAPSHOT_SEED.set(vmpx_vm::ObservedMemory::from_slots(seed));
    }
    if let Some(path) = std::env::var_os("VMPX_DUMP_VOLATILE") {
        let text: String = ranges.iter().map(|(a, b)| format!("{a:x} {b:x}\n")).collect();
        let _ = std::fs::write(path, text);
    }
    let bytes = ranges.iter().map(|(s, e)| e - s + 1).sum();
    let count = ranges.len();
    vmpx_vm::set_volatile_image_ranges(ranges);
    Ok((count, bytes))
}

/// # Errors
///
/// Fails when the input is not a PE, or unpack/discovery fails.
pub fn build_report(input_path: &str, input: &[u8], opts: &ReportOptions) -> Result<Report> {
    let input_pe = PeImage::parse(input).context("input is not a supported PE")?;
    let protection = detect_vmprotect(&input_pe);
    let packer = detect_packer(&input_pe);
    let mut warnings = protection.warnings.clone();

    let (mapped, unpacked) = match protection.image_layout {
        ImageLayout::PackedVirtualSections => {
            let result = unpack_to_mapped_image(input, UnpackLimits::default())
                .context("VMProtect section unpack failed")?;
            (result.image, true)
        }
        _ => (input.to_vec(), false),
    };
    let pe = PeImage::parse(&mapped).context("mapped analysis image failed reparse")?;
    let discovery = discover_vm_entries(&pe, VmDiscoveryLimits::default())?;
    let mut entries = discovery.entries.clone();
    let discovered = entries.len();
    let sampled = opts.sample > 0 && entries.len() > opts.sample;
    if sampled {
        entries.truncate(opts.sample);
    }

    eprintln!(
        "[report] {discovered} entries discovered; sweeping {}",
        entries.len()
    );
    let snapshot_run = if opts.serve_observed_ro {
        SNAPSHOT_SEED.get().map(|memory| {
            ObservedRun::new(memory.clone(), vmpx_vm::ObservedLoopDepths::from_depths(std::iter::empty()))
        })
    } else {
        None
    };
    if opts.serve_observed_ro && snapshot_run.is_none() {
        eprintln!("[report] --serve-observed-ro needs --volatility-snapshot; no snapshot seed was captured");
    }
    let mut outcomes = recover_all(&pe, &entries, opts, None);
    eprintln!(
        "[report] incomplete entries by stop reason: {:?}",
        LIMIT_HITS.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    );
    if let Ok(spec) = std::env::var("VMPX_BUDGET_PASSES") {
        // Sweep of the exploration caps. Each pass re-runs the entries the base pass
        // left incomplete with larger caps (`tb,visits,total,instr` multipliers,
        // passes separated by `;`). Raising a cap only lets the walk explore further:
        // completion still requires no stall and no limit hit. Newly complete entries
        // are merged, each keeping its own assumption set (so none can reach N2
        // unless its set is empty).
        let base_incomplete: Vec<usize> = outcomes
            .iter()
            .enumerate()
            .filter(|(_, o)| matches!(o, Outcome::Incomplete))
            .map(|(i, _)| i)
            .collect();
        let pending: Vec<VmEntryCandidate> = base_incomplete.iter().map(|&i| entries[i].clone()).collect();
        for pass in spec.split(';') {
            let k: Vec<usize> = pass.split(',').filter_map(|v| v.trim().parse().ok()).collect();
            let [a, b, c, d] = k[..] else { continue };
            *SCALE_OVERRIDE.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some([a, b, c, d]);
            LIMIT_HITS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
            let t0 = std::time::Instant::now();
            let rerun = recover_all(&pe, &pending, opts, None);
            *SCALE_OVERRIDE.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
            let (mut done, mut n2) = (0_usize, 0_usize);
            for (i, outcome) in base_incomplete.iter().zip(rerun) {
                if let Outcome::Complete(assumptions) = &outcome {
                    done += 1;
                    if tier_of(assumptions) == Tier::Proven {
                        n2 += 1;
                    }
                    outcomes[*i] = outcome;
                }
            }
            eprintln!(
                "[report] budget pass tb={a} visits={b} total={c} instr={d}: {done} of {} incomplete entries now complete ({n2} with empty assumptions), {:.0}s; still stopped by {:?}",
                pending.len(),
                t0.elapsed().as_secs_f64(),
                LIMIT_HITS.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
            );
        }
    }
    if let Some(run) = &snapshot_run {
        // Second pass only over entries the sound/labeled pass left incomplete, so
        // the snapshot can only ADD observed-tier completions and never relabel an
        // entry the first pass already completed.
        let indices: Vec<usize> = outcomes
            .iter()
            .enumerate()
            .filter(|(_, o)| matches!(o, Outcome::Incomplete))
            .map(|(i, _)| i)
            .collect();
        let pending: Vec<VmEntryCandidate> = indices.iter().map(|&i| entries[i].clone()).collect();
        let rerun = recover_all(&pe, &pending, opts, Some(run));
        let mut gained = 0_usize;
        for (i, outcome) in indices.into_iter().zip(rerun) {
            if matches!(outcome, Outcome::Complete(_)) {
                outcomes[i] = outcome;
                gained += 1;
            }
        }
        eprintln!("[report] snapshot-served second pass completed {gained} additional entr(ies)");
    }
    let sweep = tally(&outcomes, discovered, sampled);

    let dynamic = if opts.dynamic {
        run_dynamic(
            input_path,
            &input_pe,
            &pe,
            &entries,
            &outcomes,
            opts,
            &mut warnings,
        )
    } else {
        None
    };

    let pe_report = pe.report();
    let modules = pe_report.imports.len();
    let symbols: usize = pe_report.imports.iter().map(|m| m.entries.len()).sum();
    let packer_kind = format!("{:?}", packer.kind);

    let packing = if unpacked {
        Stage {
            status: "handled",
            detail: "VMProtect virtual-section packing: static unpack to mapped analysis image"
                .into(),
        }
    } else if packer.kind != PackerKind::None {
        Stage {
            status: "detected, not run",
            detail: format!("{packer_kind} detected; use `vmpx generic-unpack` (dynamic, bounded)"),
        }
    } else {
        Stage {
            status: "not present",
            detail: "no packed virtual sections or known packer signature".into(),
        }
    };
    let memory_protection = if unpacked {
        Stage {
            status: "handled",
            detail: "section decryption done by the static unpacker".into(),
        }
    } else {
        Stage {
            status: "not present",
            detail: "no packed/encrypted sections detected".into(),
        }
    };
    let import_resolution = match &dynamic {
        Some(d) => Stage {
            status: "partial",
            detail: format!(
                "static proof-gated reconstruction NOT run by report (use `vmpx clean`); \
                 OBSERVED (one run): {} dynamically resolved import(s)",
                d.observed_imports
            ),
        },
        None => Stage {
            status: "not run",
            detail: "protected-import resolution is dynamic-only; static proof-gated \
                     reconstruction is `vmpx clean`; observed tier needs --dynamic"
                .into(),
        },
    };
    let gate_cfgs: usize = sweep
        .assumption_sets
        .iter()
        .filter(|(k, _)| k.contains("OverApproximatedEnvironmentGate"))
        .map(|(_, n)| n)
        .sum();
    let anti_analysis = Stage {
        status: "over-approximated, not neutralized",
        detail: format!(
            "{gate_cfgs} complete CFG(s) crossed a CPUID/TSC environment gate by both-target \
             over-approximation (labeled benign); guards are not rewritten"
        ),
    };

    Ok(Report {
        schema_version: 1,
        path: input_path.to_owned(),
        packer: packer_kind,
        vmprotect_verdict: format!("{:?}", protection.verdict),
        protection_mode: format!("{:?}", protection.protection_mode.mode),
        packing,
        memory_protection,
        imports_parsed: format!(
            "{modules} module(s), {symbols} symbol(s) in the analysis image's import directory"
        ),
        import_resolution,
        anti_analysis,
        sweep,
        n3_proven_equivalent: Stage {
            status: "not wired",
            detail: "no PE-level lift->verify chain exists in vmpx-verify's public API \
                     (it checks Expr pairs); cannot sample equivalence from the CLI"
                .into(),
        },
        serve_ro_completions: Stage {
            status: "not wired",
            detail: "no serve-RO recovery API exists in this branch".into(),
        },
        observed: match &dynamic {
            Some(d) => Stage {
                status: "run",
                detail: format!(
                    "{} of {} incomplete swept entr{} complete ONLY when seeded by one detonation \
                     ({} steps, stop {}, {} observed global(s)); valid for that run/input only, never sound",
                    d.observed_tier_completions,
                    d.observed_tier_rerun_entries,
                    if d.observed_tier_rerun_entries == 1 {
                        "y"
                    } else {
                        "ies"
                    },
                    d.detonation_steps,
                    d.detonation_stop,
                    d.observed_globals
                ),
            },
            None => Stage {
                status: "not run",
                detail: "pass --dynamic (runs the contained emulator; can be very slow)".into(),
            },
        },
        dynamic,
        warnings,
    })
}

#[must_use]
pub fn render(r: &Report) -> String {
    use std::fmt::Write as _;
    let s = &r.sweep;
    let mut o = String::new();
    let _ = writeln!(o, "vmpx report: {}", r.path);
    let _ = writeln!(
        o,
        "  packer: {}   vmprotect: {}   mode: {}",
        r.packer, r.vmprotect_verdict, r.protection_mode
    );
    let scope = if s.sampled {
        format!(
            "SAMPLE: first {} of {} discovered entries, not the whole image",
            s.swept, s.discovered
        )
    } else {
        format!("all {} discovered entries", s.discovered)
    };
    let _ = writeln!(o, "\nHEADLINE (static, sound path; scope: {scope})");
    let _ = writeln!(
        o,
        "  N1 CFG-complete (any labeled assumption) : {} / {}",
        s.n1_cfg_complete, s.swept
    );
    let _ = writeln!(
        o,
        "  N2 PROVEN (empty assumption set)         : {} / {}",
        s.n2_proven, s.swept
    );
    let _ = writeln!(
        o,
        "  N3 proven-equivalent                     : {} ({})",
        r.n3_proven_equivalent.status, r.n3_proven_equivalent.detail
    );
    let _ = writeln!(o, "  N1 breakdown by assumption tier:");
    let _ = writeln!(o, "    proven (no assumptions)   : {}", s.n2_proven);
    let _ = writeln!(
        o,
        "    benign-labeled (PreferredImageBase / OverApproximatedEnvironmentGate): {}",
        s.benign_labeled
    );
    let _ = writeln!(
        o,
        "    UNPROVEN assumption (AssumesExternalStoresDisjointFromVmStack)       : {}",
        s.unproven_assumption
    );
    let _ = writeln!(
        o,
        "  not complete: incomplete {}, errors {}, panics {}",
        s.incomplete, s.errors, s.panics
    );
    for (k, n) in &s.assumption_sets {
        let _ = writeln!(o, "    {n:>4} x {k}");
    }
    let _ = writeln!(o, "\nEXPERIMENTAL TIERS (never counted in N1/N2/N3)");
    let _ = writeln!(
        o,
        "  serve-RO completions : {} ({})",
        r.serve_ro_completions.status, r.serve_ro_completions.detail
    );
    let _ = writeln!(
        o,
        "  observed-tier        : {} ({})",
        r.observed.status, r.observed.detail
    );
    let _ = writeln!(o, "\nSTAGES");
    let _ = writeln!(
        o,
        "  packing            : {} ({})",
        r.packing.status, r.packing.detail
    );
    let _ = writeln!(
        o,
        "  memory protection  : {} ({})",
        r.memory_protection.status, r.memory_protection.detail
    );
    let _ = writeln!(o, "  imports parsed     : {}", r.imports_parsed);
    let _ = writeln!(
        o,
        "  import resolution  : {} ({})",
        r.import_resolution.status, r.import_resolution.detail
    );
    let _ = writeln!(
        o,
        "  anti-analysis      : {} ({})",
        r.anti_analysis.status, r.anti_analysis.detail
    );
    for w in &r.warnings {
        let _ = writeln!(o, "  warning: {w}");
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_rank_by_weakest_assumption() {
        assert_eq!(tier_of(&[]), Tier::Proven);
        assert_eq!(
            tier_of(&[VmCfgAssumption::PreferredImageBase]),
            Tier::BenignLabeled
        );
        assert_eq!(
            tier_of(&[
                VmCfgAssumption::PreferredImageBase,
                VmCfgAssumption::AssumesExternalStoresDisjointFromVmStack
            ]),
            Tier::UnprovenAssumption
        );
        assert_eq!(
            tier_of(&[
                VmCfgAssumption::AssumesExternalStoresDisjointFromVmStack,
                VmCfgAssumption::ObservedGlobal
            ]),
            Tier::Observed
        );
    }

    #[test]
    fn tally_keeps_n1_n2_separate() {
        let outcomes = vec![
            Outcome::Complete(vec![]),
            Outcome::Complete(vec![VmCfgAssumption::PreferredImageBase]),
            Outcome::Complete(vec![
                VmCfgAssumption::AssumesExternalStoresDisjointFromVmStack,
            ]),
            Outcome::Incomplete,
            Outcome::Error,
        ];
        let s = tally(&outcomes, 9, true);
        assert_eq!(
            (
                s.n1_cfg_complete,
                s.n2_proven,
                s.benign_labeled,
                s.unproven_assumption
            ),
            (3, 1, 1, 1)
        );
        assert_eq!(
            (s.incomplete, s.errors, s.swept, s.discovered),
            (1, 1, 5, 9)
        );
    }
}
