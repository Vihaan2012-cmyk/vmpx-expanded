// In-process devirt coverage sweep over ALL VM entries of a (possibly packed)
// VMProtect image. Loads/unpacks the analysis image ONCE, discovers every VM
// entry, recovers each entry's CFG with a bounded per-entry budget (mirroring
// the `cfgdiag` example), catches per-entry panics, and tallies completeness
// plus the stall-cause distribution. READ-ONLY: no engine changes, static only.
//
// Usage: xpl_coverage [IMAGE] [SAMPLE] [THREADS] [INSTRUCTION_LIMIT] [X_DIAG_JSONL] [CAUSE_DUMP]
//   IMAGE    defaults to E:/downloads/systems.original.xpl
//   SAMPLE   entries to sweep: a number for a sample, or "all" for the full
//            image. OMITTED defaults to a SMALL memory-safe sample (50) so a
//            naive run can never OOM; the full image is explicit opt-in ("all").
//   env XPL_ENTRY_TIMEOUT_SECS  per-entry wall-clock cap (default 60, 0 = unbounded);
//            a capped entry is reported Incomplete with limit WallClock.
//   THREADS  worker threads (default 4). Use 1 for a low-memory single-threaded
//            run.
//
// Each entry is bounded by the recovery budget in `recover_one` (blocks,
// visits, decoded instructions, and cross-profile transitions), so no single
// entry can allocate without limit regardless of SAMPLE/THREADS.

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

use vmpx_pe::PeImage;
use vmpx_vm::{
    BranchRefusalDiagnostic, EnvironmentGateFork, EnvironmentGateRoute, take_environment_gate_forks, EntryProfileRecoveryLimits, ObservedBranchTargets, ObservedRun,
    VmCfgAssumption, VmCfgLimits, VmDiscoveryLimits, discover_vm_entries, recover_vm_cfg,
    recover_vm_cfg_with_observed, set_branch_refusal_diagnostics, take_branch_refusal_diagnostics,
};

const DEFAULT_IMAGE: &str = "E:/downloads/systems.original.xpl";
const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;

// Environment-sourced branches seen by every entry (rva, fork); filled from the
// worker threads, reported once at the end. Diagnostic only.
static ENV_FORKS: std::sync::Mutex<Vec<(u32, EnvironmentGateFork)>> =
    std::sync::Mutex::new(Vec::new());

// (entry rva, page number, proven) of suppressed-read image serves; audit only.
static SERVED_PAGES: std::sync::Mutex<Vec<(u32, u64, u64, u8)>> = std::sync::Mutex::new(Vec::new());
static COMPLETES: std::sync::Mutex<Vec<(u32, bool)>> = std::sync::Mutex::new(Vec::new());

// Post-init snapshot bytes for the volatile ranges (OBSERVED tier): consulted only
// where the file byte is vetoed, labeled ObservedGlobal.
static OBSERVED_SLOTS: std::sync::OnceLock<vmpx_vm::ObservedMemory> = std::sync::OnceLock::new();

enum Outcome {
    Complete { assumptions: Vec<VmCfgAssumption> },
    // Non-complete but recovery returned a CFG: distinct normalized stall
    // causes this entry exhibited, and the resource limit hit (if any).
    Incomplete {
        causes: Vec<String>,
        raw: Vec<String>,
        limit: Option<String>,
        diags: Vec<BranchRefusalDiagnostic>,
    },
    // recover_vm_cfg returned Err before producing a CFG.
    Error(String),
    Panic,
}

struct EntryResult {
    rva: u32,
    outcome: Outcome,
}

// Bucket a stall reason by stripping concrete addresses/numbers so that e.g.
// "memory_load at 0x1a2b" and "memory_load at 0x9f00" collapse to one cause.
fn normalize(reason: &str) -> String {
    let bytes = reason.as_bytes();
    let mut out = String::with_capacity(reason.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'0' && i + 1 < bytes.len() && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
            i += 2;
            while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                i += 1;
            }
            out.push('#');
        } else if c.is_ascii_digit() {
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            out.push('#');
        } else {
            out.push(c as char);
            i += 1;
        }
    }
    out
}


// Diagnostic category of one raw stall reason (see the cause-set report).
// A/B/N = runtime-dependent input; C..K = statically addressable.
fn categorize(reason: &str) -> char {
    if reason.contains("memory_load_epoch_") {
        if reason.contains("used directly") { 'A' } else { 'B' }
    } else if reason.contains("native_rax_initial") {
        'N'
    } else if reason.contains("is outside the input image") {
        'D'
    } else if reason.contains("not file-backed") || reason.contains("not executable") {
        'E'
    } else if reason.contains("non-executable data section") {
        'F'
    } else if reason.contains("different executable section") {
        'I'
    } else if reason.contains("InstructionLimit") {
        'K'
    } else if reason.contains("PartiallyRecovered") {
        'H'
    } else if reason.contains("; operand ") {
        'G'
    } else if reason.contains("expected one dispatch codec") {
        'C'
    } else {
        'J'
    }
}

fn parse_hex_after(reason: &str, marker: &str) -> Option<u64> {
    let rest = &reason[reason.find(marker)? + marker.len()..];
    let hex = rest.strip_prefix("0x")?;
    let end = hex.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(hex.len());
    u64::from_str_radix(&hex[..end], 16).ok()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn recover_one(
    pe: &PeImage<'_>,
    entry: &vmpx_vm::VmEntryCandidate,
    instruction_limit: usize,
    max_wall_time: Option<std::time::Duration>,
) -> Outcome {
    // Mirror cfgdiag's recovery budget: default CFG limits (which bound blocks,
    // visits, decoded instructions, and cross-profile transitions per entry),
    // with a bumped per-block instruction budget. Every entry is hard-bounded,
    // so none can allocate without limit.
    let limits = VmCfgLimits {
        entry: EntryProfileRecoveryLimits {
            // A per-block decode budget large enough for any real handler stream
            // block but small enough that one block's decoded-instruction vector
            // and symbolic state can't balloon memory (post-transition, a long
            // linear stream can decode before a clean boundary).
            instruction_limit,
            symbolic: vmpx_vm::SymbolicExecutionLimits {
                serve_read_only_when_suppressed: std::env::var_os("XPL_SERVE_RO").is_some(),
                prove_read_only_when_suppressed: std::env::var_os("XPL_NO_PROVEN_RO").is_none(),
                ..vmpx_vm::SymbolicExecutionLimits::default()
            },
            ..EntryProfileRecoveryLimits::default()
        },
        max_wall_time,
        max_profile_transitions: env_usize("XPL_MAX_TRANS", VmCfgLimits::default().max_profile_transitions),
        max_transition_visits: env_usize("XPL_TRANS_VISITS", VmCfgLimits::default().max_transition_visits),
        max_transition_instructions: env_usize("XPL_TRANS_INSTR", VmCfgLimits::default().max_transition_instructions),
        ..VmCfgLimits::default()
    };
    set_branch_refusal_diagnostics(true);
    let audit = std::env::var_os("XPL_SERVED_PAGES_OUT").is_some();
    vmpx_vm::set_served_page_audit(audit);
    let result = panic::catch_unwind(AssertUnwindSafe(|| match OBSERVED_SLOTS.get() {
        Some(observed) => vmpx_vm::recover_vm_cfg_with_observed_memory(pe, entry, limits, observed),
        None => recover_vm_cfg(pe, entry, limits),
    }));
    let diags = take_branch_refusal_diagnostics();
    if audit {
        // Coalesce per provenance into contiguous [start, end] runs.
        let mut runs: Vec<(u32, u64, u64, u8)> = Vec::new();
        let all = vmpx_vm::take_served_page_audit();
        for kind in 0..3u8 {
            let mut last: Option<(u64, u64)> = None;
            for (addr, _) in all.iter().copied().filter(|(_, k)| *k == kind) {
                match last {
                    Some((s0, e0)) if addr == e0 + 1 => last = Some((s0, addr)),
                    Some((s0, e0)) => {
                        runs.push((entry.rva.0, s0, e0, kind));
                        last = Some((addr, addr));
                    }
                    None => last = Some((addr, addr)),
                }
            }
            if let Some((s0, e0)) = last {
                runs.push((entry.rva.0, s0, e0, kind));
            }
        }
        SERVED_PAGES.lock().unwrap_or_else(|e| e.into_inner()).extend(runs);
    }
    {
        let forks = take_environment_gate_forks();
        if !forks.is_empty() {
            let mut sink = ENV_FORKS.lock().unwrap_or_else(|e| e.into_inner());
            sink.extend(forks.into_iter().map(|f| (entry.rva.0, f)));
        }
    }
    match result {
        Err(_) => Outcome::Panic,
        Ok(Err(error)) => Outcome::Error(normalize(&error.to_string())),
        Ok(Ok(cfg)) => {
            if cfg.all_paths_reach_vm_exit() {
                COMPLETES
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push((entry.rva.0, cfg.assumptions.is_empty()));
                Outcome::Complete { assumptions: cfg.assumptions.iter().copied().collect() }
            } else {
                let mut causes: Vec<String> = cfg
                    .stalls
                    .iter()
                    .map(|stall| normalize(&stall.reason))
                    .collect();
                causes.sort();
                causes.dedup();
                let mut raw: Vec<String> = cfg.stalls.iter().map(|stall| stall.reason.clone()).collect();
                raw.sort();
                raw.dedup();
                // A complete-but-no-exit CFG (is_complete yet no VmExit block)
                // has neither stalls nor limit; label it so it isn't invisible.
                if causes.is_empty() && cfg.limit_hit.is_none() {
                    causes.push("(no stall, no vm-exit block)".to_owned());
                }
                Outcome::Incomplete {
                    causes,
                    raw,
                    limit: cfg.limit_hit.map(|hit| format!("{hit:?}")),
                    diags,
                }
            }
        }
    }
}

fn parse_u64(token: &str) -> Option<u64> {
    token
        .strip_prefix("0x")
        .map_or_else(|| token.parse().ok(), |hex| u64::from_str_radix(hex, 16).ok())
}

fn run_bridge_pass(
    pe: &PeImage<'_>,
    entries: &[vmpx_vm::VmEntryCandidate],
    results: &[EntryResult],
    path: &str,
    instruction_limit: usize,
) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let edges: Vec<(u64, u64)> = text
        .lines()
        .filter_map(|line| {
            let mut it = line.split_whitespace();
            Some((parse_u64(it.next()?)?, parse_u64(it.next()?)?))
        })
        .collect();
    let seed = ObservedBranchTargets::from_vip_edges(edges.iter().copied());
    println!("\n==================== OBSERVED-BRIDGE PASS (EXPERIMENTAL) ====================");
    println!(
        "observed edges read: {}; far-jump branch sites kept: {}",
        edges.len(),
        seed.site_count()
    );
    let mut run = ObservedRun::default();
    run.branch_targets = seed.clone();
    let limits = VmCfgLimits {
        entry: EntryProfileRecoveryLimits { instruction_limit, ..EntryProfileRecoveryLimits::default() },
        ..VmCfgLimits::default()
    };
    let (mut retried, mut gained, mut still_stalled, mut errors) = (0usize, 0usize, 0usize, 0usize);
    let (mut stalled_sites, mut sites_with_evidence, mut near_miss) = (0usize, 0usize, 0usize);
    let seed_sites: Vec<u64> = seed.sites().collect();
    let mut engaged = 0usize;
    let mut keyed_reasons: Vec<String> = Vec::new();
    let mut gained_rvas: Vec<u32> = Vec::new();
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    for (entry, result) in entries.iter().zip(results) {
        if !matches!(result.outcome, Outcome::Incomplete { .. }) {
            continue;
        }
        retried += 1;
        let cfg = panic::catch_unwind(AssertUnwindSafe(|| {
            recover_vm_cfg_with_observed(pe, entry, limits, &run)
        }));
        match cfg {
            Ok(Ok(cfg)) => {
                for stall in &cfg.stalls {
                    stalled_sites += 1;
                    if !seed.targets_for(stall.vip.get()).is_empty() {
                        sites_with_evidence += 1;
                        if keyed_reasons.len() < 12 {
                            keyed_reasons.push(format!("{:#x} {}", entry.rva.0, stall.reason));
                        }
                    } else if seed_sites.iter().any(|s| s.abs_diff(stall.vip.get()) <= 0x40) {
                        near_miss += 1;
                    }
                }
                if cfg.assumptions.contains(&VmCfgAssumption::ObservedBranchTarget) {
                    engaged += 1;
                }
                if cfg.complete_with_observed_branch_help() {
                    gained += 1;
                    gained_rvas.push(entry.rva.0);
                } else {
                    still_stalled += 1;
                }
            }
            _ => errors += 1,
        }
    }
    panic::set_hook(hook);
    println!("  re-walked non-complete entries                    : {retried}");
    println!("  NEW complete-with-observed-help (EXPERIMENTAL tier): {gained}");
    println!("  still not complete                                : {still_stalled}");
    println!("  error/panic                                       : {errors}");
    println!(
        "  residual stalled sites: {stalled_sites}; of which keyed in observed evidence: {sites_with_evidence}; within 0x40 of an observed site but not exact (keying-offset suspects): {near_miss}"
    );
    println!("  entries where the bridge followed >=1 observed target   : {engaged}");
    for r in &keyed_reasons {
        println!("  keyed-but-unresolved: {r}");
    }
    let ex: Vec<String> = gained_rvas.iter().take(10).map(|r| format!("{r:#x}")).collect();
    println!("  e.g. {}", ex.join(", "));
    println!("  (observed-tier only: lower-bound evidence, NEVER added to the proven count)");
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_path = args.get(1).map_or(DEFAULT_IMAGE, String::as_str);
    // Memory-safe default: a small sample unless the caller explicitly asks for
    // "all" (full image) or names a sample size. 0 means "no cap" (full image).
    const DEFAULT_SAMPLE: usize = 50;
    let max_entries: usize = match args.get(2).map(String::as_str) {
        None => DEFAULT_SAMPLE,
        Some("all") => 0,
        Some(n) => n.parse().unwrap_or(DEFAULT_SAMPLE),
    };
    // A low default thread count keeps a naive multi-threaded run's peak memory
    // modest (each concurrent entry holds its own bounded recovery state over the
    // shared image). Raise it deliberately for a faster bounded sweep.
    let max_threads: usize = args
        .get(3)
        .and_then(|a| a.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(2);
    // Optional dynamic-evidence veto: file of hex page numbers (VA >> 12), one per
    // line, that a run showed to change after load; never served.
    if let Some(path) = std::env::var_os("XPL_VOLATILE_PAGES") {
        let text = std::fs::read_to_string(path).expect("read XPL_VOLATILE_PAGES");
        vmpx_vm::set_volatile_image_pages(text.lines().filter_map(|line| {
            u64::from_str_radix(line.trim().trim_start_matches("0x"), 16).ok()
        }));
    }
    if let Some(path) = std::env::var_os("XPL_OBSERVED_SLOTS") {
        let text = std::fs::read_to_string(path).expect("read XPL_OBSERVED_SLOTS");
        let slots = text.lines().filter_map(|line| {
            let (start, hex) = line.split_once(' ')?;
            let bytes = (0..hex.len() / 2)
                .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16))
                .collect::<Result<Vec<u8>, _>>()
                .ok()?;
            Some((u64::from_str_radix(start, 16).ok()?, bytes))
        });
        let _ = OBSERVED_SLOTS.set(vmpx_vm::ObservedMemory::from_slots(slots));
    }
    if let Some(path) = std::env::var_os("XPL_VOLATILE_RANGES") {
        let text = std::fs::read_to_string(path).expect("read XPL_VOLATILE_RANGES");
        vmpx_vm::set_volatile_image_ranges(text.lines().filter_map(|line| {
            let mut parts = line.split_whitespace();
            let start = u64::from_str_radix(parts.next()?, 16).ok()?;
            let end = u64::from_str_radix(parts.next()?, 16).ok()?;
            Some((start, end))
        }));
    }
    // Per-block decode budget (4th arg). Default 512 keeps a naive run memory-safe;
    // raise it deliberately (with memory headroom) to reclaim InstructionLimit stalls.
    let instruction_limit: usize = args.get(4).and_then(|a| a.parse().ok()).unwrap_or(512);
    // Per-entry wall-clock cap in seconds via XPL_ENTRY_TIMEOUT_SECS. Default 60; 0 = unbounded.
    let max_wall_time = match std::env::var("XPL_ENTRY_TIMEOUT_SECS")
        .ok()
        .and_then(|a| a.parse::<u64>().ok())
    {
        Some(0) => None,
        Some(secs) => Some(std::time::Duration::from_secs(secs)),
        None => Some(vmpx_vm::DEFAULT_MAX_WALL_TIME),
    };

    let t0 = Instant::now();
    let bytes = std::fs::read(image_path).unwrap_or_else(|e| panic!("read {image_path}: {e}"));
    let input_pe = PeImage::parse(&bytes).expect("input is not a supported PE");

    // Decide packed-ness with the same heuristic vmpx-detect uses for
    // PackedVirtualSections (>= 3 initialized virtual-only sections).
    let virtual_only = input_pe
        .sections
        .iter()
        .filter(|s| {
            s.pointer_to_raw_data.0 == 0
                && s.size_of_raw_data == 0
                && s.virtual_size != 0
                && s.characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA == 0
        })
        .count();
    let packed = virtual_only >= 3;

    // Unpack ONCE to the mapped analysis image (relayout is not needed:
    // discovery runs directly on the mapped image, as `vmpx clean` does).
    let mapped: Vec<u8> = if packed {
        println!("[load] packed ({virtual_only} virtual-only sections) -> unpacking in-process");
        vmpx_unpack::unpack_to_mapped_image(&bytes, vmpx_unpack::UnpackLimits::default())
            .expect("unpack failed")
            .image
    } else {
        println!("[load] not packed -> using input image directly");
        bytes.clone()
    };
    let pe = PeImage::parse(&mapped).expect("mapped analysis image failed to reparse");
    println!(
        "[load] analysis image {} MiB ready in {:.1}s",
        mapped.len() / (1024 * 1024),
        t0.elapsed().as_secs_f64()
    );

    // Discover ALL VM entries (scan on = default): finds the full set.
    let td = Instant::now();
    let discovery =
        discover_vm_entries(&pe, VmDiscoveryLimits::default()).expect("VM discovery failed");
    println!(
        "[discover] {} entries across {} bootstrap instances in {:.1}s",
        discovery.entries.len(),
        discovery.instances.len(),
        td.elapsed().as_secs_f64()
    );

    let mut entries: Vec<_> = discovery.entries.clone();
    let total_discovered = entries.len();
    if max_entries > 0 && entries.len() > max_entries {
        entries.truncate(max_entries);
        println!(
            "[sweep] SAMPLING first {}/{} entries (MAX_ENTRIES cap)",
            entries.len(),
            total_discovered
        );
    }
    let swept = entries.len();

    // Quiet the panic hook so one bad entry doesn't flood stderr; restore after.
    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    // Bounded parallelism: split entries across a few scoped threads sharing &pe.
    let ts = Instant::now();
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get().min(max_threads))
        .max(1);
    let chunk = entries.len().div_ceil(threads).max(1);
    let pe_ref = &pe;
    let results: Vec<EntryResult> = std::thread::scope(|scope| {
        let handles: Vec<_> = entries
            .chunks(chunk)
            .map(|slice| {
                scope.spawn(move || {
                    slice
                        .iter()
                        .map(|entry| EntryResult {
                            rva: entry.rva.0,
                            outcome: recover_one(pe_ref, entry, instruction_limit, max_wall_time),
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_default())
            .collect()
    });
    panic::set_hook(prev_hook);
    let sweep_secs = ts.elapsed().as_secs_f64();

    // ---------------- EXPERIMENTAL observed-branch bridge pass ----------------
    // Only when VMPX_OBSERVED_EDGES names a file of dynamic VIP->VIP edges
    // ("from to" per line, hex or decimal). Re-walks ONLY the non-complete
    // entries seeded with the observed far-jump targets; the proven tier above
    // is untouched. Observed-tier numbers are reported separately.
    if let Ok(path) = std::env::var("VMPX_OBSERVED_EDGES") {
        run_bridge_pass(&pe, &entries, &results, &path, instruction_limit);
    }

    // Tally.
    let mut complete = 0usize;
    let mut panics = 0usize;
    let mut errors = 0usize;
    let mut incomplete = 0usize;
    let mut limit_hits: BTreeMap<String, usize> = BTreeMap::new();
    // cause -> (entry count, example RVAs). Entries counted once per distinct cause.
    let mut causes: BTreeMap<String, (usize, Vec<u32>)> = BTreeMap::new();
    let mut error_causes: BTreeMap<String, (usize, Vec<u32>)> = BTreeMap::new();

    let add = |map: &mut BTreeMap<String, (usize, Vec<u32>)>, key: String, rva: u32| {
        let slot = map.entry(key).or_insert((0, Vec::new()));
        slot.0 += 1;
        if slot.1.len() < 5 {
            slot.1.push(rva);
        }
    };

    // Per-unrecovered-function diagnostic data: category set, raw reasons.
    let mut fn_sets: Vec<(u32, BTreeSet<char>)> = Vec::new();
    let mut d_vas: Vec<(u32, u64)> = Vec::new();
    let mut complete_sets: Vec<(u32, Vec<VmCfgAssumption>)> = Vec::new();
    for r in &results {
        match &r.outcome {
            Outcome::Complete { assumptions } => {
                complete += 1;
                complete_sets.push((r.rva, assumptions.clone()));
            }
            Outcome::Panic => panics += 1,
            Outcome::Error(msg) => {
                errors += 1;
                fn_sets.push((r.rva, BTreeSet::from(['J'])));
                add(&mut error_causes, format!("cfg_error: {msg}"), r.rva);
            }
            Outcome::Incomplete { causes: cs, raw, limit, .. } => {
                incomplete += 1;
                let mut set: BTreeSet<char> = raw.iter().map(|x| categorize(x)).collect();
                if set.is_empty() {
                    set.insert('J');
                }
                for x in raw {
                    if categorize(x) == 'D' {
                        if let Some(va) = parse_hex_after(x, "handler VA ") {
                            d_vas.push((r.rva, va));
                        }
                    }
                }
                fn_sets.push((r.rva, set));
                if let Some(hit) = limit {
                    *limit_hits.entry(hit.clone()).or_insert(0) += 1;
                }
                for c in cs {
                    add(&mut causes, c.clone(), r.rva);
                }
            }
        }
    }

    let swept_f = swept.max(1) as f64;
    println!("\n==================== COVERAGE ====================");
    println!(
        "swept {swept} entries (of {total_discovered} discovered) in {sweep_secs:.1}s ({} threads, {:.1} ms/entry avg)",
        threads,
        sweep_secs * 1000.0 / swept_f
    );
    println!("  complete (all_paths_reach_vm_exit) : {complete}");
    println!("  incomplete (stall / limit)         : {incomplete}");
    println!("  cfg_error (no CFG)                 : {errors}");
    println!("  panic                              : {panics}");

    if !limit_hits.is_empty() {
        println!("\n-- resource-limit hits (subset of incomplete) --");
        let mut lh: Vec<_> = limit_hits.iter().collect();
        lh.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for (hit, n) in lh {
            println!("  {n:>5}  {hit}");
        }
    }

    println!("\n-- stall-cause distribution (entries exhibiting each cause; an entry may show several) --");
    let mut dist: Vec<_> = causes.iter().collect();
    dist.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    for (cause, (n, examples)) in &dist {
        let ex: Vec<String> = examples.iter().map(|r| format!("{r:#x}")).collect();
        println!("  {n:>5}  {cause}");
        println!("         e.g. {}", ex.join(", "));
    }

    if !error_causes.is_empty() {
        println!("\n-- cfg_error causes --");
        let mut ed: Vec<_> = error_causes.iter().collect();
        ed.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        for (cause, (n, examples)) in &ed {
            let ex: Vec<String> = examples.iter().map(|r| format!("{r:#x}")).collect();
            println!("  {n:>5}  {cause}");
            println!("         e.g. {}", ex.join(", "));
        }
    }


    // ---------------- cause-set diagnostics (E1) ----------------
    let unrecovered = fn_sets.len();
    println!("\n==================== CAUSE-SET DIAGNOSTICS ====================");
    println!("unrecovered functions (incomplete + cfg_error): {unrecovered}");
    let mut card: BTreeMap<usize, usize> = BTreeMap::new();
    for (_, set) in &fn_sets {
        *card.entry(set.len()).or_insert(0) += 1;
    }
    println!("\n-- cardinality of distinct-category set per unrecovered function --");
    for (k, n) in &card {
        println!("  {k} categories : {n}");
    }
    let mut per_cat: BTreeMap<char, usize> = BTreeMap::new();
    for (_, set) in &fn_sets {
        for c in set {
            *per_cat.entry(*c).or_insert(0) += 1;
        }
    }
    println!("\n-- functions per category (a function counts once per category) --");
    for (c, n) in &per_cat {
        println!("  {c} : {n}");
    }
    let mut sets_hist: BTreeMap<String, usize> = BTreeMap::new();
    for (_, set) in &fn_sets {
        *sets_hist.entry(set.iter().collect()).or_insert(0) += 1;
    }
    let mut sh: Vec<_> = sets_hist.iter().collect();
    sh.sort_by_key(|(k, n)| (std::cmp::Reverse(**n), (*k).clone()));
    println!("\n-- exact category-set histogram --");
    for (k, n) in sh {
        println!("  {n:>4}  {{{k}}}");
    }
    let is_runtime = |c: &char| matches!(c, 'A' | 'B' | 'N');
    let addr_only = fn_sets.iter().filter(|(_, s)| !s.iter().any(is_runtime)).count();
    let addr_only_no_k = fn_sets
        .iter()
        .filter(|(_, s)| !s.iter().any(|c| is_runtime(c) || *c == 'K'))
        .count();
    let addr_only_ab = fn_sets.iter().filter(|(_, s)| !s.contains(&'A') && !s.contains(&'B')).count();
    let total_f = swept.max(1) as f64;
    println!("\n-- TRUE STATIC CEILING (addressable-only) --");
    println!(
        "  no A/B/N (N=native_rax_initial runtime)   : {addr_only}  -> ceiling {}/{} = {:.1}%",
        complete + addr_only,
        swept,
        (complete + addr_only) as f64 * 100.0 / total_f
    );
    println!(
        "  no A/B only (N allowed)                   : {addr_only_ab}  -> ceiling {:.1}%",
        (complete + addr_only_ab) as f64 * 100.0 / total_f
    );
    println!(
        "  no A/B/N and no K(InstructionLimit)       : {addr_only_no_k}  -> ceiling {:.1}%",
        (complete + addr_only_no_k) as f64 * 100.0 / total_f
    );
    let has_runtime = fn_sets.iter().filter(|(_, s)| s.iter().any(is_runtime)).count();
    println!("  functions with any runtime cause (A/B/N)  : {has_runtime}");

    let c_fns: Vec<_> = fn_sets.iter().filter(|(_, s)| s.contains(&'C')).collect();
    let c_de = c_fns.iter().filter(|(_, s)| s.contains(&'D') || s.contains(&'E')).count();
    let c_d = c_fns.iter().filter(|(_, s)| s.contains(&'D')).count();
    let c_e = c_fns.iter().filter(|(_, s)| s.contains(&'E')).count();
    println!("\n-- C x (D u E) co-occurrence --");
    println!(
        "  C functions: {}; also D or E: {c_de} ({:.1}%); D: {c_d}; E: {c_e}",
        c_fns.len(),
        c_de as f64 * 100.0 / c_fns.len().max(1) as f64
    );
    let c_only_de = c_fns
        .iter()
        .filter(|(_, s)| s.iter().all(|c| matches!(c, 'C' | 'D' | 'E')))
        .count();
    println!("  C functions whose set is a subset of {{C,D,E}} : {c_only_de}");
    let c_any_fns = results
        .iter()
        .filter_map(|r| match &r.outcome {
            Outcome::Incomplete { raw, .. }
                if raw.iter().any(|x| x.contains("expected one dispatch codec")) =>
            {
                Some(r.rva)
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let c_any_de = fn_sets
        .iter()
        .filter(|(rva, s)| c_any_fns.contains(rva) && (s.contains(&'D') || s.contains(&'E')))
        .count();
    println!(
        "  variant: any reason with 'expected one dispatch codec' (C or G): {}; also D/E: {c_any_de} ({:.1}%)",
        c_any_fns.len(),
        c_any_de as f64 * 100.0 / c_any_fns.len().max(1) as f64
    );

    let base = pe.optional.image_base.0;
    let soi = u64::from(pe.optional.size_of_image);
    println!("\n-- cause-D out-of-image handler VAs (base {base:#x}, SizeOfImage {soi:#x}) --");
    let d_fns: BTreeSet<u32> = d_vas.iter().map(|(r, _)| *r).collect();
    println!("  D functions: {}; (fn,VA) pairs: {}", d_fns.len(), d_vas.len());
    let mut bucket: BTreeMap<&str, usize> = BTreeMap::new();
    let mut rebase_in = 0usize;
    let mut rebase_fns: BTreeSet<u32> = BTreeSet::new();
    let alt_bases: [u64; 5] = [0x40_0000, 0x1000_0000, 0x1_4000_0000, 0x1_8000_0000, base];
    let mut rebase_kinds: BTreeMap<&str, usize> = BTreeMap::new();
    for (rva, va) in &d_vas {
        let name = if *va < base {
            if *va < soi {
                "below base: unbased RVA-like (< SizeOfImage)"
            } else if *va >= base.saturating_sub(0x1000_0000) {
                "below base: within 256 MiB under base"
            } else {
                "below base: far (scattered)"
            }
        } else {
            let delta = *va - base;
            if delta < soi {
                "in-image (unexpected)"
            } else if delta < soi + 0x10_0000 {
                "(a) just past SizeOfImage (<1 MiB)"
            } else if delta < 0x1_0000_0000 {
                "(b) base + large offset (<4 GiB)"
            } else {
                "(c) scattered wild (>= base + 4 GiB)"
            }
        };
        *bucket.entry(name).or_insert(0) += 1;
        let mut hit: Option<&str> = None;
        if *va < soi {
            hit = Some("VA itself < SizeOfImage (unbased RVA)");
        } else if (*va & 0xFFFF_FFFF) >= (base & 0xFFFF_FFFF)
            && (*va & 0xFFFF_FFFF) - (base & 0xFFFF_FFFF) < soi
        {
            hit = Some("low-32(VA) - low-32(base) in image (truncation/wrap)");
        } else if alt_bases.iter().any(|b| *va >= *b && *va - *b < soi) {
            hit = Some("VA minus a common alternate preferred base in image");
        }
        if let Some(h) = hit {
            rebase_in += 1;
            rebase_fns.insert(*rva);
            *rebase_kinds.entry(h).or_insert(0) += 1;
        }
    }
    for (k, n) in &bucket {
        println!("  {n:>5}  {k}");
    }
    println!(
        "  rebase check: {rebase_in}/{} VAs land in-image after rebase ({} functions)",
        d_vas.len(),
        rebase_fns.len()
    );
    for (k, n) in &rebase_kinds {
        println!("    {n:>5}  {k}");
    }
    let mut sample: Vec<u64> = d_vas.iter().map(|(_, v)| *v).collect();
    sample.sort_unstable();
    sample.dedup();
    println!("  distinct D VAs: {}; first 12: {:x?}", sample.len(), &sample[..sample.len().min(12)]);
    println!("  last 12: {:x?}", &sample[sample.len().saturating_sub(12)..]);
    let mut hi: BTreeMap<u64, usize> = BTreeMap::new();
    for (_, v) in &d_vas {
        *hi.entry(v >> 32).or_insert(0) += 1;
    }
    let mut hv: Vec<_> = hi.into_iter().collect();
    hv.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    println!("  high-32 histogram (top 8):");
    for (h, n) in hv.iter().take(8) {
        println!("    {n:>5}  high32={h:#x}");
    }
    // ---------------- cause-B typed flag-fork refusal split (M2) ----------------
    {
        // Per B function: the refusal label of each of its B stalls.
        let mut b_fns: Vec<(u32, BTreeSet<char>, Vec<String>)> = Vec::new();
        for (rva, set) in &fn_sets {
            if !set.contains(&'B') {
                continue;
            }
            let mut labels = Vec::new();
            for r in results.iter().filter(|r| r.rva == *rva) {
                if let Outcome::Incomplete { raw, .. } = &r.outcome {
                    for x in raw.iter().filter(|x| categorize(x) == 'B') {
                        let label = x
                            .split_once("[flagfork=")
                            .and_then(|(_, rest)| rest.rsplit_once(']').map(|(l, _)| l.to_owned()))
                            .unwrap_or_else(|| "NoTag".to_owned());
                        labels.push(label);
                    }
                }
            }
            labels.sort();
            labels.dedup();
            b_fns.push((*rva, set.clone(), labels));
        }
        let family = |l: &str| l.split('(').next().unwrap_or(l).to_owned();
        let too_many = |l: &str| -> Option<usize> {
            l.strip_prefix("TooManyFlags(")?.strip_suffix(')')?.parse().ok()
        };
        let tunable_at = |l: &str, cap: usize| {
            l.split('+').all(|c| match family(c).as_str() {
                "MultiBitMask" | "ConstantCountShift" => true,
                "TooManyFlags" => too_many(c).is_some_and(|n| n <= cap),
                _ => false,
            })
        };
        // Function class under the CURRENT cap (2): TUNABLE only if every B stall is.
        let class_of = |labels: &[String], cap: usize| -> &'static str {
            if !labels.is_empty() && labels.iter().all(|l| tunable_at(l, cap)) {
                "TUNABLE"
            } else if labels.iter().any(|l| l.split('+').any(|c| family(c) == "ResidualFullWidth")) {
                "CLOSED(ResidualFullWidth)"
            } else {
                "OTHER/MIXED"
            }
        };
        println!("\n==================== CAUSE-B FLAG-FORK REFUSAL SPLIT (M2) ====================");
        println!("cause-B functions (any B stall): {}", b_fns.len());
        let mut per_label: BTreeMap<String, usize> = BTreeMap::new();
        for (_, _, labels) in &b_fns {
            for l in labels {
                *per_label.entry(l.clone()).or_insert(0) += 1;
            }
        }
        println!("\n-- B functions per distinct refusal label (a function counts once per label) --");
        let mut pl: Vec<_> = per_label.iter().collect();
        pl.sort_by_key(|(k, n)| (std::cmp::Reverse(**n), (*k).clone()));
        for (l, n) in pl {
            println!("  {n:>4}  {l}");
        }
        let groups: [(&str, Box<dyn Fn(&BTreeSet<char>) -> bool>); 4] = [
            ("all B functions", Box::new(|_| true)),
            ("pure {B}", Box::new(|s| s.len() == 1)),
            ("{B,D}", Box::new(|s| s.len() == 2 && s.contains(&'D'))),
            ("{B,E}", Box::new(|s| s.len() == 2 && s.contains(&'E'))),
        ];
        println!("\n-- function class at current MAX_FLAGS=2 (TUNABLE = every B stall is TooManyFlags/MultiBitMask/ConstantCountShift) --");
        for (name, pred) in &groups {
            let mut tally: BTreeMap<&str, usize> = BTreeMap::new();
            let mut n = 0;
            for (_, s, labels) in b_fns.iter().filter(|(_, s, _)| pred(s)) {
                let _ = s;
                n += 1;
                *tally.entry(class_of(labels, 2)).or_insert(0) += 1;
            }
            println!("  {name}: {n}  {tally:?}");
        }
        println!("\n-- function-level first-label family (pure {{B}} / all B) --");
        for (name, pred) in &groups[..2] {
            let mut fam: BTreeMap<String, usize> = BTreeMap::new();
            for (_, _, labels) in b_fns.iter().filter(|(_, s, _)| pred(s)) {
                let mut fams: Vec<String> =
                    labels.iter().flat_map(|l| l.split('+')).map(|c| family(c)).collect();
                fams.sort();
                fams.dedup();
                *fam.entry(fams.join("+")).or_insert(0) += 1;
            }
            println!("  {name}: {fam:?}");
        }
        println!("\n-- TooManyFlags frontier-size histogram: functions by max n (all B / pure {{B}}) --");
        let max_n = |labels: &[String]| labels.iter().filter_map(|l| too_many(l)).max();
        let mut hist_all: BTreeMap<usize, usize> = BTreeMap::new();
        let mut hist_pure: BTreeMap<usize, usize> = BTreeMap::new();
        for (_, s, labels) in &b_fns {
            if let Some(n) = max_n(labels) {
                *hist_all.entry(n).or_insert(0) += 1;
                if s.len() == 1 {
                    *hist_pure.entry(n).or_insert(0) += 1;
                }
            }
        }
        println!("  all : {hist_all:?}");
        println!("  pure: {hist_pure:?}");
        println!("\n-- functions TUNABLE if MAX_FLAGS were raised to cap (upper bound: enumeration may still fail) --");
        for cap in 2..=8usize {
            let all = b_fns.iter().filter(|(_, _, l)| class_of(l, cap) == "TUNABLE").count();
            let pure = b_fns
                .iter()
                .filter(|(_, s, l)| s.len() == 1 && class_of(l, cap) == "TUNABLE")
                .count();
            let bd = b_fns
                .iter()
                .filter(|(_, s, l)| s.len() == 2 && s.contains(&'D') && class_of(l, cap) == "TUNABLE")
                .count();
            let be = b_fns
                .iter()
                .filter(|(_, s, l)| s.len() == 2 && s.contains(&'E') && class_of(l, cap) == "TUNABLE")
                .count();
            println!("  cap {cap}: all {all}, pure-B {pure}, {{B,D}} {bd}, {{B,E}} {be}");
        }
    }


    // ---------------- environment-gate fork tier ----------------
    {
        let forks = ENV_FORKS.lock().unwrap_or_else(|e| e.into_inner());
        println!("
==================== ENVIRONMENT-GATE FORKS ====================");
        let sites = |route: EnvironmentGateRoute| {
            forks
                .iter()
                .filter(|(_, f)| f.route == route)
                .map(|(_, f)| f.vip.0)
                .collect::<BTreeSet<_>>()
        };
        println!("  environment-sourced branch sites (distinct VIPs):");
        println!("    resolved by the ordinary proven path (unlabeled) : {}", sites(EnvironmentGateRoute::Primary).len());
        println!("    legacy CPUID/timestamp gate (labeled)            : {}", sites(EnvironmentGateRoute::LegacyGate).len());
        println!("    NEWLY forked by the generalized gate (labeled)   : {}", sites(EnvironmentGateRoute::Generalized).len());
        println!("    environment-sourced but REFUSED (stay stalls)    : {}", sites(EnvironmentGateRoute::Refused).len());
        let mut by_source: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
        for (_, f) in forks.iter().filter(|(_, f)| f.route == EnvironmentGateRoute::Generalized) {
            by_source.entry(format!("{:?}", f.sources)).or_default().insert(f.vip.0);
        }
        for (k, v) in &by_source {
            println!("      generalized by source set {k}: {}", v.len());
        }
        let gen_entries: BTreeSet<u32> = forks
            .iter()
            .filter(|(_, f)| f.route == EnvironmentGateRoute::Generalized)
            .map(|(r, _)| *r)
            .collect();
        let gen_complete = results
            .iter()
            .filter(|r| gen_entries.contains(&r.rva) && matches!(r.outcome, Outcome::Complete { .. }))
            .count();
        println!("  entries containing a generalized fork: {} (of which complete: {gen_complete})", gen_entries.len());
        let primary_entries: BTreeSet<u32> = forks
            .iter()
            .filter(|(_, f)| f.route == EnvironmentGateRoute::Primary)
            .map(|(r, _)| *r)
            .collect();
        let primary_proven = complete_sets
            .iter()
            .filter(|(r, a)| a.is_empty() && primary_entries.contains(r))
            .count();
        println!(
            "  entries with an unlabeled (primary-path) environment fork: {} (of which strict-PROVEN complete: {primary_proven})",
            primary_entries.len()
        );
        for (rva, f) in forks.iter().filter(|(_, f)| f.route == EnvironmentGateRoute::Generalized).take(10) {
            println!("    entry {rva:#x} vip {:#x} sources {:?} targets {:x?}", f.vip.0, f.sources, f.targets);
        }
    }

    // ---------------- assumption-tier split of the complete set ----------------
    {
        let tier = |a: &[VmCfgAssumption]| -> &'static str {
            let mut t = "PROVEN";
            for x in a {
                let this = match x {
                    VmCfgAssumption::ObservedGlobal
                    | VmCfgAssumption::ObservedRecursionDepth
                    | VmCfgAssumption::ObservedBranchTarget
                    | VmCfgAssumption::AssumesDynamicBranchTargetsComplete => "OBSERVED",
                    VmCfgAssumption::AssumesExternalStoresDisjointFromVmStack
                    | VmCfgAssumption::AssumesImageProtectionStable => "UNPROVEN-ASSUMPTION",
                    VmCfgAssumption::PreferredImageBase
                    | VmCfgAssumption::OverApproximatedEnvironmentGate => "BENIGN-LABELED",
                };
                let rank = |t: &str| match t {
                    "OBSERVED" => 3,
                    "UNPROVEN-ASSUMPTION" => 2,
                    "BENIGN-LABELED" => 1,
                    _ => 0,
                };
                if rank(this) > rank(t) {
                    t = this;
                }
            }
            t
        };
        let mut tiers: BTreeMap<&str, usize> = BTreeMap::new();
        let mut exact: BTreeMap<String, usize> = BTreeMap::new();
        for (_, a) in &complete_sets {
            *tiers.entry(tier(a)).or_insert(0) += 1;
            *exact.entry(format!("{a:?}")).or_insert(0) += 1;
        }
        println!("\n==================== ASSUMPTION-TIER SPLIT OF THE COMPLETE SET ====================");
        println!("complete (all_paths_reach_vm_exit): {complete}");
        for t in ["PROVEN", "BENIGN-LABELED", "UNPROVEN-ASSUMPTION", "OBSERVED"] {
            println!("  {t:<22}: {}", tiers.get(t).copied().unwrap_or(0));
        }
        if tiers.get("OBSERVED").copied().unwrap_or(0) > 0 {
            println!("  !!!!!! OBSERVED-tier assumptions present in a SOUND static sweep: INVESTIGATE !!!!!!");
        }
        println!("  exact assumption sets:");
        for (k, n) in &exact {
            println!("    {n:>4}  {k}");
        }
        let proven = tiers.get("PROVEN").copied().unwrap_or(0);
        let benign = tiers.get("BENIGN-LABELED").copied().unwrap_or(0);
        println!(
            "  honest proven count: {proven} strict (no assumptions); {} incl. benign labels (of {complete} reported)",
            proven + benign
        );
    }

    // ---------------- merged-X address-root classification ----------------
    {
        let base = pe.optional.image_base;
        let mut slots: BTreeMap<u32, u32> = BTreeMap::new();
        for block in &pe.report().relocations {
            for e in &block.entries {
                if e.kind == 0 {
                    continue;
                }
                let size = if e.kind == 10 { 8 } else { 4 };
                slots.insert(block.page_rva.0 + u32::from(e.offset), size);
            }
        }
        println!("\n==================== MERGED-X ADDRESS-ROOT CLASSIFICATION ====================");
        println!("image base reloc slots parsed: {}", slots.len());
        let reloc_covers = |rva: u32, width: u32| -> bool {
            let end = rva.saturating_add((width / 8).max(1));
            slots
                .range(rva.saturating_sub(7)..end)
                .any(|(s, sz)| *s < end && rva < s + sz)
        };
        let refine = |c: &vmpx_vm::RefusalCause, frame_hi: Option<u64>| -> String {
            let root = c.root_class.unwrap_or("none");
            let mut out = match root {
                "constant_global" => "CONSTANT-GLOBAL".to_owned(),
                "native_register" => "NATIVE-REGISTER".to_owned(),
                "vsp" => match (c.vsp_offset, frame_hi) {
                    (None, _) => "VSP-INDEXED".to_owned(),
                    (Some(_), None) => "VSP-UNKNOWN-FRAME".to_owned(),
                    (Some(o), Some(h)) => {
                        if o >= h.cast_signed() {
                            "VSP-ABOVE-FRAME".to_owned()
                        } else {
                            "VSP-INSIDE-FRAME".to_owned()
                        }
                    }
                },
                _ => "OTHER".to_owned(),
            };
            if c.chain_depth > 0 {
                out.push_str("+CHAIN");
            }
            out
        };
        let ab_fns: BTreeSet<u32> = fn_sets
            .iter()
            .filter(|(_, s)| s.contains(&'A') || s.contains(&'B'))
            .map(|(r, _)| *r)
            .collect();
        let jsonl_path = args.get(5).map(String::as_str).unwrap_or("xpl_x_diag.jsonl");
        let mut out = String::new();
        // per function: refined root classes of X leaves
        let mut x_fn_roots: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
        let mut leaf_hist: BTreeMap<String, usize> = BTreeMap::new();
        let mut leaf_name_hist: BTreeMap<String, usize> = BTreeMap::new();
        let mut const_sections: BTreeMap<String, usize> = BTreeMap::new();
        let mut const_reloc: BTreeMap<&str, usize> = BTreeMap::new();
        let mut fns_with_diag = 0usize;
        let mut nonx_names: BTreeMap<String, BTreeSet<u32>> = BTreeMap::new();
        let mut fn_has_nonx: BTreeSet<u32> = BTreeSet::new();
        let mut nodiag_ab = Vec::new();
        for r in &results {
            let Outcome::Incomplete { diags, raw, .. } = &r.outcome else { continue };
            if diags.is_empty() {
                continue;
            }
            fns_with_diag += 1;
            let mut js_causes = Vec::new();
            for d in diags {
                for c in &d.causes {
                    let refined = if c.class == "X" { refine(c, d.frame_hi) } else { "-".to_owned() };
                    let mut section = None;
                    let mut covered = None;
                    if c.class == "X" {
                        x_fn_roots.entry(r.rva).or_default().insert(refined.clone());
                        *leaf_hist.entry(refined.clone()).or_insert(0) += 1;
                        *leaf_name_hist.entry(c.leaf_name.clone()).or_insert(0) += 1;
                        if let Some(addr) = c.constant_address {
                            let rva = base.checked_rva_of(vmpx_pe::Va(addr)).map(|v| v.0);
                            let sec = rva.and_then(|rv| {
                                pe.sections.iter().find(|s| s.contains_mapped_rva(vmpx_pe::Rva(rv)))
                            });
                            let name = sec.map_or_else(
                                || "(outside image)".to_owned(),
                                |s| {
                                    let w = if s.characteristics & 0x8000_0000 != 0 { "W" } else { "RO" };
                                    format!("{}[{w}]", s.name)
                                },
                            );
                            let cov = rva.is_some_and(|rv| reloc_covers(rv, c.width));
                            *const_sections.entry(name.clone()).or_insert(0) += 1;
                            *const_reloc.entry(if cov { "reloc-covered" } else { "no-reloc" }).or_insert(0) += 1;
                            section = Some(name);
                            covered = Some(cov);
                        }
                    } else {
                        fn_has_nonx.insert(r.rva);
                        nonx_names.entry(c.leaf_name.clone()).or_default().insert(r.rva);
                    }
                    js_causes.push(serde_json::json!({
                        "class": c.class,
                        "leaf_name": c.leaf_name,
                        "root": c.root,
                        "root_class": refined,
                        "width": c.width,
                        "companions": c.companions,
                        "reason": c.reason,
                        "phase": c.phase,
                        "constant_address": c.constant_address.map(|a| format!("{a:#x}")),
                        "section": section,
                        "reloc_covered": covered,
                        "vsp_offset": c.vsp_offset,
                        "frame_hi": d.frame_hi,
                        "chain_depth": c.chain_depth,
                        "vip": format!("{:#x}", d.vip.0),
                    }));
                }
            }
            out.push_str(&serde_json::json!({
                "rva": format!("{:#x}", r.rva),
                "x_by_stall_label": ab_fns.contains(&r.rva),
                "first_win_stalls": raw,
                "causes": js_causes,
            }).to_string());
            out.push('\n');
        }
        for rva in &ab_fns {
            if !x_fn_roots.contains_key(rva) {
                nodiag_ab.push(*rva);
            }
        }
        std::fs::write(jsonl_path, &out).expect("write jsonl");
        println!("per-function JSONL ({fns_with_diag} functions with a diagnosed branch) -> {jsonl_path}");

        let xd: BTreeSet<u32> = x_fn_roots.keys().copied().collect();
        println!("\nX functions by first-win stall label (A or B): {}", ab_fns.len());
        println!("X functions by ALL-refusals diagnostic (>=1 X leaf) : {}", xd.len());
        println!(
            "  overlap: {}; stall-label X without any diagnosed X leaf: {}; diagnosed X not stall-labeled X: {}",
            ab_fns.intersection(&xd).count(),
            nodiag_ab.len(),
            xd.difference(&ab_fns).count()
        );
        println!("\n-- X leaves by refined address-root class (leaf count, deduped within a stall) --");
        for (k, n) in &leaf_hist {
            println!("  {n:>6}  {k}");
        }
        println!("\n-- X leaves by consuming leaf name --");
        let mut lh: Vec<_> = leaf_name_hist.iter().collect();
        lh.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for (k, n) in lh {
            println!("  {n:>6}  {k}");
        }
        println!("\n-- CONSTANT-GLOBAL leaves: section --");
        for (k, n) in &const_sections {
            println!("  {n:>6}  {k}");
        }
        println!("-- CONSTANT-GLOBAL leaves: base-relocation coverage of the slot --");
        for (k, n) in &const_reloc {
            println!("  {n:>6}  {k}");
        }
        let external = |k: &str| matches!(k.trim_end_matches("+CHAIN"), "CONSTANT-GLOBAL" | "NATIVE-REGISTER" | "VSP-ABOVE-FRAME");
        let inside = |k: &str| matches!(k.trim_end_matches("+CHAIN"), "VSP-INSIDE-FRAME" | "VSP-INDEXED" | "VSP-UNKNOWN-FRAME");
        println!("\n-- X FUNCTIONS: address-root split (function-level; a function can span classes) --");
        let mut per_fn: BTreeMap<String, usize> = BTreeMap::new();
        for roots in x_fn_roots.values() {
            for k in roots {
                *per_fn.entry(k.clone()).or_insert(0) += 1;
            }
        }
        for (k, n) in &per_fn {
            println!("  {n:>5} functions with >=1 {k} leaf");
        }
        let all_ext = x_fn_roots.values().filter(|r| r.iter().all(|k| external(k))).count();
        let has_inside = x_fn_roots.values().filter(|r| r.iter().any(|k| inside(k))).count();
        let only_inside = x_fn_roots.values().filter(|r| r.iter().all(|k| inside(k))).count();
        let direct_inside = x_fn_roots
            .values()
            .filter(|r| r.iter().any(|k| matches!(k.as_str(), "VSP-INSIDE-FRAME")))
            .count();
        let has_other = x_fn_roots
            .values()
            .filter(|r| r.iter().any(|k| !external(k) && !inside(k)))
            .count();
        println!("  X functions (diagnosed): {}", xd.len());
        println!("  ALL X leaves externally rooted (CONSTANT/NATIVE/VSP-ABOVE; genuinely closed): {all_ext}");
        println!("  >=1 VSP-inside/indexed/unknown-frame leaf (any chain; potential precision loss): {has_inside}");
        println!("  ONLY inside-frame-ish leaves (every X leaf is inside/indexed/unknown): {only_inside}");
        println!("  >=1 direct (unchained) VSP-INSIDE-FRAME leaf: {direct_inside}");
        println!("  >=1 OTHER-rooted leaf (unclassified): {has_other}");

        // ---------------- ceilings ----------------
        let proven_n = complete_sets.iter().filter(|(_, a)| a.is_empty()).count();
        let benign_n = complete_sets
            .iter()
            .filter(|(_, a)| {
                !a.is_empty()
                    && a.iter().all(|x| {
                        matches!(
                            x,
                            VmCfgAssumption::PreferredImageBase
                                | VmCfgAssumption::OverApproximatedEnvironmentGate
                        )
                    })
            })
            .count();
        let total = swept.max(1) as f64;
        let pct = |n: usize| n as f64 * 100.0 / total;
        let is_rt = |c: &char| matches!(c, 'A' | 'B' | 'N');
        let addr_only: BTreeSet<u32> =
            fn_sets.iter().filter(|(_, s)| !s.iter().any(is_rt)).map(|(r, _)| *r).collect();
        let addr_only_nodiagx: usize = addr_only.iter().filter(|r| !xd.contains(r)).count();
        // Functions whose only runtime cause is X-with-inside-frame (recoverable precision sub-case).
        let recoverable: BTreeSet<u32> = fn_sets
            .iter()
            .filter(|(r, s)| {
                x_fn_roots.get(r).is_some_and(|roots| roots.iter().all(|k| inside(k)))
                    && !s.contains(&'N')
            })
            .map(|(r, _)| *r)
            .collect();
        println!("\n==================== TRUE STATIC CEILING, THREE WAYS (swept {swept}) ====================");
        println!("addressable-only incomplete functions (no A/B/N stall label): {}", addr_only.len());
        println!("  ... of which also with no diagnosed X leaf: {addr_only_nodiagx}");
        println!(
            "  (1) as-reported : {complete} complete + {} addressable = {} -> {:.1}%",
            addr_only.len(),
            complete + addr_only.len(),
            pct(complete + addr_only.len())
        );
        println!(
            "  (2) honest proven+benign: {} + {} = {} -> {:.1}%",
            proven_n + benign_n,
            addr_only.len(),
            proven_n + benign_n + addr_only.len(),
            pct(proven_n + benign_n + addr_only.len())
        );
        println!(
            "  (3) strict proven only  : {} + {} = {} -> {:.1}%",
            proven_n,
            addr_only.len(),
            proven_n + addr_only.len(),
            pct(proven_n + addr_only.len())
        );
        println!(
            "  functions whose X leaves are ALL inside-frame-ish (recoverable precision sub-case, no N): {}",
            recoverable.len()
        );
        println!(
            "  (4) as-reported + addressable + recoverable-inside-frame X: {} -> {:.1}%",
            complete + addr_only.len() + recoverable.difference(&addr_only).count(),
            pct(complete + addr_only.len() + recoverable.difference(&addr_only).count())
        );
        println!("\n-- non-X leaves (by consuming leaf name; functions) --");
        let mut nn: Vec<_> = nonx_names.iter().map(|(k, v)| (k.clone(), v.len())).collect();
        nn.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        for (k, n) in nn.iter().take(25) {
            println!("  {n:>5}  {k}");
        }
        let _ = fn_has_nonx;
    }

    if let Ok(path) = std::env::var("XPL_RAW_DUMP") {
        let mut out = String::new();
        for r in &results {
            if let Outcome::Incomplete { raw, .. } = &r.outcome {
                for (i, x) in raw.iter().enumerate() {
                    out.push_str(&format!("{:#x}\t{}\t{}\t{}\n", r.rva, i, categorize(x), x.replace('\n', " ")));
                }
            }
        }
        std::fs::write(path, out).expect("write raw dump");
    }

    if let Some(path) = args.get(6) {
        let mut out = String::new();
        for (rva, set) in &fn_sets {
            let vas: Vec<String> =
                d_vas.iter().filter(|(r, _)| r == rva).map(|(_, v)| format!("{v:#x}")).collect();
            out.push_str(&format!("{rva:#x} {} {}\n", set.iter().collect::<String>(), vas.join(",")));
        }
        std::fs::write(path, out).expect("write dump");
    }

    if let Some(path) = std::env::var_os("XPL_SERVED_PAGES_OUT") {
        let served = SERVED_PAGES.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = String::new();
        for (rva, start, end, kind) in served.iter() {
            out.push_str(&format!("SERVED {rva:#x} {start:#x} {end:#x} {kind}\n"));
        }
        for (rva, proven) in COMPLETES.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            out.push_str(&format!("COMPLETE {rva:#x} {}\n", if *proven { "proven" } else { "assumed" }));
        }
        std::fs::write(path, out).expect("write served pages");
    }

    let pct = complete as f64 * 100.0 / swept_f;
    println!("\n==================================================");
    println!("HEADLINE: {complete}/{swept} complete ({pct:.1}%)");
    if swept != total_discovered {
        println!("(sampled {swept} of {total_discovered} discovered entries)");
    }
    println!("total wall-time {:.1}s", t0.elapsed().as_secs_f64());
}
