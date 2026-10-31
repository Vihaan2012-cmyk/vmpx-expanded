// In-process devirt coverage sweep over ALL VM entries of a (possibly packed)
// VMProtect image. Loads/unpacks the analysis image ONCE, discovers every VM
// entry, recovers each entry's CFG with a bounded per-entry budget (mirroring
// the `cfgdiag` example), catches per-entry panics, and tallies completeness
// plus the stall-cause distribution. READ-ONLY: no engine changes, static only.
//
// Usage: xpl_coverage [IMAGE] [SAMPLE] [THREADS]
//   IMAGE    defaults to E:/downloads/systems.original.xpl
//   SAMPLE   entries to sweep: a number for a sample, or "all" for the full
//            image. OMITTED defaults to a SMALL memory-safe sample (50) so a
//            naive run can never OOM; the full image is explicit opt-in ("all").
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
    EntryProfileRecoveryLimits, VmCfgLimits, VmDiscoveryLimits, discover_vm_entries, recover_vm_cfg,
};

const DEFAULT_IMAGE: &str = "E:/downloads/systems.original.xpl";
const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;

enum Outcome {
    Complete,
    // Non-complete but recovery returned a CFG: distinct normalized stall
    // causes this entry exhibited, and the resource limit hit (if any).
    Incomplete { causes: Vec<String>, raw: Vec<String>, limit: Option<String> },
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

fn recover_one(pe: &PeImage<'_>, entry: &vmpx_vm::VmEntryCandidate, instruction_limit: usize) -> Outcome {
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
            ..EntryProfileRecoveryLimits::default()
        },
        ..VmCfgLimits::default()
    };
    let result = panic::catch_unwind(AssertUnwindSafe(|| recover_vm_cfg(pe, entry, limits)));
    match result {
        Err(_) => Outcome::Panic,
        Ok(Err(error)) => Outcome::Error(normalize(&error.to_string())),
        Ok(Ok(cfg)) => {
            if cfg.all_paths_reach_vm_exit() {
                Outcome::Complete
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
                }
            }
        }
    }
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
    // Per-block decode budget (4th arg). Default 512 keeps a naive run memory-safe;
    // raise it deliberately (with memory headroom) to reclaim InstructionLimit stalls.
    let instruction_limit: usize = args.get(4).and_then(|a| a.parse().ok()).unwrap_or(512);

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
                            outcome: recover_one(pe_ref, entry, instruction_limit),
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
    for r in &results {
        match &r.outcome {
            Outcome::Complete => complete += 1,
            Outcome::Panic => panics += 1,
            Outcome::Error(msg) => {
                errors += 1;
                fn_sets.push((r.rva, BTreeSet::from(['J'])));
                add(&mut error_causes, format!("cfg_error: {msg}"), r.rva);
            }
            Outcome::Incomplete { causes: cs, raw, limit } => {
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
    if let Some(path) = args.get(5) {
        let mut out = String::new();
        for (rva, set) in &fn_sets {
            let vas: Vec<String> =
                d_vas.iter().filter(|(r, _)| r == rva).map(|(_, v)| format!("{v:#x}")).collect();
            out.push_str(&format!("{rva:#x} {} {}\n", set.iter().collect::<String>(), vas.join(",")));
        }
        std::fs::write(path, out).expect("write dump");
    }

    let pct = complete as f64 * 100.0 / swept_f;
    println!("\n==================================================");
    println!("HEADLINE: {complete}/{swept} complete ({pct:.1}%)");
    if swept != total_discovered {
        println!("(sampled {swept} of {total_discovered} discovered entries)");
    }
    println!("total wall-time {:.1}s", t0.elapsed().as_secs_f64());
}
