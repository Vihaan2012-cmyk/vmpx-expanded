//! `vmpx verify-equiv`: runs the end-to-end equivalence gate over recovered
//! functions.
//!
//! Pair checked: the recovered semantics (the lifter's decision tree, i.e. the
//! VM's symbolic exit state on every path) against the semantics read back
//! from the *recompiled* output: the replacement IR the lifter emits, after the
//! certified optimizer, evaluated by an independent IR evaluator. This checks
//! lowering, emission and optimization of what was recovered; it does not
//! re-check the recovery itself. PROVEN only when the gate proves every cell;
//! everything else is unproven.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use vmpx_detect::{ImageLayout, detect_vmprotect};
use vmpx_pe::{PeImage, Rva};
use vmpx_unpack::{UnpackLimits, unpack_to_mapped_image};
use vmpx_verify::equiv::{
    Budget, CellId, CellVerdict, EquivVerdict, FunctionReport, Method, Obligation, Unproven,
    Verifier,
};
use vmpx_verify::smtlib::{CliBackend, Flavor};
use vmpx_vmlift::{RecompiledPair, recompiled_cfg_pairs};

use crate::clean::{build_vm_inventory, export_dependency_report};
use crate::decoded_splice::lift_entry_cfg;
use crate::input_closed_functions::MIN_CONFIDENCE;

pub(crate) struct GateOptions {
    pub input: PathBuf,
    pub optimize: bool,
    pub max_functions: Option<usize>,
    pub only: Vec<u32>,
    pub use_solver: bool,
    pub exhaustive_bits: u32,
    pub timeout_ms: u32,
    pub instruction_limit: usize,
    pub strict_memory: bool,
    pub verbose: bool,
}

#[derive(Default)]
struct Tally {
    proven: usize,
    proven_structural: usize,
    proven_enumerated: usize,
    proven_solver: usize,
    proven_assuming_memory_order: usize,
    refuted: usize,
    timeout: usize,
    timeout_reasons: BTreeMap<String, usize>,
    not_lifted: BTreeMap<String, usize>,
    paths: usize,
    ops_before: usize,
    ops_after: usize,
    rewrites: usize,
}

/// Builds the gate obligation for one recovered path.
pub(crate) fn obligation_for(pair: &RecompiledPair) -> Obligation {
    let mut obligation = Obligation::from_exits(&pair.vm, &pair.recompiled);
    obligation.cells.retain(|cell| match cell.id {
        CellId::ReturnAddress => pair.return_address_attested,
        CellId::Rflags => pair.rflags_attested,
        _ => true,
    });
    for (name, recovered, recompiled) in &pair.extra_cells {
        obligation.add(CellId::Named(name.clone()), recovered.clone(), recompiled.clone());
    }
    obligation
}

pub(crate) fn load_pe(bytes: &[u8]) -> Result<Vec<u8>> {
    let input = PeImage::parse(bytes).context("input is not a supported PE")?;
    let protection = detect_vmprotect(&input);
    Ok(match protection.image_layout {
        ImageLayout::PackedVirtualSections => {
            unpack_to_mapped_image(bytes, UnpackLimits::default())?.image
        }
        _ => bytes.to_vec(),
    })
}

fn cell_kind(id: &CellId) -> String {
    match id {
        CellId::Named(name) => ["selector", "branch value", "popped bytes", "seed", "next"]
            .into_iter()
            .find(|k| name.contains(k))
            .map_or("named", |k| k)
            .to_owned(),
        CellId::Register(_) => "register".to_owned(),
        CellId::StoreAddress(_) => "store address".to_owned(),
        CellId::StoreValue(_) => "store value".to_owned(),
        other => format!("{other:?}"),
    }
}

fn first_difference<'a>(a: &'a vmpx_vm::Expr, b: &'a vmpx_vm::Expr) -> (&'a vmpx_vm::Expr, &'a vmpx_vm::Expr) {
    use vmpx_vm::ExprKind as K;
    if a == b {
        return (a, b);
    }
    let pairs: Vec<(&vmpx_vm::Expr, &vmpx_vm::Expr)> = match (&a.kind, &b.kind) {
        (K::Unary { op: x, value: p }, K::Unary { op: y, value: q }) if x == y && a.width == b.width => vec![(p, q)],
        (K::Binary { op: x, left: l1, right: r1 }, K::Binary { op: y, left: l2, right: r2 }) if x == y && a.width == b.width => {
            vec![(l1, l2), (r1, r2)]
        }
        (K::Load { address: p }, K::Load { address: q }) if a.width == b.width => vec![(p, q)],
        (K::Intrinsic { name: x, arguments: p }, K::Intrinsic { name: y, arguments: q }) if x == y && p.len() == q.len() => {
            p.iter().zip(q).collect()
        }
        (K::Resize { op: x, value: p }, K::Resize { op: y, value: q }) if x == y && a.width == b.width => vec![(p, q)],
        (K::Extract { value: p, lsb: x }, K::Extract { value: q, lsb: y }) if x == y && a.width == b.width => vec![(p, q)],
        (K::Concat { high: h1, low: l1 }, K::Concat { high: h2, low: l2 }) if a.width == b.width => vec![(h1, h2), (l1, l2)],
        _ => return (a, b),
    };
    for (x, y) in pairs {
        if x != y {
            return first_difference(x, y);
        }
    }
    (a, b)
}

fn dump_first_unproven(obligations: &[Obligation], report: &FunctionReport) {
    for (path, obligation) in report.paths.iter().zip(obligations) {
        for ((id, verdict), cell) in path.cells.iter().zip(&obligation.cells) {
            if !matches!(verdict, CellVerdict::Proven(_)) {
                let short = |e: &vmpx_vm::Expr| format!("{e:?}").chars().take(420).collect::<String>();
                let (x, y) = first_difference(&cell.vm, &cell.recovered);
                println!("    first unproven cell {id:?}\n      differs at recovered   {}\n                 recompiled  {}", short(x), short(y));
                return;
            }
        }
    }
}

fn classify_timeout(report: &FunctionReport) -> String {
    for path in &report.paths {
        if let Some(gap) = path.gaps.first() {
            return format!("modeling gap: {gap}");
        }
        for (id, verdict) in &path.cells {
            if let CellVerdict::Unproven(why) = verdict {
                return match why {
                    Unproven::Timeout(_) => {
                        format!("beyond bounded search, needs a solver [{}]", cell_kind(id))
                    }
                    Unproven::Unknown(w) => {
                        let id = match id {
                            CellId::Named(_) => "branch cell".to_owned(),
                            other => format!("{other:?}"),
                        };
                        format!("unknown ({id}): {w}")
                    }
                    Unproven::ModelingGap(w) => format!("modeling gap: {w}"),
                };
            }
        }
    }
    "no paths".to_owned()
}

fn account(tally: &mut Tally, rva: Rva, verdict: &Result<FunctionReport, String>, verbose: bool) {
    match verdict {
        Err(reason) => {
            tally.timeout += 1;
            let key: String = reason.chars().take(100).collect();
            *tally
                .timeout_reasons
                .entry(format!("recompiled IR not evaluable: {key}"))
                .or_default() += 1;
            if verbose {
                println!("  {rva}: TIMEOUT (not evaluable) {reason}");
            }
        }
        Ok(report) => match report.verdict {
            EquivVerdict::Proven => {
                tally.proven += 1;
                let methods: Vec<&Method> = report
                    .paths
                    .iter()
                    .flat_map(|p| &p.cells)
                    .filter_map(|(_, v)| match v {
                        CellVerdict::Proven(m) => Some(m),
                        _ => None,
                    })
                    .collect();
                if methods.iter().all(|m| **m == Method::Structural) {
                    tally.proven_structural += 1;
                } else if methods
                    .iter()
                    .any(|m| matches!(m, Method::Solver(n) if n != "native-bounded"))
                {
                    tally.proven_solver += 1;
                } else {
                    tally.proven_enumerated += 1;
                }
                if verbose {
                    println!("  {rva}: PROVEN ({} paths)", report.paths.len());
                }
            }
            EquivVerdict::Refuted => {
                tally.refuted += 1;
                println!("  {rva}: REFUTED");
                for path in &report.paths {
                    if let Some((cell, cex)) = path.first_counterexample() {
                        println!(
                            "    cell {cell:?}: recovered={:#x} recompiled={:#x} at {:x?}",
                            cex.vm, cex.recovered, cex.assignment
                        );
                    }
                }
            }
            EquivVerdict::Timeout => {
                tally.timeout += 1;
                let reason = classify_timeout(report);
                if verbose {
                    println!("  {rva}: TIMEOUT {reason}");
                }
                *tally.timeout_reasons.entry(reason).or_default() += 1;
            }
        },
    }
}

fn tally_total(t: &Tally) -> usize {
    t.proven + t.refuted + t.timeout
}

pub(crate) fn run(options: &GateOptions) -> Result<()> {
    let bytes = std::fs::read(&options.input)
        .with_context(|| format!("reading {}", options.input.display()))?;
    let mapped = load_pe(&bytes)?;
    let pe = PeImage::parse(&mapped).context("analysis image failed reparse")?;
    let exports = export_dependency_report(&pe);
    let mut warnings = Vec::new();
    let (inventory, _) = build_vm_inventory(&pe, &exports, &mut warnings);
    let candidates: Vec<_> = inventory
        .iter()
        .filter(|e| e.confidence >= MIN_CONFIDENCE)
        .filter(|e| options.only.is_empty() || options.only.contains(&e.rva.0))
        .collect();

    let mut verifier = Verifier::new(Budget {
        exhaustive_bits: options.exhaustive_bits,
        timeout_ms: options.timeout_ms,
    });
    let mut solver_note = "off".to_owned();
    if options.use_solver {
        match [Flavor::Z3, Flavor::Bitwuzla, Flavor::Cvc5]
            .into_iter()
            .find_map(CliBackend::discover)
        {
            Some(backend) => {
                solver_note = format!("{:?}", backend.flavor);
                verifier = verifier.with_backend(Box::new(backend));
            }
            None => solver_note = "requested, none of z3/bitwuzla/cvc5 on PATH".to_owned(),
        }
    }

    let mut tally = Tally::default();
    let mut attempted = 0_usize;
    for entry in &candidates {
        if options
            .max_functions
            .is_some_and(|max| tally_total(&tally) >= max)
        {
            break;
        }
        attempted += 1;
        let lifted = match lift_entry_cfg(&pe, entry.rva, options.instruction_limit) {
            Ok(lifted) => lifted,
            Err(reason) => {
                let key: String = reason.chars().take(100).collect();
                *tally.not_lifted.entry(key).or_default() += 1;
                if options.verbose {
                    println!("  {}: not lifted ({reason})", entry.rva);
                }
                continue;
            }
        };
        let mut assumes_memory_order = false;
        let verdict = recompiled_cfg_pairs(&lifted, options.optimize, options.strict_memory).map(|readback| {
            assumes_memory_order = readback.pairs.iter().any(|p| p.memory_order_assumed);
            tally.paths += readback.pairs.len();
            tally.ops_before += readback.stats.operations_before;
            tally.ops_after += readback.stats.operations_after;
            tally.rewrites += readback.stats.rewrites;
            let obligations: Vec<Obligation> = readback.pairs.iter().map(obligation_for).collect();
            let report = verifier.check_function(&obligations, true);
            if std::env::var_os("VMPX_GATE_DUMP").is_some() && report.verdict != EquivVerdict::Proven {
                dump_first_unproven(&obligations, &report);
            }
            report
        });
        let proven = matches!(&verdict, Ok(r) if r.verdict == EquivVerdict::Proven);
        if proven && assumes_memory_order {
            tally.proven_assuming_memory_order += 1;
        }
        account(&mut tally, entry.rva, &verdict, options.verbose);
    }

    let total = tally_total(&tally);
    println!("input: {}", options.input.display());
    println!(
        "pair: recovered semantics (lifted decision tree) vs recompiled replacement IR ({}), read back by an independent evaluator",
        if options.optimize {
            "lowered + certified-optimized"
        } else {
            "lowered, unoptimized"
        }
    );
    println!(
        "inventory {} entries; attempted {attempted}; lifted (CFG-complete) {total}; not lifted {}",
        inventory.len(),
        tally.not_lifted.values().sum::<usize>()
    );
    for (reason, count) in &tally.not_lifted {
        println!("    not lifted {count:>4}  {reason}");
    }
    println!("verified population: {total} lifted functions, {} paths", tally.paths);
    println!(
        "PROVEN {}  (structural {}, bounded-enumeration {}, external-solver {})",
        tally.proven, tally.proven_structural, tally.proven_enumerated, tally.proven_solver
    );
    println!(
        "       of the PROVEN, {} rely on loads-after-stores denoting the lifter's intended memory (not re-proven; --strict-memory removes this assumption)",
        tally.proven_assuming_memory_order
    );
    println!("REFUTED {}", tally.refuted);
    println!("TIMEOUT/unproven {}", tally.timeout);
    for (reason, count) in &tally.timeout_reasons {
        println!("    {count:>4}  {reason}");
    }
    println!(
        "recompile: {} IR ops lowered, {} after optimization ({} rewrites)",
        tally.ops_before, tally.ops_after, tally.rewrites
    );
    println!(
        "bound: backends [native-bounded{}]; exhaustive up to {} free input bits, probing beyond; {} ms/cell; solver: {solver_note}; instruction limit {}",
        if matches!(solver_note.as_str(), "Z3" | "Bitwuzla" | "Cvc5") {
            format!(", {solver_note}")
        } else {
            String::new()
        },
        options.exhaustive_bits,
        options.timeout_ms,
        options.instruction_limit
    );
    println!(
        "       Load/Intrinsic terms are free values (UNSAT-sound); PROVEN covers recovered-vs-recompiled only, with the path partition (equality chain) shared by construction"
    );
    Ok(())
}
