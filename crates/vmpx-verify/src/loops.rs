//! Inductive equivalence of two loop-structured functions (cut-point lockstep).
//!
//! Each side is a [`PairFamily`]: an entry segment plus one segment per loop cut
//! (see `differential::native_loops`). A segment is a finite path family over its
//! start state; its paths are exclusive and exhaustive, and each ends in an exit or
//! at a cut carrying the values its phis take next.
//!
//! The prover pairs the two sides' cuts and finds, per paired cut `(h, g)`, a set of
//! equalities `phi_v(j) == phi_n(i)` between the sides' phis: the invariant `I`.
//! It is the greatest fixpoint of a Houdini loop: start from every equality whose
//! carried values agree structurally at the first arrival, and drop any equality
//! the gate cannot prove at some arrival, until nothing changes.
//!
//! # Soundness
//!
//! In the final round no equality was dropped, so the gate proved, for every
//! feasible pair of paths `(p, q)` leaving a paired cut `(h, g)` (or the entry pair),
//! under `cond(p) & cond(q)[I(h, g)]`:
//!
//! * if both paths exit, the observable exit states (ABI registers, stores, calls)
//!   are equal; if both end at cuts, they end at a *paired* `(h', g')` and every
//!   equality of `I(h', g')` holds between the carried values; anything else fails
//!   the whole proof;
//! * a pair whose joint condition the gate proved unsatisfiable is vacuous.
//!
//! Take any input and initial memory. The entry segments' paths partition the
//! inputs, so exactly one entry pair `(p, q)` is taken and its joint condition
//! holds; it is feasible, hence compared. By induction on the number of segments
//! executed, the two machines are at a paired cut with `I` true of their actual phi
//! values (the entry pair needs no hypothesis; a later pair needs `I(h, g)`, which
//! the previous step established), memory is unchanged (a segment ending at a cut
//! stores nothing and calls nothing, enforced by the executors) and the carried
//! effect logs are identical. The segments from that cut again partition the state
//! space, so exactly one pair of paths is taken, feasible under `I`, and compared.
//! Either both exit with equal observable state, or both reach the same next paired
//! cut with `I` established. The machines therefore run in lockstep: one exits iff
//! the other does, with equal state, and both diverge together. Phis outside `I`
//! stay free symbols in every query, so each query holds for all their values.
//!
//! The Houdini search only chooses *which* equalities to assume; the assumed set is
//! always the one verified, so a poor choice costs completeness, never soundness.
//! A refutation inside an inductive step names a phi assignment that need not be
//! reachable, so it is never reported as REFUTED: every failure is unproven.

use std::collections::BTreeMap;

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind};
use vmpx_vmlift::ExitSemantics;

use crate::branch_paths::{and, expand_flags, is_zero, map_exit, restrict};
use crate::differential::native_loops::{NativeLoops, carried};
use crate::differential::{AbiPolicy, DiffOutcome, SegmentEnd, differential_obligation};
use crate::equiv::{Cell, CellId, CellVerdict, EquivVerdict, Obligation, Verifier};

/// Most Houdini rounds before the proof is abandoned.
const MAX_ROUNDS: usize = 64;
/// Most paired cuts.
const MAX_PAIRS: usize = 16;
/// Most (path, path) pairs one paired cut may compare.
const MAX_PATH_PAIRS: usize = 4096;

/// How a path of a segment ends.
#[derive(Clone, Debug)]
pub enum PairEnd {
    Exit(ExitSemantics),
    /// A cut, with the values the destination's phis take, aligned with its `phis`.
    Cut { header: u64, next: Vec<Expr> },
}

/// One path: a `W8` 0/1 condition over the entry symbols and the segment's phis.
#[derive(Clone, Debug)]
pub struct PairPath {
    pub cond: Expr,
    pub end: PairEnd,
}

/// The segment that starts at one cut.
#[derive(Clone, Debug)]
pub struct PairSegment {
    /// The phi symbols (`Register` expressions) of the cut.
    pub phis: Vec<Expr>,
    pub paths: Vec<PairPath>,
}

/// One side of the comparison. Contract for a producer: every segment's paths are
/// exclusive and exhaustive over its start state; a path ending at a cut changes no
/// memory and calls nothing; the phis stand for the machine state at the cut;
/// phi symbols are disjoint from the other side's.
#[derive(Clone, Debug)]
pub struct PairFamily {
    pub entry: Vec<PairPath>,
    pub loops: BTreeMap<u64, PairSegment>,
}

impl PairFamily {
    /// The native executor's cut-point family.
    ///
    /// # Errors
    /// An expression that cannot be rebuilt.
    pub fn from_native(n: &NativeLoops) -> Result<Self, String> {
        let convert = |paths: &[crate::differential::SegmentPath]| -> Result<Vec<PairPath>, String> {
            paths
                .iter()
                .map(|p| {
                    let cond = expand_flags(&p.condition()?)?;
                    let end = match &p.end {
                        SegmentEnd::Exit(f) => PairEnd::Exit(map_exit(&f.semantics, &mut expand_flags)?),
                        SegmentEnd::Cut { header, state } => PairEnd::Cut {
                            header: *header,
                            next: carried(state).iter().map(expand_flags).collect::<Result<_, _>>()?,
                        },
                    };
                    Ok(PairPath { cond, end })
                })
                .collect()
        };
        let mut loops = BTreeMap::new();
        for (h, seg) in &n.loops {
            loops.insert(*h, PairSegment { phis: seg.phis.clone(), paths: convert(&seg.paths)? });
        }
        Ok(Self { entry: convert(&n.entry)?, loops })
    }

    fn phi_names(&self) -> Vec<String> {
        self.loops.values().flat_map(|s| s.phis.iter().filter_map(register_name)).collect()
    }
}

fn register_name(e: &Expr) -> Option<String> {
    match &e.kind {
        ExprKind::Register { name } => Some(name.clone()),
        _ => None,
    }
}

type Key = (Option<u64>, Option<u64>);

struct Pair {
    /// For each of the VM side's phis, the native phis it may equal, best first.
    cands: Vec<Vec<usize>>,
}

fn fail(mut out: DiffOutcome, why: impl Into<String>) -> DiffOutcome {
    out.reason = why.into();
    out
}

/// Proves `recovered` equal to `original` by cut-point induction.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn prove_loop_equivalence(
    original: &PairFamily,
    recovered: &PairFamily,
    verifier: &Verifier,
    policy: &AbiPolicy,
) -> DiffOutcome {
    let out = DiffOutcome::unproven("");
    let (on, rn) = (original.phi_names(), recovered.phi_names());
    if on.iter().any(|a| rn.contains(a)) {
        return fail(out, "the two sides share a phi symbol name");
    }
    let zero = Expr::constant(BitWidth::W8, 0);
    let mut pairs: BTreeMap<Key, Pair> = BTreeMap::new();
    pairs.insert((None, None), Pair { cands: Vec::new() });
    let mut rounds = 0;
    let exits = loop {
        rounds += 1;
        if rounds > MAX_ROUNDS {
            return fail(out, format!("no inductive invariant within {MAX_ROUNDS} rounds"));
        }
        let mut changed = false;
        // Structural mismatches under a not-yet-verified substitution may be spurious;
        // they are fatal only in a round that changes nothing.
        let mut soft: Vec<String> = Vec::new();
        let mut exits: Vec<Obligation> = Vec::new();
        let mut keys: Vec<Key> = pairs.keys().copied().collect();
        let mut at = 0;
        while at < keys.len() {
            let key = keys[at];
            at += 1;
            let (npaths, nphis): (&[PairPath], &[Expr]) = match key.0 {
                None => (&original.entry, &[]),
                Some(h) => match original.loops.get(&h) {
                    Some(s) => (&s.paths, &s.phis),
                    None => return fail(out, format!("original has no segment for cut {h:#x}")),
                },
            };
            let (vpaths, vphis): (&[PairPath], &[Expr]) = match key.1 {
                None => (&recovered.entry, &[]),
                Some(g) => match recovered.loops.get(&g) {
                    Some(s) => (&s.paths, &s.phis),
                    None => return fail(out, format!("recovered has no segment for cut {g:#x}")),
                },
            };
            if npaths.len().saturating_mul(vpaths.len()) > MAX_PATH_PAIRS {
                return fail(out, format!("{} x {} paths exceed the pair budget", npaths.len(), vpaths.len()));
            }
            let (sigma, hypothesis) = match substitution(&pairs[&key], vphis, nphis) {
                Ok(s) => s,
                Err(e) => return fail(out, e),
            };
            for (side, conds) in [("original", npaths.iter().map(|p| p.cond.clone()).collect::<Vec<_>>()), ("recovered", {
                let mut v = Vec::new();
                for p in vpaths {
                    match p.cond.substitute(&sigma) {
                        Ok(c) => v.push(c),
                        Err(e) => return fail(out, e.to_string()),
                    }
                }
                v
            })] {
                match exhaustive(&conds, verifier) {
                    Ok(true) => {}
                    Ok(false) => return fail(out, format!("the {side} paths of a segment are not proven exhaustive")),
                    Err(e) => return fail(out, e),
                }
            }
            if std::env::var_os("VMPX_LOOP_TRACE").is_some() {
                let sig: Vec<String> = sigma.iter().map(|(k, v)| format!("{k}->{}", crate::differential::pretty(v, 2))).collect();
                eprintln!("[loops] round {rounds} pair {key:x?} sigma {sig:?}");
            }
            for (ni, np) in npaths.iter().enumerate() {
                for (vi, vp) in vpaths.iter().enumerate() {
                    let step = (|| -> Result<(), String> {
                        let vcond = vp.cond.substitute(&sigma).map_err(|e| e.to_string())?;
                        let joint = and(&and(&np.cond, &vcond)?, &hypothesis)?;
                        let dead = verifier.check_cell(&Cell {
                            id: CellId::Named("joint-path-condition".to_owned()),
                            vm: joint.clone(),
                            recovered: zero.clone(),
                        });
                        if matches!(dead, CellVerdict::Proven(_)) {
                            return Ok(());
                        }
                        match (&np.end, &vp.end) {
                            (PairEnd::Exit(orig), PairEnd::Exit(rec)) => {
                                let rec = map_exit(rec, &mut |e| e.substitute(&sigma).map_err(|x| x.to_string()))?;
                                let mut o = differential_obligation(&rec, orig, policy);
                                for cell in &mut o.cells {
                                    restrict(cell, &joint)?;
                                }
                                exits.push(o);
                                Ok(())
                            }
                            (PairEnd::Cut { header: h, next: nn }, PairEnd::Cut { header: g, next: vn }) => {
                                let dest: Key = (Some(*h), Some(*g));
                                let vnext: Vec<Expr> = vn
                                    .iter()
                                    .map(|e| e.substitute(&sigma).map_err(|x| x.to_string()))
                                    .collect::<Result<_, _>>()?;
                                if !pairs.contains_key(&dest) {
                                    if pairs.keys().any(|k| (k.0 == dest.0) != (k.1 == dest.1)) {
                                        soft.push(format!("cuts {h:#x} and {g:#x} pair inconsistently with an earlier pairing"));
                                        return Ok(());
                                    }
                                    if pairs.len() >= MAX_PAIRS {
                                        return Err("too many paired cuts".to_owned());
                                    }
                                    let (Some(ds), Some(vs)) = (original.loops.get(h), recovered.loops.get(g)) else {
                                        return Err(format!("no segment for the paired cuts {h:#x}/{g:#x}"));
                                    };
                                    if nn.len() != ds.phis.len() || vn.len() != vs.phis.len() {
                                        return Err("a carried vector does not match its cut's phis".to_owned());
                                    }
                                    let cands = (0..vn.len())
                                        .map(|j| {
                                            (0..nn.len())
                                                .filter(|&i| {
                                                    nn[i].width == vnext[j].width && nn[i].simplify() == vnext[j].simplify()
                                                })
                                                .collect()
                                        })
                                        .collect();
                                    pairs.insert(dest, Pair { cands });
                                    keys.push(dest);
                                    changed = true;
                                    return Ok(());
                                }
                                let dp = pairs.get_mut(&dest).expect("present");
                                for (j, list) in dp.cands.iter_mut().enumerate() {
                                    let mut keep = Vec::with_capacity(list.len());
                                    for &i in list.iter() {
                                        let mut cell = Cell {
                                            id: CellId::Named(format!("carry {j}")),
                                            vm: vnext[j].clone(),
                                            recovered: nn[i].clone(),
                                        };
                                        restrict(&mut cell, &joint)?;
                                        if matches!(verifier.check_cell(&cell), CellVerdict::Proven(_)) {
                                            keep.push(i);
                                        }
                                    }
                                    if keep.len() != list.len() {
                                        *list = keep;
                                        changed = true;
                                    }
                                }
                                Ok(())
                            }
                            _ => {
                                soft.push(format!(
                                    "under a feasible joint condition one side exits and the other continues (paths {ni}/{vi})"
                                ));
                                Ok(())
                            }
                        }
                    })();
                    if let Err(e) = step {
                        return fail(out, e);
                    }
                }
            }
        }
        if !changed {
            if let Some(why) = soft.into_iter().next() {
                return fail(out, why);
            }
            break exits;
        }
    };
    let mut out = out;
    out.frame_stores_dropped = 0;
    let mut out = crate::differential::conclude(out, &exits, verifier, false, false);
    if out.verdict == EquivVerdict::Refuted {
        out.verdict = EquivVerdict::Timeout;
        out.reason = "a cell failed inside an inductive step; its counterexample is not shown reachable".to_owned();
    }
    out
}

/// The invariant of a paired cut as a rewrite plus a hypothesis. Each VM phi is
/// replaced by its best surviving candidate native phi, and the hypothesis (a `W8`
/// 0/1 condition) states that every other surviving candidate equals it. Together
/// they say exactly: every surviving `phi_v(j) == phi_n(i)` holds. Assuming all of
/// them at once is what makes the greatest-fixpoint search order-independent.
fn substitution(
    pair: &Pair,
    vphis: &[Expr],
    nphis: &[Expr],
) -> Result<(BTreeMap<String, Expr>, Expr), String> {
    let mut map = BTreeMap::new();
    let mut hypothesis = Expr::constant(BitWidth::W8, 1);
    for (j, list) in pair.cands.iter().enumerate() {
        let Some(&first) = list.first() else { continue };
        let name = register_name(&vphis[j]).ok_or("a recovered phi is not a register symbol")?;
        map.insert(name, nphis[first].clone());
        for &other in &list[1..] {
            let same = Expr::binary(BinaryOp::Xor, nphis[first].clone(), nphis[other].clone()).map_err(|e| e.to_string())?;
            hypothesis = and(&hypothesis, &is_zero(&same)?)?;
        }
    }
    Ok((map, hypothesis))
}

/// True when the gate proves that at least one of the 0/1 conditions holds.
fn exhaustive(conds: &[Expr], verifier: &Verifier) -> Result<bool, String> {
    let mut any = Expr::constant(BitWidth::W8, 0);
    for c in conds {
        any = Expr::binary(BinaryOp::Or, any, c.clone()).map_err(|e| e.to_string())?;
    }
    let none = Expr::binary(BinaryOp::Xor, any, Expr::constant(BitWidth::W8, 1)).map_err(|e| e.to_string())?;
    let verdict = verifier.check_cell(&Cell {
        id: CellId::Named("segment-exhaustive".to_owned()),
        vm: none,
        recovered: Expr::constant(BitWidth::W8, 0),
    });
    Ok(matches!(verdict, CellVerdict::Proven(_)))
}
