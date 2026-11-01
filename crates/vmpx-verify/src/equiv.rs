//! Per-function end-to-end equivalence gate: PROVEN / REFUTED / TIMEOUT.
//!
//! The obligation is a set of cell pairs `(vm, recovered)` of `vmpx_vm::Expr`
//! bit-vectors: every live-out register, flags, return address, store
//! address/value and call argument of the VM's symbolic exit (from
//! `vmpx-vm`/`vmpx-vmlift`) against the recovered function's modeled exit. No
//! new symbolic engine: the VM side is whatever `vmpx-vm` already produced.
//!
//! A cell is discharged by asking whether `vm != recovered` is satisfiable
//! (structural identity -> built-in bounded search -> optional [`SmtBackend`]s).
//!
//! Fail-closed: PROVEN needs every cell UNSAT and no modeling gap. REFUTED
//! needs a counterexample that was replayed through the VM evaluator (a solver
//! "sat" is never trusted on its own). Timeout, unknown, unencodable
//! terms, count mismatches, unsupported effects, or an empty obligation all
//! leave the verdict at TIMEOUT (unproven).
//!
//! Soundness caveats (reported in [`Bound`]): `Load`/`Intrinsic` terms are
//! treated as free values per distinct expression, which is an over-approximation
//! (UNSAT stays sound) but means no counterexample can be produced for them.
//! A function is PROVEN only if its path family is `domain_complete`.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use vmpx_vm::{BitWidth, Expr, ExprKind, X64NativeRegister};
use vmpx_vmlift::ExitSemantics;

use crate::input_testing::InputTester;
use crate::smt::{fold, free_registers};

/// One observable cell of the function's live-out state.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CellId {
    Register(X64NativeRegister),
    Rflags,
    ReturnAddress,
    StoreAddress(usize),
    StoreValue(usize),
    CallArg { call: usize, arg: usize },
    Named(String),
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub id: CellId,
    pub vm: Expr,
    pub recovered: Expr,
}

/// The full proof obligation for one path: cell pairs plus modeling gaps found
/// while building it (each gap forces the verdict to stay unproven).
#[derive(Clone, Debug, Default)]
pub struct Obligation {
    pub cells: Vec<Cell>,
    pub gaps: Vec<String>,
}

impl Obligation {
    pub fn add(&mut self, id: CellId, vm: Expr, recovered: Expr) {
        self.cells.push(Cell { id, vm, recovered });
    }

    pub fn gap(&mut self, reason: impl Into<String>) {
        self.gaps.push(reason.into());
    }

    /// Builds the obligation from the VM exit and the recovered function's
    /// modeled exit. Structural mismatches the SMT layer cannot reason about
    /// (counts, widths, callee identity, frame size) become gaps.
    #[must_use]
    pub fn from_exits(vm: &ExitSemantics, rec: &ExitSemantics) -> Self {
        let mut o = Self::default();
        for (side, s) in [("vm", vm), ("recovered", rec)] {
            for reason in &s.effects.unsupported {
                o.gap(format!("{side} effect not modeled: {reason}"));
            }
        }
        let (a, b) = (&vm.exit, &rec.exit);
        for r in X64NativeRegister::ALL {
            match (a.registers.get(&r), b.registers.get(&r)) {
                (Some(x), Some(y)) => o.add(CellId::Register(r), x.clone(), y.clone()),
                (None, None) => {}
                _ => o.gap(format!("register {r:?} present on one side only")),
            }
        }
        o.add(CellId::Rflags, a.rflags.clone(), b.rflags.clone());
        o.add(
            CellId::ReturnAddress,
            a.return_address.clone(),
            b.return_address.clone(),
        );
        if a.frame_size != b.frame_size {
            o.gap(format!("frame size {} vs {}", a.frame_size, b.frame_size));
        }
        let (vs, rs) = (&vm.effects.stores, &rec.effects.stores);
        if vs.len() != rs.len() {
            o.gap(format!("store count {} vs {}", vs.len(), rs.len()));
        } else {
            for (i, (x, y)) in vs.iter().zip(rs).enumerate() {
                if x.width != y.width {
                    o.gap(format!("store {i} width differs"));
                    continue;
                }
                o.add(
                    CellId::StoreAddress(i),
                    x.address.clone(),
                    y.address.clone(),
                );
                o.add(CellId::StoreValue(i), x.value.clone(), y.value.clone());
            }
        }
        let (vc, rc) = (&vm.effects.calls, &rec.effects.calls);
        if vc.len() != rc.len() {
            o.gap(format!("call count {} vs {}", vc.len(), rc.len()));
        } else {
            for (i, (x, y)) in vc.iter().zip(rc).enumerate() {
                if x.iat_rva != y.iat_rva
                    || x.result_intrinsic != y.result_intrinsic
                    || x.arguments.len() != y.arguments.len()
                {
                    o.gap(format!("call {i} callee/arity differs"));
                    continue;
                }
                for (j, (p, q)) in x.arguments.iter().zip(&y.arguments).enumerate() {
                    o.add(CellId::CallArg { call: i, arg: j }, p.clone(), q.clone());
                }
            }
        }
        o
    }
}

/// Resource bound for one cell query (reported with every verdict).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Budget {
    /// Total free-input bits up to which the built-in backend enumerates.
    pub exhaustive_bits: u32,
    /// Wall-clock limit per cell and per backend.
    pub timeout_ms: u32,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            exhaustive_bits: 16,
            timeout_ms: 10_000,
        }
    }
}

/// Alive2-style statement of what a verdict covers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bound {
    pub budget: Budget,
    pub backends: Vec<String>,
    /// Distinct `Load`/`Intrinsic` terms treated as free values.
    pub uninterpreted_terms: usize,
    pub paths_checked: usize,
    pub domain_complete: bool,
}

/// Result of asking a backend whether `a != b` is satisfiable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SolveOutcome {
    /// No input makes the two differ.
    Unsat,
    /// A candidate model over the free registers (re-validated by the caller).
    Sat(BTreeMap<String, u64>),
    Timeout,
    Unknown(String),
}

/// Pluggable decision procedure for `a != b` over fixed-width bit-vectors.
pub trait SmtBackend {
    fn name(&self) -> String;
    fn check_distinct(&self, a: &Expr, b: &Expr, budget: &Budget) -> SolveOutcome;
}

/// Always-available backend: bounded exhaustive enumeration plus edge/random
/// input probing for counterexamples. Never needs an external solver.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeBackend;

impl SmtBackend for NativeBackend {
    fn name(&self) -> String {
        "native-bounded".to_owned()
    }

    fn check_distinct(&self, a: &Expr, b: &Expr, budget: &Budget) -> SolveOutcome {
        let Some(vars) = free_registers(a, b) else {
            return SolveOutcome::Unknown("register width clash".to_owned());
        };
        let total: u32 = vars.values().map(|w| w.bits()).sum();
        if total > budget.exhaustive_bits.min(26) {
            let probe = InputTester {
                random_inputs: 512,
                grid_cap: 512,
            };
            return match probe.test(a, b).divergence {
                Some(model) => SolveOutcome::Sat(model.into_iter().collect()),
                None => SolveOutcome::Timeout,
            };
        }
        let deadline = Instant::now() + Duration::from_millis(u64::from(budget.timeout_ms));
        let names: Vec<(&String, BitWidth)> = vars.iter().map(|(n, w)| (n, *w)).collect();
        for index in 0..(1_u64 << total) {
            if index % 1024 == 0 && Instant::now() > deadline {
                return SolveOutcome::Timeout;
            }
            let mut rest = index;
            let mut env = BTreeMap::new();
            let mut model = BTreeMap::new();
            for (name, width) in &names {
                let value = rest & (u64::MAX >> (64 - width.bits()));
                rest >>= width.bits();
                env.insert((*name).clone(), Expr::constant(*width, value));
                model.insert((*name).clone(), value);
            }
            match (fold(a, &env), fold(b, &env)) {
                (Some(x), Some(y)) if x == y => {}
                (Some(_), Some(_)) => return SolveOutcome::Sat(model),
                _ => return SolveOutcome::Unknown("term not foldable (load/intrinsic)".to_owned()),
            }
        }
        SolveOutcome::Unsat
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Method {
    Structural,
    Solver(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Counterexample {
    pub assignment: BTreeMap<String, u64>,
    pub vm: u64,
    pub recovered: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Unproven {
    Timeout(String),
    Unknown(String),
    ModelingGap(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CellVerdict {
    Proven(Method),
    Refuted(Counterexample),
    Unproven(Unproven),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EquivVerdict {
    Proven,
    Refuted,
    /// Not proven: timeout, unknown, or a modeling gap. Never treated as safe.
    Timeout,
}

#[derive(Clone, Debug)]
pub struct EquivalenceReport {
    pub verdict: EquivVerdict,
    pub cells: Vec<(CellId, CellVerdict)>,
    pub gaps: Vec<String>,
}

impl EquivalenceReport {
    #[must_use]
    pub fn first_counterexample(&self) -> Option<(&CellId, &Counterexample)> {
        self.cells.iter().find_map(|(id, v)| match v {
            CellVerdict::Refuted(c) => Some((id, c)),
            _ => None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct FunctionReport {
    pub verdict: EquivVerdict,
    pub paths: Vec<EquivalenceReport>,
    pub bound: Bound,
}

/// Runs the gate. Backends are tried in order after the structural check; the
/// [`NativeBackend`] is always first.
pub struct Verifier {
    pub budget: Budget,
    backends: Vec<Box<dyn SmtBackend>>,
    /// Verdicts of cells already decided, keyed by their simplified terms:
    /// paths of one function often share cells.
    cache: std::cell::RefCell<std::collections::HashMap<(Expr, Expr), CellVerdict>>,
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new(Budget::default())
    }
}

impl Verifier {
    #[must_use]
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            backends: vec![Box::new(NativeBackend)],
            cache: std::cell::RefCell::default(),
        }
    }

    /// Adds an extra (e.g. external solver) backend, consulted after the
    /// built-in one for cells it could not decide.
    #[must_use]
    pub fn with_backend(mut self, backend: Box<dyn SmtBackend>) -> Self {
        self.backends.push(backend);
        self
    }

    #[must_use]
    pub fn backend_names(&self) -> Vec<String> {
        self.backends.iter().map(|b| b.name()).collect()
    }

    #[must_use]
    pub fn check_cell(&self, cell: &Cell) -> CellVerdict {
        let trace = std::env::var_os("VMPX_GATE_TRACE").is_some();
        let started = Instant::now();
        if trace {
            eprintln!("[gate] cell {:?}: simplifying", cell.id);
        }
        let (a, b) = (cell.vm.simplify(), cell.recovered.simplify());
        if trace {
            eprintln!("[gate] cell {:?}: simplified in {:?}", cell.id, started.elapsed());
        }
        if a.width != b.width {
            return CellVerdict::Unproven(Unproven::ModelingGap(format!(
                "width {} vs {}",
                a.width.bits(),
                b.width.bits()
            )));
        }
        if a == b {
            return CellVerdict::Proven(Method::Structural);
        }
        let key = (a.clone(), b.clone());
        if let Some(done) = self.cache.borrow().get(&key) {
            return done.clone();
        }
        let verdict = self.decide(&a, &b);
        self.cache.borrow_mut().insert(key, verdict.clone());
        verdict
    }

    fn decide(&self, a: &Expr, b: &Expr) -> CellVerdict {
        let (a, b) = (a.clone(), b.clone());
        let mut worst = Unproven::Unknown("no backend decided".to_owned());
        for backend in &self.backends {
            match backend.check_distinct(&a, &b, &self.budget) {
                SolveOutcome::Unsat => return CellVerdict::Proven(Method::Solver(backend.name())),
                SolveOutcome::Sat(model) => match replay(&a, &b, &model) {
                    Some(cex) => return CellVerdict::Refuted(cex),
                    None => {
                        worst = Unproven::Unknown(format!(
                            "{}: model not confirmed by the VM evaluator",
                            backend.name()
                        ));
                    }
                },
                SolveOutcome::Timeout => worst = Unproven::Timeout(backend.name()),
                SolveOutcome::Unknown(why) => {
                    if !matches!(worst, Unproven::Timeout(_)) {
                        worst = Unproven::Unknown(format!("{}: {why}", backend.name()));
                    }
                }
            }
        }
        CellVerdict::Unproven(worst)
    }

    /// Composes per-cell verdicts for one path.
    #[must_use]
    pub fn check(&self, obligation: &Obligation) -> EquivalenceReport {
        let cells: Vec<(CellId, CellVerdict)> = obligation
            .cells
            .iter()
            .map(|c| (c.id.clone(), self.check_cell(c)))
            .collect();
        let refuted = cells
            .iter()
            .any(|(_, v)| matches!(v, CellVerdict::Refuted(_)));
        let all_proven = cells
            .iter()
            .all(|(_, v)| matches!(v, CellVerdict::Proven(_)));
        let verdict = if refuted {
            EquivVerdict::Refuted
        } else if all_proven && !cells.is_empty() && obligation.gaps.is_empty() {
            EquivVerdict::Proven
        } else {
            EquivVerdict::Timeout
        };
        EquivalenceReport {
            verdict,
            cells,
            gaps: obligation.gaps.clone(),
        }
    }

    /// Composes path verdicts (Dasgupta-style): PROVEN only if every path is
    /// PROVEN and the path family is domain-complete; any REFUTED path refutes.
    #[must_use]
    pub fn check_function(&self, paths: &[Obligation], domain_complete: bool) -> FunctionReport {
        let reports: Vec<EquivalenceReport> = paths.iter().map(|p| self.check(p)).collect();
        let verdict = if reports.iter().any(|r| r.verdict == EquivVerdict::Refuted) {
            EquivVerdict::Refuted
        } else if domain_complete
            && !reports.is_empty()
            && reports.iter().all(|r| r.verdict == EquivVerdict::Proven)
        {
            EquivVerdict::Proven
        } else {
            EquivVerdict::Timeout
        };
        let uninterpreted_terms = uninterpreted_terms(
            paths
                .iter()
                .flat_map(|p| &p.cells)
                .flat_map(|c| [&c.vm, &c.recovered]),
        );
        FunctionReport {
            verdict,
            paths: reports,
            bound: Bound {
                budget: self.budget.clone(),
                backends: self.backend_names(),
                uninterpreted_terms,
                paths_checked: paths.len(),
                domain_complete,
            },
        }
    }
}

/// Confirms a solver model against the VM's own evaluator; only a replay that
/// folds both sides to differing constants is a counterexample.
fn replay(a: &Expr, b: &Expr, model: &BTreeMap<String, u64>) -> Option<Counterexample> {
    let vars = free_registers(a, b)?;
    let env: BTreeMap<String, Expr> = vars
        .iter()
        .map(|(n, w)| {
            (
                n.clone(),
                Expr::constant(*w, model.get(n).copied().unwrap_or(0)),
            )
        })
        .collect();
    let (x, y) = (fold(a, &env)?, fold(b, &env)?);
    (x != y).then(|| Counterexample {
        assignment: vars
            .keys()
            .map(|n| (n.clone(), model.get(n).copied().unwrap_or(0)))
            .collect(),
        vm: x,
        recovered: y,
    })
}

/// Distinct `Load`/`Intrinsic` terms reachable from the cells (shared sub-terms
/// counted once: the terms are DAGs, a tree walk is exponential on them).
fn uninterpreted_terms<'a>(roots: impl Iterator<Item = &'a Expr>) -> usize {
    let mut seen: std::collections::HashSet<Expr> = std::collections::HashSet::new();
    let mut stack: Vec<Expr> = roots.cloned().collect();
    let mut count = 0;
    while let Some(e) = stack.pop() {
        if !seen.insert(e.clone()) {
            continue;
        }
        match &e.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => {}
            ExprKind::Load { address } => {
                count += 1;
                stack.push((**address).clone());
            }
            ExprKind::Intrinsic { arguments, .. } => {
                count += 1;
                stack.extend(arguments.iter().cloned());
            }
            ExprKind::Unary { value, .. }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => stack.push((**value).clone()),
            ExprKind::Binary { left, right, .. }
            | ExprKind::Concat { high: left, low: right } => {
                stack.push((**left).clone());
                stack.push((**right).clone());
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::{BinaryOp, NativeExitState};
    use vmpx_vmlift::{CollectedEffects, CollectedStore};

    const W: BitWidth = BitWidth::W64;

    fn reg(n: &str) -> Expr {
        Expr::register(W, n)
    }
    fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }
    fn k(v: u64) -> Expr {
        Expr::constant(W, v)
    }
    fn mul3() -> Expr {
        bin(BinaryOp::Mul, reg("a"), k(3))
    }
    fn shl_add(tail: u64) -> Expr {
        bin(
            BinaryOp::Add,
            bin(BinaryOp::Shl, reg("a"), k(1)),
            bin(BinaryOp::Add, reg("a"), k(tail)),
        )
    }
    fn one(id: CellId, vm: Expr, rec: Expr) -> Obligation {
        let mut o = Obligation::default();
        o.add(id, vm, rec);
        o
    }
    fn named(vm: Expr, rec: Expr) -> Obligation {
        one(CellId::Named("x".into()), vm, rec)
    }

    struct Fixed(SolveOutcome);
    impl SmtBackend for Fixed {
        fn name(&self) -> String {
            "fixed".into()
        }
        fn check_distinct(&self, _: &Expr, _: &Expr, _: &Budget) -> SolveOutcome {
            self.0.clone()
        }
    }

    #[test]
    fn trivially_equivalent_cells_prove() {
        // 8-bit De Morgan: exhaustive; plus a structural identity cell.
        let w = BitWidth::W8;
        let (a, b) = (Expr::register(w, "a"), Expr::register(w, "b"));
        let lhs = Expr::unary(
            vmpx_vm::UnaryOp::Not,
            bin(BinaryOp::And, a.clone(), b.clone()),
        );
        let rhs = bin(
            BinaryOp::Or,
            Expr::unary(vmpx_vm::UnaryOp::Not, a),
            Expr::unary(vmpx_vm::UnaryOp::Not, b),
        );
        let mut o = named(lhs, rhs);
        o.add(CellId::Rflags, reg("f"), reg("f"));
        let r = Verifier::default().check(&o);
        assert_eq!(r.verdict, EquivVerdict::Proven);
    }

    #[test]
    fn unequal_pair_is_refuted_with_replayed_counterexample() {
        let r = Verifier::default().check(&named(mul3(), shl_add(1)));
        assert_eq!(r.verdict, EquivVerdict::Refuted);
        let (_, cex) = r.first_counterexample().unwrap();
        assert_ne!(cex.vm, cex.recovered);
        // The counterexample really reproduces under the VM evaluator.
        let env: BTreeMap<String, Expr> = cex
            .assignment
            .iter()
            .map(|(n, v)| (n.clone(), k(*v)))
            .collect();
        assert_eq!(fold(&mul3(), &env), Some(cex.vm));
        assert_eq!(fold(&shl_add(1), &env), Some(cex.recovered));
    }

    #[test]
    fn too_hard_case_times_out_and_stays_unproven() {
        // Equivalent at 64 bits, but beyond the built-in bound and no solver.
        let r = Verifier::default().check(&named(mul3(), shl_add(0)));
        assert_eq!(r.verdict, EquivVerdict::Timeout);
        assert!(matches!(
            r.cells[0].1,
            CellVerdict::Unproven(Unproven::Timeout(_))
        ));
    }

    #[test]
    fn solver_timeout_and_unknown_never_prove() {
        for out in [SolveOutcome::Timeout, SolveOutcome::Unknown("x".into())] {
            let v = Verifier::default().with_backend(Box::new(Fixed(out)));
            let r = v.check(&named(mul3(), shl_add(0)));
            assert_eq!(r.verdict, EquivVerdict::Timeout);
        }
    }

    #[test]
    fn solver_unsat_proves_the_wide_case() {
        let v = Verifier::default().with_backend(Box::new(Fixed(SolveOutcome::Unsat)));
        let r = v.check(&named(mul3(), shl_add(0)));
        assert_eq!(r.verdict, EquivVerdict::Proven);
    }

    #[test]
    fn bogus_solver_model_is_not_a_refutation() {
        // Model says a=0 differs, but both sides evaluate equal there.
        let model = BTreeMap::from([("a".to_owned(), 0)]);
        let v = Verifier::default().with_backend(Box::new(Fixed(SolveOutcome::Sat(model))));
        let r = v.check(&named(mul3(), shl_add(0)));
        assert_eq!(r.verdict, EquivVerdict::Timeout);
    }

    #[test]
    fn one_refuted_cell_dominates_a_proven_one() {
        let mut o = named(mul3(), shl_add(1));
        o.add(CellId::Rflags, reg("f"), reg("f"));
        assert_eq!(Verifier::default().check(&o).verdict, EquivVerdict::Refuted);
    }

    #[test]
    fn modeling_gap_or_empty_obligation_is_never_proven() {
        let mut o = one(CellId::Rflags, reg("f"), reg("f"));
        o.gap("unsupported effect");
        assert_eq!(Verifier::default().check(&o).verdict, EquivVerdict::Timeout);
        assert_eq!(
            Verifier::default().check(&Obligation::default()).verdict,
            EquivVerdict::Timeout
        );
        // width mismatch is a gap, not a proof
        let o = named(Expr::register(BitWidth::W32, "a"), reg("a"));
        assert_eq!(Verifier::default().check(&o).verdict, EquivVerdict::Timeout);
    }

    #[test]
    fn unfoldable_loads_stay_unproven() {
        let l1 = Expr::load(W, reg("p"));
        let l2 = Expr::load(W, bin(BinaryOp::Add, reg("p"), k(8)));
        let r = Verifier::default().check(&named(l1, l2));
        assert_eq!(r.verdict, EquivVerdict::Timeout);
    }

    #[test]
    fn function_composition_requires_complete_domain() {
        let good = || one(CellId::Rflags, reg("f"), reg("f"));
        let v = Verifier::default();
        let paths = [good(), good()];
        assert_eq!(v.check_function(&paths, true).verdict, EquivVerdict::Proven);
        assert_eq!(
            v.check_function(&paths, false).verdict,
            EquivVerdict::Timeout
        );
        assert_eq!(v.check_function(&[], true).verdict, EquivVerdict::Timeout);
        let mixed = [good(), named(mul3(), shl_add(1))];
        let rep = v.check_function(&mixed, true);
        assert_eq!(rep.verdict, EquivVerdict::Refuted);
        assert_eq!(rep.bound.paths_checked, 2);
        assert_eq!(rep.bound.backends, vec!["native-bounded".to_owned()]);
    }

    fn exit(rax: Expr, store_value: Expr) -> ExitSemantics {
        ExitSemantics {
            exit: NativeExitState {
                schema_version: 1,
                registers: X64NativeRegister::ALL
                    .into_iter()
                    .map(|r| {
                        (
                            r,
                            if r == X64NativeRegister::Rax {
                                rax.clone()
                            } else {
                                k(0)
                            },
                        )
                    })
                    .collect(),
                rflags: k(0x246),
                return_address: k(0x1000),
                frame_size: 136,
            },
            effects: CollectedEffects {
                stores: vec![CollectedStore {
                    address: k(0x10),
                    value: store_value,
                    width: W,
                }],
                calls: Vec::new(),
                unsupported: Vec::new(),
            },
        }
    }

    #[test]
    fn exit_semantics_obligation_end_to_end() {
        let vm = exit(bin(BinaryOp::Add, reg("a"), reg("a")), k(1));
        let same = exit(bin(BinaryOp::Shl, reg("a"), k(1)), k(1));
        // Equivalent at 64 bits but beyond the built-in bound: unproven.
        let o = Obligation::from_exits(&vm, &same);
        assert_eq!(o.cells.len(), 15 + 2 + 2);
        assert_eq!(Verifier::default().check(&o).verdict, EquivVerdict::Timeout);

        let bad = exit(reg("a"), k(1));
        let r = Verifier::default().check(&Obligation::from_exits(&vm, &bad));
        assert_eq!(r.verdict, EquivVerdict::Refuted);
        assert_eq!(
            r.first_counterexample().unwrap().0,
            &CellId::Register(X64NativeRegister::Rax)
        );

        let mut unsupported = exit(reg("a"), k(1));
        unsupported.effects.unsupported.push("havoc".into());
        let o = Obligation::from_exits(&unsupported, &unsupported);
        assert!(!o.gaps.is_empty());
        assert_eq!(Verifier::default().check(&o).verdict, EquivVerdict::Timeout);
    }
}
