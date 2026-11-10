//! End-to-end ground-truth benchmark for the native-mutation tournament.
//!
//! Each case is a `(clean, obfuscated)` pair we KNOW is equal. We run the full
//! solver tournament (`Tournament`) on the obfuscated side and score the engine's
//! output against the clean ground truth. This is the piece the engine was
//! missing: it turns "built" into "built and measured".
//!
//! Equivalence is decided by EXHAUSTIVE bitvector enumeration over the whole
//! operand domain (`SmtGate::verify_bounded`, pure Rust, no z3): at `W8` with two
//! free variables that is `2^16` inputs — a genuine proof at that width, and MBA
//! identities hold at every width. The same exhaustive gate is injected as the
//! tournament's `EquivalenceBackend`, so the `Proven` tier is reachable here
//! without an SMT solver.
//!
//! Outcome per case:
//! - `Proven` / `Observed` — the solver tournament produced the simplification
//!   (and the gate proved / only observed it).
//! - `Trivial` — the IR's own structural simplifier already normalized the input
//!   to the clean form at construction, so the tournament had nothing to do.
//! - `FAIL` — the engine's output is NOT equal to the ground truth.
//!
//! Run it and read the numbers:
//! ```text
//! cargo test -p vmpx-tournament bench_mutation_report -- --nocapture
//! ```

use std::time::Instant;

use crate::{Gated, Tournament};
use vmpx_mutation::gate::{EquivalenceBackend, SmtVerdict, Tier};
use vmpx_mutation::{BitWidth, Ctx, Expr, Problem, ProblemClass};
use vmpx_verify::Verdict;
use vmpx_verify::smt::SmtGate;
use vmpx_vm::BinaryOp::{self, Add, And, Mul, Or, Sub, Xor};
use vmpx_vm::ExprKind;

const W: BitWidth = BitWidth::W8;
/// Two free variables × 8 bits — the exhaustive gate enumerates all `2^16`.
const BITS: u32 = 16;

fn reg(n: &str) -> Expr {
    Expr::register(W, n)
}
fn k(v: u64) -> Expr {
    Expr::constant(W, v)
}
fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr::binary(op, l, r).expect("well-formed binary")
}
fn x() -> Expr {
    reg("x")
}
fn y() -> Expr {
    reg("y")
}

/// `(name, clean, obfuscated)` — every pair is an exact bitvector identity,
/// asserted below before any solver runs.
fn corpus() -> Vec<(&'static str, Expr, Expr)> {
    vec![
        // --- linear MBA: real obfuscations the tournament must undo ---
        (
            "add=(x^y)+2(x&y)",
            bin(Add, x(), y()),
            bin(Add, bin(Xor, x(), y()), bin(Mul, k(2), bin(And, x(), y()))),
        ),
        (
            "add=(x|y)+(x&y)",
            bin(Add, x(), y()),
            bin(Add, bin(Or, x(), y()), bin(And, x(), y())),
        ),
        (
            "add=2(x|y)-(x^y)",
            bin(Add, x(), y()),
            bin(Sub, bin(Mul, k(2), bin(Or, x(), y())), bin(Xor, x(), y())),
        ),
        (
            "xor=(x|y)-(x&y)",
            bin(Xor, x(), y()),
            bin(Sub, bin(Or, x(), y()), bin(And, x(), y())),
        ),
        (
            "xor=x+y-2(x&y)",
            bin(Xor, x(), y()),
            bin(Sub, bin(Add, x(), y()), bin(Mul, k(2), bin(And, x(), y()))),
        ),
        (
            "and=(x|y)-(x^y)",
            bin(And, x(), y()),
            bin(Sub, bin(Or, x(), y()), bin(Xor, x(), y())),
        ),
        (
            "or=(x&y)+(x^y)",
            bin(Or, x(), y()),
            bin(Add, bin(And, x(), y()), bin(Xor, x(), y())),
        ),
        (
            "or=x+y-(x&y)",
            bin(Or, x(), y()),
            bin(Sub, bin(Add, x(), y()), bin(And, x(), y())),
        ),
        ("x=(x^y)^y", x(), bin(Xor, bin(Xor, x(), y()), y())),
        // --- multi-layer nesting (xor expanded inside the +2(x&y) form) ---
        (
            "add=nested",
            bin(Add, x(), y()),
            bin(
                Add,
                bin(Sub, bin(Or, x(), y()), bin(And, x(), y())),
                bin(Mul, k(2), bin(And, x(), y())),
            ),
        ),
        // --- nonlinear (the hard frontier) ---
        (
            "mul=2(xy)-(xy)",
            bin(Mul, x(), y()),
            bin(Sub, bin(Mul, k(2), bin(Mul, x(), y())), bin(Mul, x(), y())),
        ),
        // --- cases the IR structural simplifier normalizes at construction ---
        ("x=(x^y)^y via &", x(), bin(And, bin(Xor, bin(Xor, x(), y()), y()), x())),
        ("mul=xy+(x&0)", bin(Mul, x(), y()), bin(Add, bin(Mul, x(), y()), bin(And, x(), k(0)))),
        ("zero=x^x", k(0), bin(Xor, x(), x())),
        ("zero=x&0", k(0), bin(And, x(), k(0))),
    ]
}

fn nodes(e: &Expr) -> u64 {
    1 + match &e.kind {
        ExprKind::Binary { left, right, .. } => nodes(left) + nodes(right),
        ExprKind::Unary { value, .. } => nodes(value),
        ExprKind::Load { address } => nodes(address),
        ExprKind::Intrinsic { arguments, .. } => arguments.iter().map(nodes).sum(),
        ExprKind::Extract { value, .. } => nodes(value),
        ExprKind::Resize { value, .. } => nodes(value),
        ExprKind::Concat { high, low } => nodes(high) + nodes(low),
        _ => 0,
    }
}

/// Exhaustive bitvector equality over the operand domain (pure Rust).
fn equal(a: &Expr, b: &Expr) -> bool {
    matches!(SmtGate.verify_bounded(a, b, BITS), Verdict::Proven)
}

/// `SmtGate` as the tournament's equivalence backend: an exhaustive proof that
/// `a == b` maps to `Unsat` (sound — no input makes them differ); everything else
/// is `Unknown`, leaving refutation to the gate's own sampling.
struct ExhaustiveBackend;
impl EquivalenceBackend for ExhaustiveBackend {
    fn check_inequality(&self, original: &Expr, candidate: &Expr) -> SmtVerdict {
        match SmtGate.verify_bounded(original, candidate, BITS) {
            Verdict::Proven => SmtVerdict::Unsat,
            _ => SmtVerdict::Unknown,
        }
    }
}

struct Summary {
    correct: usize,
    proven: usize,
    trivial: usize,
    fail: usize,
    total: usize,
    reductions: Vec<i64>,
    rows: Vec<String>,
}

fn run() -> Summary {
    let ctx = Ctx { width: W };
    let cases = corpus();
    // Self-check: a mistyped identity would silently invalidate the whole
    // benchmark, so prove every pair equal before trusting any result.
    for (name, clean, obf) in &cases {
        assert!(equal(clean, obf), "corpus bug: `{name}` clean != obf");
    }

    let mut t = Tournament::default().with_backend(Box::new(ExhaustiveBackend));
    let (mut correct, mut proven, mut trivial, mut fail) = (0usize, 0usize, 0usize, 0usize);
    let mut reductions = Vec::new();
    let mut rows = Vec::new();

    for (name, clean, obf) in &cases {
        let p = Problem::new(obf.clone(), ProblemClass::Unknown, ctx.clone());
        let t0 = Instant::now();
        let got: Option<Gated> = t.run(&p);
        let ms = t0.elapsed().as_secs_f64() * 1e3;

        // The engine's output: the tournament's winner, or the (already
        // structurally normalized) input when no solver improves on it.
        let output = got
            .as_ref()
            .map(|g| g.candidate.expr.clone())
            .unwrap_or_else(|| obf.clone());
        let is_correct = equal(&output, clean);
        let nb = nodes(obf);
        let na = nodes(&output);

        let label = match &got {
            Some(g) => format!("{:?}", g.tier),
            None if is_correct => "Trivial".to_owned(),
            None => "FAIL".to_owned(),
        };
        if is_correct {
            correct += 1;
        } else {
            fail += 1;
        }
        match &got {
            Some(g) if is_correct && g.tier == Tier::Proven => proven += 1,
            None if is_correct => trivial += 1,
            _ => {}
        }
        reductions.push(nb as i64 - na as i64);
        rows.push(format!(
            "{name:<18} {label:<9} {ok:<4} {nb:>2}->{na:<2} {ms:>7.2}ms",
            ok = if is_correct { "ok" } else { "FAIL" },
        ));
    }

    Summary {
        correct,
        proven,
        trivial,
        fail,
        total: cases.len(),
        reductions,
        rows,
    }
}

fn median(mut v: Vec<i64>) -> i64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

#[test]
fn bench_mutation_report() {
    let s = run();
    println!(
        "\n=== native-mutation ground-truth benchmark ===\n\
         width W8, equivalence = exhaustive 2^16 bitvector proof (pure Rust, no z3)\n"
    );
    println!("{:<18} {:<9} {:<4} {:>6} {:>9}", "case", "outcome", "ok", "nodes", "time");
    for r in &s.rows {
        println!("{r}");
    }
    println!("---");
    println!(
        "correct {}/{}   tournament-proven {}/{}   trivially-normalized {}/{}   median node reduction {}",
        s.correct,
        s.total,
        s.proven,
        s.total,
        s.trivial,
        s.total,
        median(s.reductions.clone()),
    );

    // Can't-rot floors: the engine must stay correct on every case, the solver
    // tournament must keep proving the real obfuscations, and nothing may fail.
    assert_eq!(s.fail, 0, "an engine output did not match ground truth");
    assert_eq!(s.correct, s.total, "not every case is correct");
    assert!(
        s.proven >= 9,
        "tournament proof rate regressed to {}/{}",
        s.proven,
        s.total
    );
}
