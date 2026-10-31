//! SMT-LIB2 (QF_BV) emission for `vmpx_vm::Expr` plus an external-solver CLI
//! shim (z3, bitwuzla, cvc5) implementing [`SmtBackend`].
//!
//! Nothing here links a solver: the shim only runs when a binary is supplied
//! or discovered on PATH, so default builds and CI need none. Shift/rotate
//! semantics mirror `vmpx_vm`'s evaluator (count masked to 6 bits for 64-bit,
//! 5 bits otherwise; rotates modulo the width). `Load`/`Intrinsic` terms become
//! free constants (sound for UNSAT; any SAT model is re-validated by the caller).

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use vmpx_vm::{BinaryOp, Expr, ExprKind, ExprRef, ResizeOp, UnaryOp};

use crate::equiv::{Budget, SmtBackend, SolveOutcome};
use crate::smt::free_registers;

const MAX_DEPTH: u32 = 1000;

/// A ready-to-run query asserting `a != b`.
#[derive(Clone, Debug)]
pub struct Script {
    pub text: String,
    /// Solver symbol -> original register name, for model read-back.
    pub registers: Vec<(String, String)>,
}

/// Emits the `a != b` query, or `None` for shapes it cannot encode faithfully.
#[must_use]
pub fn emit(a: &Expr, b: &Expr) -> Option<Script> {
    if a.width != b.width {
        return None;
    }
    let vars = free_registers(a, b)?;
    let mut em = Emitter::default();
    let mut regs = Vec::new();
    for (i, (name, w)) in vars.iter().enumerate() {
        let sym = format!("r{i}");
        em.out
            .push_str(&format!("(declare-const {sym} (_ BitVec {}))\n", w.bits()));
        em.registers.insert(name.clone(), sym.clone());
        regs.push((sym, name.clone()));
    }
    let (x, y) = (em.expr(a, 0)?, em.expr(b, 0)?);
    let mut text = String::from("(set-logic QF_BV)\n");
    text.push_str(&em.out);
    text.push_str(&format!("(assert (distinct {x} {y}))\n(check-sat)\n"));
    if !regs.is_empty() {
        let list: Vec<&str> = regs.iter().map(|(s, _)| s.as_str()).collect();
        text.push_str(&format!("(get-value ({}))\n", list.join(" ")));
    }
    Some(Script {
        text,
        registers: regs,
    })
}

#[derive(Default)]
struct Emitter {
    out: String,
    registers: HashMap<String, String>,
    memo: HashMap<ExprRef, String>,
    free: HashMap<Expr, String>,
    next: usize,
}

impl Emitter {
    fn define(&mut self, bits: u32, body: &str) -> String {
        let n = format!("n{}", self.next);
        self.next += 1;
        self.out
            .push_str(&format!("(define-fun {n} () (_ BitVec {bits}) {body})\n"));
        n
    }

    fn child(&mut self, r: &ExprRef, depth: u32) -> Option<String> {
        if let Some(n) = self.memo.get(r) {
            return Some(n.clone());
        }
        let n = self.expr(r, depth + 1)?;
        self.memo.insert(r.clone(), n.clone());
        Some(n)
    }

    fn expr(&mut self, e: &Expr, depth: u32) -> Option<String> {
        if depth > MAX_DEPTH {
            return None;
        }
        let bits = e.width.bits();
        let body = match &e.kind {
            ExprKind::Const { value } => {
                return Some(format!("(_ bv{} {bits})", e.width.truncate(*value)));
            }
            ExprKind::Register { name } => return self.registers.get(name).cloned(),
            ExprKind::Unary { op, value } => {
                let v = self.child(value, depth)?;
                match op {
                    UnaryOp::Not => format!("(bvnot {v})"),
                    UnaryOp::Neg => format!("(bvneg {v})"),
                }
            }
            ExprKind::Binary { op, left, right } => {
                let (l, r) = (self.child(left, depth)?, self.child(right, depth)?);
                if left.width != e.width || right.width != e.width {
                    return None;
                }
                binary(*op, bits, &l, &r)
            }
            ExprKind::Load { .. } | ExprKind::Intrinsic { .. } => {
                if let Some(n) = self.free.get(e) {
                    return Some(n.clone());
                }
                let n = format!("u{}", self.free.len());
                self.out
                    .push_str(&format!("(declare-const {n} (_ BitVec {bits}))\n"));
                self.free.insert(e.clone(), n.clone());
                return Some(n);
            }
            ExprKind::Extract { value, lsb } => {
                let v = self.child(value, depth)?;
                let hi = lsb.checked_add(bits)?.checked_sub(1)?;
                if hi >= value.width.bits() {
                    return None;
                }
                format!("((_ extract {hi} {lsb}) {v})")
            }
            ExprKind::Concat { high, low } => {
                let (h, l) = (self.child(high, depth)?, self.child(low, depth)?);
                if high.width.bits() + low.width.bits() != bits {
                    return None;
                }
                format!("(concat {h} {l})")
            }
            ExprKind::Resize { op, value } => {
                let v = self.child(value, depth)?;
                let src = value.width.bits();
                match op {
                    ResizeOp::Truncate if bits <= src => {
                        format!("((_ extract {} 0) {v})", bits - 1)
                    }
                    ResizeOp::ZeroExtend if bits >= src => {
                        format!("((_ zero_extend {}) {v})", bits - src)
                    }
                    ResizeOp::SignExtend if bits >= src => {
                        format!("((_ sign_extend {}) {v})", bits - src)
                    }
                    _ => return None,
                }
            }
        };
        Some(self.define(bits, &body))
    }
}

fn binary(op: BinaryOp, bits: u32, l: &str, r: &str) -> String {
    let mask = if bits == 64 { 0x3F } else { 0x1F };
    let count = format!("(bvand {r} (_ bv{mask} {bits}))");
    let rot = format!("(bvand {r} (_ bv{} {bits}))", bits - 1);
    let comp = format!("(bvsub (_ bv{bits} {bits}) {rot})");
    match op {
        BinaryOp::Add => format!("(bvadd {l} {r})"),
        BinaryOp::Sub => format!("(bvsub {l} {r})"),
        BinaryOp::Mul => format!("(bvmul {l} {r})"),
        BinaryOp::Xor => format!("(bvxor {l} {r})"),
        BinaryOp::And => format!("(bvand {l} {r})"),
        BinaryOp::Or => format!("(bvor {l} {r})"),
        BinaryOp::Shl => format!("(bvshl {l} {count})"),
        BinaryOp::Shr => format!("(bvlshr {l} {count})"),
        BinaryOp::Sar => format!("(bvashr {l} {count})"),
        BinaryOp::Rol => format!("(bvor (bvshl {l} {rot}) (bvlshr {l} {comp}))"),
        BinaryOp::Ror => format!("(bvor (bvlshr {l} {rot}) (bvshl {l} {comp}))"),
    }
}

/// Parsed solver answer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Response {
    Unsat,
    Sat(BTreeMap<String, u64>),
    Unknown(String),
}

/// Parses `sat`/`unsat`/`unknown`/`timeout` plus the `get-value` model.
/// Anything unrecognised is `Unknown` (fail-closed).
#[must_use]
pub fn parse_response(text: &str, script: &Script) -> Response {
    let mut lines = text.trim_start().splitn(2, '\n');
    let head = lines.next().unwrap_or("").trim();
    match head {
        "unsat" => Response::Unsat,
        "sat" => {
            let model = lines.next().unwrap_or("");
            let mut out = BTreeMap::new();
            for (sym, name) in &script.registers {
                match find_value(model, sym) {
                    Some(v) => {
                        out.insert(name.clone(), v);
                    }
                    None => return Response::Unknown(format!("no model value for {sym}")),
                }
            }
            Response::Sat(out)
        }
        other => Response::Unknown(other.to_owned()),
    }
}

/// Reads `(sym #x..)`, `(sym #b..)` or `(sym (_ bvN W))` from a get-value reply.
fn find_value(model: &str, sym: &str) -> Option<u64> {
    let pat = format!("({sym} ");
    let rest = &model[model.find(&pat)? + pat.len()..];
    let rest = rest.trim_start();
    if let Some(h) = rest.strip_prefix("#x") {
        let digits: String = h.chars().take_while(char::is_ascii_hexdigit).collect();
        u64::from_str_radix(&digits, 16).ok()
    } else if let Some(b) = rest.strip_prefix("#b") {
        let digits: String = b.chars().take_while(|c| *c == '0' || *c == '1').collect();
        u64::from_str_radix(&digits, 2).ok()
    } else if let Some(d) = rest.strip_prefix("(_ bv") {
        let digits: String = d.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Flavor {
    Z3,
    Bitwuzla,
    Cvc5,
}

impl Flavor {
    fn binary(self) -> &'static str {
        match self {
            Self::Z3 => "z3",
            Self::Bitwuzla => "bitwuzla",
            Self::Cvc5 => "cvc5",
        }
    }

    fn args(self, timeout_ms: u32) -> Vec<String> {
        match self {
            Self::Z3 => vec!["-in".into(), "-smt2".into(), format!("-t:{timeout_ms}")],
            Self::Bitwuzla => vec![format!("--time-limit={timeout_ms}")],
            Self::Cvc5 => vec![
                "--lang=smt2".into(),
                "--produce-models".into(),
                format!("--tlimit={timeout_ms}"),
            ],
        }
    }
}

/// External SMT solver driven over stdin/stdout. Optional and off by default.
#[derive(Clone, Debug)]
pub struct CliBackend {
    pub flavor: Flavor,
    pub program: PathBuf,
}

impl CliBackend {
    #[must_use]
    pub fn new(flavor: Flavor, program: impl Into<PathBuf>) -> Self {
        Self {
            flavor,
            program: program.into(),
        }
    }

    /// Finds `flavor`'s binary on PATH by running `--version`; `None` if absent.
    #[must_use]
    pub fn discover(flavor: Flavor) -> Option<Self> {
        let program = PathBuf::from(flavor.binary());
        Command::new(&program)
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .ok()
            .filter(|s| s.success())?;
        Some(Self { flavor, program })
    }

    fn run(&self, script: &Script, timeout_ms: u32) -> Result<Option<String>, String> {
        let mut child = Command::new(&self.program)
            .args(self.flavor.args(timeout_ms))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn failed: {e}"))?;
        let mut stdin = child.stdin.take().ok_or("no stdin")?;
        let mut stdout = child.stdout.take().ok_or("no stdout")?;
        let text = script.text.clone();
        let writer = std::thread::spawn(move || {
            let _ = stdin.write_all(text.as_bytes());
        });
        let reader = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = stdout.read_to_string(&mut s);
            s
        });
        // Hard wall-clock kill a little after the solver's own soft limit.
        let deadline = Instant::now() + Duration::from_millis(u64::from(timeout_ms) + 1_000);
        let timed_out = loop {
            match child.try_wait() {
                Ok(Some(_)) => break false,
                Ok(None) if Instant::now() > deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break true;
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(5)),
                Err(e) => return Err(format!("wait failed: {e}")),
            }
        };
        let _ = writer.join();
        let out = reader.join().map_err(|_| "reader panicked".to_owned())?;
        Ok((!timed_out).then_some(out))
    }
}

impl SmtBackend for CliBackend {
    fn name(&self) -> String {
        format!("{}-cli", self.flavor.binary())
    }

    fn check_distinct(&self, a: &Expr, b: &Expr, budget: &Budget) -> SolveOutcome {
        let Some(script) = emit(a, b) else {
            return SolveOutcome::Unknown("unencodable term".to_owned());
        };
        match self.run(&script, budget.timeout_ms) {
            Err(e) => SolveOutcome::Unknown(e),
            Ok(None) => SolveOutcome::Timeout,
            Ok(Some(out)) => match parse_response(&out, &script) {
                Response::Unsat => SolveOutcome::Unsat,
                Response::Sat(m) => SolveOutcome::Sat(m),
                Response::Unknown(s) if s.contains("timeout") || s.contains("canceled") => {
                    SolveOutcome::Timeout
                }
                Response::Unknown(s) => SolveOutcome::Unknown(s),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::BitWidth;

    const W: BitWidth = BitWidth::W64;

    fn reg(n: &str) -> Expr {
        Expr::register(W, n)
    }

    #[test]
    fn emits_wellformed_query() {
        let a = Expr::binary(BinaryOp::Mul, reg("a"), Expr::constant(W, 3)).unwrap();
        let b = Expr::binary(BinaryOp::Shl, reg("a"), Expr::constant(W, 1)).unwrap();
        let s = emit(&a, &b).unwrap();
        assert!(
            s.text
                .starts_with("(set-logic QF_BV)\n(declare-const r0 (_ BitVec 64))")
        );
        assert!(s.text.contains("bvmul r0 (_ bv3 64)"));
        assert!(s.text.contains("(bvand (_ bv1 64) (_ bv63 64))") || s.text.contains("bvshl"));
        assert!(s.text.contains("(check-sat)") && s.text.contains("(get-value (r0))"));
        assert_eq!(s.registers, vec![("r0".to_owned(), "a".to_owned())]);
    }

    #[test]
    fn rotate_and_shift_follow_vm_masking() {
        let a = Expr::binary(BinaryOp::Rol, reg("a"), reg("c")).unwrap();
        let s = emit(&a, &reg("a")).unwrap();
        assert!(s.text.contains("(bvand r1 (_ bv63 64))"));
        assert!(s.text.contains("(bvsub (_ bv64 64)"));
        let w8 = BitWidth::W8;
        let sh = Expr::binary(
            BinaryOp::Shr,
            Expr::register(w8, "x"),
            Expr::register(w8, "y"),
        )
        .unwrap();
        let s = emit(&sh, &Expr::register(w8, "x")).unwrap();
        assert!(s.text.contains("(bvand r1 (_ bv31 8))"));
    }

    #[test]
    fn loads_become_free_constants_and_width_clash_is_rejected() {
        let l = Expr::load(W, reg("p"));
        let s = emit(&l, &l.clone()).unwrap();
        assert!(s.text.contains("(declare-const u0 (_ BitVec 64))"));
        assert!(emit(&Expr::register(BitWidth::W32, "a"), &reg("a")).is_none());
        assert!(emit(&reg("a"), &Expr::constant(BitWidth::W32, 1)).is_none());
    }

    #[test]
    fn shared_dag_is_linear_not_exponential() {
        let mut e = reg("a");
        for _ in 0..40 {
            let m = Expr::binary(BinaryOp::Mul, e.clone(), Expr::constant(W, 3)).unwrap();
            let s = Expr::binary(BinaryOp::Add, e, Expr::constant(W, 5)).unwrap();
            e = Expr::binary(BinaryOp::Xor, m, s).unwrap();
        }
        let s = emit(&e, &reg("a")).unwrap();
        assert!(s.text.len() < 20_000);
    }

    #[test]
    fn parses_solver_replies_fail_closed() {
        let s = Script {
            text: String::new(),
            registers: vec![("r0".into(), "a".into()), ("r1".into(), "b".into())],
        };
        assert_eq!(parse_response("unsat\n(error \"x\")", &s), Response::Unsat);
        let m = parse_response("sat\n((r0 #x00000000000000ff)\n (r1 (_ bv5 8)))\n", &s);
        assert_eq!(
            m,
            Response::Sat(BTreeMap::from([("a".into(), 255), ("b".into(), 5)]))
        );
        assert_eq!(
            parse_response("sat\n((r0 #b101) (r1 #x1))", &s),
            Response::Sat(BTreeMap::from([("a".into(), 5), ("b".into(), 1)]))
        );
        assert!(matches!(
            parse_response("sat\n((r0 #x1))", &s),
            Response::Unknown(_)
        ));
        assert!(matches!(
            parse_response("unknown", &s),
            Response::Unknown(_)
        ));
        assert!(matches!(parse_response("", &s), Response::Unknown(_)));
        assert!(matches!(
            parse_response("(error \"no\")", &s),
            Response::Unknown(_)
        ));
    }

    #[test]
    fn missing_solver_binary_is_unknown_not_proven() {
        let b = CliBackend::new(Flavor::Z3, "definitely-not-a-real-solver-binary");
        let out = b.check_distinct(&reg("a"), &reg("b"), &Budget::default());
        assert!(matches!(out, SolveOutcome::Unknown(_)));
    }
}
