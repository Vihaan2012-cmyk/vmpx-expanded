//! Synthetic benchmark: clean expressions obfuscated with known identities
//! (the inverses of the solvers' rules). Deterministic given a seed.
//!
//! Obfuscated forms are built with raw constructors so the VM's local
//! simplifier cannot fold the junk away before a solver sees it.

use vmpx_mutation::{BitWidth, Expr, ProblemClass};
use vmpx_vm::{BinaryOp, ExprKind, ExprRef, UnaryOp};

/// One benchmark item. `obfuscated` is semantically equal to `clean`.
#[derive(Clone, Debug)]
pub struct BenchProblem {
    pub clean: Expr,
    pub obfuscated: Expr,
    pub class: ProblemClass,
}

pub const BENCH_CLASSES: [ProblemClass; 5] = [
    ProblemClass::LinearMba,
    ProblemClass::NonlinearMba,
    ProblemClass::Substitution,
    ProblemClass::DeadCode,
    ProblemClass::OpaquePredicate,
];

const MAX_NODES: usize = 600;

pub struct Rng(u64);

impl Rng {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
}

fn raw_bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr {
        width: l.width,
        kind: ExprKind::Binary {
            op,
            left: ExprRef::new(l),
            right: ExprRef::new(r),
        },
    }
}

fn raw_un(op: UnaryOp, v: Expr) -> Expr {
    Expr {
        width: v.width,
        kind: ExprKind::Unary {
            op,
            value: ExprRef::new(v),
        },
    }
}

fn add(a: Expr, b: Expr) -> Expr {
    raw_bin(BinaryOp::Add, a, b)
}
fn sub(a: Expr, b: Expr) -> Expr {
    raw_bin(BinaryOp::Sub, a, b)
}
fn mul(a: Expr, b: Expr) -> Expr {
    raw_bin(BinaryOp::Mul, a, b)
}
fn xor(a: Expr, b: Expr) -> Expr {
    raw_bin(BinaryOp::Xor, a, b)
}
fn and(a: Expr, b: Expr) -> Expr {
    raw_bin(BinaryOp::And, a, b)
}
fn or(a: Expr, b: Expr) -> Expr {
    raw_bin(BinaryOp::Or, a, b)
}
fn not(a: Expr) -> Expr {
    raw_un(UnaryOp::Not, a)
}
fn neg(a: Expr) -> Expr {
    raw_un(UnaryOp::Neg, a)
}

struct Gen {
    rng: Rng,
    w: BitWidth,
}

impl Gen {
    fn var(&mut self) -> Expr {
        let n = if self.rng.chance(50) { "x" } else { "y" };
        Expr::register(self.w, n)
    }

    fn k(&self, v: u64) -> Expr {
        Expr::constant(self.w, v)
    }

    fn leaf(&mut self) -> Expr {
        if self.rng.chance(20) {
            let v = self.rng.next();
            self.k(v | 1)
        } else {
            self.var()
        }
    }

    fn bitwise_or_arith(&mut self) -> BinaryOp {
        [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Xor,
            BinaryOp::And,
            BinaryOp::Or,
        ][self.rng.below(5)]
    }

    /// Clean seed (built with the simplifying constructors).
    fn seed(&mut self, class: ProblemClass) -> Expr {
        let x = Expr::register(self.w, "x");
        let y = Expr::register(self.w, "y");
        let b = |op, l, r| Expr::binary(op, l, r).expect("same width");
        match class {
            ProblemClass::NonlinearMba => {
                let m = b(BinaryOp::Mul, x, y);
                if self.rng.chance(50) {
                    let op = self.bitwise_or_arith();
                    let l = self.var();
                    b(op, m, l)
                } else {
                    m
                }
            }
            ProblemClass::Substitution => match self.rng.below(6) {
                0 => b(BinaryOp::Add, x, y),
                1 => b(BinaryOp::Mul, x, self.k(2)),
                2 => Expr::unary(UnaryOp::Neg, x),
                3 => Expr::unary(UnaryOp::Not, y),
                4 => b(BinaryOp::Sub, x, y),
                _ => b(BinaryOp::Xor, x, y),
            },
            _ => {
                let op = self.bitwise_or_arith();
                let inner = b(op, x, y);
                if self.rng.chance(35) {
                    let op2 = self.bitwise_or_arith();
                    let l = self.leaf();
                    b(op2, inner, l)
                } else {
                    inner
                }
            }
        }
    }

    fn junk(&mut self) -> Expr {
        let op = self.bitwise_or_arith();
        raw_bin(op, self.var(), self.var())
    }

    fn other_var(e: &Expr) -> Expr {
        let name = match &e.kind {
            ExprKind::Register { name } if name == "x" => "y",
            _ => "x",
        };
        Expr::register(e.width, name)
    }

    fn linear_rule(&mut self, e: &Expr) -> Option<Expr> {
        let (a, b) = match &e.kind {
            ExprKind::Binary { left, right, .. } => ((**left).clone(), (**right).clone()),
            ExprKind::Register { .. } => {
                let y = Self::other_var(e);
                return Some(if self.rng.chance(50) {
                    add(and(e.clone(), y.clone()), and(e.clone(), not(y)))
                } else {
                    sub(or(e.clone(), y.clone()), and(y, not(e.clone())))
                });
            }
            _ => return None,
        };
        let two = self.k(2);
        let ExprKind::Binary { op, .. } = &e.kind else {
            return None;
        };
        let pick = self.rng.below(3);
        Some(match (op, pick) {
            (BinaryOp::Add, 0) => add(xor(a.clone(), b.clone()), mul(two, and(a, b))),
            (BinaryOp::Add, 1) => add(or(a.clone(), b.clone()), and(a, b)),
            (BinaryOp::Add, _) => sub(sub(a, not(b)), self.k(1)),
            (BinaryOp::Sub, 0) => sub(xor(a.clone(), b.clone()), mul(two, and(not(a), b))),
            (BinaryOp::Sub, 1) => add(add(a, not(b)), self.k(1)),
            (BinaryOp::Sub, _) => sub(and(a.clone(), not(b.clone())), and(not(a), b)),
            (BinaryOp::Xor, 0) => sub(or(a.clone(), b.clone()), and(a, b)),
            (BinaryOp::Xor, 1) => sub(add(a.clone(), b.clone()), mul(two, and(a, b))),
            (BinaryOp::Xor, _) => or(and(a.clone(), not(b.clone())), and(not(a), b)),
            (BinaryOp::And, 0) => sub(add(a.clone(), b.clone()), or(a, b)),
            (BinaryOp::And, _) => xor(or(a.clone(), b.clone()), xor(a, b)),
            (BinaryOp::Or, 0) => sub(add(a.clone(), b.clone()), and(a, b)),
            (BinaryOp::Or, 1) => add(xor(a.clone(), b.clone()), and(a, b)),
            (BinaryOp::Or, _) => add(and(a, not(b.clone())), b),
            (BinaryOp::Mul, _) => add(
                mul(and(a.clone(), b.clone()), or(a.clone(), b.clone())),
                mul(and(a.clone(), not(b.clone())), and(not(a), b)),
            ),
            _ => return None,
        })
    }

    fn substitution_rule(&mut self, e: &Expr) -> Option<Expr> {
        let zero = self.k(0);
        let one = self.k(1);
        Some(match (&e.kind, self.rng.below(5)) {
            (ExprKind::Binary { op: BinaryOp::Add, left, right }, 0) => {
                sub((**left).clone(), neg((**right).clone()))
            }
            (ExprKind::Binary { op: BinaryOp::Sub, left, right }, 0) => {
                add((**left).clone(), neg((**right).clone()))
            }
            (ExprKind::Binary { op: BinaryOp::Mul, left, right }, 0 | 1)
                if matches!(&right.kind, ExprKind::Const { value: 2 }) =>
            {
                add((**left).clone(), (**left).clone())
            }
            (ExprKind::Unary { op: UnaryOp::Neg, value }, 0 | 1) => {
                add(not((**value).clone()), one)
            }
            (_, 1) => not(not(e.clone())),
            (_, 2) => neg(neg(e.clone())),
            (_, 3) => xor(e.clone(), zero),
            (_, _) => mul(e.clone(), one),
        })
    }

    fn dead_rule(&mut self, e: &Expr) -> Option<Expr> {
        let j = self.junk();
        let zero = self.k(0);
        Some(match self.rng.below(6) {
            0 => or(e.clone(), and(j, zero)),
            1 => add(e.clone(), xor(j.clone(), j)),
            2 => and(e.clone(), or(e.clone(), j)),
            3 => sub(e.clone(), and(j, zero)),
            4 => or(e.clone(), and(e.clone(), j)),
            _ => xor(e.clone(), xor(j.clone(), j)),
        })
    }

    fn rewrite(&mut self, e: &Expr, class: ProblemClass, pct: u64) -> Expr {
        let rebuilt = match &e.kind {
            ExprKind::Binary { op, left, right } => {
                let l = self.rewrite(left, class, pct);
                let r = self.rewrite(right, class, pct);
                raw_bin(*op, l, r)
            }
            ExprKind::Unary { op, value } => raw_un(*op, self.rewrite(value, class, pct)),
            _ => e.clone(),
        };
        if !self.rng.chance(pct) {
            return rebuilt;
        }
        let out = match class {
            ProblemClass::LinearMba | ProblemClass::NonlinearMba => self.linear_rule(&rebuilt),
            ProblemClass::Substitution => self.substitution_rule(&rebuilt),
            ProblemClass::DeadCode => self.dead_rule(&rebuilt),
            _ => None,
        };
        out.unwrap_or(rebuilt)
    }

    fn obfuscate(&mut self, clean: &Expr, class: ProblemClass) -> Expr {
        let (rounds, pct) = match class {
            ProblemClass::LinearMba | ProblemClass::NonlinearMba => (2, 60),
            ProblemClass::Substitution => (2, 55),
            _ => (2, 45),
        };
        let mut cur = clean.clone();
        for _ in 0..rounds {
            cur = self.rewrite(&cur, class, pct);
        }
        cur
    }

    /// Opaque predicates: always-constant expressions over a free variable.
    fn opaque(&mut self) -> (Expr, Expr) {
        let v = self.var();
        let (one, zero) = (self.k(1), self.k(0));
        match self.rng.below(4) {
            0 => (
                zero.clone(),
                and(mul(v.clone(), add(v, one.clone())), one),
            ),
            1 => (zero, and(mul(v.clone(), sub(v, one.clone())), one)),
            2 => (one.clone(), and(or(v, one.clone()), one)),
            _ => {
                let w = self.var();
                let junk = and(mul(w.clone(), add(w, one.clone())), one);
                (self.k(0), junk)
            }
        }
    }

    fn problem(&mut self, class: ProblemClass) -> Option<BenchProblem> {
        for _ in 0..50 {
            let (clean, obfuscated) = if class == ProblemClass::OpaquePredicate {
                self.opaque()
            } else {
                let clean = self.seed(class);
                let obf = self.obfuscate(&clean, class);
                (clean, obf)
            };
            let n = obfuscated.node_count_bounded(MAX_NODES);
            if obfuscated != clean && n > clean.node_count_bounded(MAX_NODES) && n <= MAX_NODES {
                return Some(BenchProblem {
                    clean,
                    obfuscated,
                    class,
                });
            }
        }
        None
    }
}

/// `per_class` problems for each of the five benchmark classes.
#[must_use]
pub fn generate(seed: u64, per_class: usize, width: BitWidth) -> Vec<BenchProblem> {
    let mut g = Gen {
        rng: Rng::new(seed),
        w: width,
    };
    BENCH_CLASSES
        .iter()
        .flat_map(|&class| (0..per_class).filter_map(|_| g.problem(class)).collect::<Vec<_>>())
        .collect()
}
