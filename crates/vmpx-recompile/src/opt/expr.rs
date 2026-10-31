//! Width-generic bitvector expressions and the rewrite-rule verifier.
//!
//! A [`Rule`] is only usable once [`verify_rule`] returns [`Verdict::Proven`].
//! Proofs are symbolic and hold for every width up to 64:
//! - `BitSlice`: the rule uses only `not/and/or/xor` (and the constants 0 and
//!   all-ones), so it is bit-parallel and a one-bit truth table decides it.
//! - `Ring`: the rule uses only `+ - * neg not <<const`, so both sides are
//!   normalised to polynomials over Z/2^64; equal polynomials are equal modulo
//!   2^n for every n <= 64.
//!
//! Anything else is `Unknown` and never ships. A mismatch is found first by
//! evaluation (exhaustive at small widths, corner values at 16/32/64) and is
//! returned as a stored [`Counterexample`].

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Expr {
    Var(u8),
    Const(u64),
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    Shl(Box<Expr>, u8),
}

#[must_use]
pub fn var(index: u8) -> Expr {
    Expr::Var(index)
}

#[must_use]
pub fn konst(value: u64) -> Expr {
    Expr::Const(value)
}

#[must_use]
pub fn bin(op: Op, left: Expr, right: Expr) -> Expr {
    Expr::Bin(op, Box::new(left), Box::new(right))
}

#[must_use]
pub fn not(inner: Expr) -> Expr {
    Expr::Not(Box::new(inner))
}

#[must_use]
pub fn neg(inner: Expr) -> Expr {
    Expr::Neg(Box::new(inner))
}

#[must_use]
pub fn shl(inner: Expr, count: u8) -> Expr {
    Expr::Shl(Box::new(inner), count)
}

#[must_use]
pub const fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

impl Expr {
    /// Evaluates at `bits` width. `env[i]` is the value of `Var(i)`.
    #[must_use]
    pub fn eval(&self, env: &[u64], bits: u32) -> u64 {
        let m = mask(bits);
        match self {
            Self::Var(index) => env[usize::from(*index)] & m,
            Self::Const(value) => value & m,
            Self::Not(inner) => !inner.eval(env, bits) & m,
            Self::Neg(inner) => inner.eval(env, bits).wrapping_neg() & m,
            Self::Shl(inner, count) => {
                if u32::from(*count) >= bits {
                    0
                } else {
                    (inner.eval(env, bits) << count) & m
                }
            }
            Self::Bin(op, left, right) => {
                let (a, b) = (left.eval(env, bits), right.eval(env, bits));
                (match op {
                    Op::Add => a.wrapping_add(b),
                    Op::Sub => a.wrapping_sub(b),
                    Op::Mul => a.wrapping_mul(b),
                    Op::And => a & b,
                    Op::Or => a | b,
                    Op::Xor => a ^ b,
                }) & m
            }
        }
    }

    fn arity(&self) -> usize {
        match self {
            Self::Var(index) => usize::from(*index) + 1,
            Self::Const(_) => 0,
            Self::Not(inner) | Self::Neg(inner) | Self::Shl(inner, _) => inner.arity(),
            Self::Bin(_, left, right) => left.arity().max(right.arity()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rule {
    pub name: &'static str,
    pub pattern: Expr,
    pub replacement: Expr,
}

impl Rule {
    #[must_use]
    pub fn new(name: &'static str, pattern: Expr, replacement: Expr) -> Self {
        Self {
            name,
            pattern,
            replacement,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofMethod {
    BitSlice,
    Ring,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Counterexample {
    pub bits: u32,
    pub inputs: Vec<u64>,
    pub pattern_value: u64,
    pub replacement_value: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Verdict {
    Proven(ProofMethod),
    Refuted(Counterexample),
    Unknown,
}

/// Decides a rule: counterexample search first, then a symbolic proof.
#[must_use]
pub fn verify_rule(rule: &Rule) -> Verdict {
    // The replacement may only use variables the pattern binds.
    if rule.replacement.arity() > rule.pattern.arity() {
        return Verdict::Unknown;
    }
    let vars = rule.pattern.arity();
    if vars > 4 {
        return Verdict::Unknown;
    }
    if let Some(counterexample) = find_counterexample(rule, vars) {
        return Verdict::Refuted(counterexample);
    }
    if is_bitwise(&rule.pattern) && is_bitwise(&rule.replacement) && bit_slice_equal(rule, vars) {
        return Verdict::Proven(ProofMethod::BitSlice);
    }
    if let (Some(left), Some(right)) = (polynomial(&rule.pattern), polynomial(&rule.replacement))
        && left == right
    {
        return Verdict::Proven(ProofMethod::Ring);
    }
    Verdict::Unknown
}

fn find_counterexample(rule: &Rule, vars: usize) -> Option<Counterexample> {
    let check = |inputs: &[u64], bits: u32| {
        let pattern_value = rule.pattern.eval(inputs, bits);
        let replacement_value = rule.replacement.eval(inputs, bits);
        (pattern_value != replacement_value).then(|| Counterexample {
            bits,
            inputs: inputs.iter().map(|v| v & mask(bits)).collect(),
            pattern_value,
            replacement_value,
        })
    };
    let exhaustive_bits = match vars {
        0..=2 => 8,
        3 => 4,
        _ => 2,
    };
    for bits in 1..=exhaustive_bits {
        let mut inputs = vec![0u64; vars];
        loop {
            if let Some(found) = check(&inputs, bits) {
                return Some(found);
            }
            if !advance(&mut inputs, mask(bits)) {
                break;
            }
        }
    }
    const CORNERS: [u64; 12] = [
        0,
        1,
        2,
        3,
        0x7f,
        0x80,
        0xff,
        0x7fff_ffff,
        0x8000_0000,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        u64::MAX,
    ];
    for bits in [16, 32, 64] {
        let mut indices = vec![0usize; vars];
        loop {
            let inputs: Vec<u64> = indices.iter().map(|&i| CORNERS[i]).collect();
            if let Some(found) = check(&inputs, bits) {
                return Some(found);
            }
            let mut position = 0;
            while position < vars {
                indices[position] += 1;
                if indices[position] < CORNERS.len() {
                    break;
                }
                indices[position] = 0;
                position += 1;
            }
            if position == vars {
                break;
            }
        }
    }
    None
}

fn advance(inputs: &mut [u64], max: u64) -> bool {
    for value in inputs.iter_mut() {
        if *value < max {
            *value += 1;
            return true;
        }
        *value = 0;
    }
    false
}

fn is_bitwise(expr: &Expr) -> bool {
    match expr {
        Expr::Var(_) => true,
        Expr::Const(value) => *value == 0 || *value == u64::MAX,
        Expr::Not(inner) => is_bitwise(inner),
        Expr::Bin(Op::And | Op::Or | Op::Xor, left, right) => is_bitwise(left) && is_bitwise(right),
        _ => false,
    }
}

fn bit_slice_equal(rule: &Rule, vars: usize) -> bool {
    (0..1u32 << vars).all(|assignment| {
        let inputs: Vec<u64> = (0..vars)
            .map(|i| u64::from((assignment >> i) & 1))
            .collect();
        rule.pattern.eval(&inputs, 1) == rule.replacement.eval(&inputs, 1)
    })
}

type Poly = BTreeMap<Vec<u8>, u64>;

fn poly_add(left: &Poly, right: &Poly) -> Poly {
    let mut out = left.clone();
    for (monomial, coefficient) in right {
        let entry = out.entry(monomial.clone()).or_insert(0);
        *entry = entry.wrapping_add(*coefficient);
    }
    out.retain(|_, coefficient| *coefficient != 0);
    out
}

fn poly_neg(poly: &Poly) -> Poly {
    poly.iter()
        .map(|(m, c)| (m.clone(), c.wrapping_neg()))
        .collect()
}

fn poly_mul(left: &Poly, right: &Poly) -> Poly {
    let mut out = Poly::new();
    for (lm, lc) in left {
        for (rm, rc) in right {
            let mut monomial = lm.clone();
            monomial.extend_from_slice(rm);
            monomial.sort_unstable();
            let entry = out.entry(monomial).or_insert(0);
            *entry = entry.wrapping_add(lc.wrapping_mul(*rc));
        }
    }
    out.retain(|_, coefficient| *coefficient != 0);
    out
}

fn constant_poly(value: u64) -> Poly {
    let mut poly = Poly::new();
    if value != 0 {
        poly.insert(Vec::new(), value);
    }
    poly
}

fn polynomial(expr: &Expr) -> Option<Poly> {
    Some(match expr {
        Expr::Var(index) => Poly::from([(vec![*index], 1)]),
        Expr::Const(value) => constant_poly(*value),
        Expr::Neg(inner) => poly_neg(&polynomial(inner)?),
        // !x == -x - 1 in two's complement.
        Expr::Not(inner) => poly_add(&poly_neg(&polynomial(inner)?), &constant_poly(u64::MAX)),
        Expr::Shl(inner, count) if *count < 64 => {
            poly_mul(&polynomial(inner)?, &constant_poly(1u64 << count))
        }
        Expr::Bin(Op::Add, left, right) => poly_add(&polynomial(left)?, &polynomial(right)?),
        Expr::Bin(Op::Sub, left, right) => {
            poly_add(&polynomial(left)?, &poly_neg(&polynomial(right)?))
        }
        Expr::Bin(Op::Mul, left, right) => poly_mul(&polynomial(left)?, &polynomial(right)?),
        _ => return None,
    })
}

pub type Bindings = [Option<Expr>; 4];

/// Matches `pattern` against `expr`, binding pattern variables to subtrees.
#[must_use]
pub fn match_pattern(pattern: &Expr, expr: &Expr, bindings: &mut Bindings) -> bool {
    match (pattern, expr) {
        (Expr::Var(index), _) => {
            let slot = &mut bindings[usize::from(*index)];
            match slot {
                Some(bound) => bound == expr,
                None => {
                    *slot = Some(expr.clone());
                    true
                }
            }
        }
        (Expr::Const(a), Expr::Const(b)) => a == b,
        (Expr::Not(a), Expr::Not(b)) | (Expr::Neg(a), Expr::Neg(b)) => {
            match_pattern(a, b, bindings)
        }
        (Expr::Shl(a, ca), Expr::Shl(b, cb)) => ca == cb && match_pattern(a, b, bindings),
        (Expr::Bin(oa, la, ra), Expr::Bin(ob, lb, rb)) => {
            oa == ob && match_pattern(la, lb, bindings) && match_pattern(ra, rb, bindings)
        }
        _ => false,
    }
}

#[must_use]
pub fn instantiate(template: &Expr, bindings: &Bindings) -> Option<Expr> {
    Some(match template {
        Expr::Var(index) => bindings[usize::from(*index)].clone()?,
        Expr::Const(value) => Expr::Const(*value),
        Expr::Not(inner) => not(instantiate(inner, bindings)?),
        Expr::Neg(inner) => neg(instantiate(inner, bindings)?),
        Expr::Shl(inner, count) => shl(instantiate(inner, bindings)?, *count),
        Expr::Bin(op, left, right) => bin(
            *op,
            instantiate(left, bindings)?,
            instantiate(right, bindings)?,
        ),
    })
}

/// Rewrites `expr` with the first matching rule, if any.
#[must_use]
pub fn apply_rules(rules: &[Rule], expr: &Expr) -> Option<(&'static str, Expr)> {
    for rule in rules {
        let mut bindings: Bindings = [None, None, None, None];
        if match_pattern(&rule.pattern, expr, &mut bindings)
            && let Some(result) = instantiate(&rule.replacement, &bindings)
        {
            return Some((rule.name, result));
        }
    }
    None
}

/// The shipped rule set. Every entry must verify before it can be applied.
#[must_use]
pub fn standard_rules() -> Vec<Rule> {
    let x = || var(0);
    let y = || var(1);
    let zero = || konst(0);
    let ones = || konst(u64::MAX);
    vec![
        Rule::new("xor-self", bin(Op::Xor, x(), x()), zero()),
        Rule::new("sub-self", bin(Op::Sub, x(), x()), zero()),
        Rule::new("add-zero", bin(Op::Add, x(), zero()), x()),
        Rule::new("sub-zero", bin(Op::Sub, x(), zero()), x()),
        Rule::new("or-zero", bin(Op::Or, x(), zero()), x()),
        Rule::new("xor-zero", bin(Op::Xor, x(), zero()), x()),
        Rule::new("and-ones", bin(Op::And, x(), ones()), x()),
        Rule::new("and-zero", bin(Op::And, x(), zero()), zero()),
        Rule::new("or-ones", bin(Op::Or, x(), ones()), ones()),
        Rule::new("and-self", bin(Op::And, x(), x()), x()),
        Rule::new("or-self", bin(Op::Or, x(), x()), x()),
        Rule::new("mul-one", bin(Op::Mul, x(), konst(1)), x()),
        Rule::new("mul-zero", bin(Op::Mul, x(), zero()), zero()),
        // Deeper identities: not reachable from a single instruction, but kept
        // so the prover is exercised on multi-operator shapes.
        Rule::new("xor-cancel", bin(Op::Xor, bin(Op::Xor, x(), y()), y()), x()),
        Rule::new("not-not", not(not(x())), x()),
        Rule::new("neg-neg", neg(neg(x())), x()),
        Rule::new("add-neg", bin(Op::Add, x(), neg(x())), zero()),
        Rule::new("shl1-add", shl(x(), 1), bin(Op::Add, x(), x())),
        Rule::new(
            "de-morgan",
            not(bin(Op::And, x(), y())),
            bin(Op::Or, not(x()), not(y())),
        ),
    ]
}
