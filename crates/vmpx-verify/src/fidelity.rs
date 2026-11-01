//! Structural fidelity of a recovered function against the original.
//!
//! Descriptive and orthogonal to the behavioral verdict: a function can be
//! PROVEN and still be [`Fidelity::IdiomDivergent`]. Nothing here proves
//! anything and nothing here may change a verdict.
//!
//! "Structure" means the trees the gate compares, i.e. both sides after
//! `Expr::simplify`. Coarse fidelity is read off the per-cell proof method;
//! the refined class walks the two trees and names each divergence.

use std::collections::{BTreeMap, HashMap};

use vmpx_vm::fp::FpIntrinsic;
use vmpx_vm::{BinaryOp, Expr, ExprKind, ResizeOp, UnaryOp};

use crate::differential::pretty;
use crate::equiv::{CellVerdict, EquivVerdict, FunctionReport, Method, Obligation};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Fidelity {
    /// Trees identical (after the shared simplifier).
    Exact,
    /// Equal after commuting/reassociating/recognising an equivalent idiom.
    IdiomDivergent,
    /// Different structure; only a solver relates the two.
    EquivalentDifferent,
}

impl Fidelity {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Exact => "EXACT",
            Self::IdiomDivergent => "IDIOM-DIVERGENT",
            Self::EquivalentDifferent => "EQUIVALENT-DIFFERENT",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Divergence {
    Commutation,
    /// Shift pair vs `Rol`/`Ror` (or `Rol` vs `Ror`).
    RotateIdiom,
    /// `shl`/`mul`/add-self chains computing the same scale (address forms).
    LeaIdiom,
    Reassociation,
    /// Resize/extract/mask spelled differently.
    Width,
    /// `sub`/`neg`/`not` spelled through each other.
    NegationIdiom,
    /// Same boolean function spelled with different and/or/xor/not structure
    /// (De Morgan, nand-expanded or/xor): VMProtect only has nand.
    BooleanIdiom,
    /// No recognised idiom relates the two nodes.
    Structure,
}

impl Divergence {
    pub const ALL: [Self; 8] = [
        Self::Commutation,
        Self::RotateIdiom,
        Self::LeaIdiom,
        Self::Reassociation,
        Self::Width,
        Self::NegationIdiom,
        Self::BooleanIdiom,
        Self::Structure,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Commutation => "commutation",
            Self::RotateIdiom => "rotate-idiom",
            Self::LeaIdiom => "lea-idiom",
            Self::Reassociation => "reassociation",
            Self::Width => "width",
            Self::NegationIdiom => "negation-idiom",
            Self::BooleanIdiom => "boolean-idiom",
            Self::Structure => "different-structure",
        }
    }

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// Fidelity of one PROVEN function.
#[derive(Clone, Debug)]
pub struct FunctionFidelity {
    /// Worst cell by proof method alone (structural / canonical / solver).
    pub coarse: Fidelity,
    /// Worst cell after the tree diff reclassifies solver-proven cells.
    pub refined: Fidelity,
    /// Refined cell counts: exact, idiom-divergent, equivalent-different.
    pub cells: [usize; 3],
    /// Distinct divergent node pairs per type.
    pub divergences: BTreeMap<Divergence, usize>,
    /// One rendered `recovered vs original` pair per type.
    pub examples: BTreeMap<Divergence, String>,
    /// The diff budget ran out; counts are a lower bound.
    pub truncated: bool,
}

const NODE_BUDGET: usize = 300_000;

struct Differ {
    memo: HashMap<(Expr, Expr), u8>,
    budget: usize,
    counts: BTreeMap<Divergence, usize>,
    examples: BTreeMap<Divergence, String>,
    truncated: bool,
}

/// Fidelity of a function whose verdict is PROVEN; `None` otherwise or when
/// the report does not line up with the obligations.
#[must_use]
pub fn assess(obligations: &[Obligation], report: &FunctionReport) -> Option<FunctionFidelity> {
    if report.verdict != EquivVerdict::Proven || obligations.len() != report.paths.len() {
        return None;
    }
    let mut d = Differ {
        memo: HashMap::new(),
        budget: NODE_BUDGET,
        counts: BTreeMap::new(),
        examples: BTreeMap::new(),
        truncated: false,
    };
    let (mut coarse, mut refined) = (Fidelity::Exact, Fidelity::Exact);
    let mut cells = [0usize; 3];
    for (ob, rep) in obligations.iter().zip(&report.paths) {
        if ob.cells.len() != rep.cells.len() {
            return None;
        }
        for (cell, (_, verdict)) in ob.cells.iter().zip(&rep.cells) {
            let CellVerdict::Proven(method) = verdict else { return None };
            let by_method = match method {
                Method::Structural => Fidelity::Exact,
                Method::FpCanonical => Fidelity::IdiomDivergent,
                Method::Solver(_) | Method::FpSolver(_) => Fidelity::EquivalentDifferent,
            };
            coarse = coarse.max(by_method);
            let class = match by_method {
                Fidelity::Exact => Fidelity::Exact,
                _ => {
                    let mask = d.rec(&cell.vm.simplify(), &cell.recovered.simplify());
                    if by_method == Fidelity::IdiomDivergent || mask & Divergence::Structure.bit() == 0 {
                        Fidelity::IdiomDivergent
                    } else {
                        Fidelity::EquivalentDifferent
                    }
                }
            };
            cells[class as usize] += 1;
            refined = refined.max(class);
        }
    }
    Some(FunctionFidelity {
        coarse,
        refined,
        cells,
        divergences: d.counts,
        examples: d.examples,
        truncated: d.truncated,
    })
}

const FLATTEN_LIMIT: usize = 4096;

fn flatten<'a>(op: BinaryOp, e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match &e.kind {
        ExprKind::Binary { op: o, left, right } if *o == op && out.len() < FLATTEN_LIMIT => {
            flatten(op, left, out);
            flatten(op, right, out);
        }
        _ => out.push(e),
    }
}

const fn is_ac(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add | BinaryOp::Mul | BinaryOp::Xor | BinaryOp::And | BinaryOp::Or
    )
}

fn constant(e: &Expr) -> Option<u64> {
    match e.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

/// `Truncate_w(Extend(v))` with `v` already `w` wide is just `v`.
fn peel(e: &Expr) -> &Expr {
    if let ExprKind::Resize { op: ResizeOp::Truncate, value } = &e.kind
        && let ExprKind::Resize { op: ResizeOp::ZeroExtend | ResizeOp::SignExtend, value: inner } = &value.kind
        && inner.width == e.width
    {
        return peel(inner);
    }
    e
}

fn is_bitwise(e: &Expr) -> bool {
    let e = peel(e);
    matches!(
        e.kind,
        ExprKind::Binary { op: BinaryOp::And | BinaryOp::Or | BinaryOp::Xor, .. }
            | ExprKind::Unary { op: UnaryOp::Not, .. }
    )
}

/// Shape hash of the and/or/xor/not skeleton (leaves anonymous); memoised
/// because shared subterms make the tree walk exponential.
fn skeleton(e: &Expr, memo: &mut HashMap<Expr, u64>) -> u64 {
    use std::hash::{Hash, Hasher};
    let e = peel(e);
    if let Some(h) = memo.get(e) {
        return *h;
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match &e.kind {
        ExprKind::Binary { op: op @ (BinaryOp::And | BinaryOp::Or | BinaryOp::Xor), left, right } => {
            (1u8, *op as u8).hash(&mut h);
            skeleton(left, memo).hash(&mut h);
            skeleton(right, memo).hash(&mut h);
        }
        ExprKind::Unary { op: UnaryOp::Not, value } => {
            2u8.hash(&mut h);
            skeleton(value, memo).hash(&mut h);
        }
        ExprKind::Const { value } => (3u8, *value).hash(&mut h),
        _ => 4u8.hash(&mut h),
    }
    let v = h.finish();
    memo.insert(e.clone(), v);
    v
}

fn leaves(e: &Expr, out: &mut Vec<Expr>, seen: &mut std::collections::HashSet<Expr>) {
    let e = peel(e);
    if out.len() > MAX_LEAVES || !seen.insert(e.clone()) {
        return;
    }
    match &e.kind {
        ExprKind::Binary { op: BinaryOp::And | BinaryOp::Or | BinaryOp::Xor, left, right } => {
            leaves(left, out, seen);
            leaves(right, out, seen);
        }
        ExprKind::Unary { op: UnaryOp::Not, value } => leaves(value, out, seen),
        ExprKind::Const { .. } => {}
        _ => {
            if !out.contains(e) {
                out.push(e.clone());
            }
        }
    }
}

/// Bit-parallel evaluation; constants other than 0 / all-ones are not
/// position-independent, so they abort the comparison.
fn eval_bitwise(e: &Expr, env: &HashMap<Expr, u64>, memo: &mut HashMap<Expr, u64>) -> Option<u64> {
    let e = peel(e);
    if let Some(v) = memo.get(e) {
        return Some(*v);
    }
    let mask = e.width.mask();
    let v = match &e.kind {
        ExprKind::Binary { op, left, right } if matches!(op, BinaryOp::And | BinaryOp::Or | BinaryOp::Xor) => {
            let (l, r) = (eval_bitwise(left, env, memo)?, eval_bitwise(right, env, memo)?);
            match op {
                BinaryOp::And => l & r,
                BinaryOp::Or => l | r,
                _ => l ^ r,
            }
        }
        ExprKind::Unary { op: UnaryOp::Not, value } => !eval_bitwise(value, env, memo)?,
        ExprKind::Const { value } => match *value & mask {
            0 => 0,
            m if m == mask => u64::MAX,
            _ => return None,
        },
        _ => *env.get(e)?,
    } & mask;
    memo.insert(e.clone(), v);
    Some(v)
}

const MAX_LEAVES: usize = 12;

/// Variable `i`'s bit-parallel input: the exhaustive truth-table word for up
/// to 6 variables, otherwise a fixed pseudo-random word per round.
fn pattern(vars: usize, round: u64, i: usize) -> u64 {
    if vars <= 6 {
        return (0..64u32).fold(0u64, |m, p| m | (u64::from((p >> i) & 1) << p));
    }
    let mut z = (round * 64 + i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn go(cur: &mut Vec<usize>, used: &mut Vec<bool>, n: usize, out: &mut Vec<Vec<usize>>) {
        if cur.len() == n {
            out.push(cur.clone());
            return;
        }
        for k in 0..n {
            if !used[k] {
                used[k] = true;
                cur.push(k);
                go(cur, used, n, out);
                cur.pop();
                used[k] = false;
            }
        }
    }
    let mut out = Vec::new();
    go(&mut Vec::new(), &mut vec![false; n], n, &mut out);
    out
}

fn head_key(e: &Expr) -> (u8, u8) {
    match &e.kind {
        ExprKind::Const { .. } => (0, 0),
        ExprKind::Register { .. } => (1, 0),
        ExprKind::Unary { op, .. } => (2, *op as u8),
        ExprKind::Binary { op, .. } => (3, *op as u8),
        ExprKind::Load { .. } => (4, 0),
        ExprKind::Intrinsic { .. } => (5, 0),
        ExprKind::Extract { .. } => (6, 0),
        ExprKind::Concat { .. } => (7, 0),
        ExprKind::Resize { .. } => (8, 0),
    }
}

/// `x * c` shaped term as (x, c): `shl x, k`, `mul x, c`, `mul c, x`.
fn scaled_term(t: &Expr) -> Option<(&Expr, u64)> {
    match &t.kind {
        ExprKind::Binary { op: BinaryOp::Shl, left, right } => {
            let k = constant(right)?;
            (k < 64).then(|| (&**left, 1u64 << k))
        }
        ExprKind::Binary { op: BinaryOp::Mul, left, right } => {
            if let Some(c) = constant(right) {
                Some((&**left, c))
            } else {
                constant(left).map(|c| (&**right, c))
            }
        }
        _ => None,
    }
}

/// `base*coeff + konst` view of an add chain whose non-constant terms are all
/// scalings of one base. Returns (base, coeff, konst, any scaling op used).
fn linear(e: &Expr) -> Option<(&Expr, u64, u64, bool)> {
    let mut terms = Vec::new();
    flatten(BinaryOp::Add, e, &mut terms);
    let (mut base, mut coeff, mut konst, mut scaled) = (None::<&Expr>, 0u64, 0u64, false);
    let mut plain_terms = 0usize;
    for t in terms {
        if let Some(c) = constant(t) {
            konst = konst.wrapping_add(c);
            continue;
        }
        let (b, c, s) = match scaled_term(t) {
            Some((b, c)) => (b, c, true),
            None => {
                plain_terms += 1;
                (t, 1, false)
            }
        };
        match base {
            None => base = Some(b),
            Some(prev) if prev == b => {}
            Some(_) => return None,
        }
        coeff = coeff.wrapping_add(c);
        scaled |= s;
    }
    if plain_terms > 1 {
        scaled = true;
    }
    let width_mask = e.width.mask();
    Some((base?, coeff & width_mask, konst & width_mask, scaled))
}

impl Differ {
    fn rec(&mut self, a: &Expr, b: &Expr) -> u8 {
        if a == b {
            return 0;
        }
        let key = (a.clone(), b.clone());
        if let Some(mask) = self.memo.get(&key) {
            return *mask;
        }
        if self.budget == 0 {
            self.truncated = true;
            return Divergence::Structure.bit();
        }
        self.budget -= 1;
        let mut local = 0u8;
        let mask = self.node(a, b, &mut local) | local;
        for d in Divergence::ALL {
            if local & d.bit() != 0 {
                *self.counts.entry(d).or_default() += 1;
                self.examples.entry(d).or_insert_with(|| {
                    let mut s = format!("{} vs {}", pretty(a, 4), pretty(b, 4));
                    s.truncate(220);
                    s
                });
            }
        }
        self.memo.insert(key, mask);
        mask
    }

    fn node(&mut self, a: &Expr, b: &Expr, local: &mut u8) -> u8 {
        if let Some(m) = self.negation(a, b, local).or_else(|| self.negation(b, a, local)) {
            return m;
        }
        if let Some(m) = self.rotate(a, b, local).or_else(|| self.rotate(b, a, local)) {
            return m;
        }
        if let Some(m) = self.lea(a, b, local) {
            return m;
        }
        if let Some(m) = self.boolean(a, b, local) {
            return m;
        }
        if let Some(m) = self.width(a, b, local).or_else(|| self.width(b, a, local)) {
            return m;
        }
        match (&a.kind, &b.kind) {
            (
                ExprKind::Binary { op: o1, left: l1, right: r1 },
                ExprKind::Binary { op: o2, left: l2, right: r2 },
            ) if o1 == o2 => {
                if is_ac(*o1) {
                    self.ac(a, b, *o1, local)
                } else {
                    self.rec(l1, l2) | self.rec(r1, r2)
                }
            }
            (ExprKind::Unary { op: o1, value: v1 }, ExprKind::Unary { op: o2, value: v2 })
                if o1 == o2 =>
            {
                self.rec(v1, v2)
            }
            (ExprKind::Load { address: x }, ExprKind::Load { address: y }) => self.rec(x, y),
            (
                ExprKind::Intrinsic { name: n1, arguments: a1 },
                ExprKind::Intrinsic { name: n2, arguments: a2 },
            ) if n1 == n2 && a1.len() == a2.len() => {
                if let [p, q] = a1.as_slice()
                    && let [r, s] = a2.as_slice()
                    && p == s
                    && q == r
                    && FpIntrinsic::parse(n1).is_some_and(|f| f.is_commutative())
                {
                    *local |= Divergence::Commutation.bit();
                    return 0;
                }
                a1.iter().zip(a2).fold(0, |m, (x, y)| m | self.rec(x, y))
            }
            (
                ExprKind::Extract { value: v1, lsb: s1 },
                ExprKind::Extract { value: v2, lsb: s2 },
            ) if s1 == s2 && a.width == b.width => self.rec(v1, v2),
            (ExprKind::Concat { high: h1, low: l1 }, ExprKind::Concat { high: h2, low: l2 }) => {
                self.rec(h1, h2) | self.rec(l1, l2)
            }
            (ExprKind::Resize { op: o1, value: v1 }, ExprKind::Resize { op: o2, value: v2 })
                if o1 == o2 && a.width == b.width =>
            {
                self.rec(v1, v2)
            }
            _ => {
                *local |= Divergence::Structure.bit();
                0
            }
        }
    }

    fn ac(&mut self, a: &Expr, b: &Expr, op: BinaryOp, local: &mut u8) -> u8 {
        let (mut ta, mut tb) = (Vec::new(), Vec::new());
        flatten(op, a, &mut ta);
        flatten(op, b, &mut tb);
        if ta == tb {
            *local |= Divergence::Reassociation.bit();
            return 0;
        }
        let mut free: Vec<Option<&Expr>> = tb.iter().copied().map(Some).collect();
        let mut rest_a = Vec::new();
        for x in ta {
            match free.iter().position(|y| y.is_some_and(|y| y == x)) {
                Some(i) => free[i] = None,
                None => rest_a.push(x),
            }
        }
        if rest_a.is_empty() && free.iter().all(Option::is_none) {
            *local |= Divergence::Commutation.bit();
            return 0;
        }
        if rest_a.len() != free.iter().flatten().count() {
            *local |= Divergence::Structure.bit();
            return 0;
        }
        let mut mask = 0;
        for x in rest_a {
            let pick = free
                .iter()
                .position(|y| y.is_some_and(|y| head_key(y) == head_key(x)))
                .or_else(|| free.iter().position(Option::is_some));
            if let Some(i) = pick
                && let Some(y) = free[i].take()
            {
                mask |= self.rec(x, y);
            }
        }
        mask
    }

    /// `r` is `Rol`/`Ror`; `s` is the other spelling.
    fn rotate(&mut self, r: &Expr, s: &Expr, local: &mut u8) -> Option<u8> {
        let ExprKind::Binary { op: ro @ (BinaryOp::Rol | BinaryOp::Ror), left: v, right: c } = &r.kind
        else {
            return None;
        };
        let bits = u64::from(r.width.bits());
        let rot = Divergence::RotateIdiom.bit();
        match &s.kind {
            ExprKind::Binary { op: so @ (BinaryOp::Rol | BinaryOp::Ror), left: v2, right: c2 }
                if so != ro =>
            {
                if (constant(c)? + constant(c2)?) % bits != 0 {
                    return None;
                }
                *local |= rot;
                Some(self.rec(v, v2))
            }
            ExprKind::Binary { op: so @ (BinaryOp::Or | BinaryOp::Xor | BinaryOp::Add), .. } => {
                let mut terms = Vec::new();
                flatten(*so, s, &mut terms);
                let [t1, t2] = terms.as_slice() else { return None };
                let shift = |t: &Expr, want: BinaryOp| match &t.kind {
                    ExprKind::Binary { op, left, right } if *op == want => {
                        Some((left.clone(), constant(right)?))
                    }
                    _ => None,
                };
                let ((p, k1), (q, k2)) = shift(t1, BinaryOp::Shl)
                    .zip(shift(t2, BinaryOp::Shr))
                    .or_else(|| shift(t2, BinaryOp::Shl).zip(shift(t1, BinaryOp::Shr)))?;
                let k = constant(c)? % bits;
                let (shl_by, shr_by) = (k1, k2);
                let expect = if *ro == BinaryOp::Rol { (k, bits - k) } else { (bits - k, k) };
                if (shl_by, shr_by) != expect {
                    return None;
                }
                *local |= rot;
                Some(self.rec(v, &p) | self.rec(v, &q))
            }
            _ => None,
        }
    }

    /// Two and/or/xor/not trees with different skeletons that compute the same
    /// boolean function of their (paired) leaves.
    fn boolean(&mut self, a: &Expr, b: &Expr, local: &mut u8) -> Option<u8> {
        if !(is_bitwise(a) && is_bitwise(b)) || a.width != b.width {
            return None;
        }
        if let (ExprKind::Binary { op: o1, .. }, ExprKind::Binary { op: o2, .. }) = (&a.kind, &b.kind)
            && o1 == o2
        {
            let (mut ta, mut tb) = (Vec::new(), Vec::new());
            flatten(*o1, a, &mut ta);
            flatten(*o1, b, &mut tb);
            if ta.len() == tb.len() && ta.iter().all(|x| tb.contains(x)) {
                return None;
            }
        }
        if skeleton(a, &mut HashMap::new()) == skeleton(b, &mut HashMap::new()) {
            return None;
        }
        let (mut la, mut lb) = (Vec::new(), Vec::new());
        leaves(a, &mut la, &mut std::collections::HashSet::new());
        leaves(b, &mut lb, &mut std::collections::HashSet::new());
        if la.len() > MAX_LEAVES || lb.len() > MAX_LEAVES {
            return None;
        }
        let mut free: Vec<Option<Expr>> = lb.iter().cloned().map(Some).collect();
        let mut pairs: Vec<(Expr, Expr)> = Vec::new();
        let mut rest = Vec::new();
        for x in la {
            match free.iter().position(|y| y.as_ref() == Some(&x)) {
                Some(i) => {
                    free[i] = None;
                    pairs.push((x.clone(), x));
                }
                None => rest.push(x),
            }
        }
        if rest.len() != free.iter().flatten().count() {
            return None;
        }
        if rest.len() > 6 {
            return None;
        }
        let ys: Vec<Expr> = free.into_iter().flatten().collect();
        let mask = a.width.mask();
        let mut found = None;
        for perm in permutations(rest.len()) {
            let mut candidate = pairs.clone();
            candidate.extend(rest.iter().cloned().zip(perm.iter().map(|&k| ys[k].clone())));
            let mut same = true;
            for round in 0..if candidate.len() <= 6 { 1 } else { 4 } {
                let (mut env_a, mut env_b) = (HashMap::new(), HashMap::new());
                for (i, (x, y)) in candidate.iter().enumerate() {
                    let v = pattern(candidate.len(), round, i);
                    env_a.insert(x.clone(), v);
                    env_b.insert(y.clone(), v);
                }
                let va = eval_bitwise(a, &env_a, &mut HashMap::new())?;
                let vb = eval_bitwise(b, &env_b, &mut HashMap::new())?;
                same &= va & mask == vb & mask;
            }
            if same {
                found = Some(candidate);
                break;
            }
        }
        let pairs = found?;
        *local |= Divergence::BooleanIdiom.bit();
        let mut m = 0;
        for (x, y) in &pairs {
            m |= self.rec(x, y);
        }
        Some(m)
    }

    fn lea(&mut self, a: &Expr, b: &Expr, local: &mut u8) -> Option<u8> {
        let (mut ta, mut tb) = (Vec::new(), Vec::new());
        flatten(BinaryOp::Add, a, &mut ta);
        flatten(BinaryOp::Add, b, &mut tb);
        let permuted = ta.len() == tb.len() && ta.iter().all(|x| tb.contains(x));
        if permuted || (ta.len() == 1 && tb.len() == 1 && head_key(a) == head_key(b)) {
            return None;
        }
        let (ba, ca, ka, sa) = linear(a)?;
        let (bb, cb, kb, sb) = linear(b)?;
        if !(sa || sb) || (ca, ka) != (cb, kb) {
            return None;
        }
        *local |= Divergence::LeaIdiom.bit();
        Some(self.rec(ba, bb))
    }

    /// `x` is a resize/extract/low-mask wrapper the other side spells differently.
    fn width(&mut self, a: &Expr, b: &Expr, local: &mut u8) -> Option<u8> {
        let strip = |e: &Expr| -> Option<Expr> {
            match &e.kind {
                ExprKind::Resize { value, .. } | ExprKind::Extract { value, .. } => {
                    Some((**value).clone())
                }
                ExprKind::Concat { high, low } if constant(high) == Some(0) => {
                    Some((**low).clone())
                }
                _ => None,
            }
        };
        let w = Divergence::Width.bit();
        if let Some(inner) = strip(a) {
            let same_wrapper = match (&a.kind, &b.kind) {
                (ExprKind::Resize { op: o1, .. }, ExprKind::Resize { op: o2, .. }) => {
                    o1 == o2 && a.width == b.width
                }
                (ExprKind::Extract { lsb: s1, .. }, ExprKind::Extract { lsb: s2, .. }) => {
                    s1 == s2 && a.width == b.width
                }
                (ExprKind::Concat { .. }, ExprKind::Concat { .. }) => true,
                _ => false,
            };
            if !same_wrapper {
                *local |= w;
                return Some(match strip(b) {
                    Some(other) => self.rec(&inner, &other),
                    None => self.rec(&inner, b),
                });
            }
            return None;
        }
        if let ExprKind::Binary { op: BinaryOp::And, left, right } = &a.kind
            && let Some(m) = constant(right)
            && m != 0
            && m & m.wrapping_add(1) == 0
            && !matches!(b.kind, ExprKind::Binary { op: BinaryOp::And, .. })
            && strip(b).is_none()
        {
            *local |= w;
            return Some(self.rec(left, b));
        }
        None
    }

    fn negation(&mut self, a: &Expr, b: &Expr, local: &mut u8) -> Option<u8> {
        let neg = Divergence::NegationIdiom.bit();
        let ones = a.width.mask();
        match (&a.kind, &b.kind) {
            (
                ExprKind::Binary { op: BinaryOp::Sub, left: x, right: y },
                ExprKind::Binary { op: BinaryOp::Add, left: l, right: r },
            ) => {
                if let Some(c) = constant(y)
                    && constant(r) == Some(c.wrapping_neg() & ones)
                {
                    *local |= neg;
                    return Some(self.rec(x, l));
                }
                for (p, q) in [(l, r), (r, l)] {
                    if let ExprKind::Unary { op: UnaryOp::Neg, value } = &q.kind {
                        *local |= neg;
                        return Some(self.rec(x, p) | self.rec(y, value));
                    }
                }
                None
            }
            (
                ExprKind::Unary { op: UnaryOp::Not, value: x },
                ExprKind::Binary { op: BinaryOp::Xor, left: l, right: r },
            ) if constant(r) == Some(ones) => {
                *local |= neg;
                Some(self.rec(x, l))
            }
            (
                ExprKind::Unary { op: UnaryOp::Neg, value: x },
                ExprKind::Binary { op: BinaryOp::Sub, left: z, right: y },
            ) if constant(z) == Some(0) => {
                *local |= neg;
                Some(self.rec(x, y))
            }
            (
                ExprKind::Unary { op: UnaryOp::Neg, value: x },
                ExprKind::Binary { op: BinaryOp::Add, left: l, right: r },
            ) if constant(r) == Some(1) => {
                if let ExprKind::Unary { op: UnaryOp::Not, value: y } = &l.kind {
                    *local |= neg;
                    return Some(self.rec(x, y));
                }
                None
            }
            _ => None,
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
    fn k(v: u64) -> Expr {
        Expr::constant(W, v)
    }
    fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }
    fn run(a: &Expr, b: &Expr) -> BTreeMap<Divergence, usize> {
        let mut d = Differ {
            memo: HashMap::new(),
            budget: 10_000,
            counts: BTreeMap::new(),
            examples: BTreeMap::new(),
            truncated: false,
        };
        d.rec(a, b);
        d.counts
    }
    fn only(m: &BTreeMap<Divergence, usize>) -> Vec<Divergence> {
        m.keys().copied().collect()
    }

    #[test]
    fn commutation_and_reassociation() {
        let (x, y, z) = (reg("x"), reg("y"), reg("z"));
        let a = bin(BinaryOp::Xor, x.clone(), y.clone());
        let b = bin(BinaryOp::Xor, y.clone(), x.clone());
        assert_eq!(only(&run(&a, &b)), [Divergence::Commutation]);
        let l = bin(BinaryOp::Xor, bin(BinaryOp::Xor, x.clone(), y.clone()), z.clone());
        let r = bin(BinaryOp::Xor, x, bin(BinaryOp::Xor, y, z));
        assert_eq!(only(&run(&l, &r)), [Divergence::Reassociation]);
    }

    #[test]
    fn rotate_idiom_both_directions() {
        let x = reg("x");
        let ror = bin(BinaryOp::Ror, x.clone(), k(8));
        let pair = bin(
            BinaryOp::Or,
            bin(BinaryOp::Shr, x.clone(), k(8)),
            bin(BinaryOp::Shl, x.clone(), k(56)),
        );
        assert_eq!(only(&run(&ror, &pair)), [Divergence::RotateIdiom]);
        assert_eq!(only(&run(&pair, &ror)), [Divergence::RotateIdiom]);
        let rol = bin(BinaryOp::Rol, x, k(56));
        assert_eq!(only(&run(&ror, &rol)), [Divergence::RotateIdiom]);
    }

    #[test]
    fn lea_idiom() {
        let x = reg("x");
        let a = bin(BinaryOp::Add, x.clone(), bin(BinaryOp::Shl, x.clone(), k(2)));
        let b = bin(BinaryOp::Mul, x, k(5));
        assert_eq!(only(&run(&a, &b)), [Divergence::LeaIdiom]);
    }

    #[test]
    fn width_idiom() {
        let x = reg("x");
        let masked = bin(BinaryOp::And, x.clone(), k(0xFFFF_FFFF));
        let t = Expr::resize(ResizeOp::Truncate, BitWidth::W32, x).unwrap();
        let z = Expr::resize(ResizeOp::ZeroExtend, W, t).unwrap();
        assert_eq!(only(&run(&masked, &z)), [Divergence::Width]);
    }

    #[test]
    fn genuinely_different_is_structure() {
        let (x, y) = (reg("x"), reg("y"));
        let a = bin(BinaryOp::Xor, x.clone(), y.clone());
        let b = bin(BinaryOp::And, x, y);
        assert_eq!(only(&run(&a, &b)), [Divergence::Structure]);
    }

    #[test]
    fn boolean_idiom_de_morgan_and_xor_expansion() {
        let (x, y) = (reg("x"), reg("y"));
        let not = |e: Expr| Expr::unary(UnaryOp::Not, e);
        let or = bin(BinaryOp::Or, x.clone(), y.clone());
        let nand_or = not(bin(BinaryOp::And, not(x.clone()), not(y.clone())));
        assert_eq!(only(&run(&or, &nand_or)), [Divergence::BooleanIdiom]);
        let xor = bin(BinaryOp::Xor, x.clone(), y.clone());
        let expanded = bin(
            BinaryOp::And,
            bin(BinaryOp::Or, x.clone(), y.clone()),
            not(bin(BinaryOp::And, x.clone(), y.clone())),
        );
        assert_eq!(only(&run(&xor, &expanded)), [Divergence::BooleanIdiom]);
        let different = bin(BinaryOp::Or, x.clone(), reg("z"));
        assert!(run(&different, &nand_or).contains_key(&Divergence::Structure));
    }

    #[test]
    fn negation_idiom() {
        let (x, y) = (reg("x"), reg("y"));
        let a = bin(BinaryOp::Sub, x.clone(), k(8));
        let b = bin(BinaryOp::Add, x.clone(), k(8u64.wrapping_neg()));
        assert_eq!(only(&run(&a, &b)), [Divergence::NegationIdiom]);
        let c = bin(BinaryOp::Sub, x.clone(), y.clone());
        let d = bin(BinaryOp::Add, x, Expr::unary(UnaryOp::Neg, y));
        assert_eq!(only(&run(&c, &d)), [Divergence::NegationIdiom]);
    }
}
