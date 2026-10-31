//! Input sampling shared by the synthesizer and the promotion gate: random
//! values plus the fixed set {0, 1, -1, signed min, signed max}.

use std::collections::BTreeMap;

use vmpx_vm::{BitWidth, Expr};

use crate::bvutil::{mask, registers};
use crate::gate::{Counterexample, EquivalenceBackend, Judgement, PromotionGate};

const CROSS_CAP: usize = 256;

/// splitmix64 step.
pub fn next_rand(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `{0, 1, -1, min, max}` at `width` (signed min/max).
#[must_use]
pub fn fixed_values(width: BitWidth) -> [u64; 5] {
    let m = mask(width);
    [0, 1, m, 1 << (width.bits() - 1), m >> 1]
}

/// Fixed-value cross product (capped), each variable swept over the fixed set
/// with the rest random, then `n_random` fully random assignments.
#[must_use]
pub fn sample_assignments(
    vars: &BTreeMap<String, BitWidth>,
    n_random: usize,
    seed: u64,
) -> Vec<Counterexample> {
    let mut rng = seed;
    let mut out: Vec<Counterexample> = Vec::new();
    let names: Vec<(&String, BitWidth)> = vars.iter().map(|(n, w)| (n, *w)).collect();

    let total = names
        .iter()
        .try_fold(1_usize, |a, _| a.checked_mul(5))
        .filter(|t| *t <= CROSS_CAP);
    if let Some(total) = total {
        for mut k in 0..total {
            out.push(
                names
                    .iter()
                    .map(|(n, w)| {
                        let v = fixed_values(*w)[k % 5];
                        k /= 5;
                        ((*n).clone(), v)
                    })
                    .collect(),
            );
        }
    } else {
        for _ in 0..CROSS_CAP {
            let mut a = Counterexample::new();
            for (n, w) in &names {
                let i = next_rand(&mut rng) as usize % 5;
                a.insert((*n).clone(), fixed_values(*w)[i]);
            }
            out.push(a);
        }
    }
    for (n, w) in &names {
        for v in fixed_values(*w) {
            let mut a = Counterexample::new();
            for (m, mw) in &names {
                a.insert((*m).clone(), next_rand(&mut rng) & mask(*mw));
            }
            a.insert((*n).clone(), v);
            out.push(a);
        }
    }
    for _ in 0..n_random {
        let mut a = Counterexample::new();
        for (m, mw) in &names {
            a.insert((*m).clone(), next_rand(&mut rng) & mask(*mw));
        }
        out.push(a);
    }
    out
}

/// Adds sampled inputs over the free registers of `exprs` to `gate`. Returns
/// `false` (adding nothing) if a register appears at two widths.
pub fn seed_gate<B: EquivalenceBackend>(
    gate: &mut PromotionGate<B>,
    exprs: &[&Expr],
    n_random: usize,
) -> bool {
    let mut vars = BTreeMap::new();
    for e in exprs {
        let Some(r) = registers(e) else { return false };
        for (n, w) in r {
            if *vars.entry(n).or_insert(w) != w {
                return false;
            }
        }
    }
    for t in sample_assignments(&vars, n_random, 0xA076_1D64_78BD_642F) {
        gate.add_test(t);
    }
    true
}

/// Seeds the gate with samples, then judges `candidate` against `original`.
pub fn judge_sampled<B: EquivalenceBackend>(
    gate: &mut PromotionGate<B>,
    original: &Expr,
    candidate: &Expr,
) -> Judgement {
    seed_gate(gate, &[original, candidate], 64);
    gate.judge(original, candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_cover_fixed_values_and_are_deterministic() {
        let vars: BTreeMap<String, BitWidth> = [("x".to_owned(), BitWidth::W64)].into();
        let s = sample_assignments(&vars, 8, 1);
        assert_eq!(s, sample_assignments(&vars, 8, 1));
        for v in fixed_values(BitWidth::W64) {
            assert!(s.iter().any(|a| a["x"] == v));
        }
        assert_eq!(fixed_values(BitWidth::W8), [0, 1, 0xFF, 0x80, 0x7F]);
    }
}
