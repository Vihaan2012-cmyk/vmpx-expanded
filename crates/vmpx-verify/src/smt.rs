//! SMT / bitvector equivalence gate — promotion to the sound tier.

use crate::{Gate, Verdict};
use vmpx_vm::Expr;

/// Proves `original == candidate` over bitvectors.
pub struct SmtGate;

// WAVE2: WP3 — prove `original == candidate` over bitvectors. DECISION FOR THE
// WP3 AGENT: either add the `z3` crate (richest, heavy build dep) OR a
// pure-Rust bitvector decision procedure (no dep, bounded). Do NOT add z3 to
// Cargo.toml speculatively — pick one and wire it here. Fill `verify` to return
// Proven / Refuted; the stub returns Unknown so nothing is promoted to the
// sound tier until a real proof lands.
impl Gate for SmtGate {
    fn verify(&self, original: &Expr, cand: &Expr) -> Verdict {
        let _ = (original, cand);
        Verdict::Unknown
    }
}
