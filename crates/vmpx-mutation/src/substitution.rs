//! Instruction-substitution canonicalizer (pass side).

use crate::Problem;
use vmpx_vm::Expr;

// WAVE2: WP11 — peephole reversal of VMProtect's instruction-substitution
// patterns, canonicalizing a substituted expression back to its normal form.
// Intended signature:
//   pub fn canonicalize(expr: &Expr, ctx: &Ctx) -> Option<Expr>
// Pairs with the vmpx-solvers `substitution_solver` module (same WP). Stub
// reverses nothing.
#[must_use]
pub fn canonicalize(problem: &Problem) -> Option<Expr> {
    let _ = problem;
    None
}
