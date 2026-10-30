//! Dead-code / junk elimination pass.

use crate::Problem;
use vmpx_vm::Expr;

// WAVE2: WP7 — liveness + sound removal of effect-free instructions over the
// recovered CFG. Intended signature once the CFG type exists:
//   pub fn eliminate_dead_code(cfg: &mut NativeCfg) -> DceReport
// Stub operates on the simplified Expr view and removes nothing.
#[must_use]
pub fn eliminate(problem: &Problem) -> Option<Expr> {
    let _ = problem;
    None
}
