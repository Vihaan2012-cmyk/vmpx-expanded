//! Native CFG recovery — the substrate the mutation passes operate on.

use crate::Problem;

// WAVE2: WP5 — decode a mutation-protected function to a native CFG (reuse
// vmpx-disasm decode/CFG). Intended signature once the CFG type exists:
//   pub fn recover_cfg(image: &ReadOnlyImage, entry: u64) -> Result<NativeCfg, CfgError>
// Stub keeps the module declared so WP5 fills only this file.
#[must_use]
pub fn recover_cfg(problem: &Problem) -> Option<()> {
    let _ = problem;
    None
}
