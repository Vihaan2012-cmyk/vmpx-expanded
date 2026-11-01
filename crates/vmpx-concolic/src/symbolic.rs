//! Trace-guided symbolic walk: re-execute the recovered VM blocks along the block sequence the
//! emulator actually visited, recording each input-dependent branch's selector expression.
//! (The differential's `unroll.rs` forks over ALL successors; here the observed sequence picks
//! one, so cost is linear in the concrete path, not exponential in the number of branches.)

use vmpx_pe::Va;
use vmpx_vm::{
    BlockExit, CfgEdgeKind, Expr, ExprKind, SymbolicExecutionLimits, VmHandlerSemantic,
    execute_symbolic_linear_block_with_initial_state,
};
use vmpx_vmlift::{DecodedCfgInput, frame_initial_state};

/// One input-dependent branch on the walked path.
#[derive(Clone, Debug)]
pub struct Fork {
    pub block: Va,
    /// Symbolic VM branch target (the selector).
    pub selector: Expr,
    /// Recovered successors `(bytecode address, block key)`, sorted by address.
    pub succ: Vec<(u64, Va)>,
    pub chosen: usize,
    /// Position of `block` in the visit sequence.
    pub pos: usize,
}

#[derive(Debug, Default)]
pub struct GuidedPath {
    /// Block keys in execution order (constant-target hops included).
    pub blocks: Vec<Va>,
    pub forks: Vec<Fork>,
}

/// Recovered Branch/GuardedBranch successors of `key`: `(branch-target address, block key)`.
#[must_use]
pub fn successors(input: &DecodedCfgInput<'_>, key: Va) -> Vec<(u64, Va)> {
    let mut out: Vec<(u64, Va)> = Vec::new();
    for e in &input.cfg.edges {
        if e.from != key || !matches!(e.kind, CfgEdgeKind::Branch | CfgEdgeKind::GuardedBranch) {
            continue;
        }
        if let Some(a) = input.cfg.blocks.get(&e.to).and_then(|b| b.address)
            && !out.iter().any(|(_, k)| *k == e.to)
        {
            out.push((a.0, e.to));
        }
    }
    out.sort_by_key(|(v, _)| *v);
    out
}

/// Walk the CFG along `expected` (the observed block-visit sequence, entry first).
///
/// # Errors
/// Block budget, an unmodelled block, or an observed successor that the recovered CFG does not
/// contain (a static/dynamic divergence: precisely what the observed-branch bridge is for).
pub fn walk_expected(
    input: &DecodedCfgInput<'_>,
    expected: &[Va],
    max_blocks: usize,
) -> Result<GuidedPath, String> {
    let cfg = input.cfg;
    let (mut state, _) = frame_initial_state(input.frame, input.top_offset);
    let mut path = GuidedPath::default();
    if expected.first() != Some(&cfg.entry) {
        return Err(format!("observed run does not start at the CFG entry {} ({:?})", cfg.entry, expected.first()));
    }
    let mut pos = 0usize;
    loop {
        let key = expected[pos];
        if path.blocks.len() >= max_blocks {
            return Err(format!("block budget {max_blocks} exhausted"));
        }
        path.blocks.push(key);
        if pos + 1 >= expected.len() {
            return Ok(path);
        }
        let next = expected[pos + 1];
        let block = cfg.blocks.get(&key).ok_or_else(|| format!("block {key} not in CFG"))?;
        if block.exit != BlockExit::Branch {
            return Err(format!("block {key} is {:?}; observed run continues to {next}", block.exit));
        }
        let report = execute_symbolic_linear_block_with_initial_state(
            &block.instructions,
            Some(VmHandlerSemantic::UnconditionalBranch),
            SymbolicExecutionLimits::default(),
            &state,
            input.memory,
        )
        .map_err(|e| format!("block {key} symbolic exec: {e}"))?;
        if !report.native_calls.is_empty() || !report.fresh_allocations.is_empty() || !report.memory_havocs.is_empty() {
            return Err(format!("block {key} havocs memory / calls natively / allocates"));
        }
        let target = report
            .branch_target
            .as_ref()
            .ok_or_else(|| format!("block {key} branch has no symbolic target"))?
            .simplify();
        let succ = successors(input, key);
        state = report.continuation_state();
        let Some(chosen) = succ.iter().position(|(_, k)| *k == next) else {
            return Err(format!(
                "divergence: observed {key} -> {next}, recovered successors {:x?}",
                succ.iter().map(|(a, _)| a).collect::<Vec<_>>()
            ));
        };
        if !matches!(target.kind, ExprKind::Const { .. }) {
            path.forks.push(Fork { block: key, selector: target, succ, chosen, pos });
        }
        pos += 1;
    }
}
