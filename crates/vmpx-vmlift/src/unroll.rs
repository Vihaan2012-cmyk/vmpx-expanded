//! Concrete-path unrolling of a recovered VM CFG.
//!
//! The tree lift refuses a loop it cannot summarize. When every loop branch of
//! the function is decided by values that do not depend on the function's inputs
//! (a counted loop), the path through the CFG is input-independent, so executing
//! it block by block from the entry frame yields the exact exit state as
//! expressions over the entry registers, for all inputs.
//!
//! A branch whose target does not fold to a constant forks over its recovered
//! successors (the CFG proved the target is one of them), recording the
//! `selector == successor` conditions of each path, exactly as the tree lift
//! does. Budgets bound the blocks executed and the paths produced; exceeding
//! either is an error, never a partial result.

use vmpx_vm::{
    BlockExit, CfgEdgeKind, Expr, ExprKind, NativeExitState, SymbolicExecutionLimits,
    SymbolicVmInitialState, VmHandlerSemantic, execute_symbolic_linear_block_with_initial_state,
};
use vmpx_pe::Va;

use crate::decoded_lift::{RecoveredExit, first_unliftable, recover_exit};
use crate::entry_frame::frame_initial_state;
use crate::{CollectedEffects, CollectedStore, DecodedCfgInput, ExitSemantics, PathCondition};

/// One input-independent path to a VM exit. Register, flag and store values may
/// still contain `memory_load_epoch_<n>` intrinsics, resolved by the consumer
/// against `store_epochs` (the epoch right after each store committed).
#[derive(Clone, Debug)]
pub struct UnrolledPath {
    /// Forks taken on this path (`selector == value`, or `!=` for a skipped arm).
    pub conditions: Vec<PathCondition>,
    pub semantics: ExitSemantics,
    pub store_epochs: Vec<u64>,
    pub rflags_is_entry_flags: bool,
    pub return_pop: u32,
    pub blocks_executed: usize,
    /// The path returns through staged stack (not a callee return); the exit
    /// registers are those of the continuation, and `semantics` is not comparable.
    pub region_exit: bool,
    /// Forks on this path whose selector is an environment probe (cpuid, `gs`
    /// loads, constant-address loads), i.e. protector gates, not user branches.
    pub env_forks: usize,
}

#[derive(Clone, Debug)]
pub struct UnrolledFunction {
    pub paths: Vec<UnrolledPath>,
    /// Branches that folded to a single recovered successor (counted loops).
    pub branches_decided: usize,
    pub forks: usize,
    /// Arms that re-entered an environment gate already on the path (a retry
    /// or spin on a failed probe); no exit state is reachable through them.
    pub spin_arms_pruned: usize,
    /// Byte offset of the entry stack pointer from `vsp_initial`: everything
    /// below it is the VM's own frame, dead once the function returns.
    pub entry_sp_offset: i64,
}

struct Walker<'a, 'b> {
    input: &'a DecodedCfgInput<'b>,
    max_blocks: usize,
    max_paths: usize,
    blocks: usize,
    decided: usize,
    forks: usize,
    pruned: usize,
    paths: Vec<UnrolledPath>,
}

#[derive(Clone)]
struct Walk {
    state: SymbolicVmInitialState,
    stores: Vec<(CollectedStore, u64)>,
    conditions: Vec<PathCondition>,
    executed: usize,
    env_trail: Vec<Va>,
}

/// Executes the CFG's input-independent paths.
///
/// # Errors
/// The CFG is incomplete, a block is unliftable or does something the unroll
/// does not model (native call, havoc, allocation, nested entry, region exit),
/// a branch target is not one of the recovered successors, or a budget is hit.
pub fn unroll_decoded_cfg(
    input: &DecodedCfgInput<'_>,
    max_blocks: usize,
    max_paths: usize,
) -> Result<UnrolledFunction, String> {
    if !input.cfg.is_complete() {
        return Err("the CFG is incomplete".to_owned());
    }
    let (state, _) = frame_initial_state(input.frame, input.top_offset);
    let mut walker = Walker {
        input,
        max_blocks,
        max_paths,
        blocks: 0,
        decided: 0,
        forks: 0,
        pruned: 0,
        paths: Vec::new(),
    };
    walker.walk(
        input.cfg.entry,
        Walk {
            state,
            stores: Vec::new(),
            conditions: Vec::new(),
            executed: 0,
            env_trail: Vec::new(),
        },
    )?;
    Ok(UnrolledFunction {
        paths: walker.paths,
        branches_decided: walker.decided,
        forks: walker.forks,
        spin_arms_pruned: walker.pruned,
        entry_sp_offset: i64::try_from(input.top_offset)
            .unwrap_or(i64::MAX)
            .saturating_add(i64::try_from(input.frame.len() * 8).unwrap_or(i64::MAX)),
    })
}

impl Walker<'_, '_> {
    fn successors(&self, key: Va) -> Vec<(u64, Va)> {
        let cfg = self.input.cfg;
        let mut out: Vec<(u64, Va)> = Vec::new();
        for edge in &cfg.edges {
            if edge.from != key
                || !matches!(edge.kind, CfgEdgeKind::Branch | CfgEdgeKind::GuardedBranch)
            {
                continue;
            }
            if let Some(address) = cfg.blocks.get(&edge.to).and_then(|b| b.address)
                && !out.iter().any(|(_, k)| *k == edge.to)
            {
                out.push((address.0, edge.to));
            }
        }
        out.sort_by_key(|(value, _)| *value);
        out
    }

    #[allow(clippy::too_many_lines)]
    fn walk(&mut self, mut key: Va, mut walk: Walk) -> Result<(), String> {
        let input = self.input;
        let cfg = input.cfg;
        loop {
            self.blocks += 1;
            walk.executed += 1;
            if std::env::var_os("VMPX_DIFF_TRACE").is_some() && self.blocks % 50 == 0 {
                eprintln!(
                    "[unroll] blocks={} paths={} forks={} decided={} at {key} depth={}",
                    self.blocks,
                    self.paths.len(),
                    self.forks,
                    self.decided,
                    walk.executed
                );
            }
            if self.blocks > self.max_blocks {
                return Err(format!("block budget {} exhausted", self.max_blocks));
            }
            let block = cfg
                .blocks
                .get(&key)
                .ok_or_else(|| format!("block {key} is not in the recovered CFG"))?;
            if let Some((index, semantic, reason)) = first_unliftable(&block.instructions) {
                return Err(format!("block {key} instruction {index} ({semantic:?}): {reason}"));
            }
            let boundary = match block.exit {
                BlockExit::VmExit => VmHandlerSemantic::VmExit,
                BlockExit::Branch => VmHandlerSemantic::UnconditionalBranch,
                BlockExit::NativeCall => return Err(format!("block {key} makes a native call")),
                BlockExit::Stalled => return Err(format!("block {key} did not reach a boundary")),
            };
            let report = execute_symbolic_linear_block_with_initial_state(
                &block.instructions,
                Some(boundary),
                SymbolicExecutionLimits::default(),
                &walk.state,
                input.memory,
            )
            .map_err(|e| format!("block {key} symbolic execution failed: {e}"))?;
            if !report.native_calls.is_empty()
                || !report.fresh_allocations.is_empty()
                || report.memory_tainted
                || !report.memory_havocs.is_empty()
            {
                return Err(format!("block {key} havocs memory, calls natively or allocates"));
            }
            for write in &report.external_writes {
                walk.stores.push((
                    CollectedStore {
                        address: write.address.clone(),
                        value: write.value.clone(),
                        width: write.value.width,
                    },
                    write.epoch,
                ));
            }
            match block.exit {
                BlockExit::VmExit => {
                    if cfg
                        .edges
                        .iter()
                        .any(|e| e.from == key && e.kind == CfgEdgeKind::NestedEntry)
                    {
                        return Err("exit re-enters a nested VM entry".to_owned());
                    }
                    let handler = block
                        .instructions
                        .last()
                        .map_or(block.start.handler, |last| last.state_after.handler);
                    let layout = block
                        .vm_exit
                        .as_ref()
                        .or_else(|| input.exits.get(&handler))
                        .ok_or_else(|| format!("no recovered VM-exit layout for handler {handler}"))?;
                    let recovered = recover_exit(&report, layout, input.frame.len(), input.top_offset)
                        .map_err(|e| e.to_string())?;
                    let (exit, region_exit) = match recovered {
                        RecoveredExit::Callee(exit) => (exit, false),
                        RecoveredExit::Region(region) => {
                            if std::env::var_os("VMPX_DIFF_TRACE").is_some() {
                                eprintln!(
                                    "[region exit] block {key} slot_offset={} staged={} target={:?}",
                                    region.slot_offset,
                                    region.staged.len(),
                                    region.staged.first().map(|e| e.simplify())
                                );
                            }
                            (
                            crate::decoded_lift::StubExit {
                                registers: region.registers,
                                rflags: region.rflags,
                                return_pop: 0,
                            },
                            true,
                            )
                        }
                    };
                    let rflags_is_entry_flags = matches!(
                        &exit.rflags.simplify().kind,
                        ExprKind::Register { name } if name == "native_rflags_initial"
                    );
                    let (collected, store_epochs): (Vec<_>, Vec<_>) =
                        walk.stores.into_iter().unzip();
                    if self.paths.len() >= self.max_paths {
                        return Err(format!("path budget {} exhausted", self.max_paths));
                    }
                    self.paths.push(UnrolledPath {
                        conditions: walk.conditions,
                        semantics: ExitSemantics {
                            exit: NativeExitState {
                                xmm: std::collections::BTreeMap::new(),
                                schema_version: 1,
                                registers: exit.registers,
                                rflags: exit.rflags,
                                return_address: Expr::constant(vmpx_vm::BitWidth::W64, 0),
                                frame_size: 0,
                            },
                            effects: CollectedEffects {
                                stores: collected,
                                calls: Vec::new(),
                                unsupported: Vec::new(),
                            },
                        },
                        store_epochs,
                        rflags_is_entry_flags,
                        return_pop: exit.return_pop,
                        blocks_executed: walk.executed,
                        region_exit,
                        env_forks: walk.env_trail.len(),
                    });
                    return Ok(());
                }
                BlockExit::Branch => {
                    let target = report
                        .branch_target
                        .as_ref()
                        .ok_or_else(|| format!("block {key} branch has no symbolic target"))?
                        .simplify();
                    let successors = self.successors(key);
                    walk.state = report.continuation_state();
                    if let ExprKind::Const { value } = target.kind {
                        let next = successors
                            .iter()
                            .find(|(address, _)| *address == value)
                            .map(|(_, k)| *k)
                            .ok_or_else(|| {
                                format!(
                                    "block {key} branch target {value:#x} is not a recovered successor"
                                )
                            })?;
                        self.decided += 1;
                        key = next;
                        continue;
                    }
                    if successors.len() < 2 {
                        return Err(format!(
                            "block {key} branch target depends on inputs but has {} recovered successor(s)",
                            successors.len()
                        ));
                    }
                    self.forks += 1;
                    if std::env::var_os("VMPX_DIFF_TRACE").is_some() && self.forks <= 12 {
                        eprintln!(
                            "[fork {}] block {key} depth {} succ={:?} deps={:?} intr={:?}",
                            self.forks,
                            walk.executed,
                            successors.iter().map(|(v, _)| format!("{v:#x}")).collect::<Vec<_>>(),
                            target.dependencies(),
                            target.intrinsic_dependencies()
                        );
                    }
                    let env = is_environment_probe(&target);
                    if env && walk.env_trail.contains(&key) {
                        self.pruned += 1;
                        return Ok(());
                    }
                    let mut prefix = walk;
                    if env {
                        prefix.env_trail.push(key);
                    }
                    let last = successors.len() - 1;
                    for (index, (value, successor)) in successors.iter().enumerate() {
                        let mut arm = prefix.clone();
                        if index != last {
                            arm.conditions.push(PathCondition {
                                selector: target.clone(),
                                value: *value,
                                equal: true,
                            });
                        }
                        self.walk(*successor, arm)?;
                        prefix.conditions.push(PathCondition {
                            selector: target.clone(),
                            value: *value,
                            equal: false,
                        });
                    }
                    return Ok(());
                }
                BlockExit::NativeCall | BlockExit::Stalled => unreachable!("rejected above"),
            }
        }
    }
}

/// A selector that reads only the environment: a cpuid/segment-base probe, or
/// loads and flags with no dependence on the function's registers. These are the
/// protector's own gates, not branches of the protected function.
fn is_environment_probe(selector: &Expr) -> bool {
    let intrinsics = selector.intrinsic_dependencies();
    let probes = intrinsics.iter().any(|n| {
        n.starts_with("x86_cpuid") || n.starts_with("x86_gs_load") || n.starts_with("x86_fs_load")
    });
    probes
        || (selector.dependencies().is_empty()
            && intrinsics
                .iter()
                .all(|n| n.starts_with("memory_load_epoch_") || n.starts_with("vm_flags_")))
}
