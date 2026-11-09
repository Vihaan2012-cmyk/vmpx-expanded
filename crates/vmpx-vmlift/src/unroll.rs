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
    BlockExit, CfgEdgeKind, Expr, ExprKind, NativeExitState, NativeFrameSource, SymbolicExecutionLimits,
    SymbolicVmInitialState, VmHandlerSemantic, execute_symbolic_linear_block_with_initial_state,
};
use vmpx_pe::Va;

use crate::cfg_lift::compose_nested_state;
use crate::decoded_lift::{RecoveredExit, first_unliftable, recover_exit};
use crate::nested_entry::NestedEntryResolver;
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
    /// For a region exit: where it transfers to and how many stack words it stages.
    pub region: Option<RegionTarget>,
    /// Forks on this path whose selector is an environment probe (cpuid, `gs`
    /// loads, constant-address loads), i.e. protector gates, not user branches.
    pub env_forks: usize,
}

/// The computed continuation a region exit `ret`s to.
#[derive(Clone, Debug)]
pub struct RegionTarget {
    /// The address the exit transfers to (word zero of the staged continuation).
    pub target: Expr,
    /// Words staged from the `ret` slot up to the entry stack pointer; 1 means
    /// the `ret` leaves the stack exactly at the entry stack pointer.
    pub staged_words: usize,
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
    /// Region exits a consumer tried to follow and could not (why), reported next
    /// to the verdict; empty from [`unroll_decoded_cfg`] itself.
    pub notes: Vec<String>,
}

struct Walker<'a, 'b> {
    input: &'a DecodedCfgInput<'b>,
    nested: Option<&'a dyn NestedEntryResolver>,
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
    unroll_decoded_cfg_with_nested(input, None, max_blocks, max_paths)
}

/// [`unroll_decoded_cfg`] that also follows `NestedEntry` edges the resolver
/// proves: the exit's native state is composed into the nested entry's frame
/// (as the CFG lift does) and the walk continues inline. An edge the resolver
/// cannot prove keeps the old refusal.
///
/// # Errors
/// As [`unroll_decoded_cfg`]; additionally a nested entry that does not match the
/// CFG's nested block or whose state cannot be composed.
pub fn unroll_decoded_cfg_with_nested(
    input: &DecodedCfgInput<'_>,
    nested: Option<&dyn NestedEntryResolver>,
    max_blocks: usize,
    max_paths: usize,
) -> Result<UnrolledFunction, String> {
    if !input.cfg.is_complete() {
        let why = input
            .cfg
            .stalls
            .first()
            .map_or_else(|| format!("{:?}", input.cfg.limit_hit), |stall| format!("{stall:?}"));
        return Err(format!("the CFG is incomplete: {why}"));
    }
    let (state, _) = frame_initial_state(input.frame, input.top_offset);
    let mut walker = Walker {
        input,
        nested,
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
        notes: if input
            .frame
            .iter()
            .any(|cell| matches!(cell, NativeFrameSource::RelocatedConstant { .. }))
        {
            vec![
                "preferred-base frame: relocated entry-stub immediates taken at their on-disk value (relocation delta 0)"
                    .to_owned(),
            ]
        } else {
            Vec::new()
        },
    })
}

impl Walker<'_, '_> {
    fn successors(&self, key: Va) -> Vec<(u64, Va)> {
        let cfg = self.input.cfg;
        let mut out: Vec<(u64, Va)> = Vec::new();
        for edge in &cfg.edges {
            if edge.from != key
                || !matches!(
                    edge.kind,
                    // A profile transition IS this block's successor: the
                    // unconditional-branch handler pops a target and the VM
                    // continues there under a different register profile. The
                    // rolled frontend has counted it as one all along; leaving
                    // it out here made the transition block's own target "not a
                    // recovered successor".
                    CfgEdgeKind::Branch
                        | CfgEdgeKind::GuardedBranch
                        | CfgEdgeKind::ProfileTransition
                )
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
            // Same case `rolled.rs` handles: `BlockExit` has no variant for a
            // cross-profile transition, so the block records as `Stalled` while
            // the CFG carries its outgoing `ProfileTransition` edge and the
            // graph is complete through it. The edge's own documentation says
            // what it is -- an unconditional-branch handler pops a target off
            // the VM stack and the VM continues under a proven-different
            // register profile -- so it is executed as exactly that.
            let transition = block.exit == BlockExit::Stalled
                && input
                    .cfg
                    .edges
                    .iter()
                    .any(|edge| edge.from == key && edge.kind == CfgEdgeKind::ProfileTransition);
            let boundary = match if transition { BlockExit::Branch } else { block.exit } {
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
                        epoch: write.epoch,
                    },
                    write.epoch,
                ));
            }
            // The transition classification decided the boundary above, so the
            // walk has to see the SAME exit -- matching on `block.exit` here
            // sent a transition block into the `unreachable!` arm.
            match if transition { BlockExit::Branch } else { block.exit } {
                BlockExit::VmExit => {
                    let handler = block
                        .instructions
                        .last()
                        .map_or(block.start.handler, |last| last.state_after.handler);
                    let layout = block
                        .vm_exit
                        .as_ref()
                        .or_else(|| input.exits.get(&handler))
                        .ok_or_else(|| format!("no recovered VM-exit layout for handler {handler}"))?;
                    if let Some(to) = cfg
                        .edges
                        .iter()
                        .find_map(|e| (e.from == key && e.kind == CfgEdgeKind::NestedEntry).then_some(e.to))
                    {
                        let resolver = self.nested.ok_or("exit re-enters a nested VM entry")?;
                        let exit = vmpx_vm::recover_native_exit_state(&report, layout)
                            .map_err(|e| format!("nested exit state unrecovered: {e}"))?;
                        let entry = resolver
                            .resolve(&exit.return_address)
                            .ok_or("the nested VM entry is not provably resolved")?;
                        if cfg.blocks.get(&to).and_then(|b| b.address) != Some(entry.bytecode_start) {
                            return Err("the resolved nested entry is not the CFG's nested block".to_owned());
                        }
                        walk.state = compose_nested_state(&report, &exit, &entry)
                            .map_err(|e| format!("nested entry composition failed: {e}"))?;
                        key = to;
                        continue;
                    }
                    let recovered = recover_exit(&report, layout, input.frame.len(), input.top_offset)
                        .map_err(|e| e.to_string())?;
                    let mut region_target = None;
                    let (exit, region_exit) = match recovered {
                        RecoveredExit::Callee(exit) => (exit, false),
                        RecoveredExit::Trampoline(_) => {
                            return Err("the exit transfers to a computed continuation above the entry frame".into());
                        }
                        RecoveredExit::Region(region) => {
                            region_target = region.staged.first().map(|target| RegionTarget {
                                target: target.simplify(),
                                staged_words: region.staged.len(),
                            });
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
                        region: region_target,
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
pub(crate) fn is_environment_probe(selector: &Expr) -> bool {
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
