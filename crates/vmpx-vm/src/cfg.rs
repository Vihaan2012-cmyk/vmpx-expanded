//! Control-flow walk over a virtualized function.
//!
//! Decodes the entry's linear VM stream, then follows each block's terminating
//! branch to its successors while carrying the symbolic VM state. A successor is
//! followed only when the branch target is proven: a constant after symbolic
//! execution, or the image of a target over a proven-exhaustive input domain.
//! Anything unresolved stops that path and is reported; no edge is invented.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::time::{Duration, Instant};

use iced_x86::{FlowControl, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register};
use serde::Serialize;
use thiserror::Error;
use vmpx_disasm::{DirectJumpTraceLimits, trace_direct_jump_fragments};
use vmpx_pe::{PeImage, Rva, Va};

use crate::env_gate::{
    EnvironmentGateRoute, has_environment_source, record_environment_fork,
    resolve_generalized_environment_gate,
};
use crate::{
    BinaryOp, BitWidth, DecodedVmInstruction, DispatchExtractionCandidate,
    EntryProfileRecoveryError, EntryProfileRecoveryLimits, Expr, ExprError, ExprKind,
    InputDomainLimits,
    LinearDecodeCache, LinearDecodeStop, ModuloImageTargetLimits, NativeCallSelectorOutcome,
    NativeFrameSource, RecoveryStatus, RelativeDispatchState, SymbolicDomainLimits,
    SymbolicVmExecutionReport, SymbolicVmInitialState, UnaryOp, VmDiscoveryLimits, VmEntryCandidate,
    VmExitReport, VmHandlerSemantic, VmProfileRoleResolutionConfig, analyze_vm_entry_at,
    analyze_vm_transition,
    dispatch_extract::canonical,
    entry_profile::{bootstrap_global_table, entry_stream_config, prepare_entry_stream},
    resolve_vm_profile_roles,
    distinct_node_count_bounded, enumerate_symbolic_value,
    execute_symbolic_linear_block_with_initial_state,
    execute_symbolic_native_call_boundary_with_initial_state, infer_branch_input_domains,
    diagnose_branch_refusals, RefusalCause,
    native_context::{
        Provenance, caller_push_source, defines_flags, initial_provenance, is_cursor_capture,
        observe_target_push, update_provenance,
    },
    native_vm_chain::compose_native_frame,
    recover_and_decode_linear_vm_stream_cached, recover_image_table_targets,
    recover_modulo_image_targets, recover_native_exit_state, value_range, ImageTableBranchLimits,
};

/// Distinct nodes one `shift_ancestor_onto_reentry` may decompose before it
/// gives up. Each node costs an `affine_vsp_parts` decomposition of its whole
/// subtree, which allocates a map of its terms, so the work is quadratic in the
/// expression while the node COUNT -- the only thing the walker charges --
/// stays flat. That is why a state that passes every size check can still take
/// resident memory past 13 GB.
///
/// Exhausting the budget abandons the shift, so the fold simply does not
/// happen: the entry falls back to the ordinary path and, failing that, stalls.
/// Incomplete, never unsound.
const SHIFT_NODE_BUDGET: usize = 200_000;

/// 1.5 GiB per entry: far above any entry that completes, yet small enough that
/// several concurrent entries stay well inside a 10 GB machine.
const DEFAULT_MAX_MEMORY_BYTES: usize = 1536 * 1024 * 1024;

/// Default per-entry wall-clock budget. This is a safety net for a pathological
/// entry, not an exploration cap — but at 60s it was acting as one. Measured on
/// systems.xpl (814 entries, `--all --serve-ro`, 24 threads), varying ONLY this:
///
/// | budget | N1 | WallClock stops | sweep |
/// |--------|-----|-----|-------|
/// | 60s    | 448 | 62  | 430s  |
/// | 150s   | 452 | 22  | 661s  |
/// | 400s   | 460 | 0   | 684s  |
///
/// So 60s was refusing 12 entries that do finish, and at 400s no entry is
/// time-bound at all: the net has stopped deciding anything, which is where a
/// timeout belongs. The sweep cost only 1.6x (not 6.7x) because the extra time
/// is spent by a handful of entries. A lower budget only ever stops an entry
/// sooner, recorded as `WallClock`, i.e. incomplete — so this is
/// soundness-neutral in both directions. `VMPX_WALL_SECS` overrides it.
/// N2 is unaffected: on the sound path every incomplete entry stops at a stall,
/// none at the clock.
pub const DEFAULT_MAX_WALL_TIME: Duration = Duration::from_secs(400);

/// Over-estimate of the bytes one distinct hash-consed `Expr` node costs: the
/// node, its `Arc` header, the interner's weak slot and bucket, and allocator
/// overhead (a measured node is well under half this).
const EXPR_NODE_BYTES: usize = 256;
/// Per-root container overhead (`BTreeMap`/`Vec` slot plus the inline `Expr`).
const STATE_ROOT_BYTES: usize = 128;
/// Over-estimate of one decoded instruction including its heap-owned bytes.
const DECODED_INSTRUCTION_BYTES: usize = std::mem::size_of::<DecodedVmInstruction>() + 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmCfgLimits {
    pub entry: EntryProfileRecoveryLimits,
    pub input_domain: InputDomainLimits,
    pub domain: SymbolicDomainLimits,
    /// Distinct decoded basic blocks.
    pub max_blocks: usize,
    /// Symbolic block executions across all paths.
    pub max_visits: usize,
    /// Executions of one block under distinct symbolic states.
    pub max_visits_per_block: usize,
    pub max_total_instructions: usize,
    /// Distinct targets one branch may resolve to.
    pub max_targets: usize,
    /// VM-to-VM re-entries followed across the whole walk.
    pub max_nested_entries: usize,
    /// Proven cross-profile VM transitions followed per entry. Each transition
    /// opens a new profile region whose initial symbolic state is the PRIOR
    /// region's accumulated continuation state, so retained symbolic state grows
    /// with transition depth. This hard cap bounds that growth (and the extra
    /// blocks it reaches) so a pathological entry fails closed at the budget
    /// instead of allocating without limit; the sound path (no transition) never
    /// reaches it.
    pub max_profile_transitions: usize,
    /// Visited blocks allowed AFTER the first cross-profile transition, before
    /// the walk fails closed. Each retained post-transition block carries a
    /// deeper (larger) symbolic state than the last, so this — not the loose
    /// entry-global `max_visits` — is what actually bounds transition-walk
    /// memory. Counts only once a transition has fired, so an entry that follows
    /// none is bounded exactly as before (byte-for-byte unchanged).
    pub max_transition_visits: usize,
    /// Decoded instructions allowed AFTER the first cross-profile transition,
    /// before the walk fails closed. Bounds retained decoded-block memory on the
    /// transition walk; inert on a walk that followed no transition.
    pub max_transition_instructions: usize,
    /// Iterations of one loop unrolled with exact state before its changing
    /// cells are generalized to a stable over-approximation. Kept well under
    /// `max_visits_per_block`: each unrolled iteration propagates a distinct
    /// state into post-loop join blocks, so an over-large budget stalls those
    /// joins before the widened state can subsume the iterations.
    pub max_loop_unrolls: usize,
    /// Hard per-entry ceiling on a conservative estimate of the bytes the walker
    /// retains (symbolic state held in `explored`/`widened`/pending/path nodes,
    /// plus decoded-instruction buffers). Exceeding it fails the entry closed
    /// with [`CfgLimitHit::Memory`] before real memory can be exhausted. The
    /// estimate only over-counts, so an entry under the cap is unaffected.
    pub max_memory_bytes: usize,
    /// Per-entry wall-clock budget, checked between block visits. Exceeding it
    /// stops the walk and records [`CfgLimitHit::WallClock`] (never complete).
    /// `None` is unbounded. A single block visit is not interrupted.
    pub max_wall_time: Option<Duration>,
    /// EXPERIMENTAL. When true, a computed branch target that leaves the entry's
    /// proven bytecode section but lands in ANOTHER EXECUTABLE image section is
    /// followed as bytecode (labeled
    /// [`VmCfgAssumption::AssumesCrossVmSectionContinuation`]) instead of failing
    /// closed. The carried rolling-key decode at the target is the guard: a
    /// non-continuation target decodes to an invalid dispatch and still stalls.
    /// Default false: the sound path is byte-for-byte unchanged.
    pub follow_cross_vm_section: bool,
}

impl Default for VmCfgLimits {
    fn default() -> Self {
        Self {
            entry: EntryProfileRecoveryLimits::default(),
            input_domain: InputDomainLimits::default(),
            domain: SymbolicDomainLimits::default(),
            max_blocks: 512,
            max_visits: 4_096,
            // Do NOT raise. Unlike the transition and nesting caps, this one is
            // load-bearing: `visited` already counts DISTINCT states (an exact
            // repeat returns early), so a larger bound buys real exploration at
            // real cost. Measured 16 -> 64: sweep 254s -> 627s, WallClock stops
            // 12 -> 28, and N1 FELL 449 -> 442 as entries timed out against the
            // per-entry wall clock. 16 converges via `generalize_across` sooner
            // and wins.
            max_visits_per_block: 16,
            max_total_instructions: 200_000,
            max_targets: 16,
            // 8 refused 37 entries on systems.xpl once the earlier blockers
            // cleared, 3 of them blocked by nothing else; 4x is worth +3 N1.
            max_nested_entries: 32,
            // Counts DISTINCT transition sites (see `transition_sites`), so this
            // bounds how many profile regions one entry may reach, not how often
            // it re-enters them. 32 still refused entries at that bound; 8x is
            // worth a further +2 and is bounded by the per-region budget above
            // plus the entry-global and memory caps.
            max_profile_transitions: 256,
            // Per-region allowances (see `transition_budget_exhausted`). These were
            // 64 / 6000 when the budget was one flat grant for the whole walk; at
            // that size they refused 238 entries on systems.xpl and measured +3 N1
            // when scaled 8x, so the old values were simply too tight for a region
            // rather than load-bearing. Memory is bounded by `max_memory_bytes`,
            // not by these, and every one of them fails closed.
            // Raised again (8x) for a further +2: with the per-region scaling the
            // entry-global `max_visits` / `max_total_instructions` and the hard
            // `max_memory_bytes` cap bind before this sub-budget does, so it was
            // constraining exploration without being the thing bounding memory.
            max_transition_visits: 4_096,
            max_transition_instructions: 384_000,
            max_loop_unrolls: 4,
            max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
            max_wall_time: Some(DEFAULT_MAX_WALL_TIME),
            follow_cross_vm_section: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockExit {
    VmExit,
    Branch,
    NativeCall,
    Stalled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmBasicBlock {
    /// Decoder state after the first opcode dispatch; `start.vip` is the block key.
    pub start: RelativeDispatchState,
    /// Address of the first opcode when known (entry or a branch target).
    pub address: Option<Va>,
    pub instructions: Vec<DecodedVmInstruction>,
    pub exit: BlockExit,
    /// Handler of the control-flow boundary that ends the block.
    pub boundary_handler: Option<Va>,
    /// Proven call frame of the selector this `NativeCall` block decoded.
    pub native_call: Option<NativeCallSelectorOutcome>,
    /// Recovered exit layout of this `VmExit` block's handler.
    pub vm_exit: Option<VmExitReport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CfgEdgeKind {
    /// Branch target proven constant.
    Branch,
    /// One of several targets proven over an exhaustive input domain.
    GuardedBranch,
    /// Control resumes after a proven native call.
    NativeCallReturn,
    /// A proven VM exit returns into another VM entry of the same VM.
    NestedEntry,
    /// An unconditional-branch handler pops a target off the VM stack and the
    /// VM continues under a PROVEN-DIFFERENT register profile (new VIP/key/
    /// handler roles and possibly flipped bytecode direction). The successor
    /// block is decoded under that new profile.
    ProfileTransition,
    /// Back edge to a block already on the path whose entry state subsumes
    /// the re-entry state; accompanies the `Branch`/`GuardedBranch` edge.
    LoopBack,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct CfgEdge {
    pub from: Va,
    pub to: Va,
    pub kind: CfgEdgeKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CfgStall {
    /// Key of the block whose path stopped.
    pub block: Va,
    pub vip: Va,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CfgLimitHit {
    Blocks,
    Visits,
    BlockVisits,
    TotalInstructions,
    /// The tight post-cross-profile-transition exploration budget was exhausted
    /// (bounds memory: see `VmCfgLimits::max_transition_visits`). The entry fails
    /// closed; it never affects a walk that followed no transition.
    TransitionBudget,
    /// The byte-accounted per-entry memory cap was exceeded (see
    /// `VmCfgLimits::max_memory_bytes`). The entry fails closed.
    Memory,
    /// The per-entry wall-clock budget (`VmCfgLimits::max_wall_time`) expired.
    /// The entry fails closed as incomplete.
    WallClock,
}

/// DIAGNOSTIC ONLY: every offending leaf of one unresolved branch target.
#[derive(Clone, Debug)]
pub struct BranchRefusalDiagnostic {
    pub vip: Va,
    pub stall_reason: String,
    /// Offset above `vsp_initial` where the proven entry frame ends, when known.
    pub frame_hi: Option<u64>,
    pub causes: Vec<RefusalCause>,
}

thread_local! {
    static DIAG_ENABLED: Cell<bool> = const { Cell::new(false) };
    static DIAG_FRAME_HI: Cell<Option<u64>> = const { Cell::new(None) };
    static DIAG_SINK: std::cell::RefCell<Vec<BranchRefusalDiagnostic>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

pub(crate) fn diagnostics_enabled() -> bool {
    DIAG_ENABLED.with(Cell::get)
}

/// Switches the per-thread branch-refusal diagnostic collection on or off and
/// clears anything collected. Off by default; it never influences recovery.
pub fn set_branch_refusal_diagnostics(enabled: bool) {
    DIAG_ENABLED.with(|flag| flag.set(enabled));
    DIAG_SINK.with(|sink| sink.borrow_mut().clear());
}

/// Drains the diagnostics collected on this thread since the last call.
#[must_use]
pub fn take_branch_refusal_diagnostics() -> Vec<BranchRefusalDiagnostic> {
    DIAG_SINK.with(|sink| std::mem::take(&mut *sink.borrow_mut()))
}

/// A labeled assumption the recovery relied on, surfaced rather than folded
/// silently (mirrors `IdiomAssumption::AssumesCleanEnvironment`).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VmCfgAssumption {
    /// A native input register used only to relocate in-image addresses was
    /// bound to 0 for branch resolution: the image-base / relocation delta,
    /// which is 0 at the preferred base every image read already assumes.
    PreferredImageBase,
    /// An external store to a native-register-rooted or opaque address was
    /// narrowed to discard only the non-vsp-rooted tracked cells, preserving the
    /// VM's private operand stack, under the assumption that external stores do
    /// not alias that private stack (caller/parameter pointers do not point into
    /// it). Image-source reads remain withheld, so the narrowing only recovers
    /// the VM-stack-rooted VIP; it does not otherwise trust the store.
    AssumesExternalStoresDisjointFromVmStack,
    /// Generalizing or closing a loop conceded that nothing outside the VM
    /// aliases the VM stack frame. A widened state, or one re-expressed at a
    /// moved stack base, describes the same concrete memory only under that
    /// concession.
    ///
    /// Recorded SEPARATELY from `AssumesExternalStoresDisjointFromVmStack`,
    /// which is inserted only when the symbolic executor actually narrowed an
    /// external store. `frame_not_private` blocks discharging that one, so for a
    /// long time the concession was recorded only on entries that happened to
    /// have taken it — an entry that closed a strided loop without ever
    /// narrowing a store carried no label at all and counted as PROVEN. The
    /// concession is load-bearing either way, so it gets its own name.
    AssumesVmStackFramePrivate,
    /// A constant read of a read-only, file-backed image page was served from
    /// the file after an external store suppressed image reads, assuming the
    /// page protection never changes at runtime (the binary imports
    /// `VirtualProtect`, so this is not provable). EXPERIMENTAL tier: present
    /// only when `SymbolicExecutionLimits::serve_read_only_when_suppressed`
    /// is enabled.
    AssumesImageProtectionStable,
    /// A branch gated on a pure host-environment intrinsic (a VMProtect CPUID
    /// processor-signature / timestamp-counter anti-analysis gate) was resolved
    /// by a sound both-target over-approximation: the environment value is an
    /// unknown input, so both sides of the gate were enumerated and followed. A
    /// host-CPU-specific dead path is conservatively included; no reachable
    /// behavior is dropped. Also set when the gate condition is rooted in any
    /// other host-environment source (see [`crate::EnvironmentSource`]: API
    /// returns, PEB/TEB loads, constant out-of-image probes): the same
    /// both-target fork, never an assumption about the runtime value.
    OverApproximatedEnvironmentGate,
    /// A provably-null image read (a global pointer slot that is zero in the
    /// static image but runtime-initialized) was resolved using a concrete value
    /// observed on ONE detonation, instead of failing closed. EXPERIMENTAL and
    /// observed-tier: valid for that run/environment only, never sound output.
    /// Present only when recovery was driven with an observed-memory seed.
    ObservedGlobal,
    /// A recursion inlined as a VSP-changing bytecode loop (a push-down stack on
    /// the VM stack) was lifted as a BOUNDED UNROLL of exactly the depth observed
    /// on ONE detonation, instead of the non-converging fixpoint widening that
    /// otherwise fails closed at "loop changes the VM stack pointer". Engaged
    /// only when the per-iteration VSP delta is a PROVEN CONSTANT and an observed
    /// depth is present for the loop head. EXPERIMENTAL and observed-tier: valid
    /// for that run/input only, never sound output.
    ObservedRecursionDepth,
    /// A computed branch that otherwise fails closed — its constant target points
    /// into a non-executable data section (CASE-C), or its target depends on an
    /// untracked runtime memory read — was resolved to a concrete target VIP
    /// OBSERVED for that branch site on ONE detonation. The seeded target is
    /// followed ONLY when it independently validates as real in-image bytecode in
    /// the entry's proven bytecode section: observation seeds a resolution, it
    /// never turns a data address into code. EXPERIMENTAL and observed-tier:
    /// valid for that run/input only, never sound output.
    ObservedBranchTarget,
    /// Companion to [`Self::ObservedBranchTarget`]: the walker treated the
    /// dynamically observed target set of a stalled branch as ITS COMPLETE
    /// successor set. Observation is a lower bound (only branches executed on
    /// ONE run), so an unobserved arm is silently missing and the CFG is NOT
    /// proven complete. Always present whenever an observed branch target was
    /// followed; marks the result experimental-tier.
    AssumesDynamicBranchTargetsComplete,
    /// A computed VM branch target that left the entry's proven bytecode section
    /// was followed into a DIFFERENT EXECUTABLE image section, under the premise
    /// that a coherent key-driven dispatch decode at that address is a genuine
    /// continuation of this VM's bytecode (VMProtect splits one function's
    /// bytecode across executable protector sections such as `.vmp2`/`.vmp3`).
    /// The follow is taken only when the target lands in an executable section and
    /// the carried rolling key decodes a well-formed dispatch there; a wrong target
    /// decodes to garbage and still fails closed. EXPERIMENTAL and unproven:
    /// present only when `VmCfgLimits::follow_cross_vm_section` is enabled, and it
    /// keeps the entry off the proven (N2) tier.
    AssumesCrossVmSectionContinuation,
}

/// Observed back-edge trip counts per loop head, captured on ONE detonation: a
/// plain, crate-owned map from loop-head virtual address to recursion depth.
/// This is NOT a proof — it reflects a single run/input. The CLI builds it from
/// a detonation's `ObservedFacts.loop_depths`; vmpx-vm never produces it and
/// never trusts it on the sound path.
///
/// Kept deliberately free of any `vmpx-emulate` type (the crate owns this plain
/// structure, mirroring [`ObservedMemory`](crate::ObservedMemory)).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObservedLoopDepths {
    depths: BTreeMap<u64, u64>,
}

impl ObservedLoopDepths {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds the seed from `(loop_head, depth)` pairs, e.g. a detonation's
    /// observed loop depths.
    pub fn from_depths(depths: impl IntoIterator<Item = (u64, u64)>) -> Self {
        Self {
            depths: depths.into_iter().collect(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.depths.is_empty()
    }

    /// Observed recursion depth for `loop_head`, if any was recorded.
    #[must_use]
    pub fn depth_for(&self, loop_head: u64) -> Option<u64> {
        self.depths.get(&loop_head).copied()
    }
}

/// The observed-runtime seed for one experimental recovery: concrete memory and
/// loop depths captured on ONE detonation, bundled so the CLI threads a single
/// value. Both halves are empty on the sound path (so recovery is unchanged),
/// mirroring D6b's seedless default.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObservedRun {
    pub memory: crate::ObservedMemory,
    pub loop_depths: ObservedLoopDepths,
    /// Observed concrete VM branch targets per branch site (empty on the sound
    /// path). Feeds the experimental observed-branch-target bridge, which
    /// resolves a branch that otherwise fails closed (CASE-C data target or an
    /// untracked-memory-epoch target) only when the observed target still
    /// validates as real in-image bytecode.
    pub branch_targets: crate::ObservedBranchTargets,
}

impl ObservedRun {
    /// Construction stays source-compatible with the memory/loop-depths bridges;
    /// `branch_targets` defaults empty. Set the field to seed observed branch
    /// targets.
    #[must_use]
    pub fn new(memory: crate::ObservedMemory, loop_depths: ObservedLoopDepths) -> Self {
        Self {
            memory,
            loop_depths,
            branch_targets: crate::ObservedBranchTargets::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VmCfg {
    pub entry: Va,
    pub blocks: BTreeMap<Va, VmBasicBlock>,
    pub edges: Vec<CfgEdge>,
    pub stalls: Vec<CfgStall>,
    pub limit_hit: Option<CfgLimitHit>,
    pub visits: usize,
    /// Initial VM-stack cells with a proven value, and why the rest are unknown.
    pub frame_cells_known: usize,
    pub frame_note: Option<String>,
    /// Labeled assumptions relied on during recovery.
    pub assumptions: BTreeSet<VmCfgAssumption>,
    /// Why the VM-stack assumption could not be discharged, when it could not.
    /// Diagnostic only. Such an entry is CFG-complete but not proven, so this
    /// names the condition standing between it and the proven tier.
    pub discharge_refusal: Option<String>,
}

impl VmCfg {
    /// Proven call frames, keyed by the key of the block that ends in the call.
    #[must_use]
    pub fn native_calls(&self) -> BTreeMap<Va, NativeCallSelectorOutcome> {
        self.blocks
            .iter()
            .filter_map(|(key, block)| Some((*key, block.native_call.clone()?)))
            .collect()
    }

    /// Recovered exit layouts, keyed by the exit handler's address.
    #[must_use]
    pub fn exit_layouts(&self) -> BTreeMap<Va, VmExitReport> {
        self.blocks
            .values()
            .filter_map(|block| Some((block.boundary_handler?, block.vm_exit.clone()?)))
            .collect()
    }

    fn record_native_call(&mut self, key: Va, outcome: &NativeCallSelectorOutcome) {
        if let Some(block) = self.blocks.get_mut(&key)
            && block.exit == BlockExit::NativeCall
        {
            block.native_call = Some(outcome.clone());
        }
    }

    #[must_use]
    pub fn total_instructions(&self) -> usize {
        self.blocks
            .values()
            .map(|block| block.instructions.len())
            .sum()
    }

    /// Every explored path ended at a proven VM exit or re-entered a decoded
    /// block, with no stall and no resource limit hit.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.stalls.is_empty() && self.limit_hit.is_none()
    }

    /// True when any observed-tier (single-detonation) evidence helped this
    /// recovery. Such a CFG may be reported only on the experimental tier,
    /// never as proven, even when `all_paths_reach_vm_exit` holds.
    #[must_use]
    pub fn used_observed_evidence(&self) -> bool {
        self.assumptions.iter().any(|assumption| {
            matches!(
                assumption,
                VmCfgAssumption::ObservedGlobal
                    | VmCfgAssumption::ObservedRecursionDepth
                    | VmCfgAssumption::ObservedBranchTarget
                    | VmCfgAssumption::AssumesDynamicBranchTargetsComplete
            )
        })
    }

    /// Complete only thanks to observed branch targets: reaches a VM exit on
    /// every walked path, but at least one branch was resolved from dynamic
    /// evidence. The separately-counted experimental tier.
    #[must_use]
    pub fn complete_with_observed_branch_help(&self) -> bool {
        self.all_paths_reach_vm_exit()
            && self
                .assumptions
                .contains(&VmCfgAssumption::AssumesDynamicBranchTargetsComplete)
    }

    #[must_use]
    pub fn all_paths_reach_vm_exit(&self) -> bool {
        self.is_complete()
            && self
                .blocks
                .values()
                .any(|block| block.exit == BlockExit::VmExit)
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum CfgError {
    #[error("entry setup failed: {0}")]
    Entry(String),
}

impl From<EntryProfileRecoveryError> for CfgError {
    fn from(error: EntryProfileRecoveryError) -> Self {
        Self::Entry(error.to_string())
    }
}

/// If `instruction` captures RSP into `vsp`, returns the byte offset from the
/// captured VSP up to the top (last-pushed) frame cell, which sits at the live
/// RSP. `mov vsp,rsp` sets VSP = RSP, so that offset is 0. `lea vsp,[rsp+disp]`
/// sets VSP = RSP + disp, so RSP = VSP - disp and the top cell sits at VSP minus
/// disp. Only RSP-based, index-free, 64-bit, non-RSP destinations that match the
/// profile's VSP register are accepted; anything else fails closed (None).
fn vsp_capture_top_offset(instruction: &iced_x86::Instruction, vsp: Register) -> Option<u64> {
    let vsp = canonical(vsp);
    if is_cursor_capture(instruction).is_some_and(|cursor| cursor == vsp) {
        return Some(0);
    }
    if instruction.mnemonic() == Mnemonic::Lea
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op0_register() != Register::RSP
        && canonical(instruction.op0_register()) == vsp
        && instruction.memory_base() == Register::RSP
        && instruction.memory_index() == Register::None
    {
        // VSP = RSP + disp  =>  RSP = VSP - disp. The top cell sits at RSP, i.e.
        // VSP - disp; the displacement is modular, so negate it in u64.
        return Some(0_u64.wrapping_sub(instruction.memory_displacement64()));
    }
    None
}

/// True when `instruction` copies the stack pointer (or an address derived from
/// it) into a register or the stack: a register operand naming RSP, or an `lea`
/// based on it. Plain loads and stores through RSP do not expose the address.
fn exposes_stack_pointer(instruction: &iced_x86::Instruction) -> bool {
    let is_rsp = |register: Register| register.full_register() == Register::RSP;
    // A 16-bit (or narrower) read of RSP as a source operand yields no usable
    // frame address: user-mode addresses below 64K are never mapped. VMProtect
    // emits junk like `bsr ax,sp` after the VSP capture, and counting that as
    // exposure made the frame extension bail and left the image-base relocation
    // delta cell unknown, which made the whole address chain unresolvable.
    // A write to SP, a 32-bit ESP read, `lea` and `push rsp` all still expose.
    let narrow_read = |operand: u32| {
        // `cmp`/`test` write no register, so operand 0 is a READ too: a 16-bit
        // `cmp sp,r15w` only sets flags and yields no address. Every other
        // mnemonic here writes operand 0, so only its sources are exempt.
        let reads_operand_zero = matches!(instruction.mnemonic(), Mnemonic::Cmp | Mnemonic::Test);
        (operand >= 1 || reads_operand_zero)
            && instruction.op_register(operand).size() <= 2
            && matches!(
                instruction.mnemonic(),
                Mnemonic::Bsr
                    | Mnemonic::Bsf
                    | Mnemonic::Mov
                    | Mnemonic::Movzx
                    | Mnemonic::Movsx
                    | Mnemonic::Cmp
                    | Mnemonic::Test
            )
    };
    (0..instruction.op_count()).any(|operand| match instruction.op_kind(operand) {
        OpKind::Register => is_rsp(instruction.op_register(operand)) && !narrow_read(operand),
        OpKind::Memory => {
            instruction.mnemonic() == Mnemonic::Lea
                && (is_rsp(instruction.memory_base()) || is_rsp(instruction.memory_index()))
        }
        _ => false,
    })
}

/// The VM entry a nested VM exit returns into at `target`: the stub itself, or
/// the stub behind the entry flag-save (`pushfq`, then `push imm; call`). Returns
/// the RVA frame recovery must start from (the flag-save included) and the
/// candidate at the `push imm`.
#[must_use]
pub fn nested_entry_stub(pe: &PeImage<'_>, target: Rva) -> Option<(Rva, VmEntryCandidate)> {
    if let Some(candidate) = analyze_vm_entry_at(pe, target, VmDiscoveryLimits::default())
        .filter(|candidate| candidate.confidence >= NESTED_ENTRY_CONFIDENCE)
    {
        return Some((target, candidate));
    }
    let first = vmpx_disasm::decode_one(pe, target).ok()?;
    if first.instruction.mnemonic() != Mnemonic::Pushfq || std::env::var_os("VMPX_NO_PUSHFQ_NESTED").is_some() {
        return None;
    }
    let candidate = analyze_vm_entry_at(pe, first.next_rva()?, VmDiscoveryLimits::default())
        .filter(|candidate| candidate.confidence >= NESTED_ENTRY_CONFIDENCE)?;
    Some((target, candidate))
}

/// Proven contents of the VM stack at VM start, top first, recovered by
/// stepping the native entry up to the VSP capture. Returns the cells, the
/// byte offset from VSP to the top cell (nonzero when a `lea` capture places VSP
/// off the frame top), and whether any pre-capture instruction exposed the
/// stack pointer to a register or memory. Fails closed on anything it cannot
/// model (nested calls, unknown stack writes, stack-pointer arithmetic).
///
/// # Errors
///
/// Returns a note describing the first thing the stub does that is not modelled.
pub fn recover_entry_frame(
    pe: &PeImage<'_>,
    entry: Rva,
    vsp: Register,
    limits: DirectJumpTraceLimits,
) -> Result<(Vec<NativeFrameSource>, u64, bool), String> {
    let caller = trace_direct_jump_fragments(pe, entry, limits).map_err(|e| e.to_string())?;
    let caller_instructions = caller.instructions().collect::<Vec<_>>();
    let (call_index, call) = caller_instructions
        .iter()
        .enumerate()
        .find(|(_, decoded)| decoded.instruction.flow_control() == FlowControl::Call)
        .ok_or("entry trace has no direct call")?;
    let call_target = call
        .direct_target_rva(pe)
        .ok_or("entry call has no direct target")?;
    // Bottom-first model of the native stack.
    let mut stack: Vec<NativeFrameSource> = Vec::new();
    let mut exposed = false;
    let mut flags_clean = true;
    for decoded in caller_instructions.iter().copied().take(call_index) {
        let instruction = &decoded.instruction;
        exposed |= exposes_stack_pointer(instruction);
        if instruction.mnemonic() == Mnemonic::Push {
            let ordinal = u32::try_from(stack.len()).unwrap_or(u32::MAX);
            stack.push(caller_push_source(decoded, ordinal));
        } else if instruction.mnemonic() == Mnemonic::Pushfq {
            // The entry flag-save: the word is the flags the stub was entered with
            // (`native_rflags_initial`), provided nothing before it set them.
            stack.push(if flags_clean {
                NativeFrameSource::Rflags
            } else {
                NativeFrameSource::Unknown {
                    description: format!("RFLAGS modified before PUSHFQ at {}", decoded.rva),
                }
            });
        } else if instruction.stack_pointer_increment() != 0 {
            return Err(format!(
                "unproven pre-call stack mutation at {}",
                decoded.rva
            ));
        }
        if defines_flags(instruction) {
            flags_clean = false;
        }
    }
    stack.push(NativeFrameSource::ReturnAddress {
        call_site: call.rva,
        continuation: call.next_rva().ok_or("call continuation overflows")?,
    });

    let target = trace_direct_jump_fragments(pe, call_target, limits).map_err(|e| e.to_string())?;
    let relocations = relocation_slots(pe);
    let mut state = StubState {
        stack,
        provenance: initial_provenance(),
        original_flags: flags_clean,
        factory: InstructionInfoFactory::new(),
    };
    let mut remaining = target.instructions();
    let mut capture = None;
    for decoded in remaining.by_ref() {
        let instruction = &decoded.instruction;
        if let Some(top_offset) = vsp_capture_top_offset(instruction, vsp) {
            capture = Some(top_offset);
            update_provenance(&mut state.factory, instruction, &mut state.provenance);
            break;
        }
        exposed |= exposes_stack_pointer(instruction);
        state.step(decoded, false)?;
        state.track_relocated(decoded, relocations.as_ref());
    }
    let top_offset = capture.ok_or("no VSP capture reached")?;
    let mut captured = state.stack.clone();
    captured.reverse();
    // The stub keeps pushing below the captured VSP before the VM starts, so
    // those cells hold stub-written values. Any failure to model them keeps the
    // captured-frame result unchanged.
    let (cells, top_offset) =
        extend_below_capture(&mut state, remaining, relocations.as_ref(), vsp, top_offset)
            .unwrap_or((captured, top_offset));
    Ok((cells, top_offset, exposed))
}

/// Native-stub state while stepping its pushes and RSP-relative stores.
struct StubState {
    /// Bottom-first model of the native stack.
    stack: Vec<NativeFrameSource>,
    provenance: BTreeMap<Register, Provenance>,
    original_flags: bool,
    factory: InstructionInfoFactory,
}

impl StubState {
    /// Applies one stub instruction to the stack and register provenance.
    /// `keep_relocated` keeps a relocated immediate as the relocation delta
    /// instead of its on-disk constant.
    fn step(
        &mut self,
        decoded: &vmpx_disasm::DecodedInstruction,
        keep_relocated: bool,
    ) -> Result<(), String> {
        let instruction = &decoded.instruction;
        match instruction.flow_control() {
            FlowControl::Next | FlowControl::UnconditionalBranch => {}
            _ => return Err(format!("unsupported control flow at {}", decoded.rva)),
        }
        let relocated = |operand: u32| {
            (keep_relocated
                && instruction.op_kind(operand) == OpKind::Register
                && instruction.op_register(operand).size() == 8)
                .then(|| self.provenance.get(&canonical(instruction.op_register(operand))))
                .flatten()
                .and_then(|known| match known {
                    Provenance::RelocatedConstant(value) => {
                        Some(NativeFrameSource::RelocatedConstant { value: *value })
                    }
                    _ => None,
                })
        };
        let plain: BTreeMap<Register, Provenance> = self
            .provenance
            .iter()
            .map(|(register, known)| match known {
                Provenance::RelocatedConstant(value) => (*register, Provenance::Constant(*value)),
                other => (*register, other.clone()),
            })
            .collect();
        if matches!(instruction.mnemonic(), Mnemonic::Push | Mnemonic::Pushfq) {
            let source = if instruction.mnemonic() == Mnemonic::Push && let Some(source) = relocated(0) {
                source
            } else {
                observe_target_push(decoded, &plain, self.original_flags)
                    .ok_or_else(|| format!("unsupported push at {}", decoded.rva))?
                    .source
            };
            if instruction.stack_pointer_increment() != -8 {
                return Err(format!("non-qword push at {}", decoded.rva));
            }
            self.stack.push(source);
        } else {
            if instruction.stack_pointer_increment() != 0 {
                return Err(format!("unproven stack mutation at {}", decoded.rva));
            }
            let writes: Vec<_> = self
                .factory
                .info(instruction)
                .used_memory()
                .iter()
                .filter(|memory| {
                    matches!(
                        memory.access(),
                        OpAccess::Write
                            | OpAccess::CondWrite
                            | OpAccess::ReadWrite
                            | OpAccess::ReadCondWrite
                    )
                })
                .map(|memory| (memory.base(), memory.index(), memory.displacement()))
                .collect();
            if let Some(&(base, index, displacement)) = writes.first() {
                let stored = (writes.len() == 1
                    && base == Register::RSP
                    && index == Register::None
                    && instruction.mnemonic() == Mnemonic::Mov
                    && instruction.op0_kind() == OpKind::Memory
                    && displacement % 8 == 0)
                    .then(|| relocated(1).or_else(|| stored_value(instruction, &plain)))
                    .flatten()
                    .ok_or_else(|| format!("unproven memory write at {}", decoded.rva))?;
                let from_top = usize::try_from(displacement / 8)
                    .map_err(|_| format!("stack write offset at {}", decoded.rva))?;
                let position = self
                    .stack
                    .len()
                    .checked_sub(1)
                    .and_then(|top| top.checked_sub(from_top))
                    .ok_or_else(|| format!("stack write outside the frame at {}", decoded.rva))?;
                self.stack[position] = stored;
            }
        }
        update_provenance(&mut self.factory, instruction, &mut self.provenance);
        if defines_flags(instruction) {
            self.original_flags = false;
        }
        Ok(())
    }

    /// Marks the destination of a `mov r64, imm64` whose immediate a base
    /// relocation covers: its runtime value is the immediate plus the delta.
    /// With no parsed relocation table every such immediate is treated so.
    fn track_relocated(
        &mut self,
        decoded: &vmpx_disasm::DecodedInstruction,
        relocations: Option<&BTreeMap<u32, u32>>,
    ) {
        let instruction = &decoded.instruction;
        if instruction.mnemonic() != Mnemonic::Mov
            || instruction.op0_kind() != OpKind::Register
            || instruction.op1_kind() != OpKind::Immediate64
        {
            return;
        }
        let start = decoded.rva.0;
        let end = start.saturating_add(u32::try_from(decoded.bytes.len()).unwrap_or(u32::MAX));
        let covered = relocations.is_none_or(|slots| {
            slots
                .range(start.saturating_sub(7)..end)
                .any(|(slot, size)| *slot < end && start < slot.saturating_add(*size))
        });
        if covered {
            self.provenance.insert(
                canonical(instruction.op0_register()),
                Provenance::RelocatedConstant(instruction.immediate64()),
            );
        }
    }
}

/// Base-relocation slots as `rva -> width`, or `None` when the table cannot be
/// parsed.
fn relocation_slots(pe: &PeImage<'_>) -> Option<BTreeMap<u32, u32>> {
    let blocks = pe.parse_relocations().ok()?;
    let mut slots = BTreeMap::new();
    for block in &blocks {
        for entry in block.entries.iter().filter(|entry| entry.kind != 0) {
            let size = if entry.kind == 10 { 8 } else { 4 };
            slots.insert(block.page_rva.0 + u32::from(entry.offset), size);
        }
    }
    Some(slots)
}

/// Continues the stub past the VSP capture to the `mov rsp, reg` that moves the
/// native stack below VSP, returning the full frame (top first) and the
/// VSP-to-top offset. `None` (keep the captured frame) when anything in between
/// is not modelled, the pushes would place the top cell below VSP, the new stack
/// is not proven to lie at or below VSP, or the stub then writes any other memory.
fn extend_below_capture<'a>(
    state: &mut StubState,
    remaining: impl Iterator<Item = &'a vmpx_disasm::DecodedInstruction>,
    relocations: Option<&BTreeMap<u32, u32>>,
    vsp: Register,
    top_offset: u64,
) -> Option<(Vec<NativeFrameSource>, u64)> {
    let vsp = canonical(vsp);
    let before = state.stack.len();
    let mut at_or_below_vsp: BTreeSet<Register> = BTreeSet::new();
    let mut result = None;
    for decoded in remaining {
        let instruction = &decoded.instruction;
        let mut writes_memory = || {
            state
                .factory
                .info(instruction)
                .used_memory()
                .iter()
                .any(|memory| {
                    matches!(
                        memory.access(),
                        OpAccess::Write
                            | OpAccess::CondWrite
                            | OpAccess::ReadWrite
                            | OpAccess::ReadCondWrite
                    )
                })
        };
        if result.is_some() {
            // Past the redirect every write lands at or below the new stack
            // pointer, hence below the frame; nothing else may write memory.
            let push = matches!(instruction.mnemonic(), Mnemonic::Push | Mnemonic::Pushfq);
            let flow = matches!(
                instruction.flow_control(),
                FlowControl::Next | FlowControl::UnconditionalBranch | FlowControl::Return
            );
            if !flow || (!push && writes_memory()) {
                return None;
            }
            continue;
        }
        if instruction.mnemonic() == Mnemonic::Mov
            && instruction.op0_kind() == OpKind::Register
            && instruction.op1_kind() == OpKind::Register
            && instruction.op0_register() == Register::RSP
            && instruction.op1_register().size() == 8
        {
            if !at_or_below_vsp.contains(&canonical(instruction.op1_register())) {
                return None;
            }
            let pushed = u64::try_from(state.stack.len() - before).ok()?;
            let new_top = top_offset.checked_sub(pushed.checked_mul(8)?)?;
            let mut cells = state.stack.clone();
            cells.reverse();
            result = Some((cells, new_top));
            continue;
        }
        if exposes_stack_pointer(instruction) {
            return None;
        }
        let writes = |state: &mut StubState, register: Register| {
            crate::dispatch_extract::writes_register(&mut state.factory, instruction, register)
        };
        if !matches!(instruction.mnemonic(), Mnemonic::Push | Mnemonic::Pushfq)
            && writes(state, Register::RSP)
        {
            return None;
        }
        if writes(state, vsp) {
            return None;
        }
        // `and reg, imm` never increases the unsigned value, so an address at or
        // below VSP stays so (stack alignment).
        let narrowing_and = instruction.mnemonic() == Mnemonic::And
            && instruction.op0_kind() == OpKind::Register
            && crate::dispatch_extract::signed_immediate(instruction, 1).is_some();
        at_or_below_vsp.retain(|register| {
            (narrowing_and && canonical(instruction.op0_register()) == *register)
                || !writes(state, *register)
        });
        if instruction.mnemonic() == Mnemonic::Lea
            && instruction.op0_kind() == OpKind::Register
            && instruction.op0_register().size() == 8
            && canonical(instruction.memory_base()) == vsp
            && instruction.memory_index() == Register::None
            && instruction.memory_displacement64().cast_signed() <= 0
        {
            at_or_below_vsp.insert(canonical(instruction.op0_register()));
        }
        state.step(decoded, true).ok()?;
        state.track_relocated(decoded, relocations);
    }
    result
}

fn stored_value(
    instruction: &iced_x86::Instruction,
    provenance: &BTreeMap<Register, Provenance>,
) -> Option<NativeFrameSource> {
    match instruction.op1_kind() {
        OpKind::Register if instruction.op1_register().size() == 8 => {
            match provenance.get(&canonical(instruction.op1_register()))? {
                Provenance::Native(register) => Some(NativeFrameSource::Register {
                    register: *register,
                }),
                Provenance::Constant(value) | Provenance::RelocatedConstant(value) => {
                    Some(NativeFrameSource::Constant { value: *value })
                }
                Provenance::Unknown => Some(NativeFrameSource::Unknown {
                    description: "stored register of unproven value".to_owned(),
                }),
            }
        }
        _ if instruction.memory_size() == iced_x86::MemorySize::UInt64 => {
            crate::dispatch_extract::signed_immediate(instruction, 1).map(|value| {
                NativeFrameSource::Constant {
                    value: value.cast_unsigned(),
                }
            })
        }
        _ => None,
    }
}

fn frame_initial_state(
    cells: &[NativeFrameSource],
    top_offset: u64,
) -> (SymbolicVmInitialState, usize) {
    let mut initial = SymbolicVmInitialState::default();
    let base = Expr::register(BitWidth::W64, "vsp_initial");
    let mut known = 0;
    for (index, cell) in cells.iter().enumerate() {
        let Some(value) = cell.symbolic_value() else {
            continue;
        };
        let offset = top_offset.wrapping_add((index as u64).wrapping_mul(8));
        let Ok(address) = Expr::binary(
            BinaryOp::Add,
            base.clone(),
            Expr::constant(BitWidth::W64, offset),
        ) else {
            continue;
        };
        initial.memory_values.push((address, value));
        known += 1;
    }
    (initial, known)
}

struct DecodedBlock {
    /// Shared, not owned: `visit` needs the decoded buffer while it mutates the
    /// walker, and it took a full copy of it on EVERY re-visit of the block to
    /// release the borrow. Up to `max_visits_per_block` copies of a buffer that
    /// can hold `instruction_limit` instructions, per block, for nothing.
    instructions: Rc<[DecodedVmInstruction]>,
    final_state: RelativeDispatchState,
    stop: LinearDecodeStop,
    boundary: Option<VmHandlerSemantic>,
    error: Option<String>,
}

fn bytecode_section(pe: &PeImage<'_>, vip: Va) -> Option<usize> {
    let rva = pe.optional.image_base.checked_rva_of(vip)?;
    pe.sections
        .iter()
        .find(|section| section.contains_mapped_rva(rva))
        .map(|section| section.index)
}

/// Why a branch target that left the entry's bytecode section cannot be followed
/// as bytecode. VMProtect fetches bytecode from the executable protector section
/// that holds the entry stream, so a non-executable target is data (never
/// bytecode) and a different executable section is one this walk has not proven
/// holds this VM's bytecode. Either way the target is refused, not guessed;
/// `target_characteristics` is the section's flags, or `None` when unmapped.
fn out_of_bytecode_section_reason(target_characteristics: Option<u32>) -> &'static str {
    const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
    match target_characteristics {
        None => "it is not mapped to any section",
        Some(flags) if flags & IMAGE_SCN_MEM_EXECUTE == 0 => "it is in a non-executable data section",
        Some(_) => "it is in a different executable section not proven to hold this VM's bytecode",
    }
}

/// The observed branch targets for `site` that may be FOLLOWED as real
/// bytecode: each must land in the entry's proven bytecode `section` (the same
/// membership the sound branch path requires of every target). Observation
/// seeds a resolution; it can never turn a data address into code, so a target
/// in a data/different section or outside every section is dropped here. Empty
/// when there is no observation for the site or none validate — the caller then
/// keeps its exact fail-closed stall. Default/sound path: `observed` is empty,
/// so this is always empty.
fn followable_observed_targets(
    pe: &PeImage<'_>,
    section: Option<usize>,
    observed: &crate::ObservedBranchTargets,
    site: Va,
) -> Vec<Va> {
    let targets = observed.targets_for(site.get());
    // Ambiguous evidence (any target failing validation) is refused whole:
    // following only the valid subset would guess which arm is real.
    let all_valid = section.is_some()
        && targets
            .iter()
            .all(|&value| bytecode_section(pe, Va(value)) == section);
    if !all_valid {
        return Vec::new();
    }
    targets.iter().map(|&value| Va(value)).collect()
}

const NESTED_ENTRY_CONFIDENCE: u8 = 60;

struct Item {
    /// Bytes charged to the walker's live estimate while this item is queued.
    bytes: usize,
    start: RelativeDispatchState,
    address: Option<Va>,
    symbolic: SymbolicVmInitialState,
    path: Option<Rc<PathNode>>,
    /// Profile to walk this block under; `None` is the base profile. A block
    /// reached through a proven cross-profile transition carries the new
    /// profile, which then propagates to its successors.
    profile: Option<Rc<ProfileContext>>,
}

/// A block on the current walk path with the symbolic state it was entered in.
struct PathNode {
    key: Va,
    state: SymbolicVmInitialState,
    parent: Option<Rc<PathNode>>,
}

const HAVOC_PREFIX: &str = "loop_havoc_";

fn is_havoc(expr: &Expr) -> bool {
    matches!(&expr.kind, ExprKind::Register { name } if name.starts_with(HAVOC_PREFIX))
}

fn has_havoc(expr: &Expr) -> bool {
    expr.dependencies()
        .iter()
        .any(|name| name.starts_with(HAVOC_PREFIX))
}

/// The bits a loop-carried generalized unknown may set: the full width for a
/// bare havoc register (TOP), or the mask for a havoc register narrowed to a
/// proven set of bits (`loop_havoc_… & m`). `None` for anything that is not a
/// generalized unknown. The masked form carries a PROVEN-narrower domain for a
/// loop-carried value through the widening so a dependent table index resolves.
fn havoc_active_mask(expr: &Expr) -> Option<u64> {
    if is_havoc(expr) {
        return Some(expr.width.mask());
    }
    match &expr.kind {
        ExprKind::Binary {
            op: BinaryOp::And,
            left,
            right,
        } => match (&left.kind, &right.kind) {
            (ExprKind::Register { name }, ExprKind::Const { value })
            | (ExprKind::Const { value }, ExprKind::Register { name })
                if name.starts_with(HAVOC_PREFIX) =>
            {
                Some(*value & expr.width.mask())
            }
            _ => None,
        },
        _ => None,
    }
}

/// The bits `expr` can provably set within its width (its reachable one-bits),
/// from static value-range analysis with no path facts. A sound
/// over-approximation: every reachable value is a submask of this. Masks,
/// shifts, extracts and small intervals prove bits zero; an opaque register or
/// memory read proves nothing and returns the full width.
fn reachable_ones(expr: &Expr) -> u64 {
    expr.width.mask() & !value_range(expr).known_zero()
}

/// `a` holds every value `r` can take at this location. A generalized unknown
/// covers any value whose reachable bits are a subset of the unknown's proven
/// bits (a bare-TOP unknown covers everything); any other cell covers only an
/// identical expression free of generalized variables, whose meaning differs
/// between iterations.
fn covers(a: Option<&Expr>, r: Option<&Expr>) -> bool {
    match a {
        None => r.is_none(),
        Some(a) => match havoc_active_mask(a) {
            Some(mask) => match r {
                None => true,
                Some(r) => a.width == r.width && reachable_ones(r) & !mask == 0,
            },
            None => matches!(r, Some(r) if a == r && !has_havoc(a)),
        },
    }
}

/// Widens a differing loop-carried cell to a canonical unknown named for its
/// `(block, cell)` site — never a fresh name per pass. When both sides are
/// provably confined to fewer bits than the full width (a value masked by a
/// constant, a byte assigned only from a small set, a bounded accumulator), the
/// unknown carries that proven bit-mask (`loop_havoc_… & m`) instead of going to
/// full-range TOP, so a dependent multi-byte table index enumerates instead of
/// blowing its domain cross-product. The mask is the union of each side's
/// reachable bits, so it over-approximates every iteration (sound); when it
/// cannot be proven narrower it stays the full-range bare register (fail-safe).
///
/// Idempotent: the mask grows monotonically toward the full width and the
/// register name is fixed by site, so a re-widened state whose re-entry stays
/// within the carried mask equals the prior one and the loop closes (abstract-
/// interpretation widening to a fixpoint). Distinct sites get distinct names, so
/// independent unknowns are never aliased.
fn generalize(
    a: Option<&Expr>,
    r: Option<&Expr>,
    key: Va,
    cell: &str,
    changed: &mut bool,
) -> Result<Option<Expr>, String> {
    if covers(a, r) {
        return Ok(a.cloned());
    }
    let width = match (a, r) {
        (Some(a), Some(r)) if a.width != r.width => {
            return Err("a loop-carried cell changes width".to_owned());
        }
        (Some(cell), _) | (None, Some(cell)) => cell.width,
        (None, None) => return Ok(None),
    };
    *changed = true;
    let register = Expr::register(width, format!("{HAVOC_PREFIX}{:x}_{cell}", key.0));
    // A cell already carrying a mask that failed to cover the re-entry: the mask
    // is not an invariant, so go straight to full-range TOP rather than growing
    // it a bit per revisit (which would exhaust the visit budget before closing).
    if a.is_some_and(|a| havoc_active_mask(a).is_some()) {
        return Ok(Some(register));
    }
    // Otherwise carry the proven mask (union of both sides' reachable bits) only
    // when it is a narrow (<=16-bit) cell confined to 1..=4 live bits — a <=16
    // value domain, the index-byte case that shrinks the table cross-product.
    // Wider or all-zero masks barely narrow the byte yet make every loop state
    // distinct from bare TOP, only churning visits, so they keep full-range TOP
    // (fail-safe, and preserves the closing behaviour of converging loops).
    let active = a.map_or(0, reachable_ones) | r.map_or(0, reachable_ones);
    if width.bits() > 16 || !(1..=4).contains(&active.count_ones()) {
        return Ok(Some(register));
    }
    Expr::binary(BinaryOp::And, register, Expr::constant(width, active))
        .map(Some)
        .map_err(|error| error.to_string())
}

/// A short, stable token for a memory site (its address), used only to name the
/// site's canonical unknown. Collisions across distinct addresses are harmless
/// precision loss; it never aliases a memory cell with flags or a context slot
/// (those names carry a different prefix).
fn cell_token(address: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    address.hash(&mut hasher);
    format!("m{:x}", hasher.finish())
}

/// Short class of a symbolic-execution failure, so the stall histogram separates
/// the distinct underlying errors the one stall prefix would otherwise merge.
fn symbolic_failure_class(message: &str) -> &'static str {
    if message.contains("null memory read") {
        "null-read"
    } else if message.contains("exceed") {
        "resource-limit"
    } else if message.contains("not implemented") || message.contains("not modelled") {
        "unmodelled"
    } else {
        "other"
    }
}

/// A memory site's identity: its address text, or its structural hash once the
/// text of a heavily shared address DAG outgrows `SITE_KEY_BYTES`.
fn site_key(address: &Expr) -> String {
    use std::hash::BuildHasher;
    // MEASURED: lowering this to 8 KiB changed peak RSS not at all (12.26 GB vs
    // 11.29 GB on entry 0xc43d80e at an 8192 instruction limit), so the key
    // RENDERING is not what holds the memory -- `bounded_debug` already streams
    // into a bounded writer and stops. Left at 4 MiB rather than carrying a
    // behaviour change (more hash collisions merging sites) that bought nothing.
    const SITE_KEY_BYTES: usize = 4 << 20;
    crate::semantic_exec::bounded_debug(address, SITE_KEY_BYTES)
        .unwrap_or_else(|| format!("#{:016x}", crate::expr::FxBuild::default().hash_one(address)))
}

/// One memory site of a widened pair: indices of its cell in `a` and `r` (the
/// last cell wins on a repeated address) and the site's canonical-unknown token.
type MemorySite = (Option<usize>, Option<usize>, String);

/// The widened memory sites of `a` and `r` in rendered-address order. A site's
/// identity is its address's `Debug` text, which for shared address DAGs runs to
/// hundreds of KB, so the union is memoized on the two address sequences (the
/// loop fixpoints re-widen the same sites over and over).
fn memory_union(a: &SymbolicVmInitialState, r: &SymbolicVmInitialState) -> Rc<Vec<MemorySite>> {

    type Entry = (Vec<Expr>, Vec<Expr>, Rc<Vec<MemorySite>>);
    thread_local! {
        static UNIONS: std::cell::RefCell<Vec<Entry>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    const CACHED_UNIONS: usize = 32;
    let same = |cached: &[Expr], state: &SymbolicVmInitialState| {
        cached.len() == state.memory_values.len()
            && cached.iter().zip(&state.memory_values).all(|(c, (address, _))| c == address)
    };
    if let Some(hit) = UNIONS.with(|cache| {
        cache
            .borrow()
            .iter()
            .find(|(ka, kr, _)| same(ka, a) && same(kr, r))
            .map(|(_, _, union)| Rc::clone(union))
    }) {
        return hit;
    }
    let by_address = |state: &SymbolicVmInitialState| -> std::collections::HashMap<String, usize, crate::expr::FxBuild> {
        state
            .memory_values
            .iter()
            .enumerate()
            .map(|(index, entry)| (site_key(&entry.0), index))
            .collect()
    };
    let (a_memory, r_memory) = (by_address(a), by_address(r));
    let mut addresses: Vec<&String> = a_memory.keys().chain(r_memory.keys()).collect();
    addresses.sort_unstable();
    addresses.dedup();
    let union = Rc::new(
        addresses
            .into_iter()
            .map(|address| (a_memory.get(address).copied(), r_memory.get(address).copied(), cell_token(address)))
            .collect::<Vec<_>>(),
    );
    UNIONS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() >= CACHED_UNIONS {
            cache.remove(0);
        }
        cache.push((
            a.memory_values.iter().map(|(address, _)| address.clone()).collect(),
            r.memory_values.iter().map(|(address, _)| address.clone()).collect(),
            Rc::clone(&union),
        ));
    });
    union
}

/// The state covering both `a` and `r`, or `None` when `a` already covers `r`.
/// Cells that differ become fresh unknowns; the stack pointer, allocations and
/// memory precision must be loop-invariant.
fn widen_state(
    key: Va,
    a: &SymbolicVmInitialState,
    r: &SymbolicVmInitialState,
) -> Result<Option<SymbolicVmInitialState>, String> {
    if a.memory_tainted || r.memory_tainted {
        return Err("loop follows an imprecise external-memory store".to_owned());
    }
    if a.vsp != r.vsp {
        return Err("loop changes the VM stack pointer".to_owned());
    }
    if a.fresh_allocations != r.fresh_allocations {
        return Err("loop changes the fresh allocations".to_owned());
    }
    let mut changed = (r.image_reads_suppressed && !a.image_reads_suppressed)
        || (r.protection_may_have_changed && !a.protection_may_have_changed);
    let flags = generalize(Some(&a.flags), Some(&r.flags), key, "f", &mut changed)?
        .unwrap_or_else(|| a.flags.clone());
    let mut context_values = BTreeMap::new();
    let offsets: BTreeSet<i64> = a
        .context_values
        .keys()
        .chain(r.context_values.keys())
        .copied()
        .collect();
    for offset in offsets {
        let cell = generalize(
            a.context_values.get(&offset),
            r.context_values.get(&offset),
            key,
            &format!("c{offset}"),
            &mut changed,
        )?;
        if let Some(cell) = cell {
            context_values.insert(offset, cell);
        }
    }
    let mut memory_values = Vec::new();
    for (a_index, r_index, token) in memory_union(a, r).iter() {
        let a_entry = a_index.map(|index| &a.memory_values[index]);
        let r_entry = r_index.map(|index| &r.memory_values[index]);
        // A cell the re-entry tracks and the enclosing state does not: DROPPING
        // it is a widening, not a loss of soundness. An untracked, symbolically
        // addressed byte reads back as a fresh `memory_load_epoch_*` unknown
        // (`load_memory`'s symbolic miss path), and an unconstrained byte covers
        // every concrete value the re-entry's cell could hold, so the enclosing
        // state's denotation is the larger one — which is exactly what coverage
        // asks.
        //
        // This is what lets such a loop reach a fixpoint at all. Measured on
        // systems.xpl it is the largest single family of incomplete entries: a
        // loop that pushes one 8-byte word per iteration and never pops it. The
        // shift translates the whole tracked frame, but the frame's DEEP end is
        // anchored at `vsp_initial` and does not move, so the re-entry always
        // holds 8 byte-cells the shifted ancestor lacks ("memory cell count 352
        // vs 360") and a per-cell model never converges. Summarizing just that
        // anchored tail leaves a frame of FIXED size, and the loop closes in one
        // step.
        //
        // NOT for a concrete address: there the miss path consults the static
        // image BEFORE minting an unknown, so an untracked concrete byte is
        // pinned to the image value rather than unconstrained. Dropping one
        // would claim coverage of a cell we would then re-read as a stale
        // constant. The reverse direction (tracked here, untracked in the
        // re-entry) is never dropped either: there it is the RE-ENTRY whose
        // denotation is larger, so coverage genuinely has to widen.
        //
        // NOT for a cell holding a FRAME-POINTER value. Dropping one hides it:
        // the state no longer holds that pointer anywhere, an untracked load
        // could later produce it, and frame privacy would have to be conceded --
        // which is exactly what `fold_hid_frame_pointer` is for. A drop that
        // leaves `changed` clear reports `Ok(None)`, "already covered", and that
        // check never runs, so such a cell is widened the ordinary way instead.
        if a_entry.is_none()
            && let Some((address, value)) = r_entry
            && !matches!(address.kind, ExprKind::Const { .. })
            && !crate::semantic_exec::is_frame_pointer_value(value)
        {
            continue;
        }
        let location = a_entry.or(r_entry).map(|entry| entry.0.clone());
        let cell = generalize(
            a_entry.map(|entry| &entry.1),
            r_entry.map(|entry| &entry.1),
            key,
            token,
            &mut changed,
        )?;
        if let (Some(location), Some(cell)) = (location, cell) {
            memory_values.push((location, cell));
        }
    }
    if !changed {
        return Ok(None);
    }
    Ok(Some(SymbolicVmInitialState {
        vsp: a.vsp.clone(),
        flags,
        context_values,
        memory_values,
        memory_epoch: a.memory_epoch.max(r.memory_epoch),
        memory_tainted: false,
        image_reads_suppressed: a.image_reads_suppressed || r.image_reads_suppressed,
        protection_may_have_changed: a.protection_may_have_changed
            || r.protection_may_have_changed,
        next_native_call_epoch: a.next_native_call_epoch.max(r.next_native_call_epoch),
        fresh_allocations: a.fresh_allocations.clone(),
    }))
}

/// `expr` rebuilt in a canonical affine form: its non-constant terms in a stable
/// order, each with its coefficient, then one folded constant.
///
/// `VMProtect` writes the same stack address as `((base + a) - b) + c` in one
/// block and `base + c'` in another. Those are semantically identical and
/// structurally different, and `simplify` does not reassociate them —
/// [`affine_vsp_parts`] exists precisely because it does not. Every state
/// comparison in this module is structural (`Expr` is hash-consed, so `==` is a
/// pointer test), so without this the same address in two forms reads as two
/// different addresses and no fold, widen or back edge can see through it.
///
/// Semantics are preserved exactly: the result is the same wrapping sum of the
/// same terms at the same width, just written one way. Anything that is not a
/// pure Add/Sub/Neg/Const chain stays an opaque term with coefficient +-1, so a
/// non-affine expression is returned unchanged rather than approximated. Any
/// rebuild failure also returns the original.
fn canonical_affine(expr: &Expr) -> Expr {
    fn collect(expr: &Expr, positive: bool, terms: &mut Vec<(Expr, i64)>, constant: &mut u64) {
        match &expr.kind {
            ExprKind::Binary { op: BinaryOp::Add, left, right } => {
                collect(left, positive, terms, constant);
                collect(right, positive, terms, constant);
            }
            ExprKind::Binary { op: BinaryOp::Sub, left, right } => {
                collect(left, positive, terms, constant);
                collect(right, !positive, terms, constant);
            }
            ExprKind::Unary { op: UnaryOp::Neg, value } => collect(value, !positive, terms, constant),
            ExprKind::Const { value } => {
                *constant = if positive {
                    constant.wrapping_add(*value)
                } else {
                    constant.wrapping_sub(*value)
                };
            }
            _ => {
                let step = if positive { 1 } else { -1 };
                match terms.iter_mut().find(|(held, _)| held == expr) {
                    Some((_, coefficient)) => *coefficient += step,
                    None => terms.push((expr.clone(), step)),
                }
            }
        }
    }

    let width = expr.width;
    let mut terms: Vec<(Expr, i64)> = Vec::new();
    let mut constant = 0_u64;
    collect(&expr.simplify(), true, &mut terms, &mut constant);
    terms.retain(|(_, coefficient)| *coefficient != 0);
    // A stable order independent of how the expression was written. Ordering on
    // the interned hash instead measured +0 on the entry it was meant to help,
    // and it changes the canonical form, so the Debug text stays: a behavioural
    // change with no measured benefit is not worth the risk to N1.
    terms.sort_by_cached_key(|(term, _)| format!("{term:?}"));

    let mut accumulated: Option<Expr> = None;
    for (term, coefficient) in terms {
        let scaled = match coefficient {
            1 => term,
            -1 => Expr::unary(UnaryOp::Neg, term),
            other => {
                let Ok(scaled) = Expr::binary(
                    BinaryOp::Mul,
                    term,
                    Expr::constant(width, other as u64),
                ) else {
                    return expr.clone();
                };
                scaled
            }
        };
        accumulated = Some(match accumulated {
            None => scaled,
            Some(sum) => {
                let Ok(sum) = Expr::binary(BinaryOp::Add, sum, scaled) else {
                    return expr.clone();
                };
                sum
            }
        });
    }
    match (accumulated, constant) {
        (None, constant) => Expr::constant(width, constant),
        (Some(sum), 0) => sum,
        (Some(sum), constant) => Expr::binary(BinaryOp::Add, sum, Expr::constant(width, constant))
            .unwrap_or_else(|_| expr.clone()),
    }
}

/// Flattens an affine VSP expression into its non-constant terms (keyed by
/// stable debug text, carrying a signed coefficient) and the summed constant
/// offset. Add/Sub/Neg are walked with sign so the `((base + a) - b) + c`
/// nesting `VMProtect` emits for a push-down frame reduces to `{base: +1}` with a
/// single constant — the reduction `simplify` alone does not perform on `Sub`.
/// Any other node (a multiply, a load, a flag intrinsic) stays an opaque term,
/// so a non-affine VSP step cannot be mistaken for a constant adjustment.
/// Decomposes a VSP expression into non-constant terms and a constant offset.
///
/// Terms are keyed by the EXPRESSION, not by its rendered Debug text. `Expr`
/// hashes and compares through interned `ExprRef`s, both O(1), so this keys on
/// node identity instead of building a String per term — and `VMProtect` builds
/// the VSP out of an opaque `Add(Not(Or(..)), k)`, so those Strings were large.
/// The relation is the same either way (Debug rendering is faithful), so this is
/// semantics-preserving. It did NOT turn out to be the cost on the entry that
/// motivated it — 50 s per `generalize_across` call and 13 GB of transient heap
/// survive it, and that cost is still unlocated.
fn affine_vsp_parts(expr: &Expr) -> (HashMap<Expr, i64>, u64) {
    fn collect(expr: &Expr, positive: bool, terms: &mut HashMap<Expr, i64>, constant: &mut u64) {
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                collect(left, positive, terms, constant);
                collect(right, positive, terms, constant);
            }
            ExprKind::Binary {
                op: BinaryOp::Sub,
                left,
                right,
            } => {
                collect(left, positive, terms, constant);
                collect(right, !positive, terms, constant);
            }
            ExprKind::Unary {
                op: UnaryOp::Neg,
                value,
            } => collect(value, !positive, terms, constant),
            // Wrapping two's-complement sum, so the constant offset is exact
            // whether the frame grows up or down.
            ExprKind::Const { value } => {
                *constant = if positive {
                    constant.wrapping_add(*value)
                } else {
                    constant.wrapping_sub(*value)
                };
            }
            _ => {
                let coefficient = terms.entry(expr.clone()).or_insert(0);
                *coefficient += if positive { 1 } else { -1 };
            }
        }
    }
    let mut terms = HashMap::new();
    let mut constant = 0_u64;
    collect(&expr.simplify(), true, &mut terms, &mut constant);
    terms.retain(|_, coefficient| *coefficient != 0);
    (terms, constant)
}

/// The proven constant byte delta `b - a` between two VSP expressions, or `None`
/// when their non-constant parts differ (so the step is not a pure constant
/// adjustment of the same base). Width mismatch is also rejected.
fn constant_vsp_delta(a: &Expr, b: &Expr) -> Option<u64> {
    if a.width != b.width {
        return None;
    }
    let (a_terms, a_const) = affine_vsp_parts(a);
    let (b_terms, b_const) = affine_vsp_parts(b);
    (a_terms == b_terms).then(|| b_const.wrapping_sub(a_const))
}

/// The single PROVEN-CONSTANT per-iteration VSP delta of a recursion-as-loop, or
/// `None` when the delta is not a constant that is identical across every
/// observed iteration (ancestors nearest-first, then the re-entry). A zero delta
/// is rejected: the VSP is then loop-invariant, which is not this stall. This is
/// the soundness guard for the bounded unroll — without a proven constant delta
/// the loop is never unrolled, even with an observed depth.
fn proven_constant_vsp_delta(
    ancestors: &[&SymbolicVmInitialState],
    reentry: &SymbolicVmInitialState,
) -> Option<u64> {
    // Oldest -> newest: reverse the nearest-first ancestors, then the re-entry.
    // Consecutive repeats are dropped: several iterations at one VM stack depth
    // are not a stride violation, and keeping them mixed a step of 0 into the
    // chain so a clean one-cell push-down read as non-constant.
    let mut chain: Vec<&Expr> = Vec::new();
    for vsp in ancestors
        .iter()
        .rev()
        .map(|state| &state.vsp)
        .chain(std::iter::once(&reentry.vsp))
    {
        if chain.last() != Some(&vsp) {
            chain.push(vsp);
        }
    }
    let mut delta: Option<u64> = None;
    for pair in chain.windows(2) {
        let step = constant_vsp_delta(pair[0], pair[1])?;
        match delta {
            None => delta = Some(step),
            Some(previous) if previous != step => return None,
            Some(_) => {}
        }
    }
    delta.filter(|value| *value != 0)
}

#[derive(Debug, Eq, PartialEq)]
enum LoopStep {
    /// An enclosing entry state covers the re-entry state: a back edge.
    Close,
    /// An enclosing entry state IS the re-entry state with the VM stack base
    /// shifted by one constant, so the back edge closes — but the VM stack grows
    /// per level without a bound this walk can see, so the walker must concede
    /// frame privacy. Every bound it measured on the frame extent came from the
    /// finitely many stores it happened to observe, and an unbounded push-down
    /// writes below all of them; letting the proven tier discharge the
    /// external-store assumption on such an entry is exactly the error this
    /// variant exists to prevent. See [`closes_under_vm_stack_shift`].
    CloseShifted,
    /// The shifted enclosing state does NOT yet cover the re-entry, so continue
    /// from this generalization of the two. [`LoopStep::CloseShifted`] is a
    /// one-shot coverage test, which left a VSP-striding loop with no widening
    /// iteration at all: whatever failed to be covered on the first try — in the
    /// measured cases the flag word — refused the whole entry, even though the
    /// very next round would have carried a TOP that covers it. This is to
    /// `CloseShifted` what [`LoopStep::Widen`] is to [`LoopStep::Close`], and it
    /// concedes frame privacy for exactly the same reason `CloseShifted` does.
    WidenShifted(SymbolicVmInitialState),
    /// Still within the exact-unroll budget.
    Unroll,
    /// Continue from this generalization of every enclosing entry at once.
    Widen(SymbolicVmInitialState),
    Refuse(String),
    /// Bounded recursion unroll (experimental, observed-tier): the loop changes
    /// the VM stack pointer by a PROVEN CONSTANT delta and an observed depth
    /// bounds it, so this iteration is exact-unrolled past the normal budget
    /// rather than widened to a non-converging fixpoint.
    UnrollObserved,
    /// The observed recursion depth has been reached: the bounded unroll closes.
    CloseObserved,
}

/// Generalizes the nearest enclosing entry against every other enclosing entry
/// and the re-entry at once, so a cell that differs across *any* observed
/// iteration is widened to TOP in one step. Widening against only the nearest
/// neighbour lets a loop-carried cell that changes on a longer stride (an epoch
/// that only some iterations bump, a counter touched every few passes) escape
/// generalization for one visit at a time, so a state with many such cells never
/// reaches a fixpoint within the block-visit budget. Joining all observed states
/// caps the widening at the lattice height the exact unrolls already expose.
///
/// Sound: strictly more cells go to TOP than the nearest-only widen produced, so
/// the result still over-approximates every iteration (a TOP cell fails closed in
/// branch resolution and is never read as a value). `None` when the nearest entry
/// already covers all the others and the re-entry.
fn widen_to_fixpoint(
    key: Va,
    ancestors: &[&SymbolicVmInitialState],
    reentry: &SymbolicVmInitialState,
) -> Result<Option<SymbolicVmInitialState>, String> {
    let mut acc = ancestors[0].clone();
    let mut changed = false;
    let others = ancestors
        .iter()
        .skip(1)
        .copied()
        .chain(std::iter::once(reentry));
    for other in others {
        if let Some(widened) = widen_state(key, &acc, other)? {
            acc = widened;
            changed = true;
        }
    }
    Ok(changed.then_some(acc))
}

/// Folds `incoming` together with every state already seen at a block into one
/// over-approximation: a cell that differs anywhere goes to TOP, the rest stay
/// exact. Unlike [`widen_to_fixpoint`] this spans all recorded states (not a
/// single enclosing path), which converges a block re-reached through sibling
/// paths. `Err` when the states cannot be generalized (stack/allocation/taint
/// drift), which fails the re-entry closed.
fn fold_states<'s>(
    key: Va,
    states: impl IntoIterator<Item = &'s SymbolicVmInitialState>,
    incoming: &SymbolicVmInitialState,
    deadline: Option<Instant>,
) -> Result<SymbolicVmInitialState, String> {
    let mut acc = incoming.clone();
    for state in states {
        // The walk's own deadline bounds the gaps BETWEEN worklist items, not
        // the work inside one, and this fold is inside one. Measured on
        // systems.xpl at an 8192 instruction limit: a single `generalize_across`
        // ran 43s and took resident memory from 550MB to 13GB while every
        // counter the walker charges stayed flat (states 336->337,
        // interner_live 1542->1551, charged 249MB->250MB). Unbounded here is
        // what made raising the instruction limit an out-of-memory risk rather
        // than a slower sweep.
        if deadline.is_some_and(|at| Instant::now() >= at) {
            return Err("generalization exceeded the per-entry wall-clock budget".to_owned());
        }
        if let Some(widened) = widen_state(key, &acc, state)? {
            acc = widened;
        }
    }
    Ok(acc)
}

/// True when the fold HID a frame pointer: some pre-fold cell held a
/// frame-pointer value that the folded state no longer holds at that address.
/// Such a pointer could later be produced by an untracked load, so frame privacy
/// really is conceded. When no frame-pointer cell was altered, privacy survives
/// the fold and the VM-stack assumption can still be discharged — conceding it
/// unconditionally kept 115 CFG-complete entries out of the proven tier.
fn fold_hid_frame_pointer<'s>(
    before: impl Iterator<Item = &'s SymbolicVmInitialState>,
    after: &SymbolicVmInitialState,
) -> bool {
    before.into_iter().any(|state| {
        state.memory_values.iter().any(|(address, value)| {
            crate::semantic_exec::is_frame_pointer_value(value)
                && !after
                    .memory_values
                    .iter()
                    .any(|(kept, held)| kept == address && held == value)
        })
    })
}

/// Outcome of scanning a block's phases shifted onto one re-entry's depth.
enum ShiftedPhases {
    /// Nothing here shifts, or the fold refused: the ordinary same-depth fold
    /// still gets to decide.
    None,
    /// A shifted phase already covers the re-entry, so the back edge closes.
    Covered,
    /// Not covered yet; carry on from this broadened state.
    Folded(SymbolicVmInitialState),
}

/// Outcome of generalizing a re-entry against a block's accumulated states.
#[derive(Debug, Eq, PartialEq)]
enum GeneralizeAcross {
    /// The running over-approximation already covers this re-entry: it closes.
    Closed,
    /// Continue exploring from this broadened state. `hid_frame_pointer` says
    /// whether the fold discarded a frame-pointer-bearing cell.
    Continue {
        state: SymbolicVmInitialState,
        hid_frame_pointer: bool,
    },
    /// The states cannot be generalized soundly: fail closed, carrying the
    /// refusal reason so the census can tell a drifting VM stack pointer apart
    /// from every other reason a fold is refused.
    Stall(String),
}

/// True when `reentry` is exactly `ancestor` with the VM stack base shifted by
/// one constant: the same abstract state, re-reached some cells deeper.
///
/// A recursion that `VMProtect` inlines re-enters its loop head at a new VM stack
/// depth every level, and `widen_state` refuses any pair whose VSPs differ, so
/// the loop never closed and 110+ systems.xpl entries stopped here. But a state
/// that is EXACTLY the enclosing one under the uniform renaming
/// `vsp_initial := vsp_initial + delta` is bisimilar to it: a block's identity is
/// a code address, which the renaming does not touch, so both re-entries reach
/// the same blocks by the same edges, and an unresolved target stalls in both.
/// That lets the back edge close, which is all CFG completeness asks for — a back
/// edge is a path to an exit, and no trip count is involved. The VM stack still
/// grows per level at runtime; a loop in the recovered graph is the correct
/// description of that, not an approximation of it.
///
/// Deliberately an EQUALITY test and not a widening seed. Widening a shifted
/// state against an unshifted one would have to cover both, and any part of the
/// state that does NOT move with the stack would make the renamed state wrong
/// rather than conservative. Equality needs no such judgement: either every part
/// shifted by the same delta — in which case the renaming is the other state, not
/// an approximation of it — or this returns false and the refusal stands. It is
/// self-validating for the same reason: if the VSP is not rooted at
/// `vsp_initial`, the substitution leaves it alone and the equality simply fails.
fn closes_under_vm_stack_shift(
    key: Va,
    ancestor: &SymbolicVmInitialState,
    reentry: &SymbolicVmInitialState,
) -> bool {
    // Coverage, not equality, and by the SAME test the non-shifted case at the
    // top of `loop_step` uses: `widen_state(..) == Ok(None)` is this module's own
    // "the enclosing state already covers this re-entry". Equality was too
    // strict — measured, the shifted ancestor's flags are often already a
    // `loop_havoc_*` TOP from an earlier generalization while the re-entry
    // carries a precise value, and TOP covers precise. Everything new here is
    // the shift; the soundness notion is unchanged.
    shift_ancestor_onto_reentry(ancestor, reentry).is_some_and(|shifted| {
        matches!(
            widen_state(key, &canonicalized(&shifted), &canonicalized(reentry)),
            Ok(None)
        )
    })
}

/// Every expression in `state` put through [`canonical_affine`]. Substituting the
/// stack base for `base + delta` leaves nested sums like `((base + delta) + c)`
/// that are semantically the re-entry's `base + c'` but not structurally equal to
/// it, and structural equality is all `==` on a state can do. Memory cells are
/// also sorted, because two states holding the same cells in a different order
/// are the same state. Interning makes the canonical forms share, so comparing
/// them is still a pointer test.
fn canonicalized(state: &SymbolicVmInitialState) -> SymbolicVmInitialState {
    let mut memory_values: Vec<(Expr, Expr)> = state
        .memory_values
        .iter()
        .map(|(address, value)| (canonical_affine(address), canonical_affine(value)))
        .collect();
    // `site_key`, not `format!("{address:?}")`: Debug prints a shared DAG as a
    // TREE, so one heavily shared address renders to gigabytes and this sort key
    // was unbounded. Measured on entry 0xc43d80e at an 8192 instruction limit,
    // this single line was 40.8s of a 40.9s `generalize_across` and took
    // resident memory past 13GB, while every counter the walker charges stayed
    // flat -- which is why the per-entry memory cap never fired.
    //
    // `site_key` is the same bounded rendering the memory-union keys already
    // use: a bounded prefix, falling back to a structural hash when the address
    // outgrows it. The order stays deterministic, which is all a canonical form
    // needs.
    memory_values.sort_by_cached_key(|(address, _)| site_key(address));
    SymbolicVmInitialState {
        vsp: canonical_affine(&state.vsp),
        flags: canonical_affine(&state.flags),
        context_values: state
            .context_values
            .iter()
            .map(|(slot, value)| (*slot, canonical_affine(value)))
            .collect(),
        memory_values,
        fresh_allocations: state
            .fresh_allocations
            .iter()
            .map(|allocation| crate::semantic_exec::SymbolicFreshAllocation {
                allocation: canonical_affine(&allocation.allocation),
                size: canonical_affine(&allocation.size),
                zero_initialized: allocation.zero_initialized,
            })
            .collect(),
        ..state.clone()
    }
}

/// Adds `delta` to every stack-relative expression in `expr`: any subexpression
/// whose affine decomposition carries exactly `base_terms` as its non-constant
/// part is `base + k`, so it becomes `base + k + delta`.
///
/// The shift CANNOT be done by substituting the base register. `VMProtect` builds
/// the VM stack pointer out of obfuscated bitwise arithmetic — measured on these
/// entries it is `Add(Not(Or(..)), k)` — so the base is one opaque term with the
/// register buried inside it. Substituting the register rewrites the inside of
/// that term and produces a term the re-entry never had; the whole opaque term
/// has to move as a unit instead.
///
/// Top-down deliberately: a node that matches has its constant shifted and is
/// NOT descended into, because its own base term must stay byte-identical.
/// Descending as well would shift the base inside it and then shift the sum
/// again, for 2*delta. Nodes that do not match are rebuilt from shifted
/// children, so a stack address nested inside a load still moves.
/// Shared across every cell of one state: `Expr` hashes and compares through
/// interned `ExprRef`s, both O(1), so keying on the expression is keying on node
/// identity.
type ShiftMemo = std::collections::HashMap<Expr, Expr>;

fn shift_stack_relative(expr: &Expr, base_terms: &HashMap<Expr, i64>, delta: u64) -> Expr {
    shift_stack_relative_memo(expr, base_terms, delta, &mut ShiftMemo::new(), &std::cell::Cell::new(SHIFT_NODE_BUDGET))
}

/// Memoized on node identity. Expressions here are a hash-consed DAG, and the
/// unmemoized walk descended into every *edge*, so a subterm shared k ways was
/// rebuilt k times — the DAG expanded into its tree. Measured on one entry's
/// `generalize_across`: eight calls of 60-80 s each holding 11 GB of transient
/// nodes, against a per-entry cap that never fired because none of it is
/// retained. The memo makes it linear in distinct nodes.
fn shift_stack_relative_memo(
    expr: &Expr,
    base_terms: &HashMap<Expr, i64>,
    delta: u64,
    memo: &mut ShiftMemo,
    budget: &std::cell::Cell<usize>,
) -> Expr {
    if let Some(done) = memo.get(expr) {
        return done.clone();
    }
    // Out of budget: hand back the node untouched and let the caller abandon
    // the whole shift. Returning a partially shifted state would be wrong, so
    // the exhausted flag (budget stuck at zero) is what the caller checks.
    if budget.get() == 0 {
        return expr.clone();
    }
    budget.set(budget.get() - 1);
    let result = shift_stack_relative_uncached(expr, base_terms, delta, memo, budget);
    memo.insert(expr.clone(), result.clone());
    result
}

fn shift_stack_relative_uncached(
    expr: &Expr,
    base_terms: &HashMap<Expr, i64>,
    delta: u64,
    memo: &mut ShiftMemo,
    budget: &std::cell::Cell<usize>,
) -> Expr {
    let (terms, constant) = affine_vsp_parts(expr);
    if !terms.is_empty() && terms == *base_terms {
        let shifted = constant.wrapping_add(delta);
        // Rebuild as (expr - constant) + shifted, which is the same sum with one
        // different constant, without touching the opaque base term.
        let base = Expr::binary(
            BinaryOp::Sub,
            expr.clone(),
            Expr::constant(expr.width, constant),
        );
        if let Ok(base) = base
            && let Ok(result) = Expr::binary(
                BinaryOp::Add,
                base,
                Expr::constant(expr.width, shifted),
            )
        {
            return result;
        }
        return expr.clone();
    }
    let rebuilt = match &expr.kind {
        ExprKind::Binary { op, left, right } => Expr::binary(
            *op,
            shift_stack_relative_memo(left, base_terms, delta, memo, budget),
            shift_stack_relative_memo(right, base_terms, delta, memo, budget),
        )
        .ok(),
        ExprKind::Unary { op, value } => {
            Some(Expr::unary(*op, shift_stack_relative_memo(value, base_terms, delta, memo, budget)))
        }
        ExprKind::Load { address } => Some(Expr::load(
            expr.width,
            shift_stack_relative_memo(address, base_terms, delta, memo, budget),
        )),
        ExprKind::Intrinsic { name, arguments } => Some(Expr::intrinsic(
            expr.width,
            name.clone(),
            arguments
                .iter()
                .map(|argument| shift_stack_relative_memo(argument, base_terms, delta, memo, budget))
                .collect::<Vec<_>>(),
        )),
        _ => None,
    };
    rebuilt.unwrap_or_else(|| expr.clone())
}

/// `ancestor` with its whole VM stack advanced by the constant stride that
/// separates it from `reentry`, or `None` when there is no such stride.
fn shift_ancestor_onto_reentry(
    ancestor: &SymbolicVmInitialState,
    reentry: &SymbolicVmInitialState,
) -> Option<SymbolicVmInitialState> {
    let delta = constant_vsp_delta(&ancestor.vsp, &reentry.vsp).filter(|d| *d != 0)?;
    let (base_terms, _) = affine_vsp_parts(&ancestor.vsp);
    if base_terms.is_empty() {
        return None;
    }
    // One memo for the whole state: its cells share subterms heavily, so a
    // per-expression memo would rebuild the same DAG once per cell.
    let memo = std::cell::RefCell::new(ShiftMemo::new());
    let budget = std::cell::Cell::new(SHIFT_NODE_BUDGET);
    let shift = |expr: &Expr| {
        shift_stack_relative_memo(expr, &base_terms, delta, &mut memo.borrow_mut(), &budget)
    };
    let shifted = SymbolicVmInitialState {
        vsp: shift(&ancestor.vsp),
        flags: shift(&ancestor.flags),
        context_values: ancestor
            .context_values
            .iter()
            .map(|(slot, value)| (*slot, shift(value)))
            .collect(),
        memory_values: ancestor
            .memory_values
            .iter()
            .map(|(address, value)| (shift(address), shift(value)))
            .collect(),
        fresh_allocations: ancestor
            .fresh_allocations
            .iter()
            .map(|allocation| crate::semantic_exec::SymbolicFreshAllocation {
                allocation: shift(&allocation.allocation),
                size: shift(&allocation.size),
                zero_initialized: allocation.zero_initialized,
            })
            .collect(),
        ..ancestor.clone()
    };
    // A shift that ran out of budget is partial, and a partially shifted state
    // is not a sound over-approximation of the ancestor, so it is discarded
    // rather than folded against.
    (budget.get() > 0).then_some(shifted)
}

/// Diagnostic: the first field in which `ancestor` shifted by its constant VM
/// stack delta still differs from `reentry`. Names WHY a push-down level is not
/// the renaming of the enclosing one, which is the only thing that stands
/// between this stall family and a closed back edge.
fn vm_stack_shift_mismatch(
    ancestor: &SymbolicVmInitialState,
    reentry: &SymbolicVmInitialState,
) -> String {
    if constant_vsp_delta(&ancestor.vsp, &reentry.vsp).filter(|d| *d != 0).is_none() {
        return "no constant stride".to_owned();
    }
    let (base_terms, _) = affine_vsp_parts(&ancestor.vsp);
    if base_terms.is_empty() {
        return "vsp has no non-constant base term".to_owned();
    }
    let Some(a) = shift_ancestor_onto_reentry(ancestor, reentry).map(|s| canonicalized(&s)) else {
        return "substitution failed".to_owned();
    };
    let r = &canonicalized(reentry);
    // Report what the CLOSURE test decided, not a stricter one: an exact-equality
    // walk below names a field that coverage may well tolerate.
    match widen_state(Va(0), &a, r) {
        Ok(None) => return "nothing (the shift covers it)".to_owned(),
        Err(reason) => return format!("widen refuses: {reason}"),
        Ok(Some(_)) => {}
    }
    if a.vsp != r.vsp {
        let shown = |e: &Expr| format!("{e:?}").chars().take(150).collect::<String>();
        return format!(
            "shifted vsp still differs: shifted {} vs reentry {}",
            shown(&a.vsp),
            shown(&r.vsp)
        );
    }
    if !covers(Some(&a.flags), Some(&r.flags)) {
        let shown = |e: &Expr| format!("{e:?}").chars().take(100).collect::<String>();
        return format!("flags: shifted {} vs reentry {}", shown(&a.flags), shown(&r.flags));
    }
    // NOT compared: `widen_state` folds the epoch with `max` and never counts it
    // as a change, so reporting it here only buried the field that actually
    // blocks the fold. It cost two rounds of chasing the wrong thing.
    if a.next_native_call_epoch != r.next_native_call_epoch {
        return "native_call_epoch".to_owned();
    }
    if a.memory_tainted != r.memory_tainted
        || a.image_reads_suppressed != r.image_reads_suppressed
        || a.protection_may_have_changed != r.protection_may_have_changed
    {
        return "a precision flag".to_owned();
    }
    if a.fresh_allocations != r.fresh_allocations {
        return "fresh_allocations".to_owned();
    }
    if a.context_values.len() != r.context_values.len() {
        return format!(
            "context cell count {} vs {}",
            a.context_values.len(),
            r.context_values.len()
        );
    }
    if let Some(slot) = a
        .context_values
        .iter()
        .find(|(slot, value)| !covers(Some(value), r.context_values.get(slot)))
        .map(|(slot, _)| *slot)
    {
        return format!("context cell {slot:#x} not covered");
    }
    // NOT a count comparison. `widen_state` drops a symbolically addressed cell
    // the re-entry holds and this state does not -- that drop IS the widening --
    // so a differing count is no longer what blocks the fold, and reporting it
    // would bury the field that is, exactly as the epoch did before it. Only the
    // cells this state actually holds have to be covered.
    if let Some((address, _)) = a.memory_values.iter().find(|(address, value)| {
        !r.memory_values
            .iter()
            .any(|(held, got)| held == address && covers(Some(value), Some(got)))
    }) {
        let shown = format!("{address:?}").chars().take(90).collect::<String>();
        return format!("memory cell not covered at {shown}");
    }
    "nothing (would have closed)".to_owned()
}

/// Decides how a block re-entered while already on the path proceeds.
/// `ancestors` are the entry states of the enclosing visits, nearest first.
/// `observed_depth` is the recursion depth observed for this loop head on ONE
/// detonation, if any; it gates the experimental bounded-unroll bridge.
fn loop_step(
    key: Va,
    ancestors: &[&SymbolicVmInitialState],
    reentry: &SymbolicVmInitialState,
    unrolls: usize,
    observed_depth: Option<u64>,
) -> LoopStep {
    if ancestors
        .iter()
        .any(|a| matches!(widen_state(key, a, reentry), Ok(None)))
    {
        return LoopStep::Close;
    }
    if ancestors.len() < unrolls {
        return LoopStep::Unroll;
    }
    // Widen only within the re-entry's own VM-stack phase. The VSP is part of a
    // state's identity, not a dimension to join over: `widen_state` refuses any
    // pair whose VSPs differ, so one enclosing iteration sitting at a different
    // VM stack depth failed the whole re-entry closed even when the phase itself
    // converges. Measured across eight such entries: every refusal showed a
    // SINGLETON stride set of one cell (-8 or +8), never a growing
    // {8, 16, 24, ...}, so the VM stack oscillates between a bounded number of
    // depths rather than drifting. With no same-phase ancestor this is unchanged.
    let phase: Vec<&SymbolicVmInitialState> = ancestors
        .iter()
        .copied()
        .filter(|a| a.vsp == reentry.vsp)
        .collect();
    if phase.is_empty() {
        // No enclosing iteration sits at this VM stack depth, so there is nothing
        // to widen against and widening across depths only reproduces the VSP
        // refusal. Explore ONE more depth exactly instead, which assumes nothing
        // (an unroll is plain exploration) and still fails closed right after.
        //
        // Two distinct depths is the whole point: it is the least that lets
        // `proven_constant_vsp_delta` see a stride at all, and a loop that merely
        // oscillates would already have closed at the top of this function
        // against the ancestor it returned to. So reaching the refusal with two
        // depths and a constant stride is itself the proof that the VM stack
        // grows per iteration. Measured on eight such entries: the strides read
        // {-0x8, -0x10, -0x18, -0x20} from one re-entry, a clean one-cell
        // push-down recursion, which no sound widening closes without a trip
        // count — only the observed-depth bridge below can.
        let mut depths: Vec<&Expr> = Vec::new();
        for state in ancestors {
            if !depths.contains(&&state.vsp) {
                depths.push(&state.vsp);
            }
        }
        if depths.len() < 2 {
            return LoopStep::Unroll;
        }
    }
    let enclosing = if phase.is_empty() { ancestors } else { &phase };
    match widen_to_fixpoint(key, enclosing, reentry) {
        Ok(Some(widened)) => LoopStep::Widen(widened),
        Ok(None) => LoopStep::Close,
        Err(reason) => bounded_unroll_or_refuse(key, ancestors, reentry, observed_depth, reason),
    }
}

/// Diagnostic: appends the per-iteration VM-stack strides a refused loop shows,
/// and whether the stride is a PROVEN constant. A proven constant stride is the
/// only shape a sound stride summarization could ever close; an unproven or
/// mixed stride set cannot be closed without an assumption.
fn vsp_stride_note(
    ancestors: &[&SymbolicVmInitialState],
    reentry: &SymbolicVmInitialState,
    reason: String,
) -> String {
    let strides: BTreeSet<i64> = ancestors
        .iter()
        .filter(|a| a.vsp != reentry.vsp)
        .map(|a| constant_vsp_delta(&a.vsp, &reentry.vsp).map_or(i64::MIN, |d| d as i64))
        .collect();
    if strides.is_empty() {
        return reason;
    }
    let proven = proven_constant_vsp_delta(ancestors, reentry).is_some();
    // Why the nearest shifted ancestor is not the re-entry: see
    // `vm_stack_shift_mismatch`.
    let blocker = ancestors
        .iter()
        .find(|a| a.vsp != reentry.vsp)
        .map_or_else(|| "none".to_owned(), |a| vm_stack_shift_mismatch(a, reentry));
    if strides.contains(&i64::MIN) {
        format!("{reason} (vsp strides non-constant, proven={proven}, shift blocked by {blocker})")
    } else {
        format!("{reason} (vsp strides {strides:#x?}, proven={proven}, shift blocked by {blocker})")
    }
}

/// The experimental bounded-unroll bridge, reached only when the fixpoint widen
/// fails (e.g. "loop changes the VM stack pointer"). A VSP-changing recursion
/// whose per-iteration VSP delta is a PROVEN CONSTANT, bounded by an observed
/// recursion depth, unrolls exactly `depth` iterations instead of failing
/// closed. Without an observed depth, or when the delta is not a proven
/// constant, it stays the existing fail-closed refusal — no depth is ever
/// guessed, and the sound path (no observed depth) is never reached here.
fn bounded_unroll_or_refuse(
    key: Va,
    ancestors: &[&SymbolicVmInitialState],
    reentry: &SymbolicVmInitialState,
    observed_depth: Option<u64>,
    reason: String,
) -> LoopStep {
    // Checked only where the walk would otherwise refuse, and only AFTER the
    // observed bridge below has had its chance: an exact unroll to an observed
    // depth is strictly more precise output than a loop, so it must keep winning
    // where a depth exists.
    let refuse = |reason: String| -> LoopStep {
        if ancestors
            .iter()
            .any(|ancestor| closes_under_vm_stack_shift(key, ancestor, reentry))
        {
            return LoopStep::CloseShifted;
        }
        // The shift did not cover outright. Widen the shifted ancestor against
        // the re-entry and continue from that, rather than refusing: it is the
        // same generalization the non-striding path takes, and the shifted
        // ancestor sits at the re-entry's own depth by construction, so the
        // widened state does too. Converges because `generalize` goes to a
        // site-keyed TOP rather than growing a bound per revisit, so the next
        // round's coverage test sees TOP and closes; the visit, instruction and
        // wall-clock caps remain the backstop.
        for ancestor in ancestors {
            if let Some(shifted) = shift_ancestor_onto_reentry(ancestor, reentry)
                && let Ok(Some(widened)) =
                    widen_state(key, &canonicalized(&shifted), &canonicalized(reentry))
            {
                return LoopStep::WidenShifted(widened);
            }
        }
        LoopStep::Refuse(vsp_stride_note(ancestors, reentry, reason))
    };
    let Some(depth) = observed_depth else {
        return refuse(reason);
    };
    if proven_constant_vsp_delta(ancestors, reentry).is_none() {
        return refuse(reason);
    }
    // `ancestors.len()` counts the enclosing iterations already on the path; one
    // more exact unroll per remaining level up to the observed depth.
    if (ancestors.len() as u64) < depth {
        LoopStep::UnrollObserved
    } else {
        LoopStep::CloseObserved
    }
}

/// The decode profile a block is walked under. The walker holds a single base
/// profile for the entry's VM; a proven cross-profile transition attaches a new
/// one to the successor block it queues (see [`Walker::follow_cross_profile_transition`]).
/// Block/handler VAs are profile-disjoint, so the VA-keyed caches stay sound.
/// Shared by [`Rc`] so queuing a successor never deep-clones the config/roles.
struct ProfileContext {
    config: crate::HandlerCodecRecoveryConfig,
    profile: crate::VmProfile,
    direction: crate::BytecodeDirection,
    global_table: Option<crate::GlobalTableBase>,
    section: Option<usize>,
}

struct Walker<'a, 'p> {
    pe: &'a PeImage<'p>,
    limits: VmCfgLimits,
    config: crate::HandlerCodecRecoveryConfig,
    global_table: Option<crate::GlobalTableBase>,
    profile: crate::VmProfile,
    /// Total nested entries followed. Load-bearing for soundness, NOT just the
    /// cap: `discharge_vm_stack_assumption` only drops the external-store
    /// assumption when this is 0, so it must keep counting every following.
    /// Why `follow_cross_profile_transition` last refused, for the
    /// `MissingHandlerSpec` stall text. Each of its exits returns a bare
    /// `false`, so without this the biggest sound-mode stall family names a
    /// handler but never says which gate gave way.
    transition_refusal: Option<String>,
    nested_followed: usize,
    /// DISTINCT nested entry stubs followed. The cap bounds these rather than
    /// the total, because sibling paths and loop unrolls re-enter the SAME stub
    /// and were spending the budget on repeated work instead of on reaching new
    /// nested entries. Bounded as before by the entry-global visit/instruction
    /// caps and the hard per-entry memory cap, each of which fails closed.
    nested_sites: BTreeSet<Rva>,
    current: Option<Rc<PathNode>>,
    direction: crate::BytecodeDirection,
    section: Option<usize>,
    /// Profile of the block currently being visited: `None` means the base
    /// profile (every sound-path block), `Some` a profile reached through a
    /// proven cross-profile transition. Set at the top of each visit from the
    /// item being walked.
    active: Option<Rc<ProfileContext>>,
    /// Proven cross-profile transitions followed so far this entry; capped by
    /// `limits.max_profile_transitions` to bound exploration/memory.
    profile_transitions: usize,
    /// DISTINCT blocks a cross-profile transition was followed from; the cap bounds
    /// these rather than total followings, which unrolls and sibling paths burn on
    /// repeated work. Still bounded by post-transition visits, per-block visits and
    /// the hard per-entry memory cap, each of which fails closed.
    transition_sites: BTreeSet<Va>,
    /// `(total_instructions, visits)` captured when the FIRST cross-profile
    /// transition fired. `None` until then (and forever on the sound path).
    /// The post-transition budget is measured as growth beyond this baseline.
    transition_budget_base: Option<(usize, usize)>,
    cache: LinearDecodeCache,
    decoded: BTreeMap<Va, DecodedBlock>,
    explored: BTreeMap<Va, Vec<SymbolicVmInitialState>>,
    /// Per-block running over-approximation used once a block has hit the
    /// `max_visits_per_block` cap with distinct states that no path-local loop
    /// generalizes (blocks re-reached through many sibling paths, whose only drift
    /// is fresh-name bookkeeping). Incoming states fold into it so the block
    /// converges to a sound fixpoint instead of stalling at the cap.
    /// One canonical over-approximation PER VM-STACK PHASE per block. Keeping a
    /// single canonical made a re-entry at another VM stack depth hit the VSP
    /// refusal in `widen_state` and fail closed; phases converge separately.
    widened: BTreeMap<Va, Vec<SymbolicVmInitialState>>,
    cfg: VmCfg,
    total_instructions: usize,
    /// Running over-estimate of live walker bytes; see `VmCfgLimits::max_memory_bytes`.
    live_bytes: usize,
    /// Per-entry wall-clock deadline, so the generalization loops inside a
    /// single worklist item are bounded too (see `fold_states`).
    deadline: Option<Instant>,
    /// Observed runtime memory seed for the experimental tier (empty on the
    /// sound path). Consulted only to resolve provably-null image reads.
    observed: &'a crate::ObservedMemory,
    /// Flipped by [`ObservedImage`](crate::ObservedImage) when a symbolic read
    /// actually consumed an observed seed; read after each symbolic call so the
    /// recovery records the `ObservedGlobal` assumption.
    seeded: &'a Cell<bool>,
    /// Observed recursion depths per loop head (empty on the sound path).
    /// Consulted only to bound a VSP-changing recursion whose per-iteration VSP
    /// delta is a proven constant.
    observed_depths: &'a ObservedLoopDepths,
    /// Observed concrete branch targets per branch site (empty on the sound
    /// path). Consulted only at a branch that otherwise fails closed, and only
    /// for targets that still validate as real in-image bytecode.
    observed_branch_targets: &'a crate::ObservedBranchTargets,
    /// End offset above `vsp_initial` of the entry frame when it is recovered
    /// and its address provably never leaves the entry stub; `None` otherwise.
    private_frame_hi: Option<i64>,
    /// Set once a report may leak the frame address or write outside the frame,
    /// or a loop widening may have hidden such a pointer.
    /// OBSERVED: the VM stack frame really did escape — an address of it was
    /// exposed, or a store landed past its proven extent. A fact.
    frame_not_private: bool,
    /// CONCEDED: privacy was ASSUMED, to generalize or close a loop. Not the
    /// same claim as the field above, and keeping both in one flag denied the
    /// taint retry to every entry that closed a strided loop — the retry wants
    /// privacy to HOLD, and a concession is "not proven", not "false".
    frame_privacy_conceded: bool,
    /// Some report let a value derived from the frame (possibly mere data) leave it.
    frame_derived_escape: bool,
    /// Diagnostic: a report reported the frame ADDRESS escaping.
    frame_escape_seen: bool,
    /// Diagnostic: (frame_hi, written) of a store above the proven frame bound.
    frame_store_past: Option<(i64, i64)>,
    /// Some report discarded tracked memory holding a frame-pointer value.
    frame_pointer_lost: bool,
}

/// Builds the memory source for a symbolic call: the sound read-only image
/// overlaid with the observed seed (which is empty on the sound path, so the
/// overlay is then behaviourally identical to a bare [`ReadOnlyImage`]).
/// Opt-in: offer the labeled disjointness retry to EVERY tainted block, not
/// only one ending in an unconditional branch.
///
/// The restriction was never about soundness -- the retry's result carries the
/// same unproven `AssumesExternalStoresDisjointFromVmStack` label whatever the
/// boundary -- so widening it trades nothing but the number of entries that get
/// the chance.
///
/// MEASURED on systems.xpl with `--serve-ro`: +0. N1 stays at 658/814 and the
/// incomplete set stays at 154 (`BlockVisits` 5, stall 149), identical with the
/// knob on and off. The 20 remaining "branch target follows an imprecise
/// external-memory store" stalls are not reached by this retry, so widening the
/// boundary was not what held them back. Left off by default and kept only so
/// the next attempt starts from the measurement rather than the hypothesis.
fn retry_every_tainted_block() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VMPX_RETRY_TAINTED_ALL").is_ok_and(|v| v == "1"))
}

fn observed_source<'a, 'p>(
    walker: &Walker<'a, 'p>,
) -> crate::ObservedImage<'a, 'p> {
    crate::ObservedImage::new(
        crate::immutable_image::ReadOnlyImage(walker.pe),
        walker.observed,
        walker.seeded,
    )
}

/// An exit layout is carried only when its recovery was proven.
fn proven_exit_layout(report: Option<&VmExitReport>) -> Option<VmExitReport> {
    report
        .filter(|report| report.status == RecoveryStatus::Recovered)
        .cloned()
}

/// True when every occurrence of register `name` in `expr` is an operand of
/// `name + <in-image preferred-base VA>` — the signature of an image-base /
/// relocation-delta addend (added to a preferred-base image address to relocate
/// it). A single occurrence anywhere else — a bare register, a data operand, a
/// comparison — makes this false, so the caller fails closed instead of binding
/// a value it cannot prove is the delta.
fn image_base_relative_only(expr: &Expr, name: &str, is_image_va: &impl Fn(u64) -> bool) -> bool {
    let is_image_addend = |reg: &Expr, other: &Expr| {
        matches!(&reg.kind, ExprKind::Register { name: n } if n == name)
            && reg.width == BitWidth::W64
            && matches!(&other.kind, ExprKind::Const { value } if is_image_va(*value))
    };
    match &expr.kind {
        // A bare occurrence not consumed by an image-addend `Add` is a data use.
        ExprKind::Register { name: n } => n != name,
        ExprKind::Const { .. } => true,
        ExprKind::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } if is_image_addend(left, right) || is_image_addend(right, left) => true,
        ExprKind::Unary { value, .. }
        | ExprKind::Load { address: value }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => image_base_relative_only(value, name, is_image_va),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat {
            high: left,
            low: right,
        } => {
            image_base_relative_only(left, name, is_image_va)
                && image_base_relative_only(right, name, is_image_va)
        }
        ExprKind::Intrinsic { arguments, .. } => arguments
            .iter()
            .all(|argument| image_base_relative_only(argument, name, is_image_va)),
    }
}

/// Binds image-base-relative native registers in `target` to 0 under the
/// preferred-base frame, returning the rewritten target and the names bound.
/// `is_image_va` recognizes a preferred-base in-image VA.
///
/// Sound: a register bound here is proven (structurally, by
/// [`image_base_relative_only`]) to appear only as an addend relocating an
/// in-image preferred-base address. That is the relocation delta / image base,
/// which is 0 at the preferred base every image read in the tool assumes. A
/// register used as any other value is left untouched (the branch then stalls).
pub(crate) fn bind_preferred_base_registers(
    target: &Expr,
    is_image_va: &impl Fn(u64) -> bool,
) -> (Expr, BTreeSet<String>) {
    bind_preferred_base_registers_except(target, is_image_va, &BTreeSet::new())
}

/// [`bind_preferred_base_registers`] that never binds a register in `except`
/// (the environment gate passes its abstracted environment variables: one that
/// merely sits next to an image address is a free value, not the delta).
pub(crate) fn bind_preferred_base_registers_except(
    target: &Expr,
    is_image_va: &impl Fn(u64) -> bool,
    except: &BTreeSet<String>,
) -> (Expr, BTreeSet<String>) {
    const MAX_NODES: usize = 4_096;
    if target.node_count_bounded(MAX_NODES) > MAX_NODES {
        return (target.clone(), BTreeSet::new());
    }
    let mut replacements = BTreeMap::new();
    let mut bound = BTreeSet::new();
    for name in target.dependencies() {
        if !except.contains(&name) && image_base_relative_only(target, &name, is_image_va) {
            replacements.insert(name.clone(), Expr::constant(BitWidth::W64, 0));
            bound.insert(name);
        }
    }
    if replacements.is_empty() {
        return (target.clone(), bound);
    }
    match target.substitute(&replacements) {
        Ok(rewritten) => (rewritten, bound),
        // A width mismatch cannot prove the delta: leave the target unbound.
        Err(_) => (target.clone(), BTreeSet::new()),
    }
}

/// Distinct flag bits the ordinary flag-fork fallback enumerates.
const DEFAULT_MAX_FLAGS: usize = 2;

/// Targets a branch can take: the constant it simplifies to, or its image over
/// a proven-exhaustive input domain (`true` when guarded by inputs).
/// Diagnostic: with `VMPX_MEM_TRACE=1`, names any single call that ran longer
/// than five seconds. The walker-level trace samples between worklist items, so
/// it cannot fire while the walk is inside one call -- which is where the 11.7 GB
/// transient on entry 0xdb9023d lives.
fn note_slow(label: &str, start: std::time::Instant) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VMPX_MEM_TRACE").is_some()) {
        return;
    }
    let ms = start.elapsed().as_millis();
    // `VMPX_SLOW_MS` lowers the threshold: a 43s call made of many short ones
    // prints nothing at 5s, which hides exactly the distribution that matters.
    static FLOOR: std::sync::OnceLock<u128> = std::sync::OnceLock::new();
    let floor = *FLOOR.get_or_init(|| {
        std::env::var("VMPX_SLOW_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(5_000)
    });
    if ms >= floor {
        let (live, dead) = crate::expr::interner_census();
        println!("[slow] {label} {ms} ms  interner_live={live} dead={dead}");
    }
}

fn resolve_targets(target: &Expr, limits: VmCfgLimits) -> Result<(Vec<u64>, bool), String> {
    let at = std::time::Instant::now();
    let out = resolve_targets_inner(target, limits);
    note_slow("resolve_targets", at);
    out
}

fn resolve_targets_inner(target: &Expr, limits: VmCfgLimits) -> Result<(Vec<u64>, bool), String> {
    let simplified = target.simplify();
    if let ExprKind::Const { value } = simplified.kind {
        return Ok((vec![value], false));
    }
    resolve_over_inputs(&simplified, limits)
}

/// Resolves a simplified, non-constant branch target over its proven input
/// domain (or the bounded flag-fork fallback). Shared by the direct path and the
/// environment-gate over-approximation.
fn resolve_over_inputs(simplified: &Expr, limits: VmCfgLimits) -> Result<(Vec<u64>, bool), String> {
    resolve_over_inputs_capped(simplified, limits, DEFAULT_MAX_FLAGS)
}

/// [`resolve_over_inputs`] with an explicit bound on the distinct flag bits the
/// flag-fork fallback may enumerate. The environment-gate path raises it: every
/// flag there derives from unconstrained environment variables only.
pub(crate) fn resolve_over_inputs_capped(
    simplified: &Expr,
    limits: VmCfgLimits,
    max_flags: usize,
) -> Result<(Vec<u64>, bool), String> {
    let inferred = match infer_branch_input_domains(simplified, limits.input_domain) {
        Ok(inferred) => inferred,
        Err(refusal) => {
            // Sound fallback: a conditional branch whose next-VIP is a function
            // of flag bits the affine-domain path cannot characterize still
            // forks exactly when every input reaches the target through a
            // boolean, by enumerating that boolean over {0, 1}.
            return match flag_fork_analysis(simplified, limits, max_flags) {
                Ok(targets) => Ok((targets, true)),
                Err(why) => Err(format!(
                    "target depends on unproven input: {refusal} [flagfork={why}]"
                )),
            };
        }
    };
    let domain = SymbolicDomainLimits {
        max_values_per_variable: limits
            .domain
            .max_values_per_variable
            .max(limits.input_domain.max_values_per_variable),
        ..limits.domain
    };
    let report = match enumerate_symbolic_value(simplified, &inferred.domains, domain) {
        Ok(report) => report,
        Err(error) => {
            // The full input cross-product is infeasible (too many independent
            // input values), but the target may route all of its inputs through
            // only a bounded set of boolean flags — a flag-selected jump whose
            // real target set is tiny. Enumerate those flags over {0, 1} instead.
            // Sound: each substituted flag is proven single-bit, and any residual
            // input dependence fails the branch closed rather than guessing.
            return match flag_fork_analysis(simplified, limits, max_flags) {
                Ok(targets) => Ok((targets, true)),
                Err(why) => Err(format!("target enumeration failed: {error} [flagfork={why}]")),
            };
        }
    };
    if !report.complete || report.unique_values.is_empty() {
        return Err("target domain is incomplete".to_owned());
    }
    if report.unique_values.len() > limits.max_targets {
        return Err(format!(
            "branch has {} targets, exceeding limit {}",
            report.unique_values.len(),
            limits.max_targets
        ));
    }
    Ok((report.unique_values, true))
}

/// A pure host-environment intrinsic: its value is fixed by the host CPU /
/// environment, independent of the program's inputs and memory. The CPUID words
/// (processor signature etc.) and the timestamp counter are what VMProtect reads
/// for its anti-analysis gates. Statically each is an unconstrained environment
/// input, so abstracting it to a free variable is a sound over-approximation.
fn is_environment_intrinsic(name: &str) -> bool {
    name.starts_with("x86_cpuid_") || name.starts_with("read_timestamp_counter_")
}

/// A `vm_flags_*` status-word intrinsic: a pure function of its operands, adding
/// no observation of its own (the input-domain inference treats it as such).
pub(crate) fn is_flags_intrinsic(name: &str) -> bool {
    name.starts_with("vm_flags_")
}

/// Replaces every pure host-environment intrinsic (see [`is_environment_intrinsic`])
/// with a free register named by the intrinsic, dropping its (ignored or
/// constant) arguments. Equal reads share one variable; the environment value is
/// unconstrained, so the free variable's domain is a superset of the reachable
/// ones — a sound over-approximation, exactly as [`crate::abstract_epoch_loads`]
/// treats untracked memory reads.
fn abstract_environment_intrinsics(expr: &Expr) -> Result<Expr, ExprError> {
    expr.rewrite_intrinsics(&mut |width, name, _arguments| {
        is_environment_intrinsic(name).then(|| Expr::register(width, name.to_owned()))
    })
}

/// Resolves a branch whose only unknown is a VMProtect CPUID / timestamp-counter
/// anti-analysis gate, by a sound both-target over-approximation.
///
/// The host-environment intrinsics are abstracted to free variables (which also
/// unbinds the relocation-delta register they had captured as an ignored CPUID
/// sub-leaf operand), the preferred-base relocation delta is bound to 0 as
/// elsewhere, and the result is resolved over the same proven input domain /
/// flag-fork machinery. Because the environment value is unconstrained, both
/// sides of the gate are enumerated regardless of the host CPU.
///
/// The structural trigger is strict: the target must read at least one
/// environment intrinsic, and once those and the image-base delta are removed
/// nothing program-derived may remain — every surviving register is an
/// environment free variable and every surviving intrinsic is a pure flags
/// function of them. A condition touching a real program input or a memory read
/// fails this and is never forked. Returns the targets, whether the branch is
/// input-guarded, and the preferred-base registers bound, or `None` when the
/// trigger does not hold or resolution does not prove a finite target set.
fn resolve_environment_gate(
    target: &Expr,
    is_image_va: &impl Fn(u64) -> bool,
    limits: VmCfgLimits,
) -> Option<(Vec<u64>, bool, BTreeSet<String>)> {
    if !target
        .intrinsic_dependencies()
        .iter()
        .any(|name| is_environment_intrinsic(name))
    {
        return None;
    }
    let abstracted = abstract_environment_intrinsics(target).ok()?;
    let (bound, bound_registers) = bind_preferred_base_registers(&abstracted, is_image_va);
    let bound = bound.simplify();
    // Once the environment reads and the image-base delta are gone, the only
    // surviving unknowns must be the environment free variables themselves,
    // combined through pure flags functions. A surviving real program register
    // (e.g. the VM stack pointer), a memory read, or any other opaque intrinsic
    // means the target is not a pure environment gate: over-approximating it
    // would fork on program-derived state and fabricate control flow, so fail
    // closed instead.
    if !bound
        .dependencies()
        .iter()
        .all(|name| is_environment_intrinsic(name))
    {
        return None;
    }
    if !bound
        .intrinsic_dependencies()
        .iter()
        .all(|name| is_flags_intrinsic(name))
    {
        return None;
    }
    let (targets, guarded) = resolve_over_inputs(&bound, limits).ok()?;
    Some((targets, guarded, bound_registers))
}

/// An input reaching the target outside a single-bit flag: the leaf, and the
/// narrowest bit-width through which the path to it observes it (`u32::MAX`
/// when observed at full width).
struct FrontierLeaf {
    expr: Expr,
    observed_bits: u32,
}

/// Largest subexpressions, in `out`, that `target` depends on its inputs only
/// through: each is proven to lie in {0, 1}. `Err` when an input reaches the
/// target outside of such a boolean (a bare register, memory, or an opaque
/// intrinsic value), which forbids the fallback.
fn collect_flag_frontier(
    expr: &Expr,
    out: &mut Vec<Expr>,
    seen: &mut std::collections::HashSet<*const Expr>,
    observed_bits: u32,
) -> Result<(), Vec<FrontierLeaf>> {
    if !seen.insert(std::ptr::from_ref(expr)) {
        return Ok(());
    }
    // An input-independent subexpression has a fixed value; it routes no input.
    if expr.dependencies().is_empty() && expr.intrinsic_dependencies().is_empty() {
        return Ok(());
    }
    // Proven single bit: the only sound value to substitute.
    if value_range(expr).max() <= 1 {
        out.push(expr.clone());
        return Ok(());
    }
    match &expr.kind {
        ExprKind::Extract { value, .. } => {
            collect_flag_frontier(value, out, seen, observed_bits.min(expr.width.bits()))
        }
        ExprKind::Resize { value, .. } => {
            let narrowed = if expr.width.bits() < value.width.bits() {
                observed_bits.min(expr.width.bits())
            } else {
                observed_bits
            };
            collect_flag_frontier(value, out, seen, narrowed)
        }
        ExprKind::Unary { value, .. } => collect_flag_frontier(value, out, seen, observed_bits),
        ExprKind::Binary { op, left, right } => {
            let mask_bits = |side: &Expr| match side.kind {
                ExprKind::Const { value } if *op == BinaryOp::And => Some(64 - value.leading_zeros()),
                _ => None,
            };
            let left_bits = mask_bits(right).map_or(observed_bits, |m| observed_bits.min(m));
            let right_bits = mask_bits(left).map_or(observed_bits, |m| observed_bits.min(m));
            let first = collect_flag_frontier(left, out, seen, left_bits);
            merge_leaves(first, collect_flag_frontier(right, out, seen, right_bits))
        }
        ExprKind::Concat {
            high: left,
            low: right,
        } => {
            let first = collect_flag_frontier(left, out, seen, observed_bits);
            merge_leaves(first, collect_flag_frontier(right, out, seen, observed_bits))
        }
        // Input reaches the target as a full value, not through a bit.
        ExprKind::Register { .. } | ExprKind::Load { .. } | ExprKind::Intrinsic { .. } => {
            Err(vec![FrontierLeaf {
                expr: expr.clone(),
                observed_bits,
            }])
        }
        ExprKind::Const { .. } => Ok(()),
    }
}

fn merge_leaves(
    first: Result<(), Vec<FrontierLeaf>>,
    second: Result<(), Vec<FrontierLeaf>>,
) -> Result<(), Vec<FrontierLeaf>> {
    match (first, second) {
        (Ok(()), other) | (other, Ok(())) => other,
        (Err(mut a), Err(b)) => {
            a.extend(b);
            Err(a)
        }
    }
}

/// Typed reason `flag_fork_analysis` refused. Diagnostic only: rendered into the
/// stall text, never branched on.
enum FlagForkRefusal {
    /// Every offending leaf, classified, as `+`-joined labels.
    Leaves(String),
    /// The target depends on `n` > `MAX_FLAGS` distinct single-bit flags.
    TooManyFlags(usize),
    /// An input is observed through a mask/extract of `bits` bits, not a flag.
    MultiBitMask { bits: u32, leaf: String },
    /// A double shift with an input-independent count (a bit permutation).
    ConstantCountShift(String),
    /// A variable-count double shift consumed at full width.
    ResidualFullWidth(String),
    /// Some other full-width register/memory/intrinsic leaf.
    OtherLeaf(String),
    NoFlags,
    /// Size cap, substitution failure, residual non-constant, or target cap.
    Other(&'static str),
}

impl std::fmt::Display for FlagForkRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Leaves(labels) => write!(f, "{labels}"),
            Self::TooManyFlags(n) => write!(f, "TooManyFlags({n})"),
            Self::MultiBitMask { bits, leaf } => write!(f, "MultiBitMask({bits},{leaf})"),
            Self::ConstantCountShift(leaf) => write!(f, "ConstantCountShift({leaf})"),
            Self::ResidualFullWidth(leaf) => write!(f, "ResidualFullWidth({leaf})"),
            Self::OtherLeaf(leaf) => write!(f, "OtherLeaf({leaf})"),
            Self::NoFlags => write!(f, "NoFlags"),
            Self::Other(why) => write!(f, "Other({why})"),
        }
    }
}

fn classify_frontier_leaf(leaf: &FrontierLeaf) -> FlagForkRefusal {
    let family = match &leaf.expr.kind {
        ExprKind::Intrinsic { name, .. } => name
            .trim_end_matches(|c: char| c.is_ascii_digit())
            .trim_end_matches('_')
            .to_owned(),
        ExprKind::Register { .. } => "register".to_owned(),
        ExprKind::Load { .. } => "load".to_owned(),
        _ => "other".to_owned(),
    };
    let full_width = leaf.observed_bits >= leaf.expr.width.bits();
    if let ExprKind::Intrinsic { name, arguments } = &leaf.expr.kind
        && (name.starts_with("x86_double_shift_") || name.starts_with("vm_flags_double_shift_"))
    {
        let constant_count = arguments
            .get(2)
            .is_some_and(|c| c.dependencies().is_empty() && c.intrinsic_dependencies().is_empty());
        if constant_count {
            return FlagForkRefusal::ConstantCountShift(family);
        }
        if full_width {
            return FlagForkRefusal::ResidualFullWidth(family);
        }
    } else if full_width {
        return FlagForkRefusal::OtherLeaf(family);
    }
    FlagForkRefusal::MultiBitMask {
        bits: leaf.observed_bits,
        leaf: family,
    }
}

/// Replaces each subexpression equal to a flag in `assignment` with its assigned
/// bit, rebuilding and resimplifying the rest.
pub(crate) fn substitute_flags(
    expr: &Expr,
    assignment: &[(&Expr, u64)],
    memo: &mut std::collections::HashMap<*const Expr, Expr>,
) -> Result<Expr, ExprError> {
    for (flag, bit) in assignment {
        if expr == *flag {
            return Ok(Expr::constant(flag.width, *bit));
        }
    }
    let key = std::ptr::from_ref(expr);
    if let Some(done) = memo.get(&key) {
        return Ok(done.clone());
    }
    let mut go = |value: &Expr| substitute_flags(value, assignment, memo);
    let rebuilt = match &expr.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => expr.clone(),
        ExprKind::Unary { op, value } => Expr::unary(*op, go(value)?),
        ExprKind::Binary { op, left, right } => Expr::binary(*op, go(left)?, go(right)?)?,
        ExprKind::Load { address } => Expr::load(expr.width, go(address)?),
        ExprKind::Intrinsic { name, arguments } => Expr::intrinsic(
            expr.width,
            name.clone(),
            arguments
                .iter()
                .map(|argument| substitute_flags(argument, assignment, memo))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ExprKind::Extract { value, lsb } => Expr::extract(expr.width, *lsb, go(value)?)?,
        ExprKind::Concat { high, low } => Expr::concat(go(high)?, go(low)?)?,
        ExprKind::Resize { op, value } => Expr::resize(*op, expr.width, go(value)?)?,
    };
    memo.insert(key, rebuilt.clone());
    Ok(rebuilt)
}

/// Resolves a conditional branch whose next-VIP depends on the inputs only
/// through a small set of boolean flags, by enumerating every flag over {0, 1}.
///
/// Sound because each substituted flag is proven to lie in {0, 1} (so the two
/// substitutions are exhaustive for it) and every assignment is required to
/// collapse to a constant: any residual input dependence fails the branch
/// closed rather than emitting a guessed target. Over-approximation can only add
/// a target, never drop one.
#[cfg(test)]
fn flag_fork_targets(target: &Expr, limits: VmCfgLimits) -> Option<Vec<u64>> {
    flag_fork_analysis(target, limits, DEFAULT_MAX_FLAGS).ok()
}

/// Whether a computed branch target leaving the entry's bytecode section may be
/// followed. Section executability is NOT a criterion: the VM dispatcher reads
/// bytecode as data, and this sample hides bytecode in sections marked
/// non-executable (`.pdata` here carries 0 valid RUNTIME_FUNCTION entries in
/// 66688 and is high-entropy throughout, so it is not real unwind data). An
/// UNMAPPED target is still never followed, the lever is still off by default,
/// and the carried rolling-key decode at the target remains the real guard: a
/// wrong target decodes to garbage and stalls.
fn cross_section_followable(follow: bool, target_characteristics: Option<u32>) -> bool {
    follow && target_characteristics.is_some()
}

/// `flag_fork_targets` with the refusal reason retained.
fn flag_fork_analysis(
    target: &Expr,
    limits: VmCfgLimits,
    max_flags: usize,
) -> Result<Vec<u64>, FlagForkRefusal> {
    const MAX_NODES: usize = 4_096;
    if target.node_count_bounded(MAX_NODES) > MAX_NODES {
        return Err(FlagForkRefusal::Other("node_cap"));
    }
    let mut frontier = Vec::new();
    collect_flag_frontier(
        target,
        &mut frontier,
        &mut std::collections::HashSet::new(),
        u32::MAX,
    )
    .map_err(|leaves| {
        let mut labels: Vec<String> = leaves
            .iter()
            .map(|leaf| classify_frontier_leaf(leaf).to_string())
            .collect();
        labels.sort();
        labels.dedup();
        FlagForkRefusal::Leaves(labels.join("+"))
    })?;
    let mut flags: Vec<Expr> = Vec::new();
    for flag in frontier {
        if !flags.contains(&flag) {
            flags.push(flag);
        }
    }
    if flags.is_empty() {
        return Err(FlagForkRefusal::NoFlags);
    }
    if flags.len() > max_flags {
        return Err(FlagForkRefusal::TooManyFlags(flags.len()));
    }
    // A flag the range analysis proves constant must not be forked over {0, 1}:
    // the other value is infeasible and seeds a spurious branch target (observed
    // as VIP+-1 around the real target, which then decodes to a garbage handler).
    // This only drops assignments `value_range` already proves cannot occur, so
    // the over-approximation shrinks without any new assumption.
    let ranges: Vec<(u64, u64)> = flags
        .iter()
        .map(|flag| {
            let range = value_range(flag);
            (range.min(), range.max())
        })
        .collect();
    let mut targets = BTreeSet::new();
    for combination in 0_u32..(1_u32 << flags.len()) {
        let in_range = ranges.iter().enumerate().all(|(index, (lo, hi))| {
            let bit = u64::from((combination >> index) & 1);
            bit >= *lo && bit <= *hi
        });
        if !in_range {
            continue;
        }
        let assignment: Vec<(&Expr, u64)> = flags
            .iter()
            .enumerate()
            .map(|(index, flag)| (flag, u64::from((combination >> index) & 1)))
            .collect();
        let resolved = substitute_flags(target, &assignment, &mut std::collections::HashMap::new())
            .map_err(|_| FlagForkRefusal::Other("substitute"))?
            .simplify();
        match resolved.kind {
            ExprKind::Const { value } => {
                targets.insert(value);
            }
            // Residual input dependence beyond the flags: fail closed.
            _ => return Err(FlagForkRefusal::Other("residual_nonconst")),
        }
    }
    if !targets.is_empty() && targets.len() <= limits.max_targets {
        Ok(targets.into_iter().collect())
    } else {
        Err(FlagForkRefusal::Other("target_cap"))
    }
}

impl Walker<'_, '_> {
    fn stall(&mut self, block: Va, vip: Va, reason: impl Into<String>) {
        self.cfg.stalls.push(CfgStall {
            block,
            vip,
            reason: reason.into(),
        });
    }

    fn add_edge(&mut self, edge: CfgEdge) {
        if !self.cfg.edges.contains(&edge) {
            self.cfg.edges.push(edge);
        }
    }

    // Profile of the block currently being visited: the active transition
    // profile when one is attached, else the base profile. On the sound path
    // `active` is always `None`, so each accessor returns the base value and
    // behaviour is byte-for-byte unchanged.
    fn ctx_direction(&self) -> crate::BytecodeDirection {
        self.active.as_ref().map_or(self.direction, |c| c.direction)
    }

    fn ctx_section(&self) -> Option<usize> {
        self.active.as_ref().map_or(self.section, |c| c.section)
    }

    fn ctx_global_table(&self) -> Option<crate::GlobalTableBase> {
        self.active.as_ref().map_or(self.global_table, |c| c.global_table)
    }

    fn ctx_profile(&self) -> crate::VmProfile {
        self.active.as_ref().map_or(self.profile, |c| c.profile)
    }

    /// True once the post-transition exploration budget is spent. Measured as
    /// growth in visits/decoded-instructions beyond the baseline captured at the
    /// first cross-profile transition, so it is inert until one fires (and thus
    /// forever on the sound path). This is the hard memory bound on the
    /// transition walk: retained post-transition state grows with depth, so a
    /// tight node/instruction budget — not the loose entry-global caps — keeps a
    /// pathological entry from allocating without limit.
    fn transition_budget_exhausted(&self) -> bool {
        // The allowance is PER DISTINCT TRANSITION SITE, not one flat grant for
        // the whole remaining walk. Measured from the first transition only, an
        // entry that legitimately follows 30 transitions got the same 64 visits
        // and 6000 instructions as one following a single transition, so the
        // budget ran out on the NUMBER of regions rather than on any one region
        // being pathological. Per-region removes that refusal class entirely
        // (TransitionBudget stops went 238 -> 0).
        //
        // Still bounded, and every bound still fails closed: at most
        // `max_profile_transitions` distinct sites, so this tops out at that
        // multiple of the per-region allowance, with the entry-global
        // `max_visits` / `max_total_instructions` and the hard per-entry
        // `max_memory_bytes` cap all continuing to apply above it.
        let regions = self.transition_sites.len().max(1);
        let visit_budget = self.limits.max_transition_visits.saturating_mul(regions);
        let insn_budget = self
            .limits
            .max_transition_instructions
            .saturating_mul(regions);
        self.transition_budget_base.is_some_and(|(i0, v0)| {
            self.total_instructions.saturating_sub(i0) >= insn_budget
                || self.cfg.visits.saturating_sub(v0) >= visit_budget
        })
    }

    /// Conservative bytes held by `state`: distinct shared nodes (each counted
    /// once per state, so sharing across states only over-counts) plus per-root
    /// overhead. The node walk stops once it alone would bust the cap.
    fn state_bytes(&self, state: &SymbolicVmInitialState) -> usize {
        let roots = std::iter::once(&state.vsp)
            .chain(std::iter::once(&state.flags))
            .chain(state.context_values.values())
            .chain(state.memory_values.iter().flat_map(|(a, v)| [a, v]))
            .chain(
                state
                    .fresh_allocations
                    .iter()
                    .flat_map(|f| [&f.allocation, &f.size]),
            );
        let node_budget = self
            .limits
            .max_memory_bytes
            .saturating_sub(self.live_bytes)
            / EXPR_NODE_BYTES;
        let nodes = distinct_node_count_bounded(roots, node_budget);
        let root_count = 2
            + state.context_values.len()
            + 2 * state.memory_values.len()
            + 2 * state.fresh_allocations.len();
        nodes
            .saturating_mul(EXPR_NODE_BYTES)
            .saturating_add(root_count.saturating_mul(STATE_ROOT_BYTES))
    }

    /// Diagnostic: with `VMPX_MEM_TRACE=1`, prints the walker's own containers
    /// every two seconds. The per-entry cap counts the symbolic state the walker
    /// holds; an entry reached 12 GB resident against a 1.5 GB cap that never
    /// fired, so something it does not count holds the rest. Correlate these
    /// against an external RSS sample: whichever column tracks RSS is the site.
    fn mem_trace(&self) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static LAST: AtomicU64 = AtomicU64::new(0);
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VMPX_MEM_TRACE").is_some()) {
            return;
        }
        let start = START.get_or_init(std::time::Instant::now);
        let secs = start.elapsed().as_secs();
        if secs < LAST.load(Ordering::Relaxed) + 2 {
            return;
        }
        LAST.store(secs, Ordering::Relaxed);
        let states: usize = self.explored.values().map(Vec::len).sum();
        let canon: usize = self.widened.values().map(Vec::len).sum();
        let instrs: usize = self.decoded.values().map(|b| b.instructions.len()).sum();
        let (live, dead) = crate::expr::interner_census();
        println!(
            "[mem {secs:>4}s] charged={:>6}MB blocks={:<5} states={:<6} canon={:<5} decoded={:<5} \
             decoded_instrs={:<8} cfg_blocks={:<5} visits={:<6} interner_live={:<9} \
             interner_dead={:<8} interner_live_MB={:>6}",
            self.live_bytes / (1024 * 1024),
            self.explored.len(),
            states,
            canon,
            self.decoded.len(),
            instrs,
            self.cfg.blocks.len(),
            self.cfg.visits,
            live,
            dead,
            live.saturating_mul(EXPR_NODE_BYTES) / (1024 * 1024),
        );
    }

    /// Adds `bytes` to the live estimate; on exceeding the cap records
    /// `CfgLimitHit::Memory` and a stall, and returns false (fail closed).
    fn charge(&mut self, bytes: usize, key: Va, vip: Va) -> bool {
        self.live_bytes = self.live_bytes.saturating_add(bytes);
        if self.live_bytes <= self.limits.max_memory_bytes {
            return true;
        }
        self.cfg.limit_hit.get_or_insert(CfgLimitHit::Memory);
        self.stall(key, vip, "per-entry memory cap exceeded");
        false
    }

    /// Charges the symbolic state each newly queued successor carries.
    fn account_pending(&mut self, items: &mut [Item]) -> bool {
        for item in items {
            let bytes = self.state_bytes(&item.symbolic);
            item.bytes = bytes;
            let vip = item.start.vip;
            if !self.charge(bytes, vip, vip) {
                return false;
            }
        }
        true
    }

    fn decode_block(&mut self, item: &Item) -> bool {
        let key = item.start.vip;
        if self.decoded.contains_key(&key) {
            return true;
        }
        if self.cfg.blocks.len() >= self.limits.max_blocks {
            self.cfg.limit_hit.get_or_insert(CfgLimitHit::Blocks);
            return false;
        }
        let remaining = self
            .limits
            .max_total_instructions
            .saturating_sub(self.total_instructions);
        if remaining == 0 {
            self.cfg
                .limit_hit
                .get_or_insert(CfgLimitHit::TotalInstructions);
            return false;
        }
        // Disjoint field borrows: `config`/`global_table` read `active`/`config`/
        // `global_table`, the decode writes `cache` — a different field.
        let config = self.active.as_ref().map_or(&self.config, |c| &c.config);
        let global_table = self.active.as_ref().map_or(self.global_table, |c| c.global_table);
        let block = match recover_and_decode_linear_vm_stream_cached(
            self.pe,
            item.start,
            config,
            self.limits.entry.instruction_limit.min(remaining),
            global_table,
            &mut self.cache,
        ) {
            Ok(report) => {
                let decode = report.decode;
                let boundary = (decode.stop_reason == LinearDecodeStop::ControlFlowBoundary)
                    .then(|| self.cache.recoveries.get(&decode.final_state.handler))
                    .flatten()
                    .and_then(|recovery| recovery.handler_analysis.as_ref())
                    .map(|analysis| analysis.semantic);
                DecodedBlock {
                    instructions: decode.instructions.into(),
                    final_state: decode.final_state,
                    stop: decode.stop_reason,
                    boundary,
                    error: None,
                }
            }
            Err(error) => {
                let partial = error.partial_report().decode.clone();
                DecodedBlock {
                    instructions: partial.instructions.into(),
                    final_state: partial.final_state,
                    stop: partial.stop_reason,
                    boundary: None,
                    error: Some(error.to_string()),
                }
            }
        };
        self.total_instructions += block.instructions.len();
        // Decoded buffers live in `decoded` and in `cfg.blocks`. `visit` used to
        // hold a third, transient copy; it now shares the one in `decoded`.
        let decoded_bytes = block
            .instructions
            .len()
            .saturating_mul(DECODED_INSTRUCTION_BYTES)
            .saturating_mul(2);
        if !self.charge(decoded_bytes, key, item.start.vip) {
            return false;
        }
        let boundary_handler = block.boundary.map(|_| block.final_state.handler);
        let exit = match block.boundary {
            Some(VmHandlerSemantic::VmExit) => BlockExit::VmExit,
            Some(VmHandlerSemantic::UnconditionalBranch) => BlockExit::Branch,
            Some(VmHandlerSemantic::NativeCall) => BlockExit::NativeCall,
            _ => BlockExit::Stalled,
        };
        let vm_exit = (exit == BlockExit::VmExit)
            .then(|| {
                self.cache
                    .recoveries
                    .get(&block.final_state.handler)
                    .and_then(|recovery| proven_exit_layout(recovery.vm_exit.as_ref()))
            })
            .flatten();
        self.cfg.blocks.insert(
            key,
            VmBasicBlock {
                start: item.start,
                address: item.address,
                instructions: block.instructions.to_vec(),
                exit,
                boundary_handler,
                native_call: None,
                vm_exit,
            },
        );
        self.decoded.insert(key, block);
        true
    }

    /// The state to explore `item` in, or `None` when it closes a loop or
    /// the loop cannot be generalized.
    fn enter_loop(&mut self, item: &Item) -> Option<SymbolicVmInitialState> {
        let key = item.start.vip;
        let ancestors: Vec<Rc<PathNode>> =
            std::iter::successors(item.path.clone(), |node| node.parent.clone())
                .filter(|node| node.key == key)
                .collect();
        if ancestors.is_empty() {
            return Some(item.symbolic.clone());
        }
        let states: Vec<&SymbolicVmInitialState> =
            ancestors.iter().map(|node| &node.state).collect();
        let observed_depth = self.observed_depths.depth_for(key.get());
        match loop_step(
            key,
            &states,
            &item.symbolic,
            self.limits.max_loop_unrolls,
            observed_depth,
        ) {
            LoopStep::Close => {
                self.record_loop_back(item, key);
                None
            }
            LoopStep::Unroll => Some(item.symbolic.clone()),
            LoopStep::Widen(widened) => {
                self.concede_frame_privacy();
                Some(widened)
            }
            // The back edge closes, but the VM stack grows a constant amount per
            // level with no bound this walk can see. Concede frame privacy for
            // the same reason `Widen` does, and more sharply: every bound on the
            // frame extent came from the finitely many stores observed, and an
            // unbounded push-down writes below all of them. The entry can reach
            // N1 on the strength of a closed graph; it must not reach the proven
            // tier on the strength of a frame bound that does not hold.
            LoopStep::CloseShifted => {
                self.concede_frame_privacy();
                self.record_loop_back(item, key);
                None
            }
            // Same frame-privacy concession as `CloseShifted`, and the same
            // continuation as `Widen`: the loop has not closed yet, it carries on
            // from the generalization.
            LoopStep::WidenShifted(widened) => {
                self.concede_frame_privacy();
                Some(widened)
            }
            LoopStep::Refuse(reason) => {
                self.stall(key, key, format!("loop cannot be closed: {reason}"));
                None
            }
            // Experimental, observed-tier: a bounded recursion unroll keyed by an
            // observed depth. Each iteration applies the proven constant VSP
            // delta exactly; the whole bridge is labeled so the CFG never passes
            // as sound.
            LoopStep::UnrollObserved => {
                self.cfg
                    .assumptions
                    .insert(VmCfgAssumption::ObservedRecursionDepth);
                Some(item.symbolic.clone())
            }
            LoopStep::CloseObserved => {
                self.cfg
                    .assumptions
                    .insert(VmCfgAssumption::ObservedRecursionDepth);
                self.record_loop_back(item, key);
                None
            }
        }
    }

    fn record_loop_back(&mut self, item: &Item, key: Va) {
        if let Some(parent) = &item.path {
            self.add_edge(CfgEdge {
                from: parent.key,
                to: key,
                kind: CfgEdgeKind::LoopBack,
            });
        }
    }

    /// Replaces the canonical for `state`'s VM-stack phase at `key`, leaving the
    /// other phases' canonicals in place.
    fn record_canonical(&mut self, key: Va, state: SymbolicVmInitialState) {
        let phases = self.widened.entry(key).or_default();
        match phases.iter_mut().find(|held| held.vsp == state.vsp) {
            Some(slot) => *slot = state,
            None => phases.push(state),
        }
    }

    /// Generalizes a re-entry at a saturated block against its running
    /// over-approximation (seeded from every state recorded there). A cell the
    /// generalization already holds at TOP is covered, so once the varying cells
    /// saturate the block closes on every further re-entry. Pure over-approximation
    /// (cells only ever broaden to TOP), so no reachable state is dropped.
    fn generalize_across(
        &mut self,
        key: Va,
        incoming: &SymbolicVmInitialState,
    ) -> GeneralizeAcross {
        // Phase-local, for the reason recorded on `Walker::widened`: fold and widen
        // only against states at the re-entry's VM stack depth. Both lookups borrow
        // in place and hold only the owned outcome across the mutation, so a
        // saturated block no longer clones its whole recorded state set on every
        // re-entry — the cost that matters now that partitioning lets the walk
        // explore much deeper.
        let canonical = self
            .widened
            .get(&key)
            .and_then(|phases| phases.iter().find(|c| c.vsp == incoming.vsp));
        if let Some(canonical) = canonical {
            let folded = match widen_state(key, canonical, incoming) {
                Ok(None) => return GeneralizeAcross::Closed,
                Ok(Some(widened)) => {
                    let hid_frame_pointer =
                        fold_hid_frame_pointer([canonical, incoming].into_iter(), &widened);
                    Ok((widened, hid_frame_pointer))
                }
                Err(reason) => Err(reason),
            };
            return self.continue_from(key, folded);
        }
        // Counted under its own borrow so the shift below can take `&mut self`.
        let at_this_depth = self
            .explored
            .get(&key)
            .map_or(0, |states| states.iter().filter(|s| s.vsp == incoming.vsp).count());
        if at_this_depth == 0 {
            // Phase-local folding has nothing to fold against: every state
            // recorded here sits at a different VM stack depth. That is exactly
            // what a strided loop BODY looks like, and `visited` counts states
            // across all depths, so such a block piles up one state per phase and
            // dies at the ceiling without the fold ever engaging. Shift the
            // recorded phases onto this re-entry's depth — the same substitution
            // the loop HEAD already uses to close a strided back edge — and fold
            // against those instead.
            let shifted_at = std::time::Instant::now();
            let shifted = self.fold_shifted_phases(key, incoming);
            note_slow("generalize/fold_shifted_phases", shifted_at);
            if let Some(folded) = shifted {
                return folded;
            }
        }
        let states: Vec<&SymbolicVmInitialState> = self
            .explored
            .get(&key)
            .map(|states| {
                states
                    .iter()
                    .filter(|state| state.vsp == incoming.vsp)
                    .collect()
            })
            .unwrap_or_default();
        let folded = match fold_states(key, states.iter().copied(), incoming, self.deadline) {
            Ok(widened) => {
                let hid_frame_pointer =
                    fold_hid_frame_pointer(states.iter().copied().chain([incoming]), &widened);
                Ok((widened, hid_frame_pointer))
            }
            Err(reason) => Err(reason),
        };
        self.continue_from(key, folded)
    }

    /// Folds this re-entry against the block's other phases, each re-expressed at
    /// the re-entry's VM stack depth. `None` when nothing here shifts, leaving the
    /// ordinary (empty) fold to decide.
    ///
    /// The shift re-expresses an ancestor's cells relative to a moved stack base,
    /// so it describes the same concrete memory only if nothing outside the VM
    /// aliases the frame. That is a concession, not a proof, so every result here
    /// sets `frame_not_private`: it can reach N1 and never N2.
    fn fold_shifted_phases(
        &mut self,
        key: Va,
        incoming: &SymbolicVmInitialState,
    ) -> Option<GeneralizeAcross> {
        match self.scan_shifted_phases(key, incoming) {
            ShiftedPhases::None => None,
            ShiftedPhases::Covered => {
                self.concede_frame_privacy();
                Some(GeneralizeAcross::Closed)
            }
            ShiftedPhases::Folded(folded) => {
                self.concede_frame_privacy();
                // `false`, not a computed fold-hid test: this path concedes frame
                // privacy outright above, so the flag the caller ORs in is already
                // set and a test of what the fold dropped would decide nothing.
                Some(self.continue_from(key, Ok((folded, false))))
            }
        }
    }

    /// Walks the block's other phases, each re-expressed at the re-entry's VM
    /// stack depth, holding ONE shifted copy at a time.
    ///
    /// Materializing them all was measured at 171 s and 11.7 GB transient in a
    /// single call: `shift_ancestor_onto_reentry` substitutes over every cell of
    /// a state, so a vector of N shifted phases is N full copies of the state's
    /// expressions alive at once. Nothing the walker retains grows during that
    /// spike, which is why the per-entry cap never saw it -- the peak is this
    /// scan's own working set, and bounding it is the whole fix.
    fn scan_shifted_phases(&self, key: Va, incoming: &SymbolicVmInitialState) -> ShiftedPhases {
        let reentry_at = std::time::Instant::now();
        let reentry = canonicalized(incoming);
        note_slow("scan/canonicalized(incoming)", reentry_at);
        let empty: &[SymbolicVmInitialState] = &[];
        let canonicals = self.widened.get(&key).map_or(empty, Vec::as_slice);
        // A recorded canonical is already a fold of many states, so it is both
        // tighter and cheaper to shift than the raw history; fall back to the
        // history only when no phase has converged yet.
        let sources = if canonicals.is_empty() {
            self.explored.get(&key).map_or(empty, Vec::as_slice)
        } else {
            canonicals
        };
        // OLDEST first, and at most `max_visits_per_block` of them. Order is not
        // a detail here: folding oldest-first reaches a stable over-approximation
        // in a few steps, while newest-first re-widens against ever more
        // different states and the accumulator grows. Measured on one entry,
        // same budget, only the order reversed: 0.55 GB flat against 13.4 GB.
        // The bound then caps the per-call cost, since shifting one phase
        // substitutes over every cell of a state.
        let budget = self.limits.max_visits_per_block;
        let mut acc: Option<SymbolicVmInitialState> = None;
        for phase in sources.iter().take(budget) {
            // Bounded for the same reason as `fold_states`: this runs inside one
            // worklist item, where the walk's own deadline does not reach.
            if self.deadline.is_some_and(|at| Instant::now() >= at) {
                return ShiftedPhases::None;
            }
            let Some(shifted) = shift_ancestor_onto_reentry(phase, incoming) else {
                continue;
            };
            // Closing is the whole point. Folding alone does not converge a
            // strided block: the fold's result is recorded at THIS depth, the
            // next re-entry arrives one stride deeper, finds no canonical of its
            // own again, and the block still spends one state per phase until the
            // ceiling. Only a shifted phase that already COVERS the re-entry
            // subsumes the back edge.
            if matches!(widen_state(key, &canonicalized(&shifted), &reentry), Ok(None)) {
                return ShiftedPhases::Covered;
            }
            // Not covered yet: broaden so a later re-entry is.
            let base = acc.unwrap_or_else(|| incoming.clone());
            acc = Some(match widen_state(key, &base, &shifted) {
                Ok(Some(widened)) => widened,
                Ok(None) => base,
                Err(_) => return ShiftedPhases::None,
            });
        }
        acc.map_or(ShiftedPhases::None, ShiftedPhases::Folded)
    }

    /// Records the frame-privacy concession a generalization rests on: both the
    /// flag that bars the discharge and the label that names it on the result.
    /// See [`VmCfgAssumption::AssumesVmStackFramePrivate`] for why the flag alone
    /// was not enough.
    fn concede_frame_privacy(&mut self) {
        self.frame_privacy_conceded = true;
        self.cfg
            .assumptions
            .insert(VmCfgAssumption::AssumesVmStackFramePrivate);
    }

    /// Charges a generalization and records it as its phase's canonical. Split
    /// out so the lookups above can borrow `explored`/`widened` in place and end
    /// that borrow before this mutates.
    fn continue_from(
        &mut self,
        key: Va,
        folded: Result<(SymbolicVmInitialState, bool), String>,
    ) -> GeneralizeAcross {
        match folded {
            Ok((widened, hid_frame_pointer)) => {
                let bytes = self.state_bytes(&widened);
                self.charge(bytes, key, key);
                self.record_canonical(key, widened.clone());
                GeneralizeAcross::Continue {
                    state: widened,
                    hid_frame_pointer,
                }
            }
            // Never a VM-stack drift any more: both paths above are restricted to
            // the re-entry's own depth, so `widen_state` cannot refuse on the VSP
            // here. Only the loop head still sees cross-depth re-entries, and it
            // renders the strides itself.
            Err(reason) => GeneralizeAcross::Stall(reason),
        }
    }

    fn visit(&mut self, item: &Item, pending: &mut Vec<Item>) {
        self.mem_trace();
        // Walk this block (and the successors it queues) under its profile.
        // `None` is the base profile, so the sound path is unchanged.
        self.active.clone_from(&item.profile);
        let key = item.start.vip;
        // Hard memory bound on the transition walk: once the post-transition
        // budget is spent, fail closed for this entry instead of exploring (and
        // retaining) ever-deeper symbolic state. Inert until a transition fires.
        if self.transition_budget_exhausted() {
            self.cfg.limit_hit.get_or_insert(CfgLimitHit::TransitionBudget);
            self.stall(key, key, "cross-profile exploration budget exhausted");
            return;
        }
        if self.cfg.visits >= self.limits.max_visits {
            self.cfg.limit_hit.get_or_insert(CfgLimitHit::Visits);
            return;
        }
        let at = std::time::Instant::now();
        let entered = self.enter_loop(item);
        note_slow("enter_loop", at);
        let Some(mut symbolic_in) = entered else {
            return;
        };
        if self.explored.get(&key).is_some_and(|states| states.contains(&symbolic_in)) {
            return;
        }
        let visited = self.explored.get(&key).map_or(0, Vec::len);
        if visited >= self.limits.max_visits_per_block {
            // No path-local loop generalizes this block (it is re-reached through
            // many sibling paths whose only drift is fresh-name bookkeeping), so
            // the per-path widening never engaged. Fold every state seen here to a
            // single sound over-approximation and continue from it, converging the
            // block before the cap stalls it. A hard ceiling still fails closed if
            // even the generalization keeps broadening.
            if visited >= self.limits.max_visits_per_block.saturating_mul(2) {
                self.cfg.limit_hit.get_or_insert(CfgLimitHit::BlockVisits);
                self.stall(key, key, "symbolic state keeps changing at this block: ceiling");
                return;
            }
            let at = std::time::Instant::now();
            let across = self.generalize_across(key, &symbolic_in);
            note_slow("generalize_across", at);
            match across {
                GeneralizeAcross::Closed => return,
                GeneralizeAcross::Stall(reason) => {
                    self.cfg.limit_hit.get_or_insert(CfgLimitHit::BlockVisits);
                    self.stall(
                        key,
                        key,
                        format!("symbolic state keeps changing at this block: {reason}"),
                    );
                    return;
                }
                GeneralizeAcross::Continue {
                    state: widened,
                    hid_frame_pointer,
                } => {
                    // Concede frame privacy only when the fold actually hid a
                    // frame-pointer-bearing cell. Conceding on every fold cost
                    // 115 CFG-complete entries the proven tier even though no
                    // frame pointer had been lost (ptr_lost=false in all of them).
                    self.frame_privacy_conceded |= hid_frame_pointer;
                    symbolic_in = widened;
                }
            }
            if self.explored.get(&key).is_some_and(|states| states.contains(&symbolic_in)) {
                return;
            }
        }
        // Held in `explored` and the path node (which clones the containers).
        let held = self.state_bytes(&symbolic_in).saturating_mul(2);
        if !self.charge(held, key, key) {
            return;
        }
        self.explored.entry(key).or_default().push(symbolic_in.clone());
        self.cfg.visits += 1;
        self.current = Some(Rc::new(PathNode {
            key,
            state: symbolic_in.clone(),
            parent: item.path.clone(),
        }));
        let at = std::time::Instant::now();
        let decoded_ok = self.decode_block(item);
        note_slow("decode_block", at);
        if !decoded_ok {
            return;
        }
        let (final_state, stop, boundary, error, instructions) = {
            let block = &self.decoded[&key];
            (
                block.final_state,
                block.stop,
                block.boundary,
                block.error.clone(),
                Rc::clone(&block.instructions),
            )
        };
        if let Some(error) = error {
            self.stall(key, final_state.vip, format!("decode failed: {error}"));
            return;
        }
        if stop != LinearDecodeStop::ControlFlowBoundary {
            // The linear decoder stopped because the next handler has no spec
            // under the current profile. That handler may be a proven
            // cross-profile VM transition: it pops a concrete target off the VM
            // stack and re-homes dispatch into a new register profile. If so,
            // continue the walk at the target under the new profile; otherwise
            // keep the exact fail-closed stall below.
            if stop == LinearDecodeStop::MissingHandlerSpec
                && self.follow_cross_profile_transition(
                    key,
                    final_state,
                    &instructions,
                    &symbolic_in,
                    pending,
                )
            {
                return;
            }
            let detail = match stop {
                LinearDecodeStop::MissingHandlerSpec => {
                    let recovery = self.cache.recoveries.get(&final_state.handler);
                    // Reaching here means follow_cross_profile_transition already
                    // returned false, so record what the refusal had to work with:
                    // the transition count against its cap (a saturated cap is a
                    // budget refusal wearing a MissingHandlerSpec label), the roles
                    // the codec recovery was given, and where the trace stopped.
                    let roles = self.active.as_ref().map_or(&self.config, |c| &c.config);
                    format!(
                        "undecodable opcode: handler {} has no recovered specification                          (status {:?}, semantic {:?}; \
                         transitions {}/{}, vip {:?}, key {:?}, hreg {:?}, dir {:?}, \
                         trace_stop {:?}, fragments {:?}; transition refused: {}; {})",
                        final_state.handler,
                        recovery.map(|r| r.status),
                        recovery
                            .and_then(|r| r.handler_analysis.as_ref())
                            .map(|a| a.semantic),
                        self.profile_transitions,
                        self.limits.max_profile_transitions,
                        roles.roles.vip,
                        roles.roles.rolling_key,
                        roles.roles.handler,
                        roles.bytecode_direction,
                        recovery.map(|r| r.trace_stop.clone()),
                        recovery.map(|r| r.trace_fragments),
                        self.transition_refusal.as_deref().unwrap_or("not attempted"),
                        recovery.map_or(String::new(), |r| r
                            .diagnostics
                            .iter()
                            .take(8)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("; ")),
                    )
                }
                other => format!("linear decode stopped with {other:?}"),
            };
            self.stall(key, final_state.vip, detail);
            return;
        }
        let symbolic_boundary =
            boundary.filter(|semantic| *semantic != VmHandlerSemantic::NativeCall);
        self.seeded.set(false);
        let symbolic = match execute_symbolic_linear_block_with_initial_state(
            &instructions,
            symbolic_boundary,
            self.limits.entry.symbolic,
            &symbolic_in,
            Some(&observed_source(self)),
        ) {
            Ok(report)
                if report.memory_tainted
                    && (boundary == Some(VmHandlerSemantic::UnconditionalBranch)
                        || retry_every_tainted_block())
                    // Only the OBSERVED escape bars this retry. A loop
                    // generalization that conceded privacy has not shown the
                    // frame is aliased, and the retry's own result is labeled
                    // the same unproven tier, so denying it bought nothing and
                    // cost every entry that closed a strided loop.
                    && !self.frame_not_private
                    && !self.frame_pointer_lost =>
            {
                // A tainted branch block stalls; retry it once with loaded-data
                // store addresses under the labeled disjointness assumption.
                let relaxed = crate::SymbolicExecutionLimits {
                    assume_loaded_store_addresses_disjoint: true,
                    ..self.limits.entry.symbolic
                };
                self.seeded.set(false);
                match execute_symbolic_linear_block_with_initial_state(
                    &instructions,
                    symbolic_boundary,
                    relaxed,
                    &symbolic_in,
                    Some(&observed_source(self)),
                ) {
                    Ok(retry) if !retry.memory_tainted => retry,
                    _ => report,
                }
            }
            Ok(report) => report,
            Err(error) => {
                self.stall(
                    key,
                    error.vip,
                    format!(
                        "symbolic execution failed [{}]: {error}",
                        symbolic_failure_class(&error.message)
                    ),
                );
                return;
            }
        };
        self.note_frame_use(&symbolic);
        if symbolic.assumed_external_disjoint_from_vm_stack {
            self.cfg
                .assumptions
                .insert(VmCfgAssumption::AssumesExternalStoresDisjointFromVmStack);
        }
        if symbolic.served_read_only_image {
            self.cfg
                .assumptions
                .insert(VmCfgAssumption::AssumesImageProtectionStable);
        }
        if self.seeded.get() {
            self.cfg.assumptions.insert(VmCfgAssumption::ObservedGlobal);
        }
        match boundary {
            Some(VmHandlerSemantic::VmExit) => {
                self.follow_nested_entry(key, final_state.vip, &symbolic, pending);
            }
            Some(VmHandlerSemantic::UnconditionalBranch) => {
                self.follow_branch(key, final_state, &symbolic, pending);
            }
            Some(VmHandlerSemantic::NativeCall) => {
                self.follow_native_call(key, final_state, &symbolic, pending);
            }
            other => self.stall(
                key,
                final_state.vip,
                format!("unsupported control-flow boundary {other:?}"),
            ),
        }
    }

    fn branch_dispatch(&self, handler: Va) -> Result<(DispatchExtractionCandidate, Va), String> {
        let recovery = self
            .cache
            .recoveries
            .get(&handler)
            .ok_or("branch handler has no recovery")?;
        let effects = recovery
            .effects
            .as_ref()
            .filter(|effects| {
                effects.loads_vip_from_vsp
                    && effects.resets_key_from_vip
                    && effects.reinitializes_handler_base
            })
            .ok_or("branch handler does not load VIP from the VM stack and re-anchor dispatch")?;
        let _ = effects;
        let [dispatch] = recovery.dispatch_analysis.candidates.as_slice() else {
            return Err(format!(
                "branch handler has {} dispatch candidates, exactly one required",
                recovery.dispatch_analysis.candidates.len()
            ));
        };
        // A table-indexed dispatch with a static base depends only on the VIP,
        // which this handler loads from the VM stack; other codecs must be
        // proven to run after the VIP rebase.
        let static_table = dispatch
            .codec
            .table
            .is_some_and(|table| table.table_base.is_some());
        if !dispatch.vip_rebased_before_source && !static_table {
            return Err("branch dispatch does not read from the rebased VIP".to_owned());
        }
        let anchor = dispatch
            .handler_rebase
            .ok_or("branch dispatch has no handler anchor")?;
        Ok((dispatch.clone(), anchor))
    }

    /// Resolves a computed branch whose next-VIP reads a read-only image
    /// jump/const table at a proven-finite index, returning the exact concrete
    /// targets or `None` when no sound read-only table recovery applies.
    fn resolve_readonly_image_table(&self, bound_target: &Expr) -> Option<Vec<u64>> {
        let limits = ImageTableBranchLimits {
            input_domain: self.limits.input_domain,
            domain: self.limits.domain,
            ..ImageTableBranchLimits::default()
        };
        recover_image_table_targets(self.pe, bound_target, limits)
            .ok()
            .filter(|report| {
                report.complete
                    && !report.targets.is_empty()
                    && report.targets.len() <= self.limits.max_targets
            })
            .map(|report| report.targets)
    }

    #[allow(clippy::too_many_lines)]
    fn follow_branch(
        &mut self,
        key: Va,
        final_state: RelativeDispatchState,
        symbolic: &SymbolicVmExecutionReport,
        pending: &mut Vec<Item>,
    ) {
        let vip = final_state.vip;
        let Some(target) = symbolic.branch_target.as_ref() else {
            self.stall(key, vip, "branch has no symbolic target");
            return;
        };
        if symbolic.memory_tainted {
            self.stall(
                key,
                vip,
                "branch target follows an imprecise external-memory store",
            );
            return;
        }
        let (dispatch, anchor) = match self.branch_dispatch(final_state.handler) {
            Ok(found) => found,
            Err(reason) => {
                self.stall(key, vip, reason);
                return;
            }
        };
        // Bind image-base-relative native registers (the relocation delta) to 0
        // under the preferred-base frame, so a branch relocated through the
        // image base resolves instead of stalling on that residual addend.
        let is_image_va = |value: u64| bytecode_section(self.pe, Va(value)).is_some();
        let (bound_target, direct_registers) =
            bind_preferred_base_registers(target, &is_image_va);
        let image_lo = self.pe.optional.image_base.0;
        let image_hi = image_lo.saturating_add(u64::from(self.pe.optional.size_of_image));
        let in_image = |value: u64| value >= image_lo && value < image_hi;
        let mut gate_route = None;
        let mut primary_resolved = false;
        let (targets, guarded, bound_registers, mut environment_gate) =
            match resolve_targets(&bound_target, self.limits) {
                Ok((targets, guarded)) => {
                    primary_resolved = true;
                    if diagnostics_enabled() {
                        gate_route = Some(EnvironmentGateRoute::Primary);
                    }
                    (targets, guarded, direct_registers, false)
                }
                Err(reason) => {
                    // A VMProtect CPUID / timestamp-counter anti-analysis gate
                    // stalls here because the environment value is statically
                    // unknown: resolve it by a sound both-target over-approximation.
                    if let Some((targets, guarded, gate_registers)) =
                        resolve_environment_gate(target, &is_image_va, self.limits)
                    {
                        gate_route = Some(EnvironmentGateRoute::LegacyGate);
                        (targets, guarded, gate_registers, true)
                    } else if let Some(targets) = self.resolve_readonly_image_table(&bound_target) {
                        // A jump/const-table branch whose next-VIP reads a
                        // read-only image table at a proven-finite index: the
                        // table bytes are definitive, so enumerate the exact
                        // concrete targets. Relies on the preferred-base frame
                        // (the table anchor is image-base relative).
                        self.cfg
                            .assumptions
                            .insert(VmCfgAssumption::PreferredImageBase);
                        (targets, true, direct_registers.clone(), false)
                    } else if self
                        .resolve_observed_branch_targets(key, vip, &dispatch, anchor, symbolic, pending)
                    {
                        // EXPERIMENTAL: the target depends on an untracked runtime
                        // memory read, so static resolution fails closed. A target
                        // observed for this branch site (and re-validated as real
                        // bytecode) resolved it; nothing else to enumerate here.
                        return;
                    } else if let Some(gate) =
                        resolve_generalized_environment_gate(target, &is_image_va, &in_image, self.limits)
                    {
                        // The branch is gated on a host-environment / API source
                        // the narrow CPUID gate does not model: the same sound
                        // both-target fork, labeled.
                        gate_route = Some(EnvironmentGateRoute::Generalized);
                        (gate.targets, gate.guarded, gate.bound_registers, true)
                    } else {
                        let reason = format!("unresolved branch target: {reason}");
                        record_environment_fork(vip, EnvironmentGateRoute::Refused, target, &in_image, &[]);
                        if DIAG_ENABLED.with(Cell::get) {
                            let causes = diagnose_branch_refusals(
                                &bound_target.simplify(),
                                self.limits.input_domain,
                            );
                            let frame_hi = DIAG_FRAME_HI.with(Cell::get);
                            DIAG_SINK.with(|sink| {
                                sink.borrow_mut().push(BranchRefusalDiagnostic {
                                    vip,
                                    stall_reason: reason.clone(),
                                    frame_hi,
                                    causes,
                                });
                            });
                        }
                        self.stall(key, vip, reason);
                        return;
                    }
                }
            };
        if !bound_registers.is_empty() {
            self.cfg
                .assumptions
                .insert(VmCfgAssumption::PreferredImageBase);
        }
        // A fork the ordinary path resolved over an environment-sourced condition
        // (the source only fed a flag word) is the same both-arm superset: label
        // it too, so no environment fork is silent.
        if primary_resolved && targets.len() > 1 && has_environment_source(target, &in_image) {
            environment_gate = true;
        }
        if environment_gate {
            self.cfg
                .assumptions
                .insert(VmCfgAssumption::OverApproximatedEnvironmentGate);
        }
        if let Some(route) = gate_route {
            record_environment_fork(vip, route, target, &in_image, &targets);
        }
        // Consulted at most once, only when a target falls out of the bytecode
        // section (CASE-C). An observed real target for this branch site, once
        // re-validated as bytecode, supersedes the data-pointing static target.
        let mut observed_consulted = false;
        let mut observed_followed = false;
        let section = self.ctx_section();
        for value in targets {
            let target_va = Va(value);
            if section.is_none() || bytecode_section(self.pe, target_va) != section {
                if !observed_consulted {
                    observed_followed = self
                        .resolve_observed_branch_targets(key, vip, &dispatch, anchor, symbolic, pending);
                    observed_consulted = true;
                }
                if observed_followed {
                    // EXPERIMENTAL: the computed target points into data; the
                    // observed real target replaced it. Suppress the CASE-C stall.
                    continue;
                }
                let target_characteristics = self
                    .pe
                    .optional
                    .image_base
                    .checked_rva_of(target_va)
                    .and_then(|rva| {
                        self.pe
                            .sections
                            .iter()
                            .find(|section| section.contains_mapped_rva(rva))
                    })
                    .map(|section| section.characteristics);
                // EXPERIMENTAL cross-VM-section continuation (see
                // `VmCfgLimits::follow_cross_vm_section`). Labeled, so the entry
                // never reaches the proven tier.
                if cross_section_followable(
                    self.limits.follow_cross_vm_section,
                    target_characteristics,
                ) {
                    if self.follow_bytecode_target(
                        key, vip, target_va, value, &dispatch, anchor, guarded, symbolic, pending,
                    ) {
                        self.cfg
                            .assumptions
                            .insert(VmCfgAssumption::AssumesCrossVmSectionContinuation);
                    }
                    continue;
                }
                self.stall(
                    key,
                    vip,
                    format!(
                        "branch target {target_va} is outside the bytecode section: {}",
                        out_of_bytecode_section_reason(target_characteristics)
                    ),
                );
                continue;
            }
            self.follow_bytecode_target(key, vip, target_va, value, &dispatch, anchor, guarded, symbolic, pending);
        }
    }

    /// Decodes into a VALIDATED in-section bytecode `target_va`, adds the branch
    /// edge, and queues the target block. Returns false (emitting a stall) when
    /// the dispatch decode fails. Shared by the sound branch loop and the
    /// observed-branch-target bridge; the two differ only in how `target_va` was
    /// obtained — the caller must already have proven it is in-section bytecode.
    #[allow(clippy::too_many_arguments)]
    fn follow_bytecode_target(
        &mut self,
        key: Va,
        vip: Va,
        target_va: Va,
        rolling_key: u64,
        dispatch: &DispatchExtractionCandidate,
        anchor: Va,
        guarded: bool,
        symbolic: &SymbolicVmExecutionReport,
        pending: &mut Vec<Item>,
    ) -> bool {
        let first = match dispatch.codec.decode_direction(
            self.pe,
            RelativeDispatchState {
                vip: target_va,
                rolling_key,
                handler: anchor,
            },
            self.ctx_direction(),
        ) {
            Ok(first) => first,
            Err(error) => {
                self.stall(key, vip, format!("target {target_va} dispatch failed: {error}"));
                return false;
            }
        };
        self.add_edge(CfgEdge {
            from: key,
            to: first.state_after.vip,
            kind: if guarded {
                CfgEdgeKind::GuardedBranch
            } else {
                CfgEdgeKind::Branch
            },
        });
        pending.push(Item {
            bytes: 0,
            start: first.state_after,
            address: Some(target_va),
            symbolic: symbolic.continuation_state(),
            path: self.current.clone(),
            profile: self.active.clone(),
        });
        true
    }

    /// EXPERIMENTAL observed-branch-target bridge. At a branch that otherwise
    /// fails closed (a computed target into data — CASE-C — or a target gated on
    /// an untracked runtime memory read), consult the observed-run seed for the
    /// concrete target VIP(s) recorded for this branch site (keyed by the VM
    /// branch VIP). Each observed target is followed ONLY if it still validates
    /// as real in-image bytecode in the entry's proven bytecode section (see
    /// [`followable_observed_targets`]): observation seeds a resolution, it can
    /// never turn a data address into code. Each followed edge is labeled
    /// [`VmCfgAssumption::ObservedBranchTarget`]. Returns true if at least one
    /// observed target was followed (the caller then skips its fail-closed
    /// stall); false — including the whole default/sound path, where the seed is
    /// empty — leaves the fail-closed behavior byte-for-byte unchanged.
    fn resolve_observed_branch_targets(
        &mut self,
        key: Va,
        vip: Va,
        dispatch: &DispatchExtractionCandidate,
        anchor: Va,
        symbolic: &SymbolicVmExecutionReport,
        pending: &mut Vec<Item>,
    ) -> bool {
        let targets =
            followable_observed_targets(self.pe, self.ctx_section(), self.observed_branch_targets, vip);
        if targets.is_empty() || targets.len() > self.limits.max_targets {
            return false;
        }
        let mut followed = false;
        for target_va in targets {
            if self.follow_bytecode_target(
                key,
                vip,
                target_va,
                target_va.get(),
                dispatch,
                anchor,
                true,
                symbolic,
                pending,
            ) {
                followed = true;
            }
        }
        if followed {
            self.cfg
                .assumptions
                .insert(VmCfgAssumption::ObservedBranchTarget);
            self.cfg
                .assumptions
                .insert(VmCfgAssumption::AssumesDynamicBranchTargetsComplete);
        }
        followed
    }

    /// Follows a proven VM exit whose return target is another entry stub of
    /// the same VM. A target that is not provably such an entry ends the path
    /// as an ordinary native return; a proven entry that cannot be composed
    /// stalls.
    fn follow_nested_entry(
        &mut self,
        key: Va,
        vip: Va,
        symbolic: &SymbolicVmExecutionReport,
        pending: &mut Vec<Item>,
    ) {
        let Some(layout) = self
            .cfg
            .blocks
            .get(&key)
            .and_then(|block| block.vm_exit.clone())
        else {
            return;
        };
        let exit = match recover_native_exit_state(symbolic, &layout) {
            Ok(exit) => exit,
            Err(error) => {
                if std::env::var_os("VMPX_NESTED_TRACE").is_some() {
                    eprintln!("[nested] {key}: exit state unrecovered: {error}");
                }
                return;
            }
        };
        let simplified = exit.return_address.simplify();
        let target = if let ExprKind::Const { value } = simplified.kind {
            self.pe.optional.image_base.checked_rva_of(Va(value))
        } else {
            recover_modulo_image_targets(
                self.pe,
                &exit.return_address,
                ModuloImageTargetLimits::default(),
            )
            .ok()
            .filter(|report| report.complete && report.targets.len() == 1)
            .and_then(|report| report.targets[0].rva)
        };
        let Some((frame_entry, candidate)) = target.and_then(|rva| nested_entry_stub(self.pe, rva))
        else {
            return;
        };
        let at = candidate.rva;
        if !self.nested_sites.contains(&at)
            && self.nested_sites.len() >= self.limits.max_nested_entries
        {
            self.stall(
                key,
                vip,
                format!("nested VM entry {at} exceeds the nesting limit"),
            );
            return;
        }
        let setup = match prepare_entry_stream(self.pe, &candidate, self.limits.entry) {
            Ok(setup) => setup,
            Err(error) => {
                self.stall(
                    key,
                    vip,
                    format!("nested VM entry {at} setup failed: {error}"),
                );
                return;
            }
        };
        // The nested entry's OWN profile, proven by `prepare_entry_stream` (its
        // bootstrap dispatch, register roles, and bytecode direction all recover
        // or this is an `Err` stall above). When it matches the current region we
        // continue under the active profile (byte-for-byte as before); when it
        // differs we continue under the nested entry's own proven profile instead
        // of stalling. Sound: nothing is guessed — the switch rides the same
        // positive proof that recovers any root entry, and a wrong profile only
        // misdirects decoding, which then fails closed (the oracle never inflates).
        // Bounded: still one nested entry, capped by `max_nested_entries`.
        let nested_global_table = bootstrap_global_table(&setup.bootstrap);
        let nested_section = bytecode_section(self.pe, setup.bootstrap.initial_state.vip);
        let shares_profile = setup.profile == self.ctx_profile()
            && nested_global_table == self.ctx_global_table()
            && nested_section == self.ctx_section();
        let nested_profile = if shares_profile {
            self.active.clone()
        } else {
            Some(Rc::new(ProfileContext {
                config: entry_stream_config(&setup.profile, self.limits.entry),
                profile: setup.profile,
                direction: setup.profile.bytecode_direction,
                global_table: nested_global_table,
                section: nested_section,
            }))
        };
        let (cells, top_offset, _) = match recover_entry_frame(
            self.pe,
            frame_entry,
            setup.profile.effect_roles.vsp,
            self.limits.entry.trace,
        ) {
            Ok(result) => result,
            Err(error) => {
                self.stall(
                    key,
                    vip,
                    format!("nested VM entry {at} frame unproven: {error}"),
                );
                return;
            }
        };
        let continuation = cells.iter().find_map(|cell| match cell {
            NativeFrameSource::ReturnAddress { continuation, .. } => Some(*continuation),
            _ => None,
        });
        let (frame, _) = frame_initial_state(&cells, top_offset);
        let initial = match compose_native_frame(
            self.pe,
            symbolic,
            &exit,
            frame,
            cells.len(),
            continuation,
        ) {
            Ok(initial) => initial,
            Err(error) => {
                self.stall(
                    key,
                    vip,
                    format!("nested VM entry {at} composition failed: {error}"),
                );
                return;
            }
        };
        self.nested_followed += 1;
        self.nested_sites.insert(at);
        self.add_edge(CfgEdge {
            from: key,
            to: setup.first_state.vip,
            kind: CfgEdgeKind::NestedEntry,
        });
        pending.push(Item {
            bytes: 0,
            start: setup.first_state,
            address: Some(setup.bootstrap.initial_state.vip),
            symbolic: initial,
            path: self.current.clone(),
            profile: nested_profile,
        });
    }

    /// Every store the assumption narrowed targets a native-rooted or opaque
    /// address. It cannot alias the tracked VM-stack cells when the frame is the
    /// entry stub's fresh allocation (no pointer to it existed before the VM),
    /// every cell lies inside it, and the VM never exported a pointer into it.
    fn discharge_vm_stack_assumption(&mut self) {
        // Data read from the frame is benign only if no cell holding a frame
        // pointer was discarded (an untracked load could otherwise return it).
        let derived_leak = self.frame_pointer_lost && self.frame_derived_escape;
        if self.private_frame_hi.is_some()
            && !self.frame_not_private
            && !self.frame_privacy_conceded
            && !derived_leak
            && self.nested_followed == 0
        {
            self.cfg
                .assumptions
                .remove(&VmCfgAssumption::AssumesExternalStoresDisjointFromVmStack);
            return;
        }
        // Record WHICH condition refused the discharge. These entries are already
        // CFG-complete, so each one is a candidate for the proven tier and the
        // blocking condition is the lever: raising N2 means discharging here, not
        // completing more CFGs.
        self.cfg.discharge_refusal = Some(format!(
            "no_private_frame={} frame_not_private={} conceded={} [addr_escape={} store_past={}] derived_leak={} (ptr_lost={} ptr_escape={}) nested={}",
            self.private_frame_hi.is_none(),
            self.frame_not_private,
            self.frame_privacy_conceded,
            self.frame_escape_seen,
            self.frame_store_past
                .map_or_else(|| "no".to_owned(), |(hi, w)| format!("hi={hi:#x}<written={w:#x}")),
            derived_leak,
            self.frame_pointer_lost,
            self.frame_derived_escape,
            self.nested_followed,
        ));
    }

    fn note_frame_use(&mut self, report: &SymbolicVmExecutionReport) {
        let past_frame = self
            .private_frame_hi
            .zip(report.vm_stack_store_hi)
            .is_some_and(|(frame_hi, written)| written > frame_hi);
        // Which of the two reasons fired, for the discharge diagnostic.
        self.frame_escape_seen |= report.vm_stack_address_escapes;
        if past_frame {
            self.frame_store_past = self
                .private_frame_hi
                .zip(report.vm_stack_store_hi)
                .map(|(frame_hi, written)| (frame_hi, written));
        }
        self.frame_not_private |= report.vm_stack_address_escapes || past_frame;
        self.frame_derived_escape |= report.vm_stack_derived_escapes;
        self.frame_pointer_lost |= report.vm_stack_pointer_lost;
    }

    fn follow_native_call(
        &mut self,
        key: Va,
        final_state: RelativeDispatchState,
        symbolic: &SymbolicVmExecutionReport,
        pending: &mut Vec<Item>,
    ) {
        let vip = final_state.vip;
        let Some(spec) = self
            .cache
            .recoveries
            .get(&final_state.handler)
            .and_then(|recovery| recovery.boundary_spec.clone())
        else {
            self.stall(
                key,
                vip,
                "native-call boundary has no proven decode specification",
            );
            return;
        };
        let decoded = match spec.decode_direction(self.pe, final_state, self.ctx_direction()) {
            Ok(decoded) => decoded,
            Err(error) => {
                self.stall(
                    key,
                    vip,
                    format!("native-call boundary decode failed: {error}"),
                );
                return;
            }
        };
        self.seeded.set(false);
        let call = match execute_symbolic_native_call_boundary_with_initial_state(
            decoded.state_before.vip,
            &decoded.outcome,
            self.limits.entry.symbolic,
            &symbolic.continuation_state(),
            Some(&observed_source(self)),
        ) {
            Ok(call) => call,
            Err(error) => {
                self.stall(key, vip, format!("native-call effect failed: {error}"));
                return;
            }
        };
        self.note_frame_use(&call);
        if self.seeded.get() {
            self.cfg.assumptions.insert(VmCfgAssumption::ObservedGlobal);
        }
        self.cfg.record_native_call(key, &decoded.outcome);
        self.add_edge(CfgEdge {
            from: key,
            to: decoded.state_after.vip,
            kind: CfgEdgeKind::NativeCallReturn,
        });
        pending.push(Item {
            bytes: 0,
            start: decoded.state_after,
            address: None,
            symbolic: call.continuation_state(),
            path: self.current.clone(),
            profile: self.active.clone(),
        });
    }

    /// At a linear decode that stopped because the next handler has no spec
    /// under the CURRENT profile, attempt a PROVEN cross-profile VM transition:
    /// the unresolved handler pops a concrete target off the VM stack and the VM
    /// resumes under a different register profile (new VIP/key/handler roles,
    /// possibly a flipped bytecode direction). Returns `true` and queues the
    /// target block under the new profile iff [`analyze_vm_transition`] proves
    /// the popped-target shape, the popped target is concrete, and
    /// [`resolve_vm_profile_roles`] proves the new profile recovers; otherwise
    /// `false`, leaving the caller's fail-closed stall byte-for-byte unchanged.
    ///
    /// Soundness: nothing here is guessed. The transition fires only on positive
    /// proof of both the pop shape and the new roles; a non-provable shape falls
    /// through. A wrong switch cannot false-complete — it misdirects downstream
    /// decoding, which then fails closed under the wrong profile (the completeness
    /// oracle only ever stalls, never inflates). Termination is unchanged: the
    /// successor is queued like any other block and re-reaching a VA is caught by
    /// the existing VA-keyed visited/loop machinery (VAs are profile-disjoint).
    #[allow(clippy::too_many_lines)]
    fn follow_cross_profile_transition(
        &mut self,
        key: Va,
        final_state: RelativeDispatchState,
        instructions: &[DecodedVmInstruction],
        symbolic_in: &SymbolicVmInitialState,
        pending: &mut Vec<Item>,
    ) -> bool {
        // Hard per-entry budget: each transition opens a new profile region whose
        // retained symbolic state is deeper than the last, so an unbounded chain
        // allocates without limit. Beyond the budget, fail closed (the caller
        // stalls) instead of switching. Sound: a bounded walk that hits the cap
        // stalls — it never false-completes.
        if !self.transition_sites.contains(&key)
            && self.transition_sites.len() >= self.limits.max_profile_transitions
        {
            {
                self.transition_refusal = Some("transition budget spent".to_owned());
                return false;
            }
        }
        // A genuine transition handler ends a short handler-stream block; a very
        // long prefix is not a clean transition and symbolic-executing it would
        // allocate a large state for nothing. Fail closed past a modest cap.
        const MAX_TRANSITION_PREFIX: usize = 512;
        if instructions.len() > MAX_TRANSITION_PREFIX {
            {
                self.transition_refusal = Some("prefix too long for a clean transition".to_owned());
                return false;
            }
        }
        let profile = self.ctx_profile();
        let old_vsp = profile.effect_roles.vsp;
        let context_base = profile.effect_roles.context_base;
        let Some(entry_rva) = self.pe.optional.image_base.checked_rva_of(final_state.handler)
        else {
            {
                self.transition_refusal = Some("handler VA outside the image".to_owned());
                return false;
            }
        };
        // Prove the unresolved handler is a popped-target cross-profile transition.
        let Ok(transition) = analyze_vm_transition(
            self.pe,
            entry_rva,
            old_vsp,
            context_base,
            self.limits.entry.trace,
        ) else {
            {
                self.transition_refusal = Some("transition analysis failed".to_owned());
                return false;
            }
        };
        let Some(recovered) = transition.recovered else {
            {
                self.transition_refusal = Some("no popped-target transition recovered".to_owned());
                return false;
            }
        };
        debug_assert!(recovered.report.state_effects_proven);
        // Symbolic state at the transition handler: the decoded prefix executed
        // under the current profile, with no control boundary consumed.
        self.seeded.set(false);
        let Ok(symbolic) = execute_symbolic_linear_block_with_initial_state(
            instructions,
            None,
            self.limits.entry.symbolic,
            symbolic_in,
            Some(&observed_source(self)),
        ) else {
            {
                self.transition_refusal = Some("prefix symbolic execution failed".to_owned());
                return false;
            }
        };
        // Pop the concrete branch target off the VM stack.
        let Ok(pop) = execute_symbolic_linear_block_with_initial_state(
            &[],
            Some(VmHandlerSemantic::UnconditionalBranch),
            self.limits.entry.symbolic,
            &symbolic.continuation_state(),
            Some(&observed_source(self)),
        ) else {
            {
                self.transition_refusal = Some("pop of the branch target failed".to_owned());
                return false;
            }
        };
        let Some(Expr {
            kind: ExprKind::Const { value: target },
            ..
        }) = pop.branch_target
        else {
            {
                // The expression matters: if these are unresolved image reads
                // then this stall family and the `memory_load_epoch` family are
                // one cause, not two.
                self.transition_refusal = Some(format!(
                    "popped branch target is not a constant: {}",
                    pop.branch_target.as_ref().map_or_else(
                        || "none".to_owned(),
                        |target| format!("{target:?}").chars().take(110).collect::<String>(),
                    )
                ));
                return false;
            }
        };
        // Decode the first dispatch at the target under the transition's proven
        // direction and handler anchor.
        let Ok(first) = recovered.dispatch.decode_direction(
            self.pe,
            RelativeDispatchState {
                vip: Va(target),
                rolling_key: target,
                handler: recovered.handler_anchor,
            },
            recovered.report.bytecode_direction,
        ) else {
            {
                self.transition_refusal = Some("first dispatch at the target did not decode".to_owned());
                return false;
            }
        };
        // Resolve the new register profile from the target's first handler.
        let Some(first_handler_rva) = self
            .pe
            .optional
            .image_base
            .checked_rva_of(first.state_after.handler)
        else {
            {
                self.transition_refusal = Some("target handler VA outside the image".to_owned());
                return false;
            }
        };
        let Ok(roles) = resolve_vm_profile_roles(
            self.pe,
            first_handler_rva,
            &VmProfileRoleResolutionConfig {
                roles: recovered.roles,
                bytecode_direction: recovered.report.bytecode_direction,
                vsp_candidates: recovered.vsp_candidates.clone(),
                context_base,
                trace_limits: self.limits.entry.trace,
                proof_limits: self.limits.entry.dispatch_proof,
            },
        ) else {
            {
                self.transition_refusal = Some("role resolution at the target failed".to_owned());
                return false;
            }
        };
        let Some(resolved) = roles.recovered else {
            {
                self.transition_refusal = Some("no roles recovered at the target".to_owned());
                return false;
            }
        };
        // Proven switch: build the successor's per-block profile context. A new
        // profile recovers its own table anchors per handler, so none is threaded
        // through the transition (mirrors the profile-chain explorer).
        let new_profile = crate::VmProfile {
            handler_roles: resolved.handler_roles,
            effect_roles: resolved.effect_roles,
            bytecode_direction: resolved.bytecode_direction,
        };
        let context = Rc::new(ProfileContext {
            config: entry_stream_config(&new_profile, self.limits.entry),
            profile: new_profile,
            direction: resolved.bytecode_direction,
            global_table: None,
            section: bytecode_section(self.pe, Va(target)),
        });
        self.note_frame_use(&symbolic);
        self.note_frame_use(&pop);
        if symbolic.assumed_external_disjoint_from_vm_stack
            || pop.assumed_external_disjoint_from_vm_stack
        {
            self.cfg
                .assumptions
                .insert(VmCfgAssumption::AssumesExternalStoresDisjointFromVmStack);
        }
        if symbolic.served_read_only_image || pop.served_read_only_image {
            self.cfg
                .assumptions
                .insert(VmCfgAssumption::AssumesImageProtectionStable);
        }
        if self.seeded.get() {
            self.cfg.assumptions.insert(VmCfgAssumption::ObservedGlobal);
        }
        // Start the post-transition memory budget at the first switch.
        if self.transition_budget_base.is_none() {
            self.transition_budget_base = Some((self.total_instructions, self.cfg.visits));
        }
        self.profile_transitions += 1;
        self.transition_sites.insert(key);
        self.add_edge(CfgEdge {
            from: key,
            to: first.state_after.vip,
            kind: CfgEdgeKind::ProfileTransition,
        });
        pending.push(Item {
            bytes: 0,
            start: first.state_after,
            address: Some(Va(target)),
            symbolic: pop.continuation_state(),
            path: self.current.clone(),
            profile: Some(context),
        });
        true
    }
}

/// Decodes a virtualized function's blocks by following proven VM branches
/// from `entry`, using only native bootstrap data flow and handler behavior.
///
/// The initial VM stack is the frame recovered by stepping the native entry
/// (see [`recover_entry_frame`]); cells that cannot be proven stay unknown and
/// any branch depending on them stalls.
///
/// # Errors
///
/// Fails if the entry bootstrap cannot be recovered. Everything after that is
/// reported in the returned [`VmCfg`] as stalls or a hit limit.
pub fn recover_vm_cfg(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    limits: VmCfgLimits,
) -> Result<VmCfg, CfgError> {
    // Sound path: no observed seed. The empty run makes the overlay memory
    // source behaviourally identical to a bare read-only image and leaves every
    // VSP-changing recursion failing closed.
    recover_vm_cfg_with_observed(pe, entry, limits, &ObservedRun::default())
}

/// Like [`recover_vm_cfg`], but seeds provably-null image reads from concrete
/// runtime memory observed on ONE detonation. A thin wrapper over
/// [`recover_vm_cfg_with_observed`] carrying only the memory seed (no observed
/// loop depths).
///
/// # Errors
///
/// Fails if the entry bootstrap cannot be recovered; everything after is a stall
/// or hit limit in the returned [`VmCfg`].
pub fn recover_vm_cfg_with_observed_memory(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    limits: VmCfgLimits,
    observed: &crate::ObservedMemory,
) -> Result<VmCfg, CfgError> {
    let run = ObservedRun::new(observed.clone(), ObservedLoopDepths::new());
    recover_vm_cfg_with_observed(pe, entry, limits, &run)
}

/// Like [`recover_vm_cfg`], but driven by the EXPERIMENTAL observed-runtime seed
/// captured on ONE detonation (concrete memory + recursion depths).
///
/// Two observed-tier bridges may engage, each labeled on the returned CFG and
/// never the sound path:
/// - A global pointer slot that is zero in the static image but runtime-
///   initialized is resolved from [`ObservedRun::memory`], past the null-read
///   stall ([`VmCfgAssumption::ObservedGlobal`]).
/// - A recursion inlined as a VSP-changing loop, whose per-iteration VSP delta
///   is a PROVEN CONSTANT, is lifted as a BOUNDED UNROLL of exactly the depth in
///   [`ObservedRun::loop_depths`], past the "loop changes the VM stack pointer"
///   stall ([`VmCfgAssumption::ObservedRecursionDepth`]).
/// - A computed branch that otherwise fails closed — a constant target into a
///   data section (CASE-C) or a target gated on an untracked runtime memory read
///   — is resolved to a concrete target VIP in [`ObservedRun::branch_targets`],
///   but ONLY when that target re-validates as real in-image bytecode
///   ([`VmCfgAssumption::ObservedBranchTarget`]).
///
/// Pass [`ObservedRun::default`] (all empty) for the sound path, which recovers
/// exactly as [`recover_vm_cfg`].
///
/// # Errors
///
/// Fails if the entry bootstrap cannot be recovered; everything after is a stall
/// or hit limit in the returned [`VmCfg`].
pub fn recover_vm_cfg_with_observed(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    limits: VmCfgLimits,
    observed: &ObservedRun,
) -> Result<VmCfg, CfgError> {
    let setup = prepare_entry_stream(pe, entry, limits.entry)?;
    let mut private_frame_hi = None;
    let (initial, frame_cells_known, frame_note) = match recover_entry_frame(
        pe,
        entry.rva,
        setup.profile.effect_roles.vsp,
        limits.entry.trace,
    ) {
        Ok((cells, top_offset, exposed)) => {
            let (initial, known) = frame_initial_state(&cells, top_offset);
            let frame_hi = top_offset.wrapping_add((cells.len() as u64).wrapping_mul(8));
            DIAG_FRAME_HI.with(|hi| hi.set(Some(frame_hi)));
            if !exposed {
                private_frame_hi = Some(frame_hi.cast_signed());
            }
            (initial, known, None)
        }
        Err(note) => {
            DIAG_FRAME_HI.with(|hi| hi.set(None));
            (SymbolicVmInitialState::default(), 0, Some(note))
        }
    };
    let entry_address = setup.bootstrap.initial_state.vip;
    let seeded = Cell::new(false);
    let mut walker = Walker {
        pe,
        limits,
        config: entry_stream_config(&setup.profile, limits.entry),
        global_table: bootstrap_global_table(&setup.bootstrap),
        profile: setup.profile,
        transition_refusal: None,
        nested_followed: 0,
        nested_sites: BTreeSet::new(),
        current: None,
        direction: setup.profile.bytecode_direction,
        section: bytecode_section(pe, entry_address),
        active: None,
        profile_transitions: 0,
        transition_sites: BTreeSet::new(),
        transition_budget_base: None,
        cache: LinearDecodeCache::default(),
        decoded: BTreeMap::new(),
        explored: BTreeMap::new(),
        widened: BTreeMap::new(),
        cfg: VmCfg {
            entry: setup.first_state.vip,
            blocks: BTreeMap::new(),
            edges: Vec::new(),
            stalls: Vec::new(),
            limit_hit: None,
            visits: 0,
            frame_cells_known,
            frame_note,
            assumptions: BTreeSet::new(),
            discharge_refusal: None,
        },
        total_instructions: 0,
        live_bytes: 0,
        deadline: None,
        observed: &observed.memory,
        seeded: &seeded,
        observed_depths: &observed.loop_depths,
        observed_branch_targets: &observed.branch_targets,
        private_frame_hi,
        frame_not_private: false,
        frame_privacy_conceded: false,
        frame_derived_escape: false,
        frame_pointer_lost: false,
        frame_escape_seen: false,
        frame_store_past: None,
    };
    let mut pending = vec![Item {
        bytes: 0,
        start: setup.first_state,
        address: Some(entry_address),
        symbolic: initial,
        path: None,
        profile: None,
    }];
    let deadline = limits.max_wall_time.map(|budget| (Instant::now(), budget));
    // Hand the same instant to the symbolic executor: the check below bounds the
    // gaps between worklist items, not the work inside one.
    if let Some((start, budget)) = deadline {
        walker.limits.entry.symbolic.deadline = start.checked_add(budget);
        walker.deadline = start.checked_add(budget);
    }
    while let Some(item) = pending.pop() {
        if deadline.is_some_and(|(start, budget)| start.elapsed() >= budget) {
            walker.cfg.limit_hit.get_or_insert(CfgLimitHit::WallClock);
            walker.stall(
                item.start.vip,
                item.start.vip,
                "per-entry wall-clock budget exceeded",
            );
            break;
        }
        walker.live_bytes = walker.live_bytes.saturating_sub(item.bytes);
        let queued = pending.len();
        walker.visit(&item, &mut pending);
        walker.account_pending(&mut pending[queued..]);
        if walker
            .cfg
            .limit_hit
            .is_some_and(|hit| hit != CfgLimitHit::BlockVisits)
        {
            break;
        }
    }
    walker.discharge_vm_stack_assumption();
    Ok(walker.cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_one(bytes: &[u8]) -> iced_x86::Instruction {
        iced_x86::Decoder::with_ip(64, bytes, 0x1_4000_1000, iced_x86::DecoderOptions::NONE).decode()
    }

    /// A non-affine 32-bit flag operand the affine-domain path cannot
    /// characterize (an opaque intrinsic over a register), as in the oracle.
    fn opaque_operand() -> Expr {
        Expr::intrinsic(
            BitWidth::W32,
            "x86_cpuid_eax_65",
            vec![
                Expr::constant(BitWidth::W32, 1),
                Expr::register(BitWidth::W32, "native_rcx_initial"),
            ],
        )
    }

    /// The ZF bit of `vm_flags_sub_w32(operand, 4064)` as the oracle encodes it:
    /// `!( ~1 | (flags >> 6) )`, which is provably in {0, 1}.
    fn zf_bit(operand: Expr) -> Expr {
        let flags = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_sub_w32",
            vec![operand, Expr::constant(BitWidth::W32, 4064)],
        );
        let shifted =
            Expr::binary(BinaryOp::Shr, flags, Expr::constant(BitWidth::W64, 6)).unwrap();
        let masked =
            Expr::binary(BinaryOp::Or, Expr::constant(BitWidth::W64, !1_u64), shifted).unwrap();
        Expr::unary(crate::UnaryOp::Not, masked)
    }

    /// `low + bit * (high - low)`: `low` when `bit` is 0, `high` when it is 1.
    fn select(bit: Expr, low: u64, high: u64) -> Expr {
        let scaled = Expr::binary(
            BinaryOp::Mul,
            bit,
            Expr::constant(BitWidth::W64, high.wrapping_sub(low)),
        )
        .unwrap();
        Expr::binary(BinaryOp::Add, Expr::constant(BitWidth::W64, low), scaled).unwrap()
    }

    #[test]
    fn symbolic_failures_are_classified() {
        assert_eq!(
            symbolic_failure_class("proven null memory read of 1 byte(s) requires exception-flow analysis"),
            "null-read"
        );
        assert_eq!(symbolic_failure_class("instruction count exceeds limit 4096"), "resource-limit");
        assert_eq!(symbolic_failure_class("control-flow boundary is not implemented"), "unmodelled");
        assert_eq!(symbolic_failure_class("anything else"), "other");
    }

    #[test]
    fn oversized_address_site_key_falls_back_to_the_structural_hash() {
        let mut address = Expr::register(BitWidth::W64, "native_rax_initial");
        for round in 0..48_u64 {
            let mixed = Expr::binary(BinaryOp::Xor, address.clone(), Expr::constant(BitWidth::W64, round + 1)).unwrap();
            address = Expr::binary(BinaryOp::Add, address, mixed).unwrap();
        }
        let key = site_key(&address);
        assert!(key.starts_with('#') && key.len() < 32);
        assert_eq!(key, site_key(&address));
        let small = Expr::constant(BitWidth::W64, 7);
        assert_eq!(site_key(&small), format!("{small:?}"));
    }

    #[test]
    fn out_of_section_branch_target_is_classified_for_a_fail_closed_diagnostic() {
        const EXEC: u32 = 0x6000_0020; // CODE | EXECUTE | READ
        const RDATA: u32 = 0x4000_0040; // INITIALIZED_DATA | READ
        // coredll.ultra's 0x184AE lands in read-only .rdata: data, never bytecode.
        assert_eq!(
            out_of_bytecode_section_reason(Some(RDATA)),
            "it is in a non-executable data section"
        );
        // A different executable section is unproven, not followed.
        assert_eq!(
            out_of_bytecode_section_reason(Some(EXEC)),
            "it is in a different executable section not proven to hold this VM's bytecode"
        );
        // A target outside every section is unmapped.
        assert_eq!(
            out_of_bytecode_section_reason(None),
            "it is not mapped to any section"
        );
    }

    #[test]
    fn observed_branch_target_follows_only_validated_in_section_bytecode() {
        use crate::immutable_image::tests::{DATA_VA, IMAGE_BASE, RDATA_VA, two_section_pe64};
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        // The entry's bytecode section is index 0 (where RDATA_VA lives), standing
        // in for the executable protector section that holds this VM's bytecode.
        let section = bytecode_section(&pe, Va(RDATA_VA));
        assert!(section.is_some());
        let site = Va(IMAGE_BASE + 0x5000);

        // A VALID observed target inside the entry's bytecode section is followed.
        let valid = crate::ObservedBranchTargets::from_sites([(site.get(), vec![RDATA_VA])]);
        assert_eq!(
            followable_observed_targets(&pe, section, &valid, site),
            vec![Va(RDATA_VA)]
        );
        // An INVALID observed target into the writable .data section is REFUSED:
        // observation can never turn a data address into code (CASE-C soundness).
        let into_data = crate::ObservedBranchTargets::from_sites([(site.get(), vec![DATA_VA])]);
        assert!(followable_observed_targets(&pe, section, &into_data, site).is_empty());
        // An observed target outside every section is refused.
        let unmapped =
            crate::ObservedBranchTargets::from_sites([(site.get(), vec![IMAGE_BASE + 0x9000])]);
        assert!(followable_observed_targets(&pe, section, &unmapped, site).is_empty());
        // No observation for this site: empty, so the caller keeps its stall.
        assert!(followable_observed_targets(&pe, section, &valid, Va(IMAGE_BASE + 0x7777)).is_empty());
        // AMBIGUOUS evidence (one valid + one invalid target) is refused whole.
        let mixed =
            crate::ObservedBranchTargets::from_sites([(site.get(), vec![RDATA_VA, DATA_VA])]);
        assert!(followable_observed_targets(&pe, section, &mixed, site).is_empty());
        // No entry bytecode section: nothing is followable.
        assert!(followable_observed_targets(&pe, None, &valid, site).is_empty());
        // Default/sound path: an empty seed yields nothing (fail-closed unchanged).
        assert!(
            followable_observed_targets(&pe, section, &crate::ObservedBranchTargets::new(), site)
                .is_empty()
        );
    }

    fn roles(vip: Register, key: Register, handler: Register) -> crate::HandlerRegisterRoles {
        crate::HandlerRegisterRoles {
            vip,
            rolling_key: key,
            handler,
        }
    }

    fn profile(
        handler_roles: crate::HandlerRegisterRoles,
        vsp: Register,
        direction: crate::BytecodeDirection,
    ) -> crate::VmProfile {
        crate::VmProfile {
            handler_roles,
            effect_roles: crate::HandlerEffectRoles {
                vsp,
                context_base: Register::RSP,
            },
            bytecode_direction: direction,
        }
    }

    fn test_walker<'a, 'p>(
        pe: &'a PeImage<'p>,
        base: crate::VmProfile,
        observed: &'a ObservedRun,
        seeded: &'a Cell<bool>,
    ) -> Walker<'a, 'p> {
        let limits = VmCfgLimits::default();
        Walker {
            pe,
            limits,
            deadline: None,
            config: entry_stream_config(&base, limits.entry),
            global_table: None,
            profile: base,
            transition_refusal: None,
            nested_followed: 0,
            nested_sites: BTreeSet::new(),
            current: None,
            direction: base.bytecode_direction,
            section: Some(0),
            active: None,
            profile_transitions: 0,
            transition_sites: BTreeSet::new(),
            transition_budget_base: None,
            cache: LinearDecodeCache::default(),
            decoded: BTreeMap::new(),
            explored: BTreeMap::new(),
            widened: BTreeMap::new(),
            cfg: VmCfg {
                entry: Va(crate::immutable_image::tests::RDATA_VA),
                blocks: BTreeMap::new(),
                edges: Vec::new(),
                stalls: Vec::new(),
                limit_hit: None,
                visits: 0,
                frame_cells_known: 0,
                frame_note: None,
                assumptions: BTreeSet::new(),
            discharge_refusal: None,
            },
            total_instructions: 0,
            live_bytes: 0,
            observed: &observed.memory,
            seeded,
            observed_depths: &observed.loop_depths,
            observed_branch_targets: &observed.branch_targets,
            private_frame_hi: None,
            frame_not_private: false,
            frame_privacy_conceded: false,
            frame_derived_escape: false,
            frame_pointer_lost: false,
            frame_escape_seen: false,
            frame_store_past: None,
        }
    }

    #[test]
    fn stack_pointer_exposure_is_detected_only_for_copies_of_rsp() {
        let exposes = |bytes: &[u8]| exposes_stack_pointer(&decode_one(bytes));
        assert!(exposes(&[0x48, 0x89, 0xE5])); // mov rbp,rsp
        assert!(exposes(&[0x48, 0x8D, 0x6C, 0x24, 0x28])); // lea rbp,[rsp+0x28]
        assert!(exposes(&[0x54])); // push rsp
        assert!(!exposes(&[0x48, 0x89, 0x04, 0x24])); // mov [rsp],rax
        assert!(!exposes(&[0x48, 0x8B, 0x44, 0x24, 0x08])); // mov rax,[rsp+8]
        assert!(!exposes(&[0x53])); // push rbx
        // A 16-bit or narrower READ of SP yields no usable address: user-mode
        // addresses below 64K are never mapped. VMProtect emits such junk after
        // the VSP capture.
        assert!(!exposes(&[0x66, 0x0F, 0xBD, 0xC4])); // bsr ax,sp
        assert!(!exposes(&[0x66, 0x89, 0xE0])); // mov ax,sp
        assert!(!exposes(&[0x40, 0x88, 0xE0])); // mov al,spl
        // cmp/test write nothing, so operand 0 is a read as well: flags only.
        // REX.R (0x44) extends the reg field, which is where r15 sits here;
        // 0x41 is REX.B and would name r12d in r/m instead, touching no RSP.
        assert!(!exposes(&[0x66, 0x44, 0x39, 0xFC])); // cmp sp,r15w
        assert!(!exposes(&[0x66, 0x85, 0xE4])); // test sp,sp
        // Still exposure: a WRITE to SP, any 32-bit read, and the full pointer.
        assert!(exposes(&[0x66, 0x89, 0xC4])); // mov sp,ax rewrites RSP
        assert!(exposes(&[0x89, 0xE0])); // mov eax,esp keeps 32 bits
        assert!(exposes(&[0x44, 0x39, 0xFC])); // cmp esp,r15d keeps 32 bits
        assert!(exposes(&[0x4C, 0x39, 0xFC])); // cmp rsp,r15 is the full pointer
    }

    /// The assumption is discharged only for a private, un-leaked, in-frame,
    /// non-nested, non-widened walk; every other case keeps it.
    #[test]
    fn vm_stack_assumption_discharges_only_when_the_frame_is_provably_private() {
        use crate::immutable_image::tests::two_section_pe64;
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let observed = ObservedRun::default();
        let seeded = Cell::new(false);
        let base = profile(
            roles(Register::R11, Register::R13, Register::RBP),
            Register::R8,
            crate::BytecodeDirection::Descending,
        );
        let flagged = |escapes: bool, derived: bool, stored: bool, store_hi: Option<i64>| {
            let mut report = execute_symbolic_linear_block_with_initial_state(
                &[],
                None,
                crate::SymbolicExecutionLimits::default(),
                &SymbolicVmInitialState::default(),
                None,
            )
            .expect("empty block executes");
            report.vm_stack_address_escapes = escapes;
            report.vm_stack_derived_escapes = derived || escapes;
            report.vm_stack_pointer_lost = stored;
            report.vm_stack_store_hi = store_hi;
            report
        };
        let report = |escapes: bool, store_hi: Option<i64>| {
            flagged(escapes, false, false, store_hi)
        };
        let outcome = |frame_hi: Option<i64>, reports: &[SymbolicVmExecutionReport], nested: usize| {
            let mut walker = test_walker(&pe, base, &observed, &seeded);
            walker.private_frame_hi = frame_hi;
            walker.nested_followed = nested;
            walker
                .cfg
                .assumptions
                .insert(VmCfgAssumption::AssumesExternalStoresDisjointFromVmStack);
            for report in reports {
                walker.note_frame_use(report);
            }
            walker.discharge_vm_stack_assumption();
            walker.cfg.assumptions.is_empty()
        };
        let quiet = report(false, Some(0x30));
        assert!(outcome(Some(0x30), &[quiet.clone()], 0));
        assert!(outcome(Some(0x30), &[], 0));
        assert!(!outcome(None, &[quiet.clone()], 0), "frame unproven or exposed");
        assert!(!outcome(Some(0x30), &[report(true, None)], 0), "pointer leaked");
        assert!(!outcome(Some(0x30), &[report(false, Some(0x38))], 0), "write past the frame");
        assert!(!outcome(Some(0x30), &[quiet.clone()], 1), "nested entry");
        let data_only = flagged(false, true, false, Some(0x30));
        assert!(outcome(Some(0x30), &[data_only.clone()], 0), "frame data leaving is benign");
        let pointer_somewhere = flagged(false, false, true, Some(0x30));
        assert!(outcome(Some(0x30), &[pointer_somewhere.clone(), quiet], 0));
        assert!(
            !outcome(Some(0x30), &[data_only, pointer_somewhere], 0),
            "data may be a hidden frame pointer once one was stored"
        );
    }

    /// SOUND-PATH INVARIANCE: with no transition active, every profile accessor
    /// returns the base profile, so a block without a cross-profile transition
    /// decodes under exactly the inputs it did before this change. A proven
    /// transition installs a DIFFERENT profile, which then redirects the
    /// accessors (new VIP/key/handler roles and flipped bytecode direction).
    #[test]
    fn profile_context_accessors_default_to_base_and_follow_a_switch() {
        use crate::immutable_image::tests::two_section_pe64;
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let observed = ObservedRun::default();
        let seeded = Cell::new(false);
        let base = profile(
            roles(Register::R11, Register::R13, Register::RBP),
            Register::R8,
            crate::BytecodeDirection::Descending,
        );
        let mut walker = test_walker(&pe, base, &observed, &seeded);

        // No transition: base everywhere (byte-for-byte unchanged inputs).
        assert_eq!(walker.ctx_profile(), base);
        assert_eq!(walker.ctx_direction(), crate::BytecodeDirection::Descending);
        assert_eq!(walker.ctx_section(), Some(0));
        assert_eq!(walker.ctx_global_table(), None);

        // A proven switch to a disjoint profile with a flipped direction.
        let switched = profile(
            roles(Register::R15, Register::RBP, Register::R14),
            Register::R11,
            crate::BytecodeDirection::Ascending,
        );
        walker.active = Some(Rc::new(ProfileContext {
            config: entry_stream_config(&switched, walker.limits.entry),
            profile: switched,
            direction: switched.bytecode_direction,
            global_table: None,
            section: Some(1),
        }));
        assert_eq!(walker.ctx_profile(), switched);
        assert_ne!(walker.ctx_profile(), base);
        assert_eq!(walker.ctx_direction(), crate::BytecodeDirection::Ascending);
        assert_eq!(walker.ctx_section(), Some(1));
    }

    /// FAIL-CLOSED: an unresolved handler that is NOT a provable popped-target
    /// transition must not switch profiles. `analyze_vm_transition` cannot prove
    /// a transition from the synthetic image's bytes, so the walker queues
    /// nothing and adds no edge — leaving the caller's exact stall in place.
    #[test]
    fn unprovable_transition_does_not_switch_profiles() {
        use crate::immutable_image::tests::{RDATA_VA, two_section_pe64};
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let observed = ObservedRun::default();
        let seeded = Cell::new(false);
        let base = profile(
            roles(Register::R11, Register::R13, Register::RBP),
            Register::R8,
            crate::BytecodeDirection::Descending,
        );
        let mut walker = test_walker(&pe, base, &observed, &seeded);

        // The unresolved handler is an in-image VA that is not a real transition.
        let final_state = RelativeDispatchState {
            vip: Va(RDATA_VA),
            rolling_key: RDATA_VA,
            handler: Va(RDATA_VA),
        };
        let symbolic_in = SymbolicVmInitialState::default();
        let mut pending = Vec::new();
        let switched = walker.follow_cross_profile_transition(
            Va(RDATA_VA),
            final_state,
            &[],
            &symbolic_in,
            &mut pending,
        );
        assert!(!switched, "a non-provable transition must fail closed");
        assert!(pending.is_empty(), "no successor may be queued");
        assert!(walker.cfg.edges.is_empty(), "no edge may be added");
        assert!(walker.active.is_none(), "the base profile is untouched");
    }

    /// A symbolic state whose retained size exceeds the per-entry byte cap must
    /// fail the entry closed with `CfgLimitHit::Memory`; an in-cap state must
    /// charge and release exactly, leaving the entry unaffected.
    #[test]
    fn oversized_symbolic_state_trips_the_memory_cap() {
        use crate::immutable_image::tests::{RDATA_VA, two_section_pe64};
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let observed = ObservedRun::default();
        let seeded = Cell::new(false);
        let base = profile(
            roles(Register::R11, Register::R13, Register::RBP),
            Register::R8,
            crate::BytecodeDirection::Descending,
        );
        let mut walker = test_walker(&pe, base, &observed, &seeded);

        let mut big = SymbolicVmInitialState::default();
        for slot in 0..64_i64 {
            big.context_values.insert(
                slot,
                Expr::register(BitWidth::W64, format!("native_slot_{slot}_initial")),
            );
        }
        let item = |symbolic: SymbolicVmInitialState| Item {
            bytes: 0,
            start: RelativeDispatchState {
                vip: Va(RDATA_VA),
                rolling_key: RDATA_VA,
                handler: Va(RDATA_VA),
            },
            address: None,
            symbolic,
            path: None,
            profile: None,
        };

        // Default cap: the state is accounted, nothing trips.
        let mut queued = [item(big.clone())];
        assert!(walker.account_pending(&mut queued));
        assert!(queued[0].bytes >= 64 * EXPR_NODE_BYTES, "all nodes counted");
        assert_eq!(walker.live_bytes, queued[0].bytes);
        assert!(walker.cfg.limit_hit.is_none());
        assert!(walker.cfg.stalls.is_empty());

        // A cap below that state's size: fail closed with `Memory`.
        let mut walker = test_walker(&pe, base, &observed, &seeded);
        walker.limits.max_memory_bytes = 64 * EXPR_NODE_BYTES / 2;
        let mut queued = [item(big)];
        assert!(!walker.account_pending(&mut queued));
        assert_eq!(walker.cfg.limit_hit, Some(CfgLimitHit::Memory));
        assert!(!walker.cfg.is_complete(), "a memory stall is never complete");
        assert_eq!(walker.cfg.stalls.len(), 1);
    }

    /// MEMORY BUDGET: once the per-entry cross-profile-transition budget is
    /// exhausted, the walk must fail closed immediately — no further transition
    /// analysis, no successor queued, no edge — so a pathological entry stalls
    /// at the cap instead of allocating without bound.
    #[test]
    fn exhausted_transition_budget_fails_closed() {
        use crate::immutable_image::tests::{RDATA_VA, two_section_pe64};
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let observed = ObservedRun::default();
        let seeded = Cell::new(false);
        let base = profile(
            roles(Register::R11, Register::R13, Register::RBP),
            Register::R8,
            crate::BytecodeDirection::Descending,
        );
        let mut walker = test_walker(&pe, base, &observed, &seeded);
        // Saturate the budget: the next transition must be refused outright.
        walker.transition_sites = (0..walker.limits.max_profile_transitions).map(|i| Va(0x9000_0000 + i as u64)).collect();
        let final_state = RelativeDispatchState {
            vip: Va(RDATA_VA),
            rolling_key: RDATA_VA,
            handler: Va(RDATA_VA),
        };
        let symbolic_in = SymbolicVmInitialState::default();
        let mut pending = Vec::new();
        let switched = walker.follow_cross_profile_transition(
            Va(RDATA_VA),
            final_state,
            &[],
            &symbolic_in,
            &mut pending,
        );
        assert!(!switched, "an exhausted budget must fail closed");
        assert!(pending.is_empty(), "no successor may be queued at the cap");
        assert!(walker.cfg.edges.is_empty(), "no edge may be added at the cap");
    }

    #[test]
    fn flag_fork_resolves_both_outcomes_of_a_nonaffine_flag_branch() {
        let target = select(zf_bit(opaque_operand()), 0x1800_1000, 0x1800_2000).simplify();
        // The affine-domain path cannot characterize the flag operand.
        assert!(infer_branch_input_domains(&target, InputDomainLimits::default()).is_err());
        // The boolean fallback forks both outcomes exactly.
        assert_eq!(
            flag_fork_targets(&target, VmCfgLimits::default()),
            Some(vec![0x1800_1000, 0x1800_2000])
        );
        // `resolve_targets` routes through it and marks the branch guarded.
        assert_eq!(
            resolve_targets(&target, VmCfgLimits::default()),
            Ok((vec![0x1800_1000, 0x1800_2000], true))
        );
    }

    #[test]
    fn cross_section_follow_needs_the_lever_and_a_mapped_target_not_executability() {
        const EXEC: u32 = 0x2000_0000;
        const DATA: u32 = 0xC000_0040;
        // A non-executable target is followable: VM bytecode is read as data.
        assert!(cross_section_followable(true, Some(EXEC)));
        assert!(cross_section_followable(true, Some(DATA)));
        // Unmapped is never followed, and the lever still gates everything.
        assert!(!cross_section_followable(true, None));
        assert!(!cross_section_followable(false, Some(EXEC)));
        assert!(!cross_section_followable(false, Some(DATA)));
    }

    #[test]
    fn flag_fork_does_not_fork_a_flag_proven_constant() {
        // Bit 6 of `(~L & C1) & (L & C2)` is zero (C1 has bit 6 clear), but the
        // simplifier keeps the expression. Forking it over {0, 1} would add the
        // infeasible `+0x10` targets; only the real ZF fork may remain.
        let load = Expr::intrinsic(
            BitWidth::W64,
            "memory_load_epoch_1",
            vec![Expr::constant(BitWidth::W64, 0x148)],
        );
        let and = |a: Expr, b: Expr| Expr::binary(BinaryOp::And, a, b).unwrap();
        let c = |v: u64| Expr::constant(BitWidth::W64, v);
        let not_load = Expr::unary(UnaryOp::Not, load.clone());
        let inner = and(and(not_load, c(0x924a_712f)), and(load, c(0x6db5_8ed0)));
        let dead = Expr::binary(BinaryOp::Shr, and(c(0x40), inner), c(6)).unwrap();
        assert_eq!(value_range(&dead).max(), 0);
        let sum = Expr::binary(
            BinaryOp::Add,
            select(zf_bit(opaque_operand()), 0x1800_1000, 0x1800_2000),
            Expr::binary(BinaryOp::Mul, dead, c(0x10)).unwrap(),
        )
        .unwrap()
        .simplify();
        assert_eq!(
            flag_fork_targets(&sum, VmCfgLimits::default()),
            Some(vec![0x1800_1000, 0x1800_2000])
        );
    }

    #[test]
    fn flag_fork_fails_closed_on_residual_input_dependence() {
        // An input (native_r13_initial) reaches the target outside the flag,
        // exactly like the coredll oracle; the fallback must not guess.
        let target = Expr::binary(
            BinaryOp::Add,
            select(zf_bit(opaque_operand()), 0x1800_1000, 0x1800_2000),
            Expr::register(BitWidth::W64, "native_r13_initial"),
        )
        .unwrap()
        .simplify();
        assert!(target.dependencies().contains("native_r13_initial"));
        assert_eq!(flag_fork_targets(&target, VmCfgLimits::default()), None);
    }


    #[test]
    fn enumeration_explosion_recovers_a_bounded_flag_selector() {
        // A flag-selected branch whose flag operand is characterizable, so input
        // inference succeeds. With the cross-product capped below the domain size
        // (standing in for the real oracle's 256^4 blow-up), the full enumeration
        // is infeasible, yet the real target set is just the two selector outcomes
        // and the flag fork recovers them — no target dropped, none invented.
        let bit = Expr::binary(
            BinaryOp::And,
            Expr::register(BitWidth::W64, "native_rax_initial"),
            w64(1),
        )
        .unwrap();
        let target = select(bit, 0x1800_1000, 0x1800_2000).simplify();
        let limits = VmCfgLimits {
            domain: SymbolicDomainLimits {
                max_combinations: 1,
                ..SymbolicDomainLimits::default()
            },
            ..VmCfgLimits::default()
        };
        // Inference succeeds, so this exercises the enumeration-failure fallback
        // (not the inference-refusal one).
        assert!(infer_branch_input_domains(&target, limits.input_domain).is_ok());
        assert_eq!(
            resolve_targets(&target, limits),
            Ok((vec![0x1800_1000, 0x1800_2000], true))
        );
    }

    #[test]
    fn enumeration_explosion_over_a_full_value_fails_closed() {
        // The target reads a whole register value, not a bounded flag. With the
        // cross-product capped the enumeration is infeasible and the flag fork
        // cannot characterize it, so the branch fails closed rather than guessing.
        let target = Expr::binary(
            BinaryOp::Add,
            w64(0x1800_1000),
            Expr::register(BitWidth::W64, "native_rax_initial"),
        )
        .unwrap()
        .simplify();
        let limits = VmCfgLimits {
            domain: SymbolicDomainLimits {
                max_combinations: 1,
                ..SymbolicDomainLimits::default()
            },
            ..VmCfgLimits::default()
        };
        assert!(resolve_targets(&target, limits).is_err());
    }

    /// A predicate matching the oracle's bytecode-section preferred-base VAs.
    fn is_oracle_image_va(value: u64) -> bool {
        (0x1_8000_0000..0x1_9000_0000).contains(&value)
    }

    fn reg_r13() -> Expr {
        Expr::register(BitWidth::W64, "native_r13_initial")
    }

    fn image_addend(va: u64) -> Expr {
        Expr::binary(BinaryOp::Add, reg_r13(), Expr::constant(BitWidth::W64, va)).unwrap()
    }

    /// The oracle's branchless select, verbatim in shape:
    /// `r13 + select(ZF(operand), lo, hi)`, with `r13 + VA` kept under a `Not`
    /// in each branch so it survives simplification (it is never a bare addend).
    fn r13_relative_flag_select(operand: Expr, lo: u64, hi: u64) -> Expr {
        let not = |e| Expr::unary(crate::UnaryOp::Not, e);
        let or = |l, r| Expr::binary(BinaryOp::Or, l, r).unwrap();
        let inner = Expr::binary(
            BinaryOp::Add,
            zf_bit(operand),
            Expr::constant(BitWidth::W64, u64::MAX),
        )
        .unwrap();
        let left = not(or(not(inner.clone()), not(image_addend(lo))));
        let right = not(or(inner, not(image_addend(hi))));
        Expr::binary(BinaryOp::Add, left, right).unwrap()
    }

    /// The corpus CPUID-gate shape: `rcx + select(ZF(cpuid(1, rcx_low)), lo, hi)`,
    /// where native_rcx_initial is BOTH the CPUID (ignored) sub-leaf operand and
    /// the relocation-delta addend of each target. The shared register is what
    /// blocks the direct resolution — it cannot be bound while it is captured
    /// inside CPUID — and is only freed once the CPUID read is abstracted away.
    fn cpuid_subleaf_relative_gate(lo: u64, hi: u64) -> Expr {
        let not = |e| Expr::unary(crate::UnaryOp::Not, e);
        let or = |l, r| Expr::binary(BinaryOp::Or, l, r).unwrap();
        let rcx_addend = |va| {
            Expr::binary(
                BinaryOp::Add,
                Expr::register(BitWidth::W64, "native_rcx_initial"),
                w64(va),
            )
            .unwrap()
        };
        let inner = Expr::binary(BinaryOp::Add, zf_bit(opaque_operand()), w64(u64::MAX)).unwrap();
        let left = not(or(not(inner.clone()), not(rcx_addend(lo))));
        let right = not(or(inner, not(rcx_addend(hi))));
        Expr::binary(BinaryOp::Add, left, right).unwrap()
    }

    #[test]
    fn abstracting_environment_intrinsics_drops_the_captured_subleaf_register() {
        // CPUID's sub-leaf input (native_rcx_initial) is ignored; abstracting the
        // read to a free variable drops it, so a register that reached the target
        // only through CPUID is no longer a dependence.
        let operand = opaque_operand();
        assert!(operand.dependencies().contains("native_rcx_initial"));
        assert!(
            operand
                .intrinsic_dependencies()
                .iter()
                .any(|name| is_environment_intrinsic(name))
        );
        let abstracted = abstract_environment_intrinsics(&operand).unwrap();
        assert!(!abstracted.dependencies().contains("native_rcx_initial"));
        assert!(
            abstracted
                .intrinsic_dependencies()
                .iter()
                .all(|name| !is_environment_intrinsic(name))
        );
    }

    #[test]
    fn environment_gate_forks_both_targets_when_cpuid_captures_the_relocation_register() {
        let target = cpuid_subleaf_relative_gate(0x1_8009_ED37, 0x1_8009_EDB0);
        // native_rcx_initial reaches the target both inside CPUID and as the
        // addend, so the preferred-base binding cannot prove it the pure delta and
        // the direct resolution stalls.
        let (bound, names) = bind_preferred_base_registers(&target, &is_oracle_image_va);
        assert!(names.is_empty());
        assert!(resolve_targets(&bound, VmCfgLimits::default()).is_err());
        // Abstracting the CPUID read frees the register, the delta binds, and both
        // sides of the gate are enumerated as a sound over-approximation.
        let (targets, guarded, bound_registers) =
            resolve_environment_gate(&target, &is_oracle_image_va, VmCfgLimits::default())
                .expect("a pure CPUID gate resolves by both-target over-approximation");
        assert_eq!(targets, vec![0x1_8009_ED37, 0x1_8009_EDB0]);
        assert!(guarded);
        assert!(bound_registers.contains("native_rcx_initial"));
    }

    #[test]
    fn environment_gate_surfaces_its_labeled_assumption() {
        // The gate firing is exactly what follow_branch records as the surfaced,
        // labeled over-approximation assumption.
        let target = cpuid_subleaf_relative_gate(0x1_8009_ED37, 0x1_8009_EDB0);
        assert!(
            resolve_environment_gate(&target, &is_oracle_image_va, VmCfgLimits::default())
                .is_some()
        );
        let mut assumptions = BTreeSet::new();
        assumptions.insert(VmCfgAssumption::OverApproximatedEnvironmentGate);
        assert!(assumptions.contains(&VmCfgAssumption::OverApproximatedEnvironmentGate));
    }

    #[test]
    fn environment_gate_does_not_fire_on_a_real_program_input() {
        // The same select shape but gated on a real program register, with no
        // environment intrinsic: the over-approximation must not fire. The branch
        // is left to the ordinary (unchanged) resolution path.
        let operand = Expr::register(BitWidth::W32, "native_rax_initial");
        let target = r13_relative_flag_select(operand, 0x1_8009_ED37, 0x1_8009_EDB0);
        assert_eq!(
            resolve_environment_gate(&target, &is_oracle_image_va, VmCfgLimits::default()),
            None
        );
    }

    #[test]
    fn environment_gate_fails_closed_when_a_program_register_survives() {
        // CPUID is present, but a real input (native_r13_initial) reaches the
        // target outside the gate and is not an image-base addend, so a program
        // register survives abstraction: not a pure environment gate.
        let gate = cpuid_subleaf_relative_gate(0x1_8009_ED37, 0x1_8009_EDB0);
        let target = Expr::binary(BinaryOp::Add, gate, reg_r13())
            .unwrap()
            .simplify();
        assert_eq!(
            resolve_environment_gate(&target, &is_oracle_image_va, VmCfgLimits::default()),
            None
        );
    }

    #[test]
    fn image_base_relative_register_binds_and_branch_resolves() {
        // r13 is added only to in-image target VAs (the relocation delta).
        let target = r13_relative_flag_select(opaque_operand(), 0x1_8009_ED37, 0x1_8009_EDB0);
        assert!(target.dependencies().contains("native_r13_initial"));
        assert!(image_base_relative_only(
            &target,
            "native_r13_initial",
            &is_oracle_image_va
        ));
        // Binding removes the residual delta that otherwise fails the fork closed.
        let (bound, names) = bind_preferred_base_registers(&target, &is_oracle_image_va);
        assert_eq!(names, BTreeSet::from(["native_r13_initial".to_owned()]));
        assert!(!bound.dependencies().contains("native_r13_initial"));
        // The flag fork now resolves both image targets exactly.
        assert_eq!(
            resolve_targets(&bound, VmCfgLimits::default()),
            Ok((vec![0x1_8009_ED37, 0x1_8009_EDB0], true))
        );
    }

    #[test]
    fn register_used_as_data_value_is_not_bound() {
        // r13 added to a non-image constant is a general value, not a relocation
        // of an image address: it must not bind, and the branch fails closed.
        let data = Expr::binary(
            BinaryOp::Add,
            reg_r13(),
            Expr::constant(BitWidth::W64, 0x40),
        )
        .unwrap();
        let target = Expr::binary(
            BinaryOp::Add,
            select(zf_bit(opaque_operand()), 0x1_8009_ED37, 0x1_8009_EDB0),
            data,
        )
        .unwrap();
        let (bound, names) = bind_preferred_base_registers(&target, &is_oracle_image_va);
        assert!(names.is_empty());
        assert_eq!(bound, target);
        assert!(resolve_targets(&target, VmCfgLimits::default()).is_err());

        // Mixed use: one image-base addend and one bare data occurrence of the
        // same register still fails closed — it cannot be proven the pure delta.
        let mixed = Expr::binary(BinaryOp::Add, image_addend(0x1_8009_ED37), reg_r13()).unwrap();
        assert!(!image_base_relative_only(
            &mixed,
            "native_r13_initial",
            &is_oracle_image_va
        ));
        let (_, mixed_names) = bind_preferred_base_registers(&mixed, &is_oracle_image_va);
        assert!(mixed_names.is_empty());
    }

    #[test]
    fn binding_surfaces_the_preferred_base_assumption_label() {
        // A non-empty bound set is exactly what follow_branch records as the
        // labeled, surfaced assumption.
        let target = r13_relative_flag_select(opaque_operand(), 0x1_8009_ED37, 0x1_8009_EDB0);
        let (_, names) = bind_preferred_base_registers(&target, &is_oracle_image_va);
        assert!(!names.is_empty());
        let mut assumptions = BTreeSet::new();
        if !names.is_empty() {
            assumptions.insert(VmCfgAssumption::PreferredImageBase);
        }
        assert!(assumptions.contains(&VmCfgAssumption::PreferredImageBase));
    }

    #[test]
    fn flag_fork_rejects_a_value_not_proven_to_be_a_bit() {
        // `flags >> 6` is not masked to bit 0 and can hold values other than
        // {0, 1}; substituting it would be unsound, so the fallback refuses.
        let flags = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_sub_w32",
            vec![opaque_operand(), Expr::constant(BitWidth::W32, 4064)],
        );
        let not_a_bit =
            Expr::binary(BinaryOp::Shr, flags, Expr::constant(BitWidth::W64, 6)).unwrap();
        assert!(value_range(&not_a_bit).max() > 1);
        let target = select(not_a_bit, 0x1800_1000, 0x1800_2000).simplify();
        assert_eq!(flag_fork_targets(&target, VmCfgLimits::default()), None);
    }

    #[test]
    fn lea_form_vsp_capture_offsets_frame_and_bad_forms_fail_closed() {
        // `lea rbp,[rsp-28h]`: VSP = RSP - 0x28, so the top cell is at VSP+0x28.
        let lea = decode_one(&[0x48, 0x8d, 0x6c, 0x24, 0xd8]);
        assert_eq!(vsp_capture_top_offset(&lea, Register::RBP), Some(0x28));
        // `mov rbp,rsp`: VSP = RSP, top cell at VSP+0.
        let mov = decode_one(&[0x48, 0x89, 0xe5]);
        assert_eq!(vsp_capture_top_offset(&mov, Register::RBP), Some(0));

        // Fail closed: capture register disagrees with the profile's VSP.
        assert_eq!(vsp_capture_top_offset(&lea, Register::RSI), None);
        assert_eq!(vsp_capture_top_offset(&mov, Register::RSI), None);
        // Fail closed: base is not RSP (`lea rbp,[rbp-28h]`).
        let lea_rbp = decode_one(&[0x48, 0x8d, 0x6d, 0xd8]);
        assert_eq!(vsp_capture_top_offset(&lea_rbp, Register::RBP), None);
        // Fail closed: indexed address (`lea rbp,[rsp+rax-28h]`).
        let lea_indexed = decode_one(&[0x48, 0x8d, 0x6c, 0x04, 0xd8]);
        assert_eq!(vsp_capture_top_offset(&lea_indexed, Register::RBP), None);

        // The offset flows into the frame: cell i lands at vsp_initial+0x28+i*8.
        let cells = vec![
            NativeFrameSource::Register {
                register: crate::X64NativeRegister::R13,
            },
            NativeFrameSource::Register {
                register: crate::X64NativeRegister::Rcx,
            },
        ];
        let (state, known) = frame_initial_state(&cells, 0x28);
        assert_eq!(known, 2);
        let addresses: Vec<_> = state
            .memory_values
            .iter()
            .map(|(address, _)| format!("{address:?}"))
            .collect();
        let want_top = format!(
            "{:?}",
            Expr::binary(
                BinaryOp::Add,
                Expr::register(BitWidth::W64, "vsp_initial"),
                Expr::constant(BitWidth::W64, 0x28),
            )
            .unwrap()
        );
        let want_next = format!(
            "{:?}",
            Expr::binary(
                BinaryOp::Add,
                Expr::register(BitWidth::W64, "vsp_initial"),
                Expr::constant(BitWidth::W64, 0x30),
            )
            .unwrap()
        );
        assert!(addresses.contains(&want_top), "missing VSP+0x28: {addresses:?}");
        assert!(addresses.contains(&want_next), "missing VSP+0x30: {addresses:?}");
    }

    fn zf_select(register: &str, compare: u64, if_zero: u64, otherwise: u64) -> Expr {
        let rdx = Expr::register(BitWidth::W64, register);
        let low = Expr::extract(BitWidth::W32, 0, rdx).expect("extract");
        let flags = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_sub_w32",
            vec![low, Expr::constant(BitWidth::W32, compare)],
        );
        let shifted =
            Expr::binary(BinaryOp::Shr, flags, Expr::constant(BitWidth::W64, 6)).expect("shift");
        let zero =
            Expr::binary(BinaryOp::And, shifted, Expr::constant(BitWidth::W64, 1)).expect("mask");
        let delta = Expr::binary(
            BinaryOp::Mul,
            zero,
            Expr::constant(BitWidth::W64, if_zero.wrapping_sub(otherwise)),
        )
        .expect("scale");
        Expr::binary(
            BinaryOp::Add,
            delta,
            Expr::constant(BitWidth::W64, otherwise),
        )
        .expect("sum")
    }

    #[test]
    fn subtract_flag_guard_resolves_to_exactly_both_targets() {
        let target = zf_select("native_rdx_initial", 1, 0x1000, 0x2000);
        let (values, guarded) =
            resolve_targets(&target, VmCfgLimits::default()).expect("proven domain");
        assert!(guarded);
        assert_eq!(values, vec![0x1000, 0x2000]);
    }

    #[test]
    fn constant_target_is_not_guarded_and_raw_register_target_is_refused() {
        let constant = Expr::constant(BitWidth::W64, 0x4000);
        assert_eq!(
            resolve_targets(&constant, VmCfgLimits::default()),
            Ok((vec![0x4000], false))
        );
        let raw = Expr::register(BitWidth::W64, "native_rdx_initial");
        assert!(resolve_targets(&raw, VmCfgLimits::default()).is_err());
    }

    fn block(exit: BlockExit, handler: Option<Va>) -> VmBasicBlock {
        VmBasicBlock {
            start: RelativeDispatchState {
                vip: Va(0x10),
                rolling_key: 0,
                handler: Va(0x20),
            },
            address: None,
            instructions: Vec::new(),
            exit,
            boundary_handler: handler,
            native_call: None,
            vm_exit: None,
        }
    }

    fn cfg_of(blocks: Vec<(u64, VmBasicBlock)>) -> VmCfg {
        VmCfg {
            entry: Va(0x10),
            blocks: blocks.into_iter().map(|(key, b)| (Va(key), b)).collect(),
            edges: Vec::new(),
            stalls: Vec::new(),
            limit_hit: None,
            visits: 0,
            frame_cells_known: 0,
            frame_note: None,
            assumptions: BTreeSet::new(),
            discharge_refusal: None,
        }
    }

    fn exit_report(status: RecoveryStatus) -> VmExitReport {
        VmExitReport {
            schema_version: 1,
            entry: vmpx_pe::Rva(0x1000),
            status,
            stack_switch: None,
            restored_registers: Vec::new(),
            restore_slots: Vec::new(),
            flags_restore: None,
            flags_stack_offset: Some(0x70),
            native_return: None,
            return_stack_offset: Some(0x78),
            frame_size: Some(0x80),
            confidence: 100,
            evidence: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn cfg_carries_the_native_call_outcome_and_exit_layout() {
        let outcome = NativeCallSelectorOutcome {
            selector: 3,
            vsp_delta: 16,
            target_stack_offset: 0,
            register_argument_offsets: vec![8],
            stack_argument_offsets: Vec::new(),
        };
        let mut exit_block = block(BlockExit::VmExit, Some(Va(0x99)));
        exit_block.vm_exit = proven_exit_layout(Some(&exit_report(RecoveryStatus::Recovered)));
        let mut cfg = cfg_of(vec![
            (0x10, block(BlockExit::NativeCall, Some(Va(0x77)))),
            (0x20, exit_block),
            (0x30, block(BlockExit::Branch, Some(Va(0x55)))),
        ]);
        cfg.record_native_call(Va(0x10), &outcome);
        cfg.record_native_call(Va(0x30), &outcome);
        assert_eq!(cfg.native_calls().len(), 1);
        assert_eq!(cfg.native_calls()[&Va(0x10)], outcome);
        let layouts = cfg.exit_layouts();
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[&Va(0x99)].frame_size, Some(0x80));
    }

    #[test]
    fn unproven_exit_layouts_are_not_carried() {
        assert!(proven_exit_layout(None).is_none());
        for status in [RecoveryStatus::PartiallyRecovered, RecoveryStatus::Failed] {
            assert!(proven_exit_layout(Some(&exit_report(status))).is_none());
        }
    }

    fn w64(value: u64) -> Expr {
        Expr::constant(BitWidth::W64, value)
    }

    fn loop_state(vsp_offset: u64, counter: Expr) -> SymbolicVmInitialState {
        SymbolicVmInitialState {
            vsp: Expr::binary(
                BinaryOp::Add,
                Expr::register(BitWidth::W64, "vsp_initial"),
                w64(vsp_offset),
            )
            .expect("add"),
            context_values: BTreeMap::from([(0, w64(7)), (8, counter)]),
            ..SymbolicVmInitialState::default()
        }
    }

    fn increment(expr: &Expr) -> Expr {
        Expr::binary(BinaryOp::Add, expr.clone(), w64(1)).expect("add")
    }

    /// Outcome of [`walk_loop_observed`]: how the walk closed plus bookkeeping.
    #[derive(Debug, Eq, PartialEq)]
    struct LoopWalk {
        /// `loop_step` calls made until the walk closed (the closing call
        /// included).
        iterations: usize,
        /// Times the state was generalized (sound widening).
        widened: usize,
        /// Exact iterations taken through the observed bounded-unroll bridge.
        observed_unrolls: usize,
        /// The bounded unroll's observed-depth close fired.
        closed_observed: bool,
        /// The VM-stack-shift close fired (sound, but concedes frame privacy).
        closed_shifted: bool,
    }

    /// Walks one loop whose body runs `step` on the state; returns how it closed,
    /// or the refusal. `observed_depth` gates the experimental bounded unroll.
    fn walk_loop_observed(
        entry: SymbolicVmInitialState,
        step: impl Fn(&SymbolicVmInitialState) -> SymbolicVmInitialState,
        unrolls: usize,
        observed_depth: Option<u64>,
    ) -> Result<LoopWalk, String> {
        let key = Va(0x1000);
        let mut ancestors = vec![entry];
        let mut walk = LoopWalk {
            iterations: 0,
            widened: 0,
            observed_unrolls: 0,
            closed_observed: false,
            closed_shifted: false,
        };
        for iteration in 1..=64 {
            let reentry = step(ancestors.last().expect("entry"));
            let nearest_first: Vec<_> = ancestors.iter().rev().collect();
            match loop_step(key, &nearest_first, &reentry, unrolls, observed_depth) {
                LoopStep::Close => {
                    walk.iterations = iteration;
                    return Ok(walk);
                }
                LoopStep::CloseObserved => {
                    walk.iterations = iteration;
                    walk.closed_observed = true;
                    return Ok(walk);
                }
                LoopStep::Unroll => ancestors.push(reentry),
                LoopStep::UnrollObserved => {
                    walk.observed_unrolls += 1;
                    ancestors.push(reentry);
                }
                LoopStep::Widen(state) | LoopStep::WidenShifted(state) => {
                    walk.widened += 1;
                    ancestors.push(state);
                }
                LoopStep::CloseShifted => {
                    walk.iterations = iteration;
                    walk.closed_shifted = true;
                    return Ok(walk);
                }
                LoopStep::Refuse(reason) => return Err(reason),
            }
        }
        Err("did not terminate".to_owned())
    }

    /// Walks one loop whose body runs `step` on the state; returns the steps
    /// taken until the back edge closes, or the refusal. Sound path: no observed
    /// depth.
    fn walk_loop(
        entry: SymbolicVmInitialState,
        step: impl Fn(&SymbolicVmInitialState) -> SymbolicVmInitialState,
        unrolls: usize,
    ) -> Result<(usize, usize), String> {
        walk_loop_observed(entry, step, unrolls, None).map(|walk| (walk.iterations, walk.widened))
    }

    #[test]
    fn invariant_loop_closes_on_first_reentry() {
        let entry = loop_state(0, w64(0));
        assert_eq!(walk_loop(entry, Clone::clone, 4), Ok((1, 0)));
    }

    #[test]
    fn counting_loop_is_generalized_once_then_closes() {
        let entry = loop_state(0, w64(0));
        let step = |state: &SymbolicVmInitialState| {
            let mut next = state.clone();
            next.context_values
                .insert(8, increment(&state.context_values[&8]));
            next
        };
        let (iterations, widened) = walk_loop(entry, step, 3).expect("converges");
        assert_eq!(widened, 1);
        assert_eq!(iterations, 4, "two exact unrolls, one widening, one close");
    }

    #[test]
    fn folding_concedes_frame_privacy_only_when_it_hides_a_frame_pointer() {
        let w = BitWidth::W64;
        let addr = |n: u64| Expr::constant(w, n);
        // A frame POINTER: vsp_initial occurs outside an untracked load's args.
        let pointer = Expr::register(w, "vsp_initial");
        // Frame DATA: vsp_initial occurs only inside the load's address.
        let data = Expr::intrinsic(w, "memory_load_epoch_0", vec![pointer.clone()]);
        assert!(crate::semantic_exec::is_frame_pointer_value(&pointer));
        assert!(!crate::semantic_exec::is_frame_pointer_value(&data));

        let state = |cells: Vec<(Expr, Expr)>| SymbolicVmInitialState {
            memory_values: cells,
            ..SymbolicVmInitialState::default()
        };

        // Pointer cell survives the fold unchanged: privacy is NOT conceded.
        let before = state(vec![(addr(0x10), pointer.clone())]);
        let after = state(vec![(addr(0x10), pointer.clone())]);
        assert!(!fold_hid_frame_pointer([&before].into_iter(), &after));

        // Pointer cell DROPPED by the fold: privacy IS conceded.
        let after = state(vec![]);
        assert!(fold_hid_frame_pointer([&before].into_iter(), &after));

        // Pointer cell BROADENED to an unknown: privacy IS conceded.
        let after = state(vec![(addr(0x10), Expr::register(w, "havoc_1"))]);
        assert!(fold_hid_frame_pointer([&before].into_iter(), &after));

        // Pointer cell kept, but re-homed to another address: conceded.
        let after = state(vec![(addr(0x18), pointer.clone())]);
        assert!(fold_hid_frame_pointer([&before].into_iter(), &after));

        // Only a NON-pointer cell is dropped: privacy survives.
        let before = state(vec![(addr(0x10), data.clone()), (addr(0x18), pointer.clone())]);
        let after = state(vec![(addr(0x18), pointer)]);
        assert!(!fold_hid_frame_pointer([&before].into_iter(), &after));

        // A pointer in ANY pre-fold state counts, not just the first.
        let other = state(vec![(addr(0x20), Expr::register(w, "vsp_initial"))]);
        let after = state(vec![]);
        assert!(fold_hid_frame_pointer([&other].into_iter(), &after));
    }

    #[test]
    fn widened_loop_state_keeps_invariant_cells_exact() {
        let a = loop_state(0, w64(0));
        let r = loop_state(0, w64(1));
        let widened = widen_state(Va(0x1000), &a, &r)
            .expect("widens")
            .expect("changed");
        assert_eq!(widened.context_values[&0], w64(7));
        assert!(is_havoc(&widened.context_values[&8]));
        assert_eq!(widened.vsp, a.vsp);
        assert!(matches!(widen_state(Va(0x1000), &widened, &r), Ok(None)));
    }

    #[test]
    fn re_entry_that_lost_image_or_protection_proof_is_not_subsumed() {
        let a = loop_state(0, w64(0));
        for lose in [
            |s: &mut SymbolicVmInitialState| s.image_reads_suppressed = true,
            |s: &mut SymbolicVmInitialState| s.protection_may_have_changed = true,
        ] {
            let mut r = loop_state(0, w64(0));
            lose(&mut r);
            let widened = widen_state(Va(0x1000), &a, &r)
                .expect("widens")
                .expect("a state with weaker facts must be re-analysed");
            assert!(widened.image_reads_suppressed || widened.protection_may_have_changed);
            assert!(matches!(widen_state(Va(0x1000), &widened, &r), Ok(None)));
        }
    }

    #[test]
    fn expression_over_a_generalized_variable_is_never_assumed_invariant() {
        let havoc = Expr::register(BitWidth::W64, format!("{HAVOC_PREFIX}1000_1"));
        let derived = increment(&havoc);
        let a = loop_state(0, derived.clone());
        let r = loop_state(0, derived);
        let widened = widen_state(Va(0x1000), &a, &r)
            .expect("widens")
            .expect("the same text means a different value next iteration");
        assert!(is_havoc(&widened.context_values[&8]));
    }

    #[test]
    fn loop_that_only_moves_the_stack_pointer_closes_as_a_back_edge() {
        // The VSP advances a constant 1 byte per iteration and NOTHING else
        // changes, so each re-entry is the enclosing state under the uniform
        // renaming of the stack base: bisimilar, and the back edge closes. It
        // used to be refused, which cost 110+ systems.xpl entries for a loop the
        // graph can describe exactly.
        let entry = loop_state(0, w64(0));
        let step = |state: &SymbolicVmInitialState| {
            let mut next = state.clone();
            next.vsp = increment(&state.vsp);
            next
        };
        let walk = walk_loop_observed(entry, step, 2, None).expect("closes as a loop");
        assert!(walk.closed_shifted, "closed through the stack-shift test");
        assert_eq!(walk.widened, 0, "no widening is involved in an exact shift");
    }

    #[test]
    fn stack_shift_close_concludes_nothing_about_a_cell_that_also_moves() {
        // Same constant VSP stride, but a context cell drifts too, so the
        // re-entry is NOT the renaming of the enclosing state. What this guards
        // is that NOTHING is concluded from the shift about the drifting cell.
        // It used to be a flat refusal; the drift is now generalized instead,
        // which concludes strictly less about the cell rather than more. The
        // pairing with `loop_that_only_moves_the_stack_pointer_...` is the whole
        // point: a pure shift closes with NO widening, a shift with a drift
        // cannot close until the drift has been widened away.
        let step = |state: &SymbolicVmInitialState| {
            let mut next = state.clone();
            next.vsp = increment(&state.vsp);
            next.context_values.insert(0, increment(&state.context_values[&0]));
            next
        };
        let walk = walk_loop_observed(loop_state(0, w64(0)), step, 2, None)
            .expect("the drift is widened away rather than refused");
        assert!(walk.closed_shifted, "still closes through the stack-shift test");
        assert!(walk.widened >= 1, "the drifting cell had to be generalized first");
    }

    /// A step that decrements VSP by a proven constant `delta` bytes each
    /// iteration (the recursion-as-loop push-down frame), like rmath_fib's −64.
    fn constant_vsp_shift(delta: u64) -> impl Fn(&SymbolicVmInitialState) -> SymbolicVmInitialState {
        move |state| {
            let mut next = state.clone();
            next.vsp = Expr::binary(BinaryOp::Add, state.vsp.clone(), w64(delta)).expect("add");
            next
        }
    }

    #[test]
    fn bounded_unroll_from_observed_depth_unrolls_exactly_and_closes() {
        // A VSP-changing loop with a proven constant −64 delta: with an observed
        // depth of 5 it unrolls exactly 5 iterations and the back edge closes via
        // the observed bridge instead of stalling.
        let entry = SymbolicVmInitialState::default();
        let minus_64 = (-64_i64) as u64;
        let walk = walk_loop_observed(entry, constant_vsp_shift(minus_64), 2, Some(5))
            .expect("bounded unroll closes");
        assert!(walk.closed_observed, "closed through the observed bridge");
        assert_eq!(walk.iterations, 5, "exactly the observed depth of iterations");
        assert_eq!(walk.widened, 0, "a bounded unroll never widens");
    }

    #[test]
    fn vsp_loop_without_observed_depth_closes_as_a_loop_and_never_unrolls() {
        // The same constant-delta VSP loop with no observed depth. The point this
        // test has always guarded is that NO DEPTH IS GUESSED, and that still
        // holds: the walk closes the back edge instead, which is depth-free, and
        // takes neither of the observed-tier unroll paths.
        let entry = SymbolicVmInitialState::default();
        let minus_64 = (-64_i64) as u64;
        let walk = walk_loop_observed(entry, constant_vsp_shift(minus_64), 2, None)
            .expect("closes as a loop");
        assert!(walk.closed_shifted);
        assert!(!walk.closed_observed, "no observed bridge without a depth");
        assert_eq!(walk.observed_unrolls, 0, "no depth is ever guessed");
    }

    #[test]
    fn non_constant_vsp_delta_is_never_unrolled_even_with_depth() {
        // VSP that changes non-constantly (doubles each iteration): the
        // proven-constant-delta guard refuses to bounded-unroll even when an
        // observed depth is present.
        let entry = SymbolicVmInitialState::default();
        let step = |state: &SymbolicVmInitialState| {
            let mut next = state.clone();
            next.vsp = Expr::binary(BinaryOp::Mul, state.vsp.clone(), w64(2)).expect("mul");
            next
        };
        let reason = walk_loop_observed(entry, step, 2, Some(5))
            .expect_err("non-constant delta is not liftable");
        assert!(reason.contains("stack pointer"), "{reason}");
    }

    #[test]
    fn proven_constant_vsp_delta_detects_constant_and_rejects_drift() {
        let s = |vsp: Expr| SymbolicVmInitialState {
            vsp,
            ..SymbolicVmInitialState::default()
        };
        let base = Expr::register(BitWidth::W64, "vsp_initial");
        let shifted = |k: u64| {
            Expr::binary(BinaryOp::Add, base.clone(), w64(k))
                .expect("add")
        };
        // vsp_initial, -64, -128: a proven constant −64 per iteration.
        let a = s(base.clone());
        let r1 = s(shifted((-64_i64) as u64));
        let reentry = s(shifted((-128_i64) as u64));
        assert_eq!(
            proven_constant_vsp_delta(&[&r1, &a], &reentry),
            Some((-64_i64) as u64)
        );
        // A push-down level whose state is otherwise identical closes the back
        // edge; one that differs in a cell does not, and the refusal stands.
        let pushed = |offset: u64, cell: Expr| SymbolicVmInitialState {
            vsp: shifted(offset),
            memory_values: vec![(shifted(offset), cell)],
            ..SymbolicVmInitialState::default()
        };
        let level1 = pushed((-8_i64) as u64, w64(7));
        let level2 = pushed((-16_i64) as u64, w64(7));
        assert!(closes_under_vm_stack_shift(Va(0x1000), &level1, &level2));
        assert!(!closes_under_vm_stack_shift(
            Va(0x1000),
            &level1,
            &pushed((-16_i64) as u64, w64(9))
        ));
        // Equal VSPs are the ordinary case, handled by `widen_state`, not here.
        assert!(!closes_under_vm_stack_shift(Va(0x1000), &level1, &level1));

        // Iterations that linger at one depth are not a stride violation: the
        // same chain with a repeat still proves −64.
        assert_eq!(
            proven_constant_vsp_delta(&[&r1, &r1, &a, &a], &reentry),
            Some((-64_i64) as u64)
        );
        // A non-constant step (doubling) has no proven constant delta.
        let doubled = s(Expr::binary(BinaryOp::Mul, base.clone(), w64(2)).expect("mul"));
        assert_eq!(proven_constant_vsp_delta(&[&a], &doubled), None);
        // An invariant VSP (zero delta) is not this stall.
        assert_eq!(proven_constant_vsp_delta(&[&a], &s(base.clone())), None);
    }

    #[test]
    fn loop_after_an_imprecise_store_is_refused() {
        let entry = loop_state(0, w64(0));
        let step = |state: &SymbolicVmInitialState| {
            let mut next = state.clone();
            next.memory_tainted = true;
            next
        };
        let reason = walk_loop(entry, step, 1).expect_err("tainted state");
        assert!(reason.contains("imprecise"), "{reason}");
    }

    #[test]
    fn generalized_cell_uses_a_stable_name_so_rewidening_is_a_fixpoint() {
        let a = loop_state(0, w64(0));
        let w1 = widen_state(Va(0x1000), &a, &loop_state(0, w64(1)))
            .expect("widens")
            .expect("changed");
        let w2 = widen_state(Va(0x1000), &a, &loop_state(0, w64(2)))
            .expect("widens")
            .expect("changed");
        // Same (block, cell) site generalizes to one canonical name, never a
        // fresh one per pass, so the widened state stops changing.
        assert_eq!(w1.context_values[&8], w2.context_values[&8]);
        // Re-widening the generalized state against any further value closes.
        assert!(matches!(
            widen_state(Va(0x1000), &w1, &loop_state(0, w64(9))),
            Ok(None)
        ));
    }

    #[test]
    fn branch_over_a_generalized_cell_fails_closed() {
        // A cell widened to TOP is a bare generalized register: an unconstrained
        // value. A branch target that is this value must resolve to no concrete
        // target (fail closed) rather than silently pick one.
        let top = Expr::register(BitWidth::W64, format!("{HAVOC_PREFIX}1000_f"));
        assert!(resolve_targets(&top, VmCfgLimits::default()).is_err());
    }

    #[test]
    fn widen_to_fixpoint_generalizes_a_cell_that_only_differs_across_a_stride() {
        // The counter agrees between the nearest ancestor and the re-entry but
        // differs from an older ancestor (it oscillates). Nearest-only widening
        // would keep it exact and never converge; joining all enclosing entries
        // generalizes it in one step.
        let older = loop_state(0, w64(5));
        let nearest = loop_state(0, w64(9));
        let reentry = loop_state(0, w64(9));
        assert!(matches!(
            widen_state(Va(0x1000), &nearest, &reentry),
            Ok(None)
        ));
        let states = [&nearest, &older];
        let widened = widen_to_fixpoint(Va(0x1000), &states, &reentry)
            .expect("widens")
            .expect("changed");
        assert!(is_havoc(&widened.context_values[&8]));
        assert_eq!(widened.context_values[&0], w64(7));
    }

    #[test]
    fn fold_states_generalizes_the_varying_cell_then_covers_every_reentry() {
        // A block re-reached with distinct counter values folds to one state whose
        // counter is TOP and whose invariant cell stays exact; that fixpoint then
        // covers any further counter value, so the block closes instead of piling
        // up states until the visit cap.
        let states = [
            loop_state(0, w64(0)),
            loop_state(0, w64(1)),
            loop_state(0, w64(2)),
        ];
        let incoming = loop_state(0, w64(3));
        let canonical = fold_states(Va(0x1000), &states, &incoming, None).expect("folds");
        assert!(is_havoc(&canonical.context_values[&8]));
        assert_eq!(canonical.context_values[&0], w64(7));
        assert!(matches!(
            widen_state(Va(0x1000), &canonical, &loop_state(0, w64(99))),
            Ok(None)
        ));
    }

    /// The ceiling family: a block whose every recorded state sits at a
    /// different VM stack depth. Phase-local folding has no sibling to fold
    /// against, so before the shift such a block spent one state per phase and
    /// died at `max_visits_per_block * 2` with "symbolic state keeps changing".
    /// Shifting a recorded phase onto the re-entry's depth closes it instead —
    /// and concedes frame privacy, so it can reach N1 and never N2.
    #[test]
    fn a_block_re_reached_only_at_new_stack_depths_closes_by_shifting_a_phase() {
        use crate::immutable_image::tests::{RDATA_VA, two_section_pe64};
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let observed = ObservedRun::default();
        let seeded = Cell::new(false);
        let base = profile(
            roles(Register::R11, Register::R13, Register::RBP),
            Register::R8,
            crate::BytecodeDirection::Descending,
        );
        let mut walker = test_walker(&pe, base, &observed, &seeded);
        let key = Va(RDATA_VA);
        // Four phases, 8 bytes apart, identical but for the stack base.
        for offset in [0, 8, 16, 24] {
            walker.explored.entry(key).or_default().push(loop_state(offset, w64(7)));
        }
        let incoming = loop_state(32, w64(7));
        assert!(
            walker.explored[&key].iter().all(|s| s.vsp != incoming.vsp),
            "the point of the fixture is that no recorded state shares the re-entry's depth"
        );
        assert_eq!(walker.generalize_across(key, &incoming), GeneralizeAcross::Closed);
        assert!(
            walker.frame_privacy_conceded,
            "closing by shifting the stack base concedes frame privacy"
        );
        assert!(
            !walker.frame_not_private,
            "a concession is not an observed escape: conflating them denied the \
             taint retry to every entry that closed a strided loop"
        );
    }

    /// The shift must not rescue a block whose drift is not a pure stack stride:
    /// a re-entry that also changes a tracked cell still fails closed, because
    /// re-expressing the ancestor at the new depth does not cover it.
    #[test]
    fn shifting_a_phase_does_not_close_a_block_whose_cells_also_move() {
        use crate::immutable_image::tests::{RDATA_VA, two_section_pe64};
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let observed = ObservedRun::default();
        let seeded = Cell::new(false);
        let base = profile(
            roles(Register::R11, Register::R13, Register::RBP),
            Register::R8,
            crate::BytecodeDirection::Descending,
        );
        let mut walker = test_walker(&pe, base, &observed, &seeded);
        let key = Va(RDATA_VA);
        walker.explored.entry(key).or_default().push(loop_state(0, w64(7)));
        let outcome = walker.generalize_across(key, &loop_state(8, w64(9)));
        assert_ne!(outcome, GeneralizeAcross::Closed, "{outcome:?}");
    }

    #[test]
    fn fold_states_refuses_a_stack_pointer_that_moves() {
        // Generalization is sound only when the stack pointer is loop-invariant;
        // a block whose re-entries move it cannot be folded and fails closed.
        let states = [loop_state(0, w64(0))];
        let incoming = loop_state(8, w64(0));
        let refusal = fold_states(Va(0x1000), &states, &incoming, None).expect_err("moves the VSP");
        assert!(refusal.contains("stack pointer"), "{refusal}");
        // The census needs the drift itself, not just the refusal: a growing VM
        // stack is a different lever from one that merely oscillates, and only
        // the stride set tells them apart.
        let noted = vsp_stride_note(&[&states[0]], &incoming, refusal);
        assert!(noted.contains("0x8,"), "{noted}");
        assert!(noted.contains("proven=true"), "{noted}");
    }

    fn byte_cell_state(cell: Expr) -> SymbolicVmInitialState {
        SymbolicVmInitialState {
            context_values: BTreeMap::from([(8, cell)]),
            ..SymbolicVmInitialState::default()
        }
    }

    #[test]
    fn provably_masked_loop_byte_keeps_a_narrow_domain_through_widening() {
        // A loop-carried byte re-masked to its low bits each iteration
        // (`x & 0x0f`) is provably confined to [0, 15]. Widening must carry that
        // proven mask instead of widening to the full [0, 0xffff] TOP, so a table
        // index built from it enumerates instead of overflowing its domain.
        let w = BitWidth::W16;
        let masked = |name: &str| {
            Expr::binary(BinaryOp::And, Expr::register(w, name), Expr::constant(w, 0x0f))
                .expect("same width")
        };
        let widened = widen_state(
            Va(0x2000),
            &byte_cell_state(masked("a")),
            &byte_cell_state(masked("b")),
        )
        .expect("widens")
        .expect("the byte differs between iterations");
        let cell = widened.context_values[&8].clone();
        // The generalized unknown carries the proven 4-bit mask, not full width.
        assert_eq!(havoc_active_mask(&cell), Some(0x0f));
        // A third value within the mask is covered: the narrow unknown is a
        // fixpoint, so the loop closes without widening to full range.
        assert!(matches!(
            widen_state(Va(0x2000), &widened, &byte_cell_state(masked("c"))),
            Ok(None)
        ));
        // The dependent index now has a 16-value domain (the submasks of 0x0f),
        // small enough to enumerate, rather than the full 256-value byte.
        let index =
            Expr::binary(BinaryOp::Mul, cell, Expr::constant(w, 0x10)).expect("same width");
        let inferred = infer_branch_input_domains(&index, InputDomainLimits::default())
            .expect("a masked byte is a bounded input domain");
        assert!(
            inferred.domains.values().all(|values| values.len() <= 16),
            "a 4-bit-masked index byte must not enumerate the full 256-value range"
        );
    }

    #[test]
    fn unprovable_loop_byte_widens_to_full_range_fail_safe() {
        // A loop-carried byte with no provable bound (an opaque value that is not
        // masked) must keep the full-range bare-TOP widening: narrowing it would
        // drop reachable values and miss real branch targets (unsound). A branch
        // over the full-range unknown fails closed.
        let w = BitWidth::W16;
        let widened = widen_state(
            Va(0x3000),
            &byte_cell_state(Expr::register(w, "p")),
            &byte_cell_state(Expr::register(w, "q")),
        )
        .expect("widens")
        .expect("the byte differs between iterations");
        let cell = &widened.context_values[&8];
        // No mask could be proven, so the cell stays a bare full-width unknown.
        assert!(is_havoc(cell));
        assert_eq!(havoc_active_mask(cell), Some(w.mask()));
    }
}
