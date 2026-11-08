use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::Serialize;
use thiserror::Error;
use vmpx_pe::{PeImage, Va};

use crate::{
    AppliedNativeCallEffect, BinaryOp, BitWidth, DecodedVmInstruction, Expr, ExprError, ExprKind,
    NativeCallArgumentAlias, NativeCallArgumentDescriptor, NativeCallArgumentLocation,
    NativeCallEffectSummary, NativeCallEpochAllocator, NativeCallImportResolution,
    NativeCallInvocationDescriptor, NativeCallMemoryDecision, NativeCallMemoryEffect,
    NativeCallResultEffect, NativeCallSelectorOutcome, NativeCallTargetDescriptor,
    NativeCallValueSource, ResizeOp, TableChecksumMetadata, UnaryOp, VmHandlerSemantic,
    VmSemanticMetadata, apply_native_call_effect, distinct_node_count_bounded,
    resolve_native_call_import, summarize_known_import,
};

thread_local! {
    static SERVED_READ_AUDIT: std::cell::RefCell<Option<BTreeSet<(u64, u8)>>> =
        const { std::cell::RefCell::new(None) };
}

/// Image bytes served as constants, by provenance (audit only).
pub const SERVED_UNSUPPRESSED: u8 = 0;
pub const SERVED_PROVEN_SUPPRESSED: u8 = 1;
pub const SERVED_LABELED_SUPPRESSED: u8 = 2;

/// DIAGNOSTIC ONLY: start (or stop) recording, per thread, the byte addresses of
/// image bytes served as constants, with their provenance, so a run can be
/// audited against runtime evidence.
pub fn set_served_page_audit(enabled: bool) {
    SERVED_READ_AUDIT.with(|sink| *sink.borrow_mut() = enabled.then(BTreeSet::new));
}

/// Drains the `(address, provenance)` audit recorded on this thread.
#[must_use]
pub fn take_served_page_audit() -> Vec<(u64, u8)> {
    SERVED_READ_AUDIT.with(|sink| {
        sink.borrow_mut()
            .as_mut()
            .map(|set| std::mem::take(set).into_iter().collect())
            .unwrap_or_default()
    })
}

fn note_served_read(address: u64, kind: u8) {
    SERVED_READ_AUDIT.with(|sink| {
        if let Some(set) = sink.borrow_mut().as_mut() {
            set.insert((address, kind));
        }
    });
}

const WIN64_ARGUMENT_REGISTERS: [&str; 4] = ["RCX", "RDX", "R8", "R9"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SymbolicExecutionLimits {
    /// Top of the entry stub's fresh frame, in `vsp_initial` coordinates.
    ///
    /// A tracked cell at or above this is NOT in the VM's private allocation --
    /// it is caller memory an external pointer may alias -- so the narrowed
    /// havoc discards it. That makes the discharge's "every cell lies inside the
    /// frame" obligation hold directly.
    pub private_frame_hi: Option<i64>,
    pub max_instructions: usize,
    pub max_memory_bytes: usize,
    pub max_expression_nodes: usize,
    /// EXPERIMENTAL: serve constant reads of read-only file-backed image bytes
    /// even after an external store suppressed image reads. Unsound if the
    /// page is re-protected writable at runtime; the block is labeled
    /// `AssumesImageProtectionStable`.
    pub serve_read_only_when_suppressed: bool,
    /// Serve declared read-only file-backed pages after an external store when
    /// no call that may change page protection has run on the path (a proof,
    /// no assumption). Off only to A/B the baseline.
    pub prove_read_only_when_suppressed: bool,
    /// OBSERVED TIER: after an external store suppressed image reads, serve a
    /// constant read of a non-writable image address from the observed seed (a
    /// post-init snapshot) when the file byte is withheld. Valid for the captured
    /// state only; consulting the seed labels the block `ObservedGlobal`, so it
    /// never reaches the proven tier.
    pub serve_observed_when_suppressed: bool,
    /// Treat an external store address that is data read from the VM frame (not a
    /// frame pointer) like any other opaque pointer: the labeled
    /// `AssumesExternalStoresDisjointFromVmStack` narrowing instead of the
    /// tainting havoc. Only while no frame pointer has escaped or been lost.
    pub assume_loaded_store_addresses_disjoint: bool,
    /// Instant past which this block's execution gives up, so the per-entry
    /// wall-clock budget bounds ONE block and not merely the gaps between
    /// blocks. The CFG walk checked its deadline only between worklist items, so
    /// a single block whose expressions blow up superlinearly ran unbounded --
    /// which is why the per-block instruction limit could not be raised past 512
    /// despite 102 entries being refused on it: doubling it hung the sweep
    /// instead of failing closed. `None` leaves execution untimed.
    pub deadline: Option<std::time::Instant>,
}

impl Default for SymbolicExecutionLimits {
    fn default() -> Self {
        Self {
            private_frame_hi: None,
            max_instructions: 4_096,
            max_memory_bytes: 1_048_576,
            max_expression_nodes: 200_000,
            serve_read_only_when_suppressed: false,
            prove_read_only_when_suppressed: true,
            serve_observed_when_suppressed: false,
            assume_loaded_store_addresses_disjoint: false,
            deadline: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolicExecutionStop {
    EndOfStream,
    ControlFlowBoundary,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SymbolicMemoryWrite {
    pub instruction_index: usize,
    pub address: Expr,
    pub value: Expr,
    /// Memory epoch immediately after this store committed. Reads whose epoch is
    /// at least this value observe the store; reads below it precede it.
    pub epoch: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SymbolicMemoryHavoc {
    pub instruction_index: usize,
    pub cause: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SymbolicFreshAllocation {
    pub allocation: Expr,
    pub size: Expr,
    pub zero_initialized: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SymbolicNativeCall {
    pub instruction_index: usize,
    pub invocation: NativeCallInvocationDescriptor,
    pub import_resolution: Option<NativeCallImportResolution>,
    pub applied_effect: AppliedNativeCallEffect,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SymbolicVmExecutionReport {
    pub schema_version: u32,
    pub executed_instructions: usize,
    pub stop_reason: SymbolicExecutionStop,
    pub final_vsp: Expr,
    pub final_vsp_offset: Option<i64>,
    pub final_flags: Expr,
    pub branch_target: Option<Expr>,
    pub branch_target_dependencies: Vec<String>,
    pub context_cells: BTreeMap<i64, Expr>,
    /// Byte-addressed memory state retained for sound continuation into a
    /// successor block. Entries include VM-stack and proven external writes.
    pub memory_values: Vec<(Expr, Expr)>,
    pub memory_epoch: u64,
    /// True after an external store with an address that could alias any
    /// tracked VM-memory cell. Precise pre-store memory is discarded then.
    pub memory_tainted: bool,
    /// True once image-source reads can no longer be trusted because an external
    /// store to a non-VM-stack address may have overwritten an in-image byte.
    /// Unlike `memory_tainted` this does not block branch resolution or resume:
    /// the VM-private operand stack is intact, only original image bytes are
    /// withheld (const loads return fresh unknowns). Carried across blocks.
    pub image_reads_suppressed: bool,
    /// True when this block narrowed an unprovable external store under the
    /// labeled `AssumesExternalStoresDisjointFromVmStack` assumption instead of
    /// the broad havoc. Surfaced so the CFG can record the assumption.
    pub assumed_external_disjoint_from_vm_stack: bool,
    /// Whether the narrowed havoc preserved a cell rooted at an allocation this
    /// frame made. Set only when such a cell was actually retained, so with
    /// none present the run is byte-for-byte what it was before.
    pub assumed_external_disjoint_from_fresh_allocations: bool,
    /// True when a suppressed constant read was served from a read-only
    /// file-backed page; the CFG records `AssumesImageProtectionStable`.
    pub served_read_only_image: bool,
    /// True once a native call that is not a trusted allocation contract has run
    /// on this path since VM entry: the callee could change page protections.
    /// While false, no instruction on the path can make a non-writable image
    /// page writable, so a store can never alter a declared read-only page
    /// (it faults) and such pages are served from the file even after an
    /// external store, with no assumption. Carried across blocks; OR-merged.
    pub protection_may_have_changed: bool,
    /// True when a value derived from `vsp_initial` was stored to an address that
    /// is not a tracked VM-stack cell, or passed to a native call as target or
    /// argument: the frame address may then be reachable from outside the VM.
    pub vm_stack_address_escapes: bool,
    /// Superset of `vm_stack_address_escapes`: also set when a value merely
    /// derived from `vsp_initial` (e.g. data loaded from an untracked frame
    /// cell) leaves the frame. Only trusted to be benign when no frame pointer
    /// was discarded from tracked memory (`vm_stack_pointer_lost`).
    pub vm_stack_derived_escapes: bool,
    /// True when tracked memory holding a frame-pointer value was discarded, so a
    /// later untracked load from the frame could return that pointer.
    pub vm_stack_pointer_lost: bool,
    /// Highest end offset above `vsp_initial` of any VM-stack cell written by
    /// this block, or `None` if it wrote none.
    pub vm_stack_store_hi: Option<i64>,
    pub external_writes: Vec<SymbolicMemoryWrite>,
    pub memory_havocs: Vec<SymbolicMemoryHavoc>,
    pub native_calls: Vec<SymbolicNativeCall>,
    pub fresh_allocations: Vec<SymbolicFreshAllocation>,
    pub next_native_call_epoch: u64,
    pub peak_expression_nodes: usize,
    pub tracked_memory_bytes: usize,
    pub resolved_image_bytes: usize,
}

#[derive(Clone, Debug, Eq, Error, PartialEq, Serialize)]
pub enum SymbolicMemoryReadError {
    #[error("tracked symbolic memory has no byte at address {address:?}")]
    MissingByte { address: Expr },
    #[error("tracked symbolic memory address arithmetic failed: {message}")]
    Address { message: String },
    #[error("tracked symbolic memory value assembly failed: {message}")]
    Assembly { message: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SymbolicVmInitialState {
    pub vsp: Expr,
    pub flags: Expr,
    pub context_values: BTreeMap<i64, Expr>,
    pub memory_values: Vec<(Expr, Expr)>,
    pub memory_epoch: u64,
    pub memory_tainted: bool,
    /// See `SymbolicVmExecutionReport::image_reads_suppressed`: carried into a
    /// successor block so its image-source reads stay withheld after a narrowed
    /// external store.
    pub image_reads_suppressed: bool,
    /// See `SymbolicVmExecutionReport::protection_may_have_changed`.
    pub protection_may_have_changed: bool,
    pub next_native_call_epoch: u64,
    pub fresh_allocations: Vec<SymbolicFreshAllocation>,
}

impl Default for SymbolicVmInitialState {
    fn default() -> Self {
        Self {
            vsp: Expr::register(BitWidth::W64, "vsp_initial"),
            flags: Expr::register(BitWidth::W64, "flags_initial"),
            context_values: BTreeMap::new(),
            memory_values: Vec::new(),
            memory_epoch: 0,
            memory_tainted: false,
            image_reads_suppressed: false,
            protection_may_have_changed: false,
            next_native_call_epoch: 0,
            fresh_allocations: Vec::new(),
        }
    }
}

impl SymbolicVmInitialState {
    /// Substitutes concrete or symbolic path constraints into a continuation
    /// state before exploring a specialized successor.
    ///
    /// # Errors
    ///
    /// Returns an expression-width error if a replacement is incompatible with
    /// the dependency it replaces.
    pub fn substitute(&self, replacements: &BTreeMap<String, Expr>) -> Result<Self, ExprError> {
        Ok(Self {
            vsp: self.vsp.substitute(replacements)?,
            flags: self.flags.substitute(replacements)?,
            context_values: self
                .context_values
                .iter()
                .map(|(slot, value)| Ok((*slot, value.substitute(replacements)?)))
                .collect::<Result<_, ExprError>>()?,
            memory_values: self
                .memory_values
                .iter()
                .map(|(address, value)| {
                    Ok((
                        address.substitute(replacements)?,
                        value.substitute(replacements)?,
                    ))
                })
                .collect::<Result<_, ExprError>>()?,
            memory_epoch: self.memory_epoch,
            memory_tainted: self.memory_tainted,
            image_reads_suppressed: self.image_reads_suppressed,
            protection_may_have_changed: self.protection_may_have_changed,
            next_native_call_epoch: self.next_native_call_epoch,
            fresh_allocations: self
                .fresh_allocations
                .iter()
                .map(|allocation| {
                    Ok(SymbolicFreshAllocation {
                        allocation: allocation.allocation.substitute(replacements)?,
                        size: allocation.size.substitute(replacements)?,
                        zero_initialized: allocation.zero_initialized,
                    })
                })
                .collect::<Result<_, ExprError>>()?,
        })
    }
}

impl SymbolicVmExecutionReport {
    /// Builds an exact successor-block state from this block's final machine
    /// state. The returned VSP is the post-boundary VSP when a supported
    /// boundary semantic was executed.
    #[must_use]
    pub fn continuation_state(&self) -> SymbolicVmInitialState {
        SymbolicVmInitialState {
            vsp: self.final_vsp.clone(),
            flags: self.final_flags.clone(),
            context_values: self.context_cells.clone(),
            memory_values: self.memory_values.clone(),
            memory_epoch: self.memory_epoch,
            memory_tainted: self.memory_tainted,
            image_reads_suppressed: self.image_reads_suppressed,
            protection_may_have_changed: self.protection_may_have_changed,
            next_native_call_epoch: self.next_native_call_epoch,
            fresh_allocations: self.fresh_allocations.clone(),
        }
    }

    /// Reads a width-sized value from the exact byte-addressed memory snapshot
    /// retained at the end of symbolic execution.
    ///
    /// # Errors
    ///
    /// Returns a structured error if address arithmetic fails, any byte is
    /// absent, or the byte vector cannot be reassembled at the requested width.
    pub fn read_tracked_memory(
        &self,
        address: Expr,
        width: BitWidth,
    ) -> Result<Expr, SymbolicMemoryReadError> {
        let address = canonicalize_pointer(address);
        let mut bytes = Vec::with_capacity(width.bytes() as usize);
        for byte in 0..width.bytes() {
            let byte_address =
                add_pointer_offset(address.clone(), i64::from(byte)).map_err(|error| {
                    SymbolicMemoryReadError::Address {
                        message: error.to_string(),
                    }
                })?;
            let value = self
                .memory_values
                .iter()
                .find(|(candidate, _)| candidate == &byte_address)
                .map(|(_, value)| value.clone())
                .ok_or(SymbolicMemoryReadError::MissingByte {
                    address: byte_address,
                })?;
            bytes.push(value);
        }
        combine_little_endian(&bytes).map_err(|error| SymbolicMemoryReadError::Assembly {
            message: error.to_string(),
        })
    }
}

pub trait SymbolicMemorySource {
    fn read_byte(&self, address: Va) -> Option<u8>;

    /// Observed runtime value for a slot that is zero or absent in the static
    /// image. EXPERIMENTAL: a byte seen on ONE detonation, not a proof; valid
    /// for one run/environment only. The sound path leaves this `None`; only an
    /// observed overlay returns a byte, and the symbolic executor consults it
    /// solely to resolve a provably-null image read it would otherwise fail on.
    fn observed_byte(&self, _address: Va) -> Option<u8> {
        None
    }

    /// Byte of a read-only, file-backed page: determinate in the file and
    /// immune to stores, provided the page protection never changes.
    fn read_only_byte(&self, _address: Va) -> Option<u8> {
        None
    }

    /// True when `address` lies in a readable, non-writable image section (or the
    /// headers), regardless of any volatility veto on its bytes.
    fn is_read_only_address(&self, _address: Va) -> bool {
        false
    }

    /// PROOF that nothing in the analyzed image can change page protection (see
    /// `image_cannot_reprotect`). While true and no call on the path can either,
    /// a non-writable page keeps its loader protection, so no store can alter
    /// it. The default `false` keeps every serve a labeled assumption.
    fn protection_stability_proven(&self) -> bool {
        false
    }

    fn resolve_native_call_import(&self, _target: u64) -> Option<NativeCallImportResolution> {
        None
    }
}

impl SymbolicMemorySource for PeImage<'_> {
    fn read_byte(&self, address: Va) -> Option<u8> {
        let rva = self.optional.image_base.checked_rva_of(address)?;
        let offset = usize::try_from(self.rva_to_file_offset(rva).ok()?.get()).ok()?;
        self.data().get(offset).copied()
    }

    fn resolve_native_call_import(&self, target: u64) -> Option<NativeCallImportResolution> {
        let imports = self.report().imports;
        Some(resolve_native_call_import(
            &imports,
            self.optional.image_base,
            target,
        ))
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq, Serialize)]
#[error(
    "symbolic VM execution failed at instruction {instruction_index} ({vip}, {semantic:?}): {message}"
)]
pub struct SymbolicExecutionError {
    pub instruction_index: usize,
    pub vip: Va,
    pub semantic: VmHandlerSemantic,
    pub message: String,
}

#[derive(Default)]
struct SymbolicByteMemory {
    bytes: BTreeMap<i64, Expr>,
    touched_cells: BTreeSet<i64>,
    seeded_cells: BTreeSet<i64>,
    prefix: String,
}

impl SymbolicByteMemory {
    fn new(prefix: impl Into<String>) -> Self {
        Self {
            bytes: BTreeMap::new(),
            touched_cells: BTreeSet::new(),
            seeded_cells: BTreeSet::new(),
            prefix: prefix.into(),
        }
    }

    fn seed_cell(&mut self, cell: i64) -> Result<(), String> {
        self.touched_cells.insert(cell);
        if self.seeded_cells.insert(cell) {
            let initial = Expr::register(BitWidth::W64, format!("{}[{cell:+#x}]", self.prefix));
            for byte in 0_u32..8 {
                let offset = cell
                    .checked_add(i64::from(byte))
                    .ok_or_else(|| "symbolic memory cell offset overflows".to_owned())?;
                let value = Expr::extract(BitWidth::W8, byte * 8, initial.clone())
                    .map_err(|error| error.to_string())?;
                self.bytes.entry(offset).or_insert(value);
            }
        }
        Ok(())
    }

    fn load(&mut self, offset: i64, width: BitWidth) -> Result<Expr, String> {
        let end = offset
            .checked_add(i64::from(width.bytes()))
            .ok_or_else(|| "symbolic memory access overflows".to_owned())?;
        let first_cell = offset.div_euclid(8) * 8;
        let last_cell = end
            .checked_sub(1)
            .ok_or_else(|| "symbolic memory access underflows".to_owned())?
            .div_euclid(8)
            * 8;
        let mut cell = first_cell;
        loop {
            self.seed_cell(cell)?;
            if cell == last_cell {
                break;
            }
            cell = cell
                .checked_add(8)
                .ok_or_else(|| "symbolic memory cell traversal overflows".to_owned())?;
        }
        let mut bytes = Vec::with_capacity(width.bytes() as usize);
        for byte in 0..width.bytes() {
            let byte_offset = offset
                .checked_add(i64::from(byte))
                .ok_or_else(|| "symbolic memory byte offset overflows".to_owned())?;
            let value = self
                .bytes
                .get(&byte_offset)
                .cloned()
                .ok_or_else(|| "symbolic memory seed is incomplete".to_owned())?;
            bytes.push(value);
        }
        combine_little_endian(&bytes).map_err(|error| error.to_string())
    }

    fn store(&mut self, offset: i64, value: &Expr) -> Result<(), String> {
        for byte in 0..value.width.bytes() {
            let byte_offset = offset
                .checked_add(i64::from(byte))
                .ok_or_else(|| "symbolic memory byte offset overflows".to_owned())?;
            let byte_value = Expr::extract(BitWidth::W8, byte * 8, value.clone())
                .map_err(|error| error.to_string())?;
            self.bytes.insert(byte_offset, byte_value);
            self.touched_cells.insert(byte_offset.div_euclid(8) * 8);
        }
        Ok(())
    }
}

struct SymbolicMachine<'a> {
    vsp: Expr,
    vsp_offset: Option<i64>,
    flags: Expr,
    memory: HashMap<Expr, Expr>,
    context: SymbolicByteMemory,
    external_writes: Vec<SymbolicMemoryWrite>,
    memory_havocs: Vec<SymbolicMemoryHavoc>,
    native_calls: Vec<SymbolicNativeCall>,
    fresh_allocations: Vec<SymbolicFreshAllocation>,
    native_call_epochs: NativeCallEpochAllocator,
    peak_expression_nodes: usize,
    resolved_image_addresses: BTreeSet<Va>,
    memory_source: Option<&'a dyn SymbolicMemorySource>,
    memory_epoch: u64,
    /// Bumped by any store this machine cannot place at a concrete byte, so it
    /// could reach anything. Names unresolved loads AT A CONCRETE ADDRESS,
    /// together with that byte's own version below.
    opaque_store_epoch: u64,
    /// Per concrete byte address, how many times it has been stored.
    ///
    /// `memory_epoch` counts EVERY store, and naming unresolved loads from it
    /// meant the same load of the same address read as a different value on
    /// each pass: a VM handler's ordinary pushes and pops race it forward by
    /// ~1000 per loop iteration, so two iterations never compared equal and no
    /// loop carrying an unresolved load could converge. A store to one concrete
    /// byte cannot change a different concrete byte, so it must not rename that
    /// byte's loads.
    byte_versions: BTreeMap<u64, u64>,
    /// Address of the most recent image byte served as ZERO with no observed
    /// seed behind it. A pointer slot that is zero in the file but initialised
    /// at run time produces the null-read stall several instructions later, by
    /// which point the address is just `0` and the slot is unrecoverable -- so
    /// the slot is recorded here and reported with the refusal.
    last_zero_image_read: Option<u64>,
    memory_tainted: bool,
    image_reads_suppressed: bool,
    assumed_external_disjoint_from_vm_stack: bool,
    assumed_external_disjoint_from_fresh_allocations: bool,
    serve_read_only_when_suppressed: bool,
    assume_loaded_store_addresses_disjoint: bool,
    prove_read_only_when_suppressed: bool,
    serve_observed_when_suppressed: bool,
    served_read_only_image: bool,
    protection_may_have_changed: bool,
    image_protection_stable_cache: Option<bool>,
    vm_stack_address_escapes: bool,
    vm_stack_derived_escapes: bool,
    vm_stack_pointer_lost: bool,
    frame_pointer_in_memory: bool,
    vm_stack_store_hi: Option<i64>,
    limits_private_frame_hi: Option<i64>,
    checksum_byte_limit: usize,
}

impl<'a> SymbolicMachine<'a> {
    fn new(
        initial: &SymbolicVmInitialState,
        memory_source: Option<&'a dyn SymbolicMemorySource>,
        limits: SymbolicExecutionLimits,
    ) -> Result<Self, String> {
        if initial.vsp.width != BitWidth::W64 {
            return Err("initial VSP is not 64 bits".to_owned());
        }
        if initial.flags.width != BitWidth::W64 {
            return Err("initial flags value is not 64 bits".to_owned());
        }
        if initial.memory_tainted {
            return Err(
                "initial memory is tainted by an unbounded alias and cannot be resumed soundly"
                    .to_owned(),
            );
        }
        let initial_memory_bytes = initial
            .context_values
            .values()
            .chain(initial.memory_values.iter().map(|(_, value)| value))
            .try_fold(0_usize, |total, value| {
                total.checked_add(value.width.bytes() as usize)
            })
            .ok_or_else(|| "initial symbolic memory size overflows".to_owned())?;
        if initial_memory_bytes > limits.max_memory_bytes {
            return Err(format!(
                "initial symbolic memory exceeds {} bytes",
                limits.max_memory_bytes
            ));
        }
        let mut initial_expression_nodes = 0_usize;
        for expression in std::iter::once(&initial.vsp)
            .chain(std::iter::once(&initial.flags))
            .chain(initial.context_values.values())
            .chain(
                initial
                    .memory_values
                    .iter()
                    .flat_map(|(address, value)| [address, value]),
            )
            .chain(
                initial
                    .fresh_allocations
                    .iter()
                    .flat_map(|allocation| [&allocation.allocation, &allocation.size]),
            )
        {
            let nodes = expression.node_count_bounded(limits.max_expression_nodes);
            initial_expression_nodes = initial_expression_nodes.saturating_add(nodes);
            if nodes > limits.max_expression_nodes
                || initial_expression_nodes > limits.max_expression_nodes
            {
                return Err(format!(
                    "initial expressions exceed aggregate limit {}",
                    limits.max_expression_nodes
                ));
            }
        }
        let mut machine = Self {
            vsp: initial.vsp.clone(),
            vsp_offset: pointer_offset(&initial.vsp),
            flags: initial.flags.clone(),
            memory: HashMap::new(),
            context: SymbolicByteMemory::new("context_initial"),
            external_writes: Vec::new(),
            memory_havocs: Vec::new(),
            native_calls: Vec::new(),
            fresh_allocations: initial.fresh_allocations.clone(),
            native_call_epochs: NativeCallEpochAllocator::new(initial.next_native_call_epoch),
            peak_expression_nodes: 1,
            resolved_image_addresses: BTreeSet::new(),
            memory_source,
            memory_epoch: initial.memory_epoch,
            opaque_store_epoch: initial.memory_epoch,
            byte_versions: BTreeMap::new(),
            last_zero_image_read: None,
            memory_tainted: initial.memory_tainted,
            image_reads_suppressed: initial.image_reads_suppressed,
            assumed_external_disjoint_from_vm_stack: false,
            assumed_external_disjoint_from_fresh_allocations: false,
            serve_read_only_when_suppressed: limits.serve_read_only_when_suppressed,
            assume_loaded_store_addresses_disjoint: limits.assume_loaded_store_addresses_disjoint,
            prove_read_only_when_suppressed: limits.prove_read_only_when_suppressed,
            serve_observed_when_suppressed: limits.serve_observed_when_suppressed,
            served_read_only_image: false,
            protection_may_have_changed: initial.protection_may_have_changed,
            image_protection_stable_cache: None,
            vm_stack_address_escapes: false,
            vm_stack_derived_escapes: false,
            vm_stack_pointer_lost: false,
            frame_pointer_in_memory: initial
                .memory_values
                .iter()
                .any(|(_, value)| is_frame_pointer_value(value)),
            vm_stack_store_hi: None,
            limits_private_frame_hi: limits.private_frame_hi,
            checksum_byte_limit: limits.max_memory_bytes,
        };
        for (offset, value) in &initial.context_values {
            machine.context.store(*offset, value)?;
        }
        for (address, value) in &initial.memory_values {
            if address.width != BitWidth::W64 {
                return Err("initial memory address is not 64 bits".to_owned());
            }
            let address = canonicalize_pointer(address.clone());
            for byte in 0..value.width.bytes() {
                let byte_address = add_pointer_offset(address.clone(), i64::from(byte))
                    .map_err(|error| error.to_string())?;
                if machine.memory.contains_key(&byte_address) {
                    return Err("initial memory contains overlapping addresses".to_owned());
                }
            }
            machine.write_memory_bytes(&address, value)?;
        }
        // Initial memory is a snapshot, not a sequence of runtime writes.
        machine.memory_epoch = initial.memory_epoch;
        machine.check_limits(limits)?;
        Ok(machine)
    }

    fn adjust_vsp(&mut self, delta: i32) -> Result<(), String> {
        self.vsp_offset = self
            .vsp_offset
            .and_then(|current| current.checked_add(i64::from(delta)));
        self.vsp = add_pointer_offset(self.vsp.clone(), i64::from(delta))
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn load_stack(&mut self, displacement: i64, width: BitWidth) -> Result<Expr, String> {
        let address = add_pointer_offset(self.vsp.clone(), displacement)
            .map_err(|error| error.to_string())?;
        self.load_memory(address, width)
    }

    fn store_stack(&mut self, displacement: i64, value: &Expr) -> Result<(), String> {
        let address = add_pointer_offset(self.vsp.clone(), displacement)
            .map_err(|error| error.to_string())?;
        self.store_memory(&address, value)
    }

    fn image_protection_stable(&mut self) -> bool {
        let source = self.memory_source;
        *self.image_protection_stable_cache.get_or_insert_with(|| {
            source.is_some_and(|source| source.protection_stability_proven())
        })
    }

    fn load_memory(&mut self, address: Expr, width: BitWidth) -> Result<Expr, String> {
        let address = canonicalize_pointer(address);
        let mut bytes = Vec::with_capacity(width.bytes() as usize);
        let mut all_unknown = true;
        for byte in 0..width.bytes() {
            let byte_address = add_pointer_offset(address.clone(), i64::from(byte))
                .map_err(|error| error.to_string())?;
            if let Some(value) = self.memory.get(&byte_address) {
                all_unknown = false;
                bytes.push(value.clone());
            } else if let ExprKind::Const { value } = byte_address.kind
                && !self.memory_tainted
                && !self.image_reads_suppressed
            {
                let byte_va = Va(value);
                let image_byte = self
                    .memory_source
                    .and_then(|source| source.read_byte(byte_va));
                // A slot that is zero (or absent) in the static image but
                // runtime-initialized is the documented null-read stall: the
                // image proves only the pre-init zero. If an observed-runtime
                // seed records this slot, use it so recovery continues past the
                // null read. EXPERIMENTAL, observed-labeled (not sound output);
                // a non-zero image byte is genuinely resolvable and never
                // overridden.
                //
                // MEASURED: this does NOT reach the null reads that block 25
                // entries on systems.xpl. Their slots (0x378135cd4, 0x377e387a2,
                // 0x377fcdf9c) are in .vmp2, which is read-only and so eligible
                // for the seed -- and the seed still does not carry them. Two
                // post-init snapshots, one at 39.1B steps and one at 120B, give
                // a byte-identical seed (7365 bytes over 1559 ranges), because
                // the protector finishes writing its read-only sections early
                // (idata writes stop at 16.8B). The slots are zero at every
                // depth, so they are not runtime-initialized globals at all.
                //
                // Those reads are therefore real null dereferences -- VMProtect
                // reading address 0 on purpose -- and the refusal's own wording
                // is the prescription: they need exception-flow analysis, not a
                // deeper snapshot. The observed bridge cannot close them.
                let seeded = if image_byte.is_none_or(|byte| byte == 0) {
                    self.memory_source
                        .and_then(|source| source.observed_byte(byte_va))
                } else {
                    None
                };
                if let Some(byte) = seeded {
                    all_unknown = false;
                    self.resolved_image_addresses.insert(byte_va);
                    bytes.push(Expr::constant(BitWidth::W8, u64::from(byte)));
                } else if let Some(byte) = image_byte {
                    all_unknown = false;
                    if byte == 0 {
                        self.last_zero_image_read = Some(value);
                    }
                    note_served_read(value, SERVED_UNSUPPRESSED);
                    self.resolved_image_addresses.insert(byte_va);
                    bytes.push(Expr::constant(BitWidth::W8, u64::from(byte)));
                } else {
                    bytes.push(Expr::intrinsic(
                        BitWidth::W8,
                        format!("memory_load_epoch_{}", self.memory_epoch),
                        vec![byte_address],
                    ));
                }
            } else if let ExprKind::Const { value } = byte_address.kind
                && self.prove_read_only_when_suppressed
                && self.image_reads_suppressed
                && !self.protection_may_have_changed
                && !self.memory_tainted
                && self.image_protection_stable()
                && let Some(byte) = self
                    .memory_source
                    .and_then(|source| source.read_only_byte(Va(value)))
            {
                // PROVEN: only a non-writable file-backed page reaches here, and no
                // instruction on this path since VM entry can have changed page
                // protection, so an external store cannot have altered it.
                all_unknown = false;
                note_served_read(value, SERVED_PROVEN_SUPPRESSED);
                self.resolved_image_addresses.insert(Va(value));
                bytes.push(Expr::constant(BitWidth::W8, u64::from(byte)));
            } else if let ExprKind::Const { value } = byte_address.kind
                && self.serve_read_only_when_suppressed
                && (!self.memory_tainted || serve_read_only_when_tainted())
                && let Some(byte) = self
                    .memory_source
                    .and_then(|source| source.read_only_byte(Va(value)))
            {
                all_unknown = false;
                self.served_read_only_image = true;
                note_served_read(value, SERVED_LABELED_SUPPRESSED);
                self.resolved_image_addresses.insert(Va(value));
                bytes.push(Expr::constant(BitWidth::W8, u64::from(byte)));
            } else if let ExprKind::Const { value } = byte_address.kind
                && self.serve_observed_when_suppressed
                && self.image_reads_suppressed
                && !self.memory_tainted
                && let Some(source) = self.memory_source
                && source.is_read_only_address(Va(value))
                && let Some(byte) = source.observed_byte(Va(value))
            {
                // OBSERVED: a non-writable image address whose file byte is withheld
                // (volatile) read from the post-init snapshot. `observed_byte`
                // flips the source's seeded flag, which labels the CFG
                // `ObservedGlobal` (observed tier, never proven).
                all_unknown = false;
                self.resolved_image_addresses.insert(Va(value));
                bytes.push(Expr::constant(BitWidth::W8, u64::from(byte)));
            } else if let ExprKind::Const { value: at } = byte_address.kind {
                // A concrete byte: only a store that could reach THIS byte may
                // rename it. The `memory_load_epoch_` prefix is kept because
                // consumers match on it.
                let version = self.byte_versions.get(&at).copied().unwrap_or(0);
                bytes.push(Expr::intrinsic(
                    BitWidth::W8,
                    format!("memory_load_epoch_{}_{version}", self.opaque_store_epoch),
                    vec![byte_address],
                ));
            } else {
                // A symbolic address could be any byte, so every store still
                // renames it: that is exactly what `memory_epoch` counts.
                bytes.push(Expr::intrinsic(
                    BitWidth::W8,
                    format!("memory_load_epoch_{}", self.memory_epoch),
                    vec![byte_address],
                ));
            }
        }
        if all_unknown {
            Ok(Expr::intrinsic(
                width,
                format!("memory_load_epoch_{}", self.memory_epoch),
                vec![address],
            ))
        } else {
            combine_little_endian(&bytes).map_err(|error| error.to_string())
        }
    }

    fn advance_memory_epoch(&mut self) -> Result<(), String> {
        self.memory_epoch = self
            .memory_epoch
            .checked_add(1)
            .ok_or_else(|| "symbolic memory epoch overflows".to_owned())?;
        Ok(())
    }

    fn write_memory_bytes(&mut self, address: &Expr, value: &Expr) -> Result<(), String> {
        for byte in 0..value.width.bytes() {
            let byte_address = add_pointer_offset(address.clone(), i64::from(byte))
                .map_err(|error| error.to_string())?;
            // Version the byte so that if this cell is later discarded, loads of
            // it cannot reuse a name minted before the store. A byte we cannot
            // place could be any byte, so that invalidates every concrete name.
            match byte_address.kind {
                ExprKind::Const { value: at } => {
                    *self.byte_versions.entry(at).or_default() += 1;
                }
                _ => self.opaque_store_epoch = self.opaque_store_epoch.wrapping_add(1),
            }
            self.memory.insert(
                byte_address,
                Expr::extract(BitWidth::W8, byte * 8, value.clone())
                    .map_err(|error| error.to_string())?,
            );
        }
        Ok(())
    }

    /// A value leaving the frame exposes a frame pointer only if it IS one;
    /// data merely read from the frame does not. Anything not provably
    /// pointer-free counts as a pointer.
    fn note_value_leaves_frame(&mut self, value: &Expr) {
        if depends_on_vsp_root(value) {
            self.vm_stack_derived_escapes = true;
            if is_frame_pointer_value(value) {
                self.vm_stack_address_escapes = true;
            }
        }
    }

    /// Discards tracked cells not satisfying `keep`, recording whether a
    /// frame-pointer value was among them.
    fn retain_memory(&mut self, keep: impl Fn(&Expr) -> bool) {
        if self.frame_pointer_in_memory {
            self.vm_stack_pointer_lost |= self
                .memory
                .iter()
                .any(|(address, value)| !keep(address) && is_frame_pointer_value(value));
        }
        self.memory.retain(|address, _| keep(address));
    }

    fn store_memory(&mut self, address: &Expr, value: &Expr) -> Result<(), String> {
        if is_frame_pointer_value(value) {
            self.frame_pointer_in_memory = true;
        }
        let address = canonicalize_pointer(address.clone());
        if let Some(offset) = pointer_offset(&address) {
            let end = offset.saturating_add(i64::from(value.width.bytes()));
            self.vm_stack_store_hi = self.vm_stack_store_hi.max(Some(end));
        }
        self.advance_memory_epoch()?;
        self.write_memory_bytes(&address, value)
    }

    /// Sound over-approximation of a store whose address cannot be placed in a
    /// tracked region.
    ///
    /// When the address could reach the VM's private operand stack (it still
    /// references `vsp_initial`), the whole byte-addressed memory is discarded
    /// and tainted: loads afterwards return fresh unknowns, never a stale cached
    /// or image-source byte, and the block cannot resume or branch. This is the
    /// unconditional sound fallback.
    ///
    /// Otherwise the address is native-register-rooted or opaque. Under the
    /// labeled assumption that external stores do not alias the VM's private
    /// operand stack (its internal stack is private; caller/parameter pointers
    /// do not point into it), only the cells that could genuinely alias such a
    /// pointer are discarded (concrete/image/allocation cells); the vsp-rooted
    /// VM operand-stack cells are preserved, so the post-store VIP stays precise.
    /// Image-source reads are still withheld (the store may hit an in-image
    /// byte), but branch resolution and resume continue because the VM stack is
    /// intact. The vreg file is a separate namespace and is untouched either way.
    fn havoc_external_store(
        &mut self,
        instruction_index: usize,
        address: &Expr,
        width: BitWidth,
    ) -> Result<(), String> {
        self.advance_memory_epoch()?;
        // The address could not be placed, so it could have reached any byte:
        // concrete names minted before it are no longer valid either.
        self.opaque_store_epoch = self.opaque_store_epoch.wrapping_add(1);
        // Proof path: when the store address provably lies within a bounded byte
        // range over a single proven-region root (vsp_initial or a fresh
        // allocation), preserve exactly the tracked cells proven disjoint from
        // that range instead of havocing them. This is a proof, not an
        // assumption: a cell at `root + d` (one byte) aliases the store iff
        // `d` falls in `[slo, shi)`, so cells outside it (and cells in a
        // different proven-disjoint region) survive precisely. Branch-relevant
        // vsp-rooted state below/above an array-store window thus stays precise
        // and the post-store branch resolves.
        if let Some((root, slo, shi)) = proven_store_byte_range(address, width) {
            self.retain_memory(|cell_address| {
                store_range_disjoint_from_cell(&root, slo, shi, cell_address)
            });
            // The store base is a symbolic region pointer; it cannot be proven
            // disjoint from concrete in-image addresses, so image-source reads
            // stay withheld (this does not block branch resolution or resume).
            self.image_reads_suppressed = true;
            self.memory_havocs.push(SymbolicMemoryHavoc {
                instruction_index,
                cause: "external store proven within a bounded range over a single VM-region \
                        base; range-disjoint tracked cells preserved"
                    .to_owned(),
            });
            return Ok(());
        }
        // A frame pointer can only be read back from a tracked cell, so with none
        // lost or escaped an untracked read is caller data, not a frame pointer.
        let loaded_data_address = self.assume_loaded_store_addresses_disjoint
            && !is_frame_pointer_value(address)
            && !self.vm_stack_address_escapes
            && !self.vm_stack_pointer_lost;
        if address.dependencies().contains("vsp_initial") && !loaded_data_address {
            // The address may alias the VM operand stack itself; keep the broad,
            // assumption-free havoc.
            self.retain_memory(|_| false);
            self.memory_tainted = true;
            self.memory_havocs.push(SymbolicMemoryHavoc {
                instruction_index,
                // Which sub-condition refused: the entry-level retry gate can be
                // widened without effect while this one still holds, so the two
                // have to be distinguishable from the stall text alone.
                cause: format!(
                    "external store address may alias tracked VM memory                      (assume_disjoint={} is_frame_ptr={} stack_escapes={} ptr_lost={})",
                    self.assume_loaded_store_addresses_disjoint,
                    is_frame_pointer_value(address),
                    self.vm_stack_address_escapes,
                    self.vm_stack_pointer_lost,
                ) + &format!(
                    " offset={:?} unbounded_at={}",
                    pointer_offset(address).map(|offset| format!("{offset:#x}")),
                    // Which node kind `address_bounds` has no arm for. The
                    // address renders to more than 400 bytes, so the text is
                    // useless; the failing kind is the actionable part.
                    unbounded_node_kind(address).unwrap_or("<bounded>"),
                ),
            });
            return Ok(());
        }
        // Preserve the cells rooted at a proven region, discard the rest.
        //
        // This kept only `vsp_initial`-rooted cells until now, which threw away
        // every cell rooted at an allocation the frame itself made -- and those
        // discards are where the `memory_load_epoch` cascade starts: a later
        // read of a discarded cell returns an opaque value, that value lands in
        // the next store's address, the address has no provable range, and the
        // store havocs again. `proven_store_byte_range` already treats a
        // fresh-allocation return as a proven region root, so this is the same
        // root set applied to what survives a havoc rather than a new one.
        //
        // It is a SEPARATE concession from the VM-stack one and carries its own
        // label, because an opaque external store could land inside an
        // allocation this frame made exactly as it could land in the operand
        // stack, and because the VM-stack assumption is discharged for a
        // provably private frame -- which says nothing about an allocation
        // whose pointer the frame may have handed out.
        // Out-of-frame vsp-rooted cells are caller memory; an external store
        // may alias them, so they cannot be preserved on trust.
        let frame_hi = frame_scoped_retain()
            .then_some(self.limits_private_frame_hi)
            .flatten();
        let in_frame = move |address: &Expr| {
            frame_hi.is_none_or(|hi| pointer_offset(address).is_some_and(|off| off < hi))
        };
        let keep_allocations = retain_fresh_allocations();
        let retained_fresh_allocation = keep_allocations
            && self.memory.iter().any(|(address, _)| {
                pointer_offset(address).is_none() && fresh_allocation_offset(address).is_some()
            });
        self.retain_memory(|cell_address| {
            (pointer_offset(cell_address).is_some() && in_frame(cell_address))
                || (keep_allocations && fresh_allocation_offset(cell_address).is_some())
        });
        self.image_reads_suppressed = true;
        self.assumed_external_disjoint_from_vm_stack = true;
        self.assumed_external_disjoint_from_fresh_allocations |= retained_fresh_allocation;
        self.memory_havocs.push(SymbolicMemoryHavoc {
            instruction_index,
            cause: format!(
                "external store assumed disjoint from the VM operand stack; \
                 non-region-rooted cells discarded, vsp-rooted cells preserved \
                 (fresh-allocation cells preserved: {retained_fresh_allocation})"
            ),
        });
        Ok(())
    }

    fn load_external(&mut self, address: Expr, width: BitWidth) -> Result<Expr, String> {
        let address = canonicalize_pointer(address);
        if matches!(address.kind, ExprKind::Const { value: 0 }) {
            return Err(format!(
                "proven null memory read of {} byte(s) requires exception-flow analysis \
                 [zero slot {}]",
                width.bytes(),
                self.last_zero_image_read
                    .map_or_else(|| "none".to_owned(), |slot| format!("{slot:#x}")),
            ));
        }
        self.load_memory(address, width)
    }

    fn store_external(
        &mut self,
        instruction_index: usize,
        address: Expr,
        value: Expr,
    ) -> Result<(), String> {
        let address = canonicalize_pointer(address);
        if pointer_offset(&address).is_none() {
            self.note_value_leaves_frame(&value);
        }
        if is_proven_region_pointer(&address) {
            // The address provably lies in a tracked region (VM stack / proven
            // allocation / concrete). Update tracked memory precisely.
            self.store_memory(&address, &value)?;
        } else {
            // The address cannot be placed in a tracked region, so it may alias
            // any byte-addressed tracked cell. We cannot prove disjointness, so
            // over-approximate soundly rather than assume no alias: discard the
            // precise byte-addressed memory and mark it tainted. Tainting
            // disables image-source reads (so a later load cannot read a stale
            // original image byte for an address this write may have changed)
            // and refuses cross-block resumption. Where the address cannot reach
            // the VM operand stack it narrows this under a labeled assumption and
            // preserves the vsp-rooted cells. The VM-private vreg file is a
            // separate namespace the store path cannot reach, so it is kept.
            // The write itself is still recorded below, so the lift preserves it.
            self.havoc_external_store(instruction_index, &address, value.width)?;
        }
        self.external_writes.push(SymbolicMemoryWrite {
            instruction_index,
            address,
            value,
            epoch: self.memory_epoch,
        });
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn apply_native_call(
        &mut self,
        instruction_index: usize,
        outcome: &NativeCallSelectorOutcome,
    ) -> Result<(), String> {
        let target = self.load_stack(i64::from(outcome.target_stack_offset), BitWidth::W64)?;
        self.note_value_leaves_frame(&target);
        let argument_capacity = outcome
            .register_argument_offsets
            .len()
            .saturating_add(outcome.stack_argument_offsets.len());
        let mut arguments = Vec::with_capacity(argument_capacity);
        let mut descriptors = Vec::with_capacity(argument_capacity);
        for (index, offset) in outcome.register_argument_offsets.iter().enumerate() {
            let value = self.load_stack(i64::from(*offset), BitWidth::W64)?;
            self.note_value_leaves_frame(&value);
            let ordinal = u16::try_from(index)
                .map_err(|_| "native-call register argument ordinal overflows".to_owned())?;
            let name = WIN64_ARGUMENT_REGISTERS.get(index).ok_or_else(|| {
                "native-call proof exceeds four Win64 register arguments".to_owned()
            })?;
            descriptors.push(NativeCallArgumentDescriptor {
                ordinal,
                location: NativeCallArgumentLocation::Register {
                    name: (*name).to_owned(),
                },
                source: NativeCallValueSource::VmStack {
                    offset: i64::from(*offset),
                    width: BitWidth::W64,
                },
                alias: if is_fresh_allocation_root(&value) {
                    NativeCallArgumentAlias::ProvenDisjointFromTrackedMemory
                } else {
                    NativeCallArgumentAlias::MayAliasTrackedMemory
                },
                value: value.clone(),
            });
            arguments.push(value);
        }
        for (stack_index, offset) in outcome.stack_argument_offsets.iter().enumerate() {
            let value = self.load_stack(i64::from(*offset), BitWidth::W64)?;
            self.note_value_leaves_frame(&value);
            let logical_index = outcome
                .register_argument_offsets
                .len()
                .checked_add(stack_index)
                .ok_or_else(|| "native-call argument ordinal overflows".to_owned())?;
            let ordinal = u16::try_from(logical_index)
                .map_err(|_| "native-call stack argument ordinal overflows".to_owned())?;
            let stack_index = u32::try_from(stack_index)
                .map_err(|_| "native-call stack argument index overflows".to_owned())?;
            let abi_offset = stack_index
                .checked_mul(8)
                .and_then(|offset| offset.checked_add(0x20))
                .ok_or_else(|| "native-call ABI stack offset overflows".to_owned())?;
            descriptors.push(NativeCallArgumentDescriptor {
                ordinal,
                location: NativeCallArgumentLocation::Stack { offset: abi_offset },
                source: NativeCallValueSource::VmStack {
                    offset: i64::from(*offset),
                    width: BitWidth::W64,
                },
                alias: if is_fresh_allocation_root(&value) {
                    NativeCallArgumentAlias::ProvenDisjointFromTrackedMemory
                } else {
                    NativeCallArgumentAlias::MayAliasTrackedMemory
                },
                value: value.clone(),
            });
            arguments.push(value);
        }

        let invocation = NativeCallInvocationDescriptor {
            target: NativeCallTargetDescriptor {
                source: NativeCallValueSource::VmStack {
                    offset: i64::from(outcome.target_stack_offset),
                    width: BitWidth::W64,
                },
                value: target.clone(),
            },
            arguments: descriptors,
        };
        let import_resolution = if let ExprKind::Const { value } = target.kind {
            self.memory_source
                .and_then(|source| source.resolve_native_call_import(value))
        } else {
            None
        };
        let summary = match &import_resolution {
            Some(NativeCallImportResolution::Unique { import, .. }) => {
                summarize_known_import(import, &arguments)
            }
            _ => None,
        }
        .unwrap_or(NativeCallEffectSummary {
            result: NativeCallResultEffect::Opaque {
                width: BitWidth::W64,
            },
            memory: NativeCallMemoryEffect::Opaque,
        });
        let call_epoch = self
            .native_call_epochs
            .claim()
            .map_err(|error| error.to_string())?;
        let applied = apply_native_call_effect(&invocation, &summary, call_epoch)
            .map_err(|error| error.to_string())?;

        match &applied.memory {
            NativeCallMemoryDecision::HavocTracked { reason } => {
                self.protection_may_have_changed = true;
                self.advance_memory_epoch()?;
                self.opaque_store_epoch = self.opaque_store_epoch.wrapping_add(1);
                self.retain_memory(|_| false);
                self.memory_tainted = false;
                let touched_context = self.context.touched_cells.clone();
                self.context = SymbolicByteMemory::new(format!(
                    "context_after_native_call_{}",
                    call_epoch.get()
                ));
                for cell in touched_context {
                    self.context.seed_cell(cell)?;
                }
                self.memory_havocs.push(SymbolicMemoryHavoc {
                    instruction_index,
                    cause: reason.clone(),
                });
            }
            NativeCallMemoryDecision::PreserveTracked => {
                self.protection_may_have_changed = true;
            }
            NativeCallMemoryDecision::PreserveTrackedAndCreateFreshAllocation {
                allocation,
                size,
                zero_initialized,
            } => self.fresh_allocations.push(SymbolicFreshAllocation {
                allocation: allocation.clone(),
                size: size.clone(),
                zero_initialized: *zero_initialized,
            }),
        }

        self.adjust_vsp(outcome.vsp_delta)?;
        self.store_stack(0, &applied.result)?;
        self.flags = Expr::intrinsic(
            BitWidth::W64,
            format!("native_call_flags_epoch_{}", call_epoch.get()),
            Vec::new(),
        );
        self.native_calls.push(SymbolicNativeCall {
            instruction_index,
            invocation,
            import_resolution,
            applied_effect: applied,
        });
        Ok(())
    }

    fn check_limits(&mut self, limits: SymbolicExecutionLimits) -> Result<(), String> {
        let tracked_memory_bytes = self.memory.len().saturating_add(self.context.bytes.len());
        if tracked_memory_bytes > limits.max_memory_bytes {
            return Err(format!(
                "tracked memory exceeds {} bytes",
                limits.max_memory_bytes
            ));
        }
        let roots = std::iter::once(&self.vsp)
            .chain(std::iter::once(&self.flags))
            .chain(self.memory.keys())
            .chain(self.memory.values())
            .chain(self.context.bytes.values())
            .chain(
                self.external_writes
                    .iter()
                    .flat_map(|write| [&write.address, &write.value]),
            )
            .chain(
                self.fresh_allocations
                    .iter()
                    .flat_map(|allocation| [&allocation.allocation, &allocation.size]),
            );
        let live = distinct_node_count_bounded(roots, limits.max_expression_nodes);
        if live > limits.max_expression_nodes {
            return Err(format!(
                "aggregate expressions exceed {} nodes",
                limits.max_expression_nodes
            ));
        }
        self.peak_expression_nodes = self.peak_expression_nodes.max(live);
        Ok(())
    }
}

/// Total rendered bytes allowed for the address sort keys of one block.
const MAX_SORT_KEY_BYTES: usize = 64 * 1024 * 1024;

struct BoundedKey {
    text: String,
    remaining: usize,
}

impl std::fmt::Write for BoundedKey {
    fn write_str(&mut self, chunk: &str) -> std::fmt::Result {
        self.remaining = self.remaining.checked_sub(chunk.len()).ok_or(std::fmt::Error)?;
        self.text.push_str(chunk);
        Ok(())
    }
}

/// Up to `limit` bytes of `expression`'s `Debug` text, stopping there.
///
/// `format!("{e:?}").chars().take(n)` renders the WHOLE tree first and then
/// throws nearly all of it away. `Debug` prints a hash-consed DAG as a tree, so
/// for a shared expression that is megabytes of text for an n-character
/// snippet. `BoundedKey` already keeps what it wrote before going over budget.
pub(crate) fn debug_prefix(expression: &Expr, limit: usize) -> String {
    use std::fmt::Write;
    let mut key = BoundedKey { text: String::new(), remaining: limit };
    let _ = write!(key, "{expression:?}");
    key.text
}

/// `Debug` text of `expression`, or `None` when it renders past `limit` bytes.
pub(crate) fn bounded_debug(expression: &Expr, limit: usize) -> Option<String> {
    use std::fmt::Write;
    let mut key = BoundedKey { text: String::new(), remaining: limit };
    write!(key, "{expression:?}").ok()?;
    Some(key.text)
}

/// Orders tracked memory by the rendered address. `Debug` prints a shared DAG
/// as a tree, so a heavily shared address can render exponentially large; the
/// rendering is bounded and the set falls back to structural-hash order rather
/// than exhausting RAM.
///
/// The rendered order (or the over-budget failure) depends only on the address
/// set, so it is memoized on the addresses in a canonical hash order: successive
/// blocks mostly track the same addresses and would otherwise re-render and
/// re-sort hundreds of KB of text per address each time.
fn sorted_memory_values(
    memory: HashMap<Expr, Expr>,
) -> Result<Vec<(Expr, Expr)>, String> {
    use std::fmt::Write;
    use std::hash::BuildHasher;
    type Order = Option<Vec<usize>>;
    thread_local! {
        static ORDERS: std::cell::RefCell<Vec<(Vec<Expr>, Order)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    const CACHED_ORDERS: usize = 64;
    let hasher = crate::expr::FxBuild::default();
    let mut items: Vec<(u64, Expr, Expr)> = memory
        .into_iter()
        .map(|(address, value)| (hasher.hash_one(&address), address, value))
        .collect();
    items.sort_by_key(|item| item.0);
    let cached = ORDERS.with(|cache| {
        cache
            .borrow()
            .iter()
            .find(|(addresses, _)| {
                addresses.len() == items.len()
                    && addresses.iter().zip(&items).all(|(cached, item)| *cached == item.1)
            })
            .map(|(_, order)| order.clone())
    });
    let order = if let Some(order) = cached {
        order
    } else {
        let mut remaining = MAX_SORT_KEY_BYTES;
        let mut keyed = Vec::with_capacity(items.len());
        let mut failed = false;
        for (index, (_, address, _)) in items.iter().enumerate() {
            let mut key = BoundedKey {
                text: String::new(),
                remaining,
            };
            if write!(key, "{address:?}").is_err() {
                failed = true;
                break;
            }
            remaining = key.remaining;
            keyed.push((key.text, index));
        }
        let order = if failed {
            None
        } else {
            keyed.sort_by(|left, right| left.0.cmp(&right.0));
            Some(keyed.into_iter().map(|(_, index)| index).collect())
        };
        ORDERS.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() >= CACHED_ORDERS {
                cache.remove(0);
            }
            cache.push((items.iter().map(|item| item.1.clone()).collect(), order.clone()));
        });
        order
    };
    // Over-budget address text falls back to the structural-hash order the items
    // are already in: the order is canonical either way and never semantic.
    let order = order.unwrap_or_else(|| (0..items.len()).collect());
    Ok(order
        .into_iter()
        .map(|index| (items[index].1.clone(), items[index].2.clone()))
        .collect())
}

fn combine_little_endian(bytes: &[Expr]) -> Result<Expr, ExprError> {
    if let [byte] = bytes {
        Ok(byte.clone())
    } else {
        let midpoint = bytes.len() / 2;
        let low = combine_little_endian(&bytes[..midpoint])?;
        let high = combine_little_endian(&bytes[midpoint..])?;
        Expr::concat(high, low)
    }
}

fn add_pointer_offset(pointer: Expr, offset: i64) -> Result<Expr, ExprError> {
    Expr::binary(
        BinaryOp::Add,
        pointer,
        Expr::constant(BitWidth::W64, u64::from_ne_bytes(offset.to_ne_bytes())),
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PointerAffine {
    root: Option<Expr>,
    coefficient: i8,
    constant: u64,
}

fn is_fresh_allocation_root(expression: &Expr) -> bool {
    matches!(
        &expression.kind,
        ExprKind::Intrinsic { name, .. }
            if name.starts_with("fresh_allocation_return_epoch_")
    )
}

fn normalize_pointer_affine(mut affine: PointerAffine) -> PointerAffine {
    if affine.coefficient == 0 {
        affine.root = None;
    }
    affine
}

enum PointerRootMerge {
    Compatible(Option<Expr>),
    Incompatible,
}

fn combine_pointer_roots(left: &PointerAffine, right: &PointerAffine) -> PointerRootMerge {
    match (&left.root, &right.root) {
        (None, None) => PointerRootMerge::Compatible(None),
        (Some(root), None) | (None, Some(root)) => PointerRootMerge::Compatible(Some(root.clone())),
        (Some(left), Some(right)) if left == right => {
            PointerRootMerge::Compatible(Some(left.clone()))
        }
        (Some(_), Some(_)) => PointerRootMerge::Incompatible,
    }
}

fn pointer_affine(expression: &Expr) -> Option<PointerAffine> {
    match &expression.kind {
        ExprKind::Register { name } if name == "vsp_initial" => Some(PointerAffine {
            root: Some(expression.clone()),
            coefficient: 1,
            constant: 0,
        }),
        ExprKind::Intrinsic { .. } if is_fresh_allocation_root(expression) => Some(PointerAffine {
            root: Some(expression.clone()),
            coefficient: 1,
            constant: 0,
        }),
        ExprKind::Const { value } => Some(PointerAffine {
            root: None,
            coefficient: 0,
            constant: *value,
        }),
        ExprKind::Unary { op, value } => {
            let affine = pointer_affine(value)?;
            match op {
                UnaryOp::Not => Some(normalize_pointer_affine(PointerAffine {
                    root: affine.root,
                    coefficient: affine.coefficient.checked_neg()?,
                    constant: !affine.constant,
                })),
                UnaryOp::Neg => Some(normalize_pointer_affine(PointerAffine {
                    root: affine.root,
                    coefficient: affine.coefficient.checked_neg()?,
                    constant: affine.constant.wrapping_neg(),
                })),
            }
        }
        ExprKind::Binary {
            op: BinaryOp::Add | BinaryOp::Sub,
            left,
            right,
        } => {
            let left = pointer_affine(left)?;
            let right = pointer_affine(right)?;
            let PointerRootMerge::Compatible(root) = combine_pointer_roots(&left, &right) else {
                return None;
            };
            let (coefficient, constant) = if matches!(
                &expression.kind,
                ExprKind::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ) {
                (
                    left.coefficient.checked_add(right.coefficient)?,
                    left.constant.wrapping_add(right.constant),
                )
            } else {
                (
                    left.coefficient.checked_sub(right.coefficient)?,
                    left.constant.wrapping_sub(right.constant),
                )
            };
            (-1..=1).contains(&coefficient).then(|| {
                normalize_pointer_affine(PointerAffine {
                    root,
                    coefficient,
                    constant,
                })
            })
        }
        ExprKind::Binary {
            op: BinaryOp::And | BinaryOp::Or,
            left,
            right,
        } if left == right => pointer_affine(left),
        _ => None,
    }
}

fn canonicalize_pointer(expression: Expr) -> Expr {
    match pointer_affine(&expression) {
        Some(PointerAffine {
            root: None,
            coefficient: 0,
            constant,
        }) => Expr::constant(BitWidth::W64, constant),
        Some(PointerAffine {
            root: Some(root),
            coefficient: 1,
            constant,
        }) => Expr::binary(BinaryOp::Add, root, Expr::constant(BitWidth::W64, constant))
            .expect("same-width affine pointer expression"),
        _ => expression,
    }
}

/// Canonicalizes a proven affine symbolic pointer into `root + constant` form.
/// Non-affine expressions are returned unchanged.
#[must_use]
pub fn canonicalize_symbolic_pointer(expression: &Expr) -> Expr {
    canonicalize_pointer(expression.clone())
}

fn depends_on_vsp_root(expression: &Expr) -> bool {
    expression.dependencies().contains("vsp_initial")
}

/// True unless `vsp_initial` occurs only as (part of) the address of an untracked
/// memory read, i.e. the value is data read from the frame, not a frame pointer.
pub(crate) fn is_frame_pointer_value(expression: &Expr) -> bool {
    expression.register_outside_intrinsic_args("vsp_initial", |name| {
        name.starts_with("memory_load_epoch_")
    })
}

/// Whether the narrowed havoc keeps allocation-rooted cells.
///
/// On by default; `VMPX_NO_FRESH_ALLOC_RETAIN=1` restores the vsp-only
/// retention. The switch exists so the change can be measured against itself in
/// one build rather than across two, which is how a change that looked like a
/// gain on one metric and a loss on another got caught before.
/// Whether a read-only image byte is still served after a BROAD havoc.
///
/// A store to a non-writable page faults, so only a protection change could
/// have altered it -- which is exactly what `AssumesImageProtectionStable`
/// already concedes for the suppressed case. Measured: this is the sole
/// remaining blocker on 5 store-class entries, whose broad havoc sets
/// `memory_tainted` and so stops their constant image reads being served.
/// `VMPX_NO_SERVE_RO_TAINTED=1` restores the stricter gate.
fn serve_read_only_when_tainted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VMPX_NO_SERVE_RO_TAINTED").is_none())
}

/// Whether the narrowed havoc keeps only cells inside the private frame.
///
/// OPT-IN (`VMPX_FRAME_SCOPED_RETAIN=1`), because on its own it measured N1
/// 733 -> 732 and N2 unchanged: discarding out-of-frame cells costs precision
/// and buys nothing until the discharge stops using `past_frame` as its proxy
/// for "every cell lies inside the frame". It exists because it is the
/// prerequisite for that change, which is worth 119 entries on N2.
fn frame_scoped_retain() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VMPX_FRAME_SCOPED_RETAIN").is_ok_and(|v| v == "1"))
}

fn retain_fresh_allocations() -> bool {
    static CHOICE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CHOICE.get_or_init(|| std::env::var_os("VMPX_NO_FRESH_ALLOC_RETAIN").is_none())
}

/// A cell address's byte offset from a fresh-allocation root, if it has one.
///
/// The allocation-rooted counterpart of [`pointer_offset`], which only accepts
/// `vsp_initial`. Both roots are what [`proven_store_byte_range`] already calls
/// a proven region root, via `is_proven_region_root`.
fn fresh_allocation_offset(expression: &Expr) -> Option<i64> {
    let PointerAffine {
        root: Some(root),
        coefficient: 1,
        constant,
    } = pointer_affine(expression)?
    else {
        return None;
    };
    is_fresh_allocation_root(&root).then(|| i64::from_ne_bytes(constant.to_ne_bytes()))
}

fn pointer_offset(expression: &Expr) -> Option<i64> {
    let PointerAffine {
        root: Some(root),
        coefficient: 1,
        constant,
    } = pointer_affine(expression)?
    else {
        return None;
    };
    matches!(&root.kind, ExprKind::Register { name } if name == "vsp_initial")
        .then(|| i64::from_ne_bytes(constant.to_ne_bytes()))
}

/// Returns the signed constant offset of an affine pointer rooted at one named
/// symbolic register. Expressions with another root, multiple roots, or a
/// non-unit coefficient are intentionally rejected.
#[must_use]
pub fn symbolic_pointer_offset(expression: &Expr, root_name: &str) -> Option<i64> {
    let PointerAffine {
        root: Some(root),
        coefficient: 1,
        constant,
    } = pointer_affine(expression)?
    else {
        return None;
    };
    matches!(&root.kind, ExprKind::Register { name } if name == root_name)
        .then(|| i64::from_ne_bytes(constant.to_ne_bytes()))
}

fn is_proven_region_pointer(expression: &Expr) -> bool {
    match pointer_affine(expression) {
        Some(PointerAffine {
            root: None,
            coefficient: 0,
            ..
        }) => true,
        Some(PointerAffine {
            root: Some(root),
            coefficient: 1,
            ..
        }) => {
            matches!(&root.kind, ExprKind::Register { name } if name == "vsp_initial")
                || is_fresh_allocation_root(&root)
        }
        _ => false,
    }
}

fn is_proven_region_root(root: &Expr) -> bool {
    matches!(&root.kind, ExprKind::Register { name } if name == "vsp_initial")
        || is_fresh_allocation_root(root)
}

/// A proven bound on a pointer/index expression's 64-bit value: it equals
/// `root_value + k` for some signed `k` in `[lo, hi]` (when `root` is `Some`),
/// or the value itself lies in `[lo, hi]` (when `root` is `None`). Intervals are
/// over-approximations; any form that cannot be bounded returns `None` so the
/// caller fails closed to the conservative havoc.
#[derive(Clone)]
struct AddressBounds {
    root: Option<Expr>,
    lo: i64,
    hi: i64,
}

/// Unsigned value range `[0, 2^bits - 1]` of a sub-expression of the given
/// width. Returns `None` for 64-bit (unbounded for our signed-offset model).
fn unsigned_width_range(width: BitWidth) -> Option<(i64, i64)> {
    let bits = width.bits();
    if bits >= 64 {
        return None;
    }
    i64::try_from((1u128 << bits) - 1).ok().map(|hi| (0, hi))
}

#[allow(clippy::too_many_lines)]
/// The kind of the first node `address_bounds` has no arm for, or `None` when
/// the whole address bounds.
///
/// `address_bounds` is structural: one unsupported node anywhere makes the whole
/// address unbounded, and the store then havocs memory. These addresses render
/// past any sane budget, so the failing KIND is the only part worth printing --
/// it names the arm to add.
fn unbounded_node_kind(expression: &Expr) -> Option<&'static str> {
    if address_bounds(expression).is_some() {
        return None;
    }
    let children: Vec<&Expr> = match &expression.kind {
        ExprKind::Binary { left, right, .. } => vec![left, right],
        ExprKind::Unary { value, .. } | ExprKind::Resize { value, .. } => vec![value],
        ExprKind::Concat { high, low } => vec![high, low],
        ExprKind::Extract { value, .. } => vec![value],
        ExprKind::Load { address } => vec![address],
        _ => Vec::new(),
    };
    // Deepest unsupported node first: a parent only fails because a child did.
    for child in children {
        if let Some(kind) = unbounded_node_kind(child) {
            return Some(kind);
        }
    }
    Some(match &expression.kind {
        ExprKind::Binary { op, .. } => match op {
            BinaryOp::Add => "Add",
            BinaryOp::Sub => "Sub",
            BinaryOp::Mul => "Mul",
            BinaryOp::Shl => "Shl",
            BinaryOp::And => "And",
            BinaryOp::Or => "Or",
            BinaryOp::Xor => "Xor",
            // The OP, not just "other". Naming the intrinsic is what turned
            // this diagnostic actionable once already; a bare "Binary(other)"
            // hides which arm `address_bounds` is missing.
            other => {
                return Some(Box::leak(format!("Binary:{other:?}").into_boxed_str()));
            }
        },
        ExprKind::Unary { .. } => "Unary",
        ExprKind::Resize { op, .. } => match op {
            ResizeOp::SignExtend => "SignExtend",
            ResizeOp::ZeroExtend => "ZeroExtend",
            ResizeOp::Truncate => "Truncate",
        },
        ExprKind::Load { .. } => "Load",
        // The NAME, not just the kind: closing this class means proving
        // `f(vsp + k) = vsp + g(k)` for the specific functions VMProtect emits,
        // so which ones they are is the whole scope of that work.
        // For an opaque read, also classify WHERE it read from: that separates
        // a cell a havoc discarded from one never written.
        ExprKind::Intrinsic { name, arguments } if name.starts_with("memory_load_epoch_") => {
            let site = arguments.first().map_or("none", |address| {
                if matches!(address.kind, ExprKind::Const { .. }) {
                    "image-const"
                } else if pointer_offset(address).is_some() {
                    "vsp-rooted"
                } else if address.dependencies().contains("vsp_initial") {
                    "vsp-derived"
                } else if unbounded_node_kind(address)
                    .is_some_and(|kind| kind.starts_with("Intrinsic:memory_load_epoch_"))
                {
                    "opaque-chain"
                } else {
                    "other"
                }
            });
            return Some(Box::leak(
                format!("Intrinsic:memory_load_epoch@{site}").into_boxed_str(),
            ));
        }
        ExprKind::Intrinsic { name, .. } => return Some(Box::leak(
            format!("Intrinsic:{name}").into_boxed_str(),
        )),
        ExprKind::Extract { .. } => "Extract",
        ExprKind::Concat { .. } => "Concat",
        ExprKind::Register { .. } => "Register",
        ExprKind::Const { .. } => "Const",
        _ => "other",
    })
}

fn address_bounds(expression: &Expr) -> Option<AddressBounds> {
    match &expression.kind {
        ExprKind::Register { name } if name == "vsp_initial" => Some(AddressBounds {
            root: Some(expression.clone()),
            lo: 0,
            hi: 0,
        }),
        ExprKind::Intrinsic { .. } if is_fresh_allocation_root(expression) => Some(AddressBounds {
            root: Some(expression.clone()),
            lo: 0,
            hi: 0,
        }),
        ExprKind::Const { value } => {
            let signed = i64::from_ne_bytes(value.to_ne_bytes());
            Some(AddressBounds {
                root: None,
                lo: signed,
                hi: signed,
            })
        }
        // A bare symbolic register below 64 bits, or any truncation, is an
        // unconstrained value of its own width, i.e. bounded by that unsigned
        // range.
        ExprKind::Register { .. }
        | ExprKind::Resize {
            op: ResizeOp::Truncate,
            ..
        } => {
            let (lo, hi) = unsigned_width_range(expression.width)?;
            Some(AddressBounds {
                root: None,
                lo,
                hi,
            })
        }
        ExprKind::Resize {
            op: ResizeOp::ZeroExtend,
            value,
        } => {
            // A zero-extended value stays in the unsigned range of its SOURCE
            // width; intersect with any tighter bound the operand proves.
            let (wlo, whi) = unsigned_width_range(value.width)?;
            let inner = address_bounds(value);
            let (lo, hi) = match inner {
                Some(AddressBounds {
                    root: None,
                    lo,
                    hi,
                }) => (lo.max(wlo), hi.min(whi)),
                _ => (wlo, whi),
            };
            (lo <= hi).then_some(AddressBounds {
                root: None,
                lo,
                hi,
            })
        }
        // `concat(0, low)` is a zero-extension of `low`.
        ExprKind::Concat { high, low } if matches!(high.kind, ExprKind::Const { value: 0 }) => {
            address_bounds(low)
        }
        // `concat(high, low)` is `(high << width(low)) + low`: the halves do not
        // overlap. A rootless bounded high half therefore contributes a bounded
        // displacement and the low half's root survives, which is what
        // `proven_store_byte_range` needs. A wider range is still sound here --
        // it preserves fewer cells, never more.
        ExprKind::Concat { high, low } => {
            let width = u32::from(low.width.bits());
            if width >= 64 {
                return None;
            }
            let high = address_bounds(high)?;
            if high.root.is_some() {
                return None;
            }
            let low = address_bounds(low)?;
            // `checked_shl` validates only the shift amount, not value
            // overflow, so a wide high half would wrap into a WRONG bound --
            // and a wrong bound preserves cells the store can reach.
            let scale = 1i64.checked_shl(width)?;
            let shift = |value: i64| value.checked_mul(scale);
            Some(AddressBounds {
                root: low.root,
                lo: shift(high.lo)?.checked_add(low.lo)?,
                hi: shift(high.hi)?.checked_add(low.hi)?,
            })
        }
        // `x >> c` on a rootless non-negative range. A rooted operand is
        // refused: shifting a root is not `root + constant`.
        ExprKind::Binary {
            op: BinaryOp::Shr,
            left,
            right,
        } => {
            let ExprKind::Const { value: shift } = right.kind else {
                return None;
            };
            let shift = u32::try_from(shift).ok()?;
            if shift >= 64 {
                return None;
            }
            let base = address_bounds(left)?;
            if base.root.is_some() || base.lo < 0 {
                return None;
            }
            Some(AddressBounds {
                root: None,
                lo: base.lo >> shift,
                hi: base.hi >> shift,
            })
        }
        ExprKind::Binary {
            op: op @ (BinaryOp::Add | BinaryOp::Sub),
            left,
            right,
        } => {
            let left = address_bounds(left)?;
            let right = address_bounds(right)?;
            if matches!(op, BinaryOp::Add) {
                // At most one side may carry a root (two roots would be a
                // non-unit coefficient we cannot represent as an offset).
                let ((None, root) | (root, None)) = (left.root, right.root) else {
                    return None;
                };
                Some(AddressBounds {
                    root,
                    lo: left.lo.checked_add(right.lo)?,
                    hi: left.hi.checked_add(right.hi)?,
                })
            } else {
                // Subtracting a rooted term is not a constant offset.
                if right.root.is_some() {
                    return None;
                }
                Some(AddressBounds {
                    root: left.root,
                    lo: left.lo.checked_sub(right.hi)?,
                    hi: left.hi.checked_sub(right.lo)?,
                })
            }
        }
        // Scaling an index (`index << c` / `index * c`) keeps a rootless range.
        ExprKind::Binary {
            op: BinaryOp::Shl,
            left,
            right,
        } => {
            let ExprKind::Const { value: shift } = right.kind else {
                return None;
            };
            let base = address_bounds(left)?;
            if base.root.is_some() || shift >= 63 {
                return None;
            }
            let scale = 1i64.checked_shl(u32::try_from(shift).ok()?)?;
            scale_rootless(base.lo, base.hi, scale)
        }
        ExprKind::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            let (base, scalar) = match (&left.kind, &right.kind) {
                (ExprKind::Const { value }, _) => (address_bounds(right)?, *value),
                (_, ExprKind::Const { value }) => (address_bounds(left)?, *value),
                _ => return None,
            };
            if base.root.is_some() {
                return None;
            }
            scale_rootless(base.lo, base.hi, i64::try_from(scalar).ok()?)
        }
        // `index & mask` with a small positive mask bounds the value to [0, mask].
        ExprKind::Binary {
            op: BinaryOp::And,
            left,
            right,
        } => {
            let mask = match (&left.kind, &right.kind) {
                (ExprKind::Const { value }, _) | (_, ExprKind::Const { value }) => *value,
                _ => return None,
            };
            let hi = i64::try_from(mask).ok()?;
            Some(AddressBounds {
                root: None,
                lo: 0,
                hi,
            })
        }
        // No arm, deliberately, and NOT because the missing arm is arithmetic.
        //
        // Measured on the affected entries, the node that stops the bound is
        // `memory_load_epoch_N` in 31 of 45 occurrences -- the opaque value a
        // read of UNTRACKED memory returns. So the store address is not
        // `f(vsp + k)` for some VMProtect function f whose semantics could be
        // modelled; it depends on a value the recovery already lost. There is
        // no g to prove.
        //
        // That makes this class and the "branch target depends on unproven
        // input" class one cause, not two: 96 of 98 of those refuse on
        // `OtherLeaf(memory_load_epoch)` as well. And it is circular -- a store
        // with an unbounded address havocs memory, later reads of it come back
        // opaque, those reads feed the next store's address, which havocs
        // again. Closing either class means not starting that cascade, which is
        // memory precision, not intrinsic semantics.
        //
        // Bounding an unsupported node by its own width
        // was tried and reverted: it is sound as a VALUE bound but it discards
        // the ROOT, and `proven_store_byte_range` needs one, so it cannot reach
        // the proof path and measured +0.
        //
        // It could not have worked. Measured on systems.xpl, the node that stops
        // the bound is an `Intrinsic` in 31 of 45 occurrences (then
        // Binary(other) 7, Concat 6, Register 1), and in these addresses
        // `vsp_initial` sits INSIDE that intrinsic. The address is therefore
        // `f(frame_pointer)` for an opaque `f`, which has no provable byte range
        // relative to the VM stack root at all. The havoc those stores take is
        // not conservative, it is the only sound answer without knowing `f`.
        _ => None,
    }
}

fn scale_rootless(lo: i64, hi: i64, scale: i64) -> Option<AddressBounds> {
    if scale < 0 {
        return None;
    }
    Some(AddressBounds {
        root: None,
        lo: lo.checked_mul(scale)?,
        hi: hi.checked_mul(scale)?,
    })
}

/// Proves the half-open byte range `[slo, shi)` (offsets relative to a single
/// proven-region `root`) that a `width`-wide store provably stays within.
/// Returns `None` unless the address reduces to `root + [lo, hi]` over a
/// `vsp_initial` / fresh-allocation root with representable signed bounds.
fn proven_store_byte_range(address: &Expr, width: BitWidth) -> Option<(Expr, i64, i64)> {
    let bounds = address_bounds(address)?;
    let root = bounds.root?;
    if !is_proven_region_root(&root) {
        return None;
    }
    let slo = bounds.lo;
    let shi = bounds.hi.checked_add(i64::from(width.bytes()))?;
    (slo < shi).then_some((root, slo, shi))
}

/// Whether a one-byte tracked cell is proven disjoint from a store occupying
/// `[slo, shi)` over `store_root`. A same-root cell is disjoint iff its offset
/// is outside the range; a cell in a different proven-region (another named
/// base / distinct fresh allocation) is disjoint by region; everything else
/// (concrete cells, unprovable roots) is treated as possibly aliasing and
/// discarded.
fn store_range_disjoint_from_cell(
    store_root: &Expr,
    slo: i64,
    shi: i64,
    cell_address: &Expr,
) -> bool {
    let Some(PointerAffine {
        root: Some(cell_root),
        coefficient: 1,
        constant,
    }) = pointer_affine(cell_address)
    else {
        return false;
    };
    if !is_proven_region_root(&cell_root) {
        return false;
    }
    if &cell_root == store_root {
        let offset = i64::from_ne_bytes(constant.to_ne_bytes());
        offset < slo || offset >= shi
    } else {
        // Two distinct proven regions (vsp stack vs a fresh allocation, or two
        // distinct fresh allocations) never alias.
        true
    }
}

/// Splits a double-width or wide-operand result into two same-width halves;
/// 64-bit operations use two independent intrinsics because no 128-bit width exists.
fn wide_pair(
    width: BitWidth,
    name: &str,
    arguments: &[Expr],
    labels: (&str, &str),
) -> Result<(Expr, Expr), String> {
    if width == BitWidth::W64 {
        return Ok((
            Expr::intrinsic(width, format!("{name}_{}", labels.0), arguments.to_vec()),
            Expr::intrinsic(width, format!("{name}_{}", labels.1), arguments.to_vec()),
        ));
    }
    let packed = Expr::intrinsic(BitWidth::W64, name.to_owned(), arguments.to_vec());
    let first = Expr::extract(width, 0, packed.clone()).map_err(|error| error.to_string())?;
    let second = Expr::extract(width, width.bits(), packed).map_err(|error| error.to_string())?;
    Ok((first, second))
}

fn resized_to(width: BitWidth, value: Expr) -> Result<Expr, ExprError> {
    if value.width == width {
        Ok(value)
    } else if value.width.bits() < width.bits() {
        Expr::resize(ResizeOp::ZeroExtend, width, value)
    } else {
        Expr::resize(ResizeOp::Truncate, width, value)
    }
}

fn operation_width(instruction: &DecodedVmInstruction) -> Result<BitWidth, String> {
    instruction
        .operation_width
        .ok_or_else(|| "semantic operation width is unknown".to_owned())
}

fn expected_delta(instruction: &DecodedVmInstruction, expected: i32) -> Result<(), String> {
    if instruction.vsp_delta == Some(expected) {
        Ok(())
    } else {
        Err(format!(
            "recovered VSP delta {:?} does not match semantic delta {expected}",
            instruction.vsp_delta
        ))
    }
}

fn arithmetic_result(semantic: VmHandlerSemantic, left: Expr, right: Expr) -> Result<Expr, String> {
    match semantic {
        VmHandlerSemantic::Add => Expr::binary(BinaryOp::Add, left, right),
        VmHandlerSemantic::Sub => Expr::binary(BinaryOp::Sub, left, right),
        VmHandlerSemantic::Nor => Expr::binary(
            BinaryOp::And,
            Expr::unary(UnaryOp::Not, left),
            Expr::unary(UnaryOp::Not, right),
        ),
        VmHandlerSemantic::Nand => Expr::binary(
            BinaryOp::Or,
            Expr::unary(UnaryOp::Not, left),
            Expr::unary(UnaryOp::Not, right),
        ),
        _ => return Err(format!("unsupported arithmetic semantic {semantic:?}")),
    }
    .map_err(|error| error.to_string())
}

fn expression_constant(expression: &Expr) -> Option<u64> {
    match &expression.kind {
        ExprKind::Const { value } => Some(*value),
        _ => None,
    }
}

fn checksum_finalize(expression: Expr, metadata: &TableChecksumMetadata) -> Result<Expr, String> {
    let expression = if metadata.final_not {
        Expr::unary(UnaryOp::Not, expression)
    } else {
        expression
    };
    Expr::binary(
        BinaryOp::Xor,
        expression,
        Expr::constant(BitWidth::W32, u64::from(metadata.final_xor)),
    )
    .map_err(|error| error.to_string())
}

fn checksum_recurrence_intrinsic(
    machine: &SymbolicMachine<'_>,
    metadata: &TableChecksumMetadata,
    accumulator: Expr,
    pointer: Expr,
    length: Expr,
) -> Expr {
    let table = Expr::constant(BitWidth::W64, metadata.table_address.get());
    let memory_snapshot = Expr::intrinsic(
        BitWidth::W64,
        format!("memory_snapshot_epoch_{}", machine.memory_epoch),
        vec![pointer.clone(), length.clone(), table.clone()],
    );
    Expr::intrinsic(
        BitWidth::W32,
        "table_checksum_low_byte_xor_recurrence_w32",
        vec![
            accumulator,
            pointer,
            length,
            table,
            Expr::constant(BitWidth::W8, u64::from(metadata.shift_bits)),
            Expr::constant(
                BitWidth::W8,
                u64::from(matches!(metadata.shift, crate::ChecksumShift::LogicalRight)),
            ),
            Expr::constant(
                BitWidth::W8,
                u64::from(metadata.pointer_stride.cast_unsigned()),
            ),
            Expr::constant(BitWidth::W32, u64::from(metadata.iteration_xor)),
            memory_snapshot,
        ],
    )
}

fn checksum_pointer_at(pointer: u64, stride: i8, index: usize) -> Result<Va, String> {
    let index = i64::try_from(index)
        .map_err(|_| "table checksum input index exceeds signed address range".to_owned())?;
    let offset = index
        .checked_mul(i64::from(stride))
        .ok_or_else(|| "table checksum pointer stride overflows".to_owned())?;
    Va(pointer)
        .checked_add_signed(offset)
        .ok_or_else(|| "table checksum input address overflows".to_owned())
}

fn shifted_checksum(accumulator: u32, metadata: &TableChecksumMetadata) -> u32 {
    match metadata.shift {
        crate::ChecksumShift::LogicalLeft => {
            accumulator.wrapping_shl(u32::from(metadata.shift_bits))
        }
        crate::ChecksumShift::LogicalRight => {
            accumulator.wrapping_shr(u32::from(metadata.shift_bits))
        }
    }
}

#[allow(clippy::too_many_lines)]
fn execute_table_checksum(
    machine: &mut SymbolicMachine<'_>,
    instruction: &DecodedVmInstruction,
    width: BitWidth,
    metadata: &TableChecksumMetadata,
) -> Result<(), String> {
    metadata.validate().map_err(|error| error.to_string())?;
    if width != metadata.checksum_width {
        return Err(format!(
            "decoded checksum width {width:?} differs from metadata width {:?}",
            metadata.checksum_width
        ));
    }
    if instruction.vsp_delta != Some(metadata.vsp_delta) {
        return Err(format!(
            "recovered VSP delta {:?} differs from checksum metadata delta {}",
            instruction.vsp_delta, metadata.vsp_delta
        ));
    }

    let accumulator = match metadata.accumulator_source {
        crate::ChecksumAccumulatorSource::Constant { value } => {
            Expr::constant(BitWidth::W32, u64::from(value))
        }
        crate::ChecksumAccumulatorSource::VmStack { offset } => {
            machine.load_stack(i64::from(offset), BitWidth::W32)?
        }
    };
    let accumulator = Expr::binary(
        BinaryOp::Xor,
        accumulator,
        Expr::constant(BitWidth::W32, u64::from(metadata.initial_xor)),
    )
    .map_err(|error| error.to_string())?;
    let pointer = machine.load_stack(i64::from(metadata.pointer_stack_offset), BitWidth::W64)?;
    let length = machine.load_stack(
        i64::from(metadata.length_stack_offset),
        metadata.length_width,
    )?;

    let result = if expression_constant(&length) == Some(0) {
        checksum_finalize(accumulator, metadata)?
    } else if let (Some(mut checksum), Some(pointer_value), Some(length_value)) = (
        expression_constant(&accumulator).and_then(|value| u32::try_from(value).ok()),
        expression_constant(&pointer),
        expression_constant(&length),
    ) {
        let length_usize = usize::try_from(length_value)
            .map_err(|_| "table checksum length cannot be represented".to_owned())?;
        if length_usize > machine.checksum_byte_limit {
            return Err(format!(
                "table checksum length {length_usize} exceeds byte limit {}",
                machine.checksum_byte_limit
            ));
        }
        let mut symbolic_suffix = None;
        for index in 0..length_usize {
            let input_address = checksum_pointer_at(pointer_value, metadata.pointer_stride, index)?;
            let input = machine.load_external(
                Expr::constant(BitWidth::W64, input_address.get()),
                BitWidth::W8,
            )?;
            let Some(input) =
                expression_constant(&input).and_then(|value| u8::try_from(value).ok())
            else {
                let remaining = length_usize
                    .checked_sub(index)
                    .ok_or_else(|| "table checksum remaining length underflows".to_owned())?;
                let remaining = u64::try_from(remaining).map_err(|_| {
                    "table checksum remaining length cannot be represented".to_owned()
                })?;
                symbolic_suffix = Some(checksum_recurrence_intrinsic(
                    machine,
                    metadata,
                    Expr::constant(BitWidth::W32, u64::from(checksum)),
                    Expr::constant(BitWidth::W64, input_address.get()),
                    Expr::constant(BitWidth::W64, remaining),
                ));
                break;
            };
            let table_index = u64::from(checksum.to_le_bytes()[0] ^ input);
            let table_offset = table_index
                .checked_mul(u64::from(metadata.table_entry_width.bytes()))
                .ok_or_else(|| "table checksum table offset overflows".to_owned())?;
            let table_address = metadata
                .table_address
                .checked_add(table_offset)
                .ok_or_else(|| "table checksum table address overflows".to_owned())?;
            let entry = machine.load_external(
                Expr::constant(BitWidth::W64, table_address.get()),
                metadata.table_entry_width,
            )?;
            let Some(entry) =
                expression_constant(&entry).and_then(|value| u32::try_from(value).ok())
            else {
                let remaining = length_usize
                    .checked_sub(index)
                    .ok_or_else(|| "table checksum remaining length underflows".to_owned())?;
                let remaining = u64::try_from(remaining).map_err(|_| {
                    "table checksum remaining length cannot be represented".to_owned()
                })?;
                symbolic_suffix = Some(checksum_recurrence_intrinsic(
                    machine,
                    metadata,
                    Expr::constant(BitWidth::W32, u64::from(checksum)),
                    Expr::constant(BitWidth::W64, input_address.get()),
                    Expr::constant(BitWidth::W64, remaining),
                ));
                break;
            };
            checksum = shifted_checksum(checksum, metadata) ^ entry ^ metadata.iteration_xor;
        }
        checksum_finalize(
            symbolic_suffix.unwrap_or_else(|| Expr::constant(BitWidth::W32, u64::from(checksum))),
            metadata,
        )?
    } else {
        checksum_finalize(
            checksum_recurrence_intrinsic(machine, metadata, accumulator, pointer, length),
            metadata,
        )?
    };

    machine.adjust_vsp(metadata.vsp_delta)?;
    machine.store_stack(i64::from(metadata.result_stack_offset), &result)
}

#[allow(clippy::too_many_lines)]
fn execute_instruction(
    machine: &mut SymbolicMachine<'_>,
    instruction_index: usize,
    instruction: &DecodedVmInstruction,
) -> Result<(), String> {
    if instruction.semantic == VmHandlerSemantic::Dispatch {
        expected_delta(instruction, 0)?;
        return Ok(());
    }
    let width = operation_width(instruction)?;
    match instruction.semantic {
        VmHandlerSemantic::ConstantPush => {
            let operand = instruction
                .operand
                .ok_or_else(|| "constant push has no decoded operand".to_owned())?;
            let input_width = instruction.input_width.unwrap_or(width);
            let value = Expr::constant(input_width, operand);
            let value = match instruction.resize {
                Some(operation) => {
                    Expr::resize(operation, width, value).map_err(|error| error.to_string())?
                }
                None if input_width == width => value,
                None => {
                    return Err(
                        "constant-push input/output widths differ without a proven resize"
                            .to_owned(),
                    );
                }
            };
            let delta = i32::try_from(width.bytes())
                .ok()
                .and_then(i32::checked_neg)
                .ok_or_else(|| "constant-push width exceeds VSP range".to_owned())?;
            expected_delta(instruction, delta)?;
            machine.adjust_vsp(delta)?;
            machine.store_stack(0, &value)?;
        }
        VmHandlerSemantic::VirtualRegisterRead => {
            let slot = instruction
                .operand
                .and_then(|slot| i64::try_from(slot).ok())
                .ok_or_else(|| "context read has no bounded slot operand".to_owned())?;
            let input_width = instruction.input_width.unwrap_or(width);
            let value = machine.context.load(slot, input_width)?;
            let value = match instruction.resize {
                Some(operation) => {
                    Expr::resize(operation, width, value).map_err(|error| error.to_string())?
                }
                None if input_width == width => value,
                None => {
                    return Err(
                        "context-read input/output widths differ without a proven resize"
                            .to_owned(),
                    );
                }
            };
            let delta = i32::try_from(width.bytes())
                .ok()
                .and_then(i32::checked_neg)
                .ok_or_else(|| "context-read width exceeds VSP range".to_owned())?;
            expected_delta(instruction, delta)?;
            machine.adjust_vsp(delta)?;
            machine.store_stack(0, &value)?;
        }
        VmHandlerSemantic::VirtualRegisterWrite => {
            let slot = instruction
                .operand
                .and_then(|slot| i64::try_from(slot).ok())
                .ok_or_else(|| "context write has no bounded slot operand".to_owned())?;
            let input_width = instruction.input_width.unwrap_or(width);
            let value = machine.load_stack(0, input_width)?;
            let value = match instruction.resize {
                Some(operation) => {
                    Expr::resize(operation, width, value).map_err(|error| error.to_string())?
                }
                None if input_width == width => value,
                None => {
                    return Err(
                        "context-write input/output widths differ without a proven resize"
                            .to_owned(),
                    );
                }
            };
            let delta = i32::try_from(input_width.bytes())
                .map_err(|_| "context-write width exceeds VSP range".to_owned())?;
            expected_delta(instruction, delta)?;
            machine.context.store(slot, &value)?;
            machine.adjust_vsp(delta)?;
        }
        VmHandlerSemantic::ReadTimestampCounter => {
            expected_delta(instruction, -8)?;
            let counter = Expr::intrinsic(
                BitWidth::W64,
                format!("read_timestamp_counter_{instruction_index}"),
                Vec::new(),
            );
            let high = Expr::extract(BitWidth::W32, 32, counter.clone())
                .map_err(|error| error.to_string())?;
            let low =
                Expr::extract(BitWidth::W32, 0, counter).map_err(|error| error.to_string())?;
            machine.adjust_vsp(-8)?;
            machine.store_stack(0, &high)?;
            machine.store_stack(4, &low)?;
        }
        VmHandlerSemantic::UnsignedDivide => {
            let bytes = width.bytes();
            let high = machine.load_stack(0, width)?;
            let low = machine.load_stack(i64::from(bytes), width)?;
            let divisor = machine.load_stack(i64::from(bytes.saturating_mul(2)), width)?;
            let arguments = vec![high, low, divisor];
            let (quotient, remainder) = wide_pair(
                width,
                &format!("x86_unsigned_divide_{width:?}_{instruction_index}").to_lowercase(),
                &arguments,
                ("quotient", "remainder"),
            )?;
            let flagged_delta = i32::try_from(bytes)
                .ok()
                .and_then(|bytes| bytes.checked_sub(8))
                .ok_or_else(|| "division width exceeds VSP range".to_owned())?;
            machine.flags = Expr::intrinsic(
                BitWidth::W64,
                format!("vm_flags_unsigned_divide_{width:?}_{instruction_index}").to_lowercase(),
                arguments,
            );
            if instruction.vsp_delta == Some(flagged_delta) {
                machine.adjust_vsp(flagged_delta)?;
                machine.store_stack(8, &remainder)?;
                machine.store_stack(i64::from(8_u32.saturating_add(bytes)), &quotient)?;
                let flags = machine.flags.clone();
                machine.store_stack(0, &flags)?;
            } else if instruction.vsp_delta == i32::try_from(bytes).ok() {
                machine.adjust_vsp(i32::try_from(bytes).unwrap_or(0))?;
                machine.store_stack(0, &remainder)?;
                machine.store_stack(i64::from(bytes), &quotient)?;
            } else {
                return Err(format!(
                    "recovered VSP delta {:?} does not match a division layout",
                    instruction.vsp_delta
                ));
            }
        }
        VmHandlerSemantic::Mul | VmHandlerSemantic::SignedMultiply => {
            let kind = if instruction.semantic == VmHandlerSemantic::Mul {
                "unsigned"
            } else {
                "signed"
            };
            let bytes = width.bytes();
            let multiplier = machine.load_stack(0, width)?;
            let accumulator = machine.load_stack(i64::from(bytes), width)?;
            let arguments = vec![multiplier, accumulator];
            let (low, high) = wide_pair(
                width,
                &format!("x86_{kind}_multiply_{width:?}_{instruction_index}").to_lowercase(),
                &arguments,
                ("low", "high"),
            )?;
            expected_delta(instruction, -8)?;
            machine.adjust_vsp(-8)?;
            machine.store_stack(8, &high)?;
            machine.store_stack(i64::from(8_u32.saturating_add(bytes)), &low)?;
            machine.flags = Expr::intrinsic(
                BitWidth::W64,
                format!("vm_flags_{kind}_multiply_{width:?}_{instruction_index}").to_lowercase(),
                arguments,
            );
            let flags = machine.flags.clone();
            machine.store_stack(0, &flags)?;
        }
        VmHandlerSemantic::Add
        | VmHandlerSemantic::Sub
        | VmHandlerSemantic::Nor
        | VmHandlerSemantic::Nand => {
            let input_width = instruction.input_width.unwrap_or(width);
            let left_slot = machine.load_stack(0, width)?;
            let right_slot = machine.load_stack(i64::from(width.bytes()), width)?;
            let left = resized_to(input_width, left_slot).map_err(|error| error.to_string())?;
            let right = resized_to(input_width, right_slot).map_err(|error| error.to_string())?;
            let result = arithmetic_result(instruction.semantic, left.clone(), right.clone())?;
            let result = match instruction.resize {
                Some(operation) => {
                    Expr::resize(operation, width, result).map_err(|error| error.to_string())?
                }
                None if input_width == width => result,
                None => {
                    return Err(
                        "arithmetic input/output widths differ without a proven resize".to_owned(),
                    );
                }
            };
            let delta = i32::try_from(width.bytes())
                .ok()
                .and_then(|bytes| bytes.checked_sub(8))
                .ok_or_else(|| "arithmetic width exceeds VSP range".to_owned())?;
            expected_delta(instruction, delta)?;
            machine.adjust_vsp(delta)?;
            machine.store_stack(8, &result)?;
            machine.flags = Expr::intrinsic(
                BitWidth::W64,
                format!("vm_flags_{:?}_{:?}", instruction.semantic, input_width).to_lowercase(),
                vec![left, right],
            );
            let flags = machine.flags.clone();
            machine.store_stack(0, &flags)?;
        }
        VmHandlerSemantic::ShiftLeftDouble | VmHandlerSemantic::ShiftRightDouble => {
            let bytes = width.bytes();
            let delta = i32::try_from(bytes)
                .ok()
                .and_then(|bytes| bytes.checked_sub(6))
                .ok_or_else(|| "double-shift width exceeds VSP range".to_owned())?;
            expected_delta(instruction, delta)?;
            let destination = machine.load_stack(0, width)?;
            let source = machine.load_stack(i64::from(bytes), width)?;
            let count = machine.load_stack(i64::from(bytes.saturating_mul(2)), BitWidth::W8)?;
            let direction = if instruction.semantic == VmHandlerSemantic::ShiftLeftDouble {
                "left"
            } else {
                "right"
            };
            let arguments = vec![destination, source, count];
            let result = Expr::intrinsic(
                width,
                format!("x86_double_shift_{direction}_{width:?}_{instruction_index}")
                    .to_lowercase(),
                arguments.clone(),
            );
            machine.adjust_vsp(delta)?;
            machine.store_stack(8, &result)?;
            machine.flags = Expr::intrinsic(
                BitWidth::W64,
                format!("vm_flags_double_shift_{direction}_{width:?}_{instruction_index}")
                    .to_lowercase(),
                arguments,
            );
            let flags = machine.flags.clone();
            machine.store_stack(0, &flags)?;
        }
        VmHandlerSemantic::Cpuid => {
            // Two layouts share one executor: the two-operand form (-8) reads a
            // leaf and a subleaf from the stack; the leaf-only form (-12) reads
            // only the leaf, and its subleaf is the native ECX on entry — an
            // environment value modelled as a fresh opaque input. Either way the
            // four result dwords are opaque, CPU-dependent intrinsics, so the
            // leaf-only subleaf being unknown only widens an already-opaque
            // result and stays sound.
            let delta = instruction
                .vsp_delta
                .ok_or_else(|| "cpuid VSP delta is unknown".to_owned())?;
            let leaf = machine.load_stack(0, BitWidth::W32)?;
            let subleaf = match delta {
                -8 => machine.load_stack(4, BitWidth::W32)?,
                -12 => Expr::intrinsic(
                    BitWidth::W32,
                    format!("x86_cpuid_subleaf_{instruction_index}"),
                    Vec::new(),
                ),
                other => {
                    return Err(format!("unsupported cpuid VSP delta {other}"));
                }
            };
            machine.adjust_vsp(delta)?;
            for (displacement, register) in [(12, "eax"), (8, "ebx"), (4, "ecx"), (0, "edx")] {
                let value = Expr::intrinsic(
                    BitWidth::W32,
                    format!("x86_cpuid_{register}_{instruction_index}"),
                    vec![leaf.clone(), subleaf.clone()],
                );
                machine.store_stack(displacement, &value)?;
            }
        }
        VmHandlerSemantic::GsRelativeLoad => {
            expected_delta(instruction, 0)?;
            let pointer = machine.load_stack(0, BitWidth::W64)?;
            let value = Expr::intrinsic(
                BitWidth::W64,
                format!("x86_gs_load_{instruction_index}"),
                vec![pointer],
            );
            machine.store_stack(0, &value)?;
        }
        VmHandlerSemantic::PopFlags => {
            expected_delta(instruction, 8)?;
            let value = machine.load_stack(0, BitWidth::W64)?;
            machine.adjust_vsp(8)?;
            machine.flags = Expr::intrinsic(
                BitWidth::W64,
                format!("x86_popfq_{instruction_index}"),
                vec![value],
            );
        }
        VmHandlerSemantic::ShiftLeft
        | VmHandlerSemantic::ShiftRightLogical
        | VmHandlerSemantic::ShiftRightArithmetic
        | VmHandlerSemantic::RotateLeft
        | VmHandlerSemantic::RotateRight => {
            let delta = instruction
                .vsp_delta
                .ok_or_else(|| "shift VSP delta is unknown".to_owned())?;
            let count_bytes = delta
                .checked_add(8)
                .and_then(|bytes| u32::try_from(bytes).ok())
                .and_then(BitWidth::from_bytes)
                .ok_or_else(|| "shift count slot has unsupported width".to_owned())?;
            let input_width = instruction.input_width.unwrap_or(width);
            if (input_width != width) != instruction.resize.is_some()
                || instruction
                    .resize
                    .is_some_and(|resize| resize != ResizeOp::ZeroExtend)
            {
                return Err("shift input/output widths differ without a proven resize".to_owned());
            }
            let value = resized_to(input_width, machine.load_stack(0, width)?)
                .map_err(|error| error.to_string())?;
            let count = resized_to(
                input_width,
                machine.load_stack(i64::from(width.bytes()), count_bytes)?,
            )
            .map_err(|error| error.to_string())?;
            let operation = match instruction.semantic {
                VmHandlerSemantic::ShiftLeft => BinaryOp::Shl,
                VmHandlerSemantic::ShiftRightLogical => BinaryOp::Shr,
                VmHandlerSemantic::ShiftRightArithmetic => BinaryOp::Sar,
                VmHandlerSemantic::RotateLeft => BinaryOp::Rol,
                VmHandlerSemantic::RotateRight => BinaryOp::Ror,
                _ => unreachable!(),
            };
            let result = Expr::binary(operation, value.clone(), count.clone())
                .map_err(|error| error.to_string())?;
            let result = if input_width == width {
                result
            } else {
                Expr::resize(ResizeOp::ZeroExtend, width, result)
                    .map_err(|error| error.to_string())?
            };
            machine.adjust_vsp(delta)?;
            machine.store_stack(8, &result)?;
            machine.flags = Expr::intrinsic(
                BitWidth::W64,
                format!("vm_flags_{:?}_{:?}", instruction.semantic, input_width).to_lowercase(),
                vec![value, count],
            );
            let flags = machine.flags.clone();
            machine.store_stack(0, &flags)?;
        }
        VmHandlerSemantic::MemoryLoad => {
            let pointer = machine.load_stack(0, BitWidth::W64)?;
            let input_width = instruction.input_width.unwrap_or(width);
            let value = machine.load_external(pointer, input_width)?;
            let value = match instruction.resize {
                Some(operation) => {
                    Expr::resize(operation, width, value).map_err(|error| error.to_string())?
                }
                None if input_width == width => value,
                None => {
                    return Err(
                        "memory-load input/output widths differ without a proven resize".to_owned(),
                    );
                }
            };
            let delta = i32::try_from(8_u32.saturating_sub(width.bytes()))
                .map_err(|_| "memory-load width exceeds VSP range".to_owned())?;
            expected_delta(instruction, delta)?;
            machine.adjust_vsp(delta)?;
            machine.store_stack(0, &value)?;
        }
        VmHandlerSemantic::MemoryStore => {
            let pointer = machine.load_stack(0, BitWidth::W64)?;
            let value = machine.load_stack(8, width)?;
            let slot_bytes = if width == BitWidth::W8 && instruction.vsp_delta == Some(10) {
                2
            } else {
                width.bytes()
            };
            let delta = i32::try_from(8_u32.saturating_add(slot_bytes))
                .map_err(|_| "memory-store width exceeds VSP range".to_owned())?;
            expected_delta(instruction, delta)?;
            machine.store_external(instruction_index, pointer, value)?;
            machine.adjust_vsp(delta)?;
        }
        VmHandlerSemantic::TableChecksum => {
            let metadata = instruction
                .semantic_metadata
                .as_ref()
                .ok_or_else(|| "table checksum has no recovered semantic metadata".to_owned())?;
            let VmSemanticMetadata::TableChecksum(metadata) = metadata;
            execute_table_checksum(machine, instruction, width, metadata)?;
        }
        VmHandlerSemantic::Push => {
            let delta = i32::try_from(width.bytes())
                .ok()
                .and_then(i32::checked_neg)
                .ok_or_else(|| "push width exceeds VSP range".to_owned())?;
            expected_delta(instruction, delta)?;
            let prior_vsp = machine.vsp.clone();
            let pushed = Expr::extract(width, 0, prior_vsp).map_err(|error| error.to_string())?;
            machine.adjust_vsp(delta)?;
            machine.store_stack(0, &pushed)?;
        }
        VmHandlerSemantic::Pop => {
            if width != BitWidth::W64 {
                return Err("partial-width VSP reload is not modelled symbolically".to_owned());
            }
            let restored = machine.load_stack(0, BitWidth::W64)?;
            machine.vsp_offset = pointer_offset(&restored);
            machine.vsp = restored;
        }
        unsupported => {
            return Err(format!(
                "semantic {unsupported:?} is not implemented by the symbolic executor"
            ));
        }
    }
    Ok(())
}

/// Symbolically executes a proven straight-line VM block and resolves its
/// control-flow target from VM stack/context state when possible.
///
/// # Errors
///
/// Returns the exact instruction and VIP for inconsistent recovered metadata,
/// unsupported semantics, symbolic-VSP loss, expression failure, or a resource
/// limit. It never guesses a missing operation.
#[allow(clippy::too_many_lines)]
pub fn execute_symbolic_linear_block(
    instructions: &[DecodedVmInstruction],
    boundary: Option<VmHandlerSemantic>,
    limits: SymbolicExecutionLimits,
) -> Result<SymbolicVmExecutionReport, SymbolicExecutionError> {
    execute_symbolic_linear_block_with_initial_state(
        instructions,
        boundary,
        limits,
        &SymbolicVmInitialState::default(),
        None,
    )
}

/// Symbolically executes a proven straight-line VM block with recovered entry
/// state and an optional immutable image-memory source.
///
/// # Errors
///
/// Returns a structured error for invalid initial state, inconsistent handler
/// effects, unresolved semantics, or resource exhaustion.
#[allow(clippy::too_many_lines)]
pub fn execute_symbolic_linear_block_with_initial_state(
    instructions: &[DecodedVmInstruction],
    boundary: Option<VmHandlerSemantic>,
    limits: SymbolicExecutionLimits,
    initial: &SymbolicVmInitialState,
    memory_source: Option<&dyn SymbolicMemorySource>,
) -> Result<SymbolicVmExecutionReport, SymbolicExecutionError> {
    let initial_vip = instructions
        .first()
        .map_or(Va(0), |instruction| instruction.state_before.vip);
    let mut machine = SymbolicMachine::new(initial, memory_source, limits).map_err(|message| {
        SymbolicExecutionError {
            instruction_index: 0,
            vip: initial_vip,
            semantic: instructions
                .first()
                .map_or(VmHandlerSemantic::Unknown, |instruction| {
                    instruction.semantic
                }),
            message,
        }
    })?;
    for (instruction_index, instruction) in instructions.iter().enumerate() {
        // Sampled, not per-instruction: a clock read costs more than many of the
        // steps it guards, and a block is capped at a few thousand instructions.
        if instruction_index % 32 == 0
            && limits.deadline.is_some_and(|at| std::time::Instant::now() >= at)
        {
            return Err(SymbolicExecutionError {
                instruction_index,
                vip: instruction.state_before.vip,
                semantic: instruction.semantic,
                message: "per-entry wall-clock budget exceeded".to_owned(),
            });
        }
        if instruction_index >= limits.max_instructions {
            return Err(SymbolicExecutionError {
                instruction_index,
                vip: instruction.state_before.vip,
                semantic: instruction.semantic,
                message: format!(
                    "instruction count exceeds limit {}",
                    limits.max_instructions
                ),
            });
        }
        if let Err(message) = execute_instruction(&mut machine, instruction_index, instruction)
            .and_then(|()| machine.check_limits(limits))
        {
            return Err(SymbolicExecutionError {
                instruction_index,
                vip: instruction.state_before.vip,
                semantic: instruction.semantic,
                message,
            });
        }
    }

    let branch_target = match boundary {
        None | Some(VmHandlerSemantic::VmExit) => None,
        Some(VmHandlerSemantic::UnconditionalBranch) => {
            let target =
                machine
                    .load_stack(0, BitWidth::W64)
                    .map_err(|message| SymbolicExecutionError {
                        instruction_index: instructions.len(),
                        vip: instructions
                            .last()
                            .map_or(Va(0), |instruction| instruction.state_after.vip),
                        semantic: VmHandlerSemantic::UnconditionalBranch,
                        message,
                    })?;
            machine
                .adjust_vsp(8)
                .map_err(|message| SymbolicExecutionError {
                    instruction_index: instructions.len(),
                    vip: instructions
                        .last()
                        .map_or(Va(0), |instruction| instruction.state_after.vip),
                    semantic: VmHandlerSemantic::UnconditionalBranch,
                    message,
                })?;
            Some(target)
        }
        Some(semantic) => {
            return Err(SymbolicExecutionError {
                instruction_index: instructions.len(),
                vip: instructions
                    .last()
                    .map_or(Va(0), |instruction| instruction.state_after.vip),
                semantic,
                message: "control-flow boundary is not implemented".to_owned(),
            });
        }
    };
    machine
        .check_limits(limits)
        .map_err(|message| SymbolicExecutionError {
            instruction_index: instructions.len(),
            vip: instructions
                .last()
                .map_or(Va(0), |instruction| instruction.state_after.vip),
            semantic: boundary.unwrap_or(VmHandlerSemantic::Unknown),
            message,
        })?;

    let mut context_cells = BTreeMap::new();
    for cell in machine.context.touched_cells.clone() {
        let value =
            machine
                .context
                .load(cell, BitWidth::W64)
                .map_err(|error| SymbolicExecutionError {
                    instruction_index: instructions.len(),
                    vip: instructions
                        .last()
                        .map_or(Va(0), |instruction| instruction.state_after.vip),
                    semantic: boundary.unwrap_or(VmHandlerSemantic::Unknown),
                    message: error,
                })?;
        context_cells.insert(cell, value);
    }
    machine
        .check_limits(limits)
        .map_err(|message| SymbolicExecutionError {
            instruction_index: instructions.len(),
            vip: instructions
                .last()
                .map_or(Va(0), |instruction| instruction.state_after.vip),
            semantic: boundary.unwrap_or(VmHandlerSemantic::Unknown),
            message,
        })?;
    if let Some(target) = &branch_target {
        let nodes = target.node_count_bounded(limits.max_expression_nodes);
        if nodes > limits.max_expression_nodes {
            return Err(SymbolicExecutionError {
                instruction_index: instructions.len(),
                vip: instructions
                    .last()
                    .map_or(Va(0), |instruction| instruction.state_after.vip),
                semantic: boundary.unwrap_or(VmHandlerSemantic::Unknown),
                message: format!(
                    "branch-target expression exceeds {} nodes",
                    limits.max_expression_nodes
                ),
            });
        }
        machine.peak_expression_nodes = machine.peak_expression_nodes.max(nodes);
    }
    let branch_target_dependencies = branch_target.as_ref().map_or_else(Vec::new, |target| {
        target.dependencies().into_iter().collect()
    });
    let tracked_memory_bytes = machine
        .memory
        .len()
        .saturating_add(machine.context.bytes.len());
    let next_native_call_epoch = machine.native_call_epochs.next();
    let memory_values = sorted_memory_values(machine.memory).map_err(|message| {
        SymbolicExecutionError {
            instruction_index: instructions.len(),
            vip: instructions
                .last()
                .map_or(Va(0), |instruction| instruction.state_after.vip),
            semantic: boundary.unwrap_or(VmHandlerSemantic::Unknown),
            message,
        }
    })?;
    Ok(SymbolicVmExecutionReport {
        schema_version: 4,
        executed_instructions: instructions.len(),
        stop_reason: if boundary.is_some() {
            SymbolicExecutionStop::ControlFlowBoundary
        } else {
            SymbolicExecutionStop::EndOfStream
        },
        final_vsp: machine.vsp,
        final_vsp_offset: machine.vsp_offset,
        final_flags: machine.flags,
        branch_target,
        branch_target_dependencies,
        context_cells,
        memory_values,
        memory_epoch: machine.memory_epoch,
        memory_tainted: machine.memory_tainted,
        image_reads_suppressed: machine.image_reads_suppressed,
        assumed_external_disjoint_from_vm_stack: machine
            .assumed_external_disjoint_from_vm_stack,
        assumed_external_disjoint_from_fresh_allocations: machine
            .assumed_external_disjoint_from_fresh_allocations,
        served_read_only_image: machine.served_read_only_image,
        protection_may_have_changed: machine.protection_may_have_changed,
        vm_stack_address_escapes: machine.vm_stack_address_escapes,
        vm_stack_derived_escapes: machine.vm_stack_derived_escapes,
        vm_stack_pointer_lost: machine.vm_stack_pointer_lost,
        vm_stack_store_hi: machine.vm_stack_store_hi,
        external_writes: machine.external_writes,
        memory_havocs: machine.memory_havocs,
        native_calls: machine.native_calls,
        fresh_allocations: machine.fresh_allocations,
        next_native_call_epoch,
        peak_expression_nodes: machine.peak_expression_nodes,
        tracked_memory_bytes,
        resolved_image_bytes: machine.resolved_image_addresses.len(),
    })
}

/// Applies one exhaustively proven native-call bridge outcome. Unknown or
/// ambiguous targets conservatively havoc tracked memory. A uniquely resolved
/// import receives a non-opaque effect only when it exists in the trusted API
/// summary registry.
///
/// # Errors
///
/// Returns a structured error if the carried state is invalid, the proven
/// call-frame accesses cannot be represented, or a resource limit is reached.
pub fn execute_symbolic_native_call_boundary_with_initial_state(
    boundary_vip: Va,
    outcome: &NativeCallSelectorOutcome,
    limits: SymbolicExecutionLimits,
    initial: &SymbolicVmInitialState,
    memory_source: Option<&dyn SymbolicMemorySource>,
) -> Result<SymbolicVmExecutionReport, SymbolicExecutionError> {
    let mut machine = SymbolicMachine::new(initial, memory_source, limits).map_err(|message| {
        SymbolicExecutionError {
            instruction_index: 0,
            vip: boundary_vip,
            semantic: VmHandlerSemantic::NativeCall,
            message,
        }
    })?;
    machine
        .apply_native_call(0, outcome)
        .and_then(|()| machine.check_limits(limits))
        .map_err(|message| SymbolicExecutionError {
            instruction_index: 0,
            vip: boundary_vip,
            semantic: VmHandlerSemantic::NativeCall,
            message,
        })?;

    let mut context_cells = BTreeMap::new();
    for cell in machine.context.touched_cells.clone() {
        let value = machine
            .context
            .load(cell, BitWidth::W64)
            .map_err(|message| SymbolicExecutionError {
                instruction_index: 0,
                vip: boundary_vip,
                semantic: VmHandlerSemantic::NativeCall,
                message,
            })?;
        context_cells.insert(cell, value);
    }
    machine
        .check_limits(limits)
        .map_err(|message| SymbolicExecutionError {
            instruction_index: 0,
            vip: boundary_vip,
            semantic: VmHandlerSemantic::NativeCall,
            message,
        })?;
    let tracked_memory_bytes = machine
        .memory
        .len()
        .saturating_add(machine.context.bytes.len());
    let next_native_call_epoch = machine.native_call_epochs.next();
    let memory_values =
        sorted_memory_values(machine.memory).map_err(|message| SymbolicExecutionError {
            instruction_index: 0,
            vip: boundary_vip,
            semantic: VmHandlerSemantic::NativeCall,
            message,
        })?;
    Ok(SymbolicVmExecutionReport {
        schema_version: 4,
        executed_instructions: 1,
        stop_reason: SymbolicExecutionStop::ControlFlowBoundary,
        final_vsp: machine.vsp,
        final_vsp_offset: machine.vsp_offset,
        final_flags: machine.flags,
        branch_target: None,
        branch_target_dependencies: Vec::new(),
        context_cells,
        memory_values,
        memory_epoch: machine.memory_epoch,
        memory_tainted: machine.memory_tainted,
        image_reads_suppressed: machine.image_reads_suppressed,
        assumed_external_disjoint_from_vm_stack: machine
            .assumed_external_disjoint_from_vm_stack,
        assumed_external_disjoint_from_fresh_allocations: machine
            .assumed_external_disjoint_from_fresh_allocations,
        served_read_only_image: machine.served_read_only_image,
        protection_may_have_changed: machine.protection_may_have_changed,
        vm_stack_address_escapes: machine.vm_stack_address_escapes,
        vm_stack_derived_escapes: machine.vm_stack_derived_escapes,
        vm_stack_pointer_lost: machine.vm_stack_pointer_lost,
        vm_stack_store_hi: machine.vm_stack_store_hi,
        external_writes: machine.external_writes,
        memory_havocs: machine.memory_havocs,
        native_calls: machine.native_calls,
        fresh_allocations: machine.fresh_allocations,
        next_native_call_epoch,
        peak_expression_nodes: machine.peak_expression_nodes,
        tracked_memory_bytes,
        resolved_image_bytes: machine.resolved_image_addresses.len(),
    })
}

#[cfg(test)]
mod tests {
    /// A heavily shared DAG has few distinct nodes but renders as an
    /// exponentially large tree; sorting its address must fall back to hash order
    /// instead of allocating the tree.
    #[test]
    fn shared_dag_address_sort_key_falls_back_to_hash_order() {
        let mut address = Expr::register(BitWidth::W64, "native_rax_initial");
        for round in 0..48_u64 {
            let mixed = Expr::binary(
                BinaryOp::Xor,
                address.clone(),
                Expr::constant(BitWidth::W64, round + 1),
            )
            .unwrap();
            address = Expr::binary(BinaryOp::Add, address, mixed).unwrap();
        }
        assert!(address.node_count_bounded(1_000_000) < 10_000, "DAG stays small");
        let mut memory = HashMap::new();
        memory.insert(address.clone(), Expr::constant(BitWidth::W8, 0));
        memory.insert(Expr::constant(BitWidth::W64, 7), Expr::constant(BitWidth::W8, 1));
        let first = sorted_memory_values(memory.clone()).unwrap();
        let second = sorted_memory_values(memory).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first, second);
        assert!(bounded_debug(&address, 1 << 20).is_none());

        let mut small = HashMap::new();
        small.insert(Expr::constant(BitWidth::W64, 2), Expr::constant(BitWidth::W8, 0));
        small.insert(Expr::constant(BitWidth::W64, 1), Expr::constant(BitWidth::W8, 0));
        let sorted = sorted_memory_values(small).unwrap();
        assert_eq!(sorted.len(), 2);
        assert!(format!("{:?}", sorted[0].0) < format!("{:?}", sorted[1].0));
    }

    use crate::{EffectKnowledge, RelativeDispatchState};

    use super::*;

    #[test]
    fn continuation_state_substitution_specializes_all_retained_expressions() {
        let dependency = Expr::register(BitWidth::W64, "path_input");
        let state = SymbolicVmInitialState {
            vsp: dependency.clone(),
            flags: dependency.clone(),
            context_values: BTreeMap::from([(8, dependency.clone())]),
            memory_values: vec![(dependency.clone(), dependency.clone())],
            memory_epoch: 3,
            memory_tainted: false,
            image_reads_suppressed: false,
            protection_may_have_changed: true,
            next_native_call_epoch: 5,
            fresh_allocations: vec![SymbolicFreshAllocation {
                allocation: dependency.clone(),
                size: dependency,
                zero_initialized: true,
            }],
        };
        let constant = Expr::constant(BitWidth::W64, 7);
        let specialized = state
            .substitute(&BTreeMap::from([(
                "path_input".to_owned(),
                constant.clone(),
            )]))
            .expect("replacement width matches");
        assert_eq!(specialized.vsp, constant);
        assert!(specialized.protection_may_have_changed);
        assert!(
            specialized
                .context_values
                .values()
                .all(|value| { matches!(value.kind, ExprKind::Const { value: 7 }) })
        );
        assert!(specialized.memory_values.iter().all(|(address, value)| {
            matches!(address.kind, ExprKind::Const { value: 7 })
                && matches!(value.kind, ExprKind::Const { value: 7 })
        }));
        assert!(specialized.fresh_allocations.iter().all(|allocation| {
            matches!(allocation.allocation.kind, ExprKind::Const { value: 7 })
                && matches!(allocation.size.kind, ExprKind::Const { value: 7 })
        }));
    }

    fn instruction(
        semantic: VmHandlerSemantic,
        width: BitWidth,
        operand: Option<u64>,
        vsp_delta: Option<i32>,
    ) -> DecodedVmInstruction {
        DecodedVmInstruction {
            state_before: RelativeDispatchState {
                vip: Va(0x1_4000_1000),
                rolling_key: 0,
                handler: Va(0x1_4000_2000),
            },
            encoded_bytes: Vec::new(),
            semantic,
            semantic_metadata: None,
            input_width: None,
            operation_width: Some(width),
            resize: None,
            operand,
            vsp_delta,
            flags_effect: EffectKnowledge::Preserved,
            confidence: 100,
            state_after: RelativeDispatchState {
                vip: Va(0x1_4000_1004),
                rolling_key: 0,
                handler: Va(0x1_4000_3000),
            },
        }
    }

    #[derive(Default)]
    struct TestMemorySource {
        bytes: BTreeMap<u64, u8>,
    }

    impl TestMemorySource {
        fn write(&mut self, address: u64, bytes: &[u8]) {
            for (offset, byte) in bytes.iter().copied().enumerate() {
                let offset = u64::try_from(offset).expect("small test-memory offset");
                self.bytes.insert(address + offset, byte);
            }
        }
    }

    impl SymbolicMemorySource for TestMemorySource {
        fn read_byte(&self, address: Va) -> Option<u8> {
            self.bytes.get(&address.get()).copied()
        }
    }

    fn checksum_metadata() -> TableChecksumMetadata {
        TableChecksumMetadata {
            checksum_width: BitWidth::W32,
            accumulator_source: crate::ChecksumAccumulatorSource::Constant { value: 0 },
            initial_xor: u32::MAX,
            pointer_stack_offset: 0,
            length_stack_offset: 8,
            length_width: BitWidth::W64,
            pointer_stride: 1,
            table_address: Va(0x1_4000_8000),
            table_entries: 256,
            table_entry_width: BitWidth::W32,
            table_index: crate::ChecksumTableIndex::AccumulatorLowByteXorInputByte,
            shift: crate::ChecksumShift::LogicalRight,
            shift_bits: 8,
            combine: crate::ChecksumCombine::ShiftedAccumulatorXorTableEntry,
            iteration_xor: 0,
            final_not: true,
            final_xor: 0,
            result_stack_offset: 0,
            vsp_delta: 12,
        }
    }

    fn checksum_instruction(metadata: TableChecksumMetadata) -> DecodedVmInstruction {
        let mut instruction = instruction(
            VmHandlerSemantic::TableChecksum,
            BitWidth::W32,
            None,
            Some(metadata.vsp_delta),
        );
        instruction.semantic_metadata = Some(VmSemanticMetadata::TableChecksum(metadata));
        instruction
    }

    fn stack_address(offset: i64) -> Expr {
        add_pointer_offset(Expr::register(BitWidth::W64, "vsp_initial"), offset)
            .expect("bounded test stack offset")
    }

    fn checksum_result(
        report: &SymbolicVmExecutionReport,
        memory_source: Option<&dyn SymbolicMemorySource>,
    ) -> Expr {
        let continuation = report.continuation_state();
        let mut machine = SymbolicMachine::new(
            &continuation,
            memory_source,
            SymbolicExecutionLimits::default(),
        )
        .expect("checksum continuation state");
        machine
            .load_stack(0, BitWidth::W32)
            .expect("checksum result on VM stack")
    }

    #[test]
    fn table_checksum_evaluates_exactly_from_concrete_memory() {
        let mut metadata = checksum_metadata();
        metadata.iteration_xor = 0x1122_3344;
        let iteration_xor = metadata.iteration_xor;
        let input_address = 0x1_4000_A000;
        let input = [0x10_u8, 0x20, 0x30, 0xFF, 0x05];
        let mut source = TestMemorySource::default();
        source.write(input_address, &input);
        let mut table = [0_u32; 256];
        for (index, entry) in table.iter_mut().enumerate() {
            let index = u32::try_from(index).expect("table index fits u32");
            *entry = index.wrapping_mul(0x0102_0408) ^ 0xA5A5_5A5A;
            let address = metadata.table_address.get() + u64::from(index) * 4;
            source.write(address, &entry.to_le_bytes());
        }
        let initial = SymbolicVmInitialState {
            memory_values: vec![
                (
                    stack_address(0),
                    Expr::constant(BitWidth::W64, input_address),
                ),
                (
                    stack_address(8),
                    Expr::constant(
                        BitWidth::W64,
                        u64::try_from(input.len()).expect("small input length"),
                    ),
                ),
            ],
            ..SymbolicVmInitialState::default()
        };
        let report = execute_symbolic_linear_block_with_initial_state(
            &[checksum_instruction(metadata)],
            None,
            SymbolicExecutionLimits::default(),
            &initial,
            Some(&source),
        )
        .expect("concrete checksum must execute");

        let mut expected = u32::MAX;
        for byte in input {
            let table_index = usize::from(expected.to_le_bytes()[0] ^ byte);
            expected = expected.wrapping_shr(8) ^ table[table_index] ^ iteration_xor;
        }
        expected = !expected;
        assert_eq!(
            checksum_result(&report, Some(&source)),
            Expr::constant(BitWidth::W32, u64::from(expected))
        );
        assert_eq!(report.final_vsp_offset, Some(12));
        assert!(report.resolved_image_bytes >= input.len());
    }

    #[test]
    fn symbolic_checksum_retains_pointer_length_and_memory_epoch() {
        let metadata = checksum_metadata();
        let pointer = Expr::register(BitWidth::W64, "checksum_pointer");
        let length = Expr::register(BitWidth::W64, "checksum_length");
        let initial = SymbolicVmInitialState {
            memory_values: vec![
                (stack_address(0), pointer.clone()),
                (stack_address(8), length.clone()),
            ],
            memory_epoch: 11,
            ..SymbolicVmInitialState::default()
        };
        let report = execute_symbolic_linear_block_with_initial_state(
            &[checksum_instruction(metadata)],
            None,
            SymbolicExecutionLimits::default(),
            &initial,
            None,
        )
        .expect("symbolic checksum must retain an intrinsic");
        let result = checksum_result(&report, None);
        let dependencies = result.dependencies();
        let rendered = format!("{result:?}");

        assert!(dependencies.contains("checksum_pointer"));
        assert!(dependencies.contains("checksum_length"));
        assert!(rendered.contains("table_checksum_low_byte_xor_recurrence_w32"));
        assert!(rendered.contains("memory_snapshot_epoch_11"));
        assert!(!matches!(result.kind, ExprKind::Const { .. }));
    }

    #[test]
    fn unreadable_concrete_memory_never_becomes_a_guessed_constant() {
        let metadata = checksum_metadata();
        let initial = SymbolicVmInitialState {
            memory_values: vec![
                (
                    stack_address(0),
                    Expr::constant(BitWidth::W64, 0x1_5000_0000),
                ),
                (stack_address(8), Expr::constant(BitWidth::W64, 1)),
            ],
            ..SymbolicVmInitialState::default()
        };
        let report = execute_symbolic_linear_block_with_initial_state(
            &[checksum_instruction(metadata)],
            None,
            SymbolicExecutionLimits::default(),
            &initial,
            None,
        )
        .expect("missing bytes must remain symbolic");
        let result = checksum_result(&report, None);

        assert!(!matches!(result.kind, ExprKind::Const { .. }));
        assert!(format!("{result:?}").contains("table_checksum_low_byte_xor_recurrence_w32"));
    }

    #[test]
    fn zero_length_checksum_does_not_require_a_concrete_pointer() {
        let metadata = checksum_metadata();
        let initial = SymbolicVmInitialState {
            memory_values: vec![
                (
                    stack_address(0),
                    Expr::register(BitWidth::W64, "unused_pointer"),
                ),
                (stack_address(8), Expr::constant(BitWidth::W64, 0)),
            ],
            ..SymbolicVmInitialState::default()
        };
        let report = execute_symbolic_linear_block_with_initial_state(
            &[checksum_instruction(metadata)],
            None,
            SymbolicExecutionLimits::default(),
            &initial,
            None,
        )
        .expect("zero-length checksum is exact");

        assert_eq!(
            checksum_result(&report, None),
            Expr::constant(BitWidth::W32, 0)
        );
        assert_eq!(report.resolved_image_bytes, 0);
    }

    #[test]
    fn resolves_constant_branch_target_through_context() {
        let instructions = vec![
            instruction(
                VmHandlerSemantic::ConstantPush,
                BitWidth::W64,
                Some(0x1_4000_5000),
                Some(-8),
            ),
            instruction(
                VmHandlerSemantic::VirtualRegisterWrite,
                BitWidth::W64,
                Some(0x98),
                Some(8),
            ),
            instruction(
                VmHandlerSemantic::VirtualRegisterRead,
                BitWidth::W64,
                Some(0x98),
                Some(-8),
            ),
        ];
        let report = execute_symbolic_linear_block(
            &instructions,
            Some(VmHandlerSemantic::UnconditionalBranch),
            SymbolicExecutionLimits::default(),
        )
        .expect("symbolic block must execute");

        assert_eq!(
            report.branch_target,
            Some(Expr::constant(BitWidth::W64, 0x1_4000_5000))
        );
        assert_eq!(report.final_vsp_offset, Some(0));
    }

    #[test]
    fn partial_context_write_preserves_unknown_upper_dword() {
        let instructions = vec![
            instruction(
                VmHandlerSemantic::ConstantPush,
                BitWidth::W32,
                Some(0xAABB_CCDD),
                Some(-4),
            ),
            instruction(
                VmHandlerSemantic::VirtualRegisterWrite,
                BitWidth::W32,
                Some(0x98),
                Some(4),
            ),
            instruction(
                VmHandlerSemantic::VirtualRegisterRead,
                BitWidth::W64,
                Some(0x98),
                Some(-8),
            ),
        ];
        let report = execute_symbolic_linear_block(
            &instructions,
            Some(VmHandlerSemantic::UnconditionalBranch),
            SymbolicExecutionLimits::default(),
        )
        .expect("symbolic block must execute");
        let target = report.branch_target.expect("branch target");

        assert_eq!(target.width, BitWidth::W64);
        assert!(target.dependencies().contains("context_initial[+0x98]"));
    }

    #[test]
    fn zero_extending_context_read_uses_input_and_stack_widths() {
        let mut initial = SymbolicVmInitialState::default();
        initial
            .context_values
            .insert(0x20, Expr::constant(BitWidth::W8, 0xAB));
        let mut machine = SymbolicMachine::new(&initial, None, SymbolicExecutionLimits::default())
            .expect("symbolic machine");
        let mut read = instruction(
            VmHandlerSemantic::VirtualRegisterRead,
            BitWidth::W16,
            Some(0x20),
            Some(-2),
        );
        read.input_width = Some(BitWidth::W8);
        read.resize = Some(ResizeOp::ZeroExtend);

        execute_instruction(&mut machine, 0, &read).expect("resized context read");

        assert_eq!(
            machine.load_stack(0, BitWidth::W16).expect("word result"),
            Expr::constant(BitWidth::W16, 0xAB)
        );
    }

    #[test]
    fn push_and_pop_restore_relative_vsp() {
        let instructions = vec![
            instruction(VmHandlerSemantic::Push, BitWidth::W64, None, Some(-8)),
            instruction(VmHandlerSemantic::Pop, BitWidth::W64, None, None),
        ];
        let report =
            execute_symbolic_linear_block(&instructions, None, SymbolicExecutionLimits::default())
                .expect("symbolic block must execute");

        assert_eq!(report.final_vsp_offset, Some(0));
        assert_eq!(
            report.final_vsp,
            Expr::register(BitWidth::W64, "vsp_initial")
        );
    }

    // (c) Two provably-disjoint stack-relative stores: an array store bounded to
    // `vsp_initial + [64, 1088)` (index is a zero-extended byte scaled by 4) is
    // proven disjoint from a tracked cell below it. The disjoint cell is
    // preserved precisely, no assumption is taken, memory is not tainted, and the
    // post-store branch state survives.
    #[test]
    fn bounded_array_store_preserves_range_disjoint_stack_cell() {
        let mut machine = default_machine();
        // Branch-relevant cell well below the array window.
        machine
            .store_memory(&stack_address(-8), &Expr::constant(BitWidth::W64, 0x1234))
            .expect("seed disjoint low cell");
        // A cell inside the array window must be discarded.
        machine
            .store_memory(&stack_address(80), &Expr::constant(BitWidth::W64, 0x99))
            .expect("seed in-window cell");
        // Address: (zext_W8(index) << 2) + (vsp_initial + 64).
        let index = Expr::resize(
            ResizeOp::ZeroExtend,
            BitWidth::W64,
            Expr::register(BitWidth::W8, "array_index"),
        )
        .expect("zero-extend index");
        let scaled = Expr::binary(BinaryOp::Shl, index, Expr::constant(BitWidth::W64, 2))
            .expect("scale index");
        let address = Expr::binary(BinaryOp::Add, scaled, stack_address(64)).expect("array address");
        let range = proven_store_byte_range(&address, BitWidth::W32).expect("bounded range");
        assert_eq!((range.1, range.2), (64, 64 + 1020 + 4));
        machine
            .store_external(5, address, Expr::constant(BitWidth::W32, 0x55))
            .expect("bounded store narrows");
        assert!(!machine.memory_tainted);
        assert!(!machine.assumed_external_disjoint_from_vm_stack);
        assert_eq!(machine.memory_havocs.len(), 1);
        // The disjoint low cell survives precisely.
        let preserved = machine
            .load_memory(stack_address(-8), BitWidth::W64)
            .expect("load preserved cell");
        assert!(matches!(preserved.kind, ExprKind::Const { value: 0x1234 }));
        // The in-window cell was discarded (reads back as a fresh unknown).
        let discarded = machine
            .load_memory(stack_address(80), BitWidth::W64)
            .expect("load discarded cell");
        assert!(is_free_read(&discarded));
    }

    // (d) Fail-closed: a vsp-rooted store with an UNPROVABLE (full-width symbolic)
    // offset cannot be bounded to a range, so no cell is proven disjoint and the
    // conservative full havoc (taint) is kept — a possibly-aliasing store never
    // preserves a cell.
    #[test]
    fn unprovable_offset_store_still_full_havocs() {
        let mut machine = default_machine();
        machine
            .store_memory(&stack_address(-8), &Expr::constant(BitWidth::W64, 0x1234))
            .expect("seed cell");
        // vsp_initial + opaque 64-bit index: no provable bound on the index.
        let address = Expr::binary(
            BinaryOp::Add,
            Expr::register(BitWidth::W64, "vsp_initial"),
            Expr::register(BitWidth::W64, "opaque_index"),
        )
        .expect("symbolic offset address");
        assert!(proven_store_byte_range(&address, BitWidth::W32).is_none());
        machine
            .store_external(5, address, Expr::constant(BitWidth::W32, 0x55))
            .expect("unprovable store over-approximated");
        assert!(machine.memory_tainted);
        let reloaded = machine
            .load_memory(stack_address(-8), BitWidth::W64)
            .expect("load");
        assert!(is_free_read(&reloaded));
    }

    #[test]
    fn a_null_dereference_of_a_vm_stack_slot_is_an_exception_boundary() {
        // A VM handler can store zero into a stack slot, pop a pointer to that
        // slot, and then dereference it. The load must surface as an exception
        // boundary rather than an opaque value, because the protector's own
        // handler is what recovers control on such a path.
        let store = instruction(
            VmHandlerSemantic::MemoryStore,
            BitWidth::W64,
            None,
            Some(16),
        );
        let mut machine = SymbolicMachine::new(
            &SymbolicVmInitialState::default(),
            None,
            SymbolicExecutionLimits::default(),
        )
        .expect("machine");
        // Push a pointer to vsp_initial - 8, then store zero through it.
        machine.adjust_vsp(-8).expect("reserve the slot");
        machine
            .store_stack(0, &Expr::constant(BitWidth::W64, 0))
            .expect("zero the slot");
        machine.adjust_vsp(-8).expect("reserve the pointer slot");
        let pointer = Expr::binary(
            BinaryOp::Add,
            Expr::register(BitWidth::W64, "vsp_initial"),
            Expr::constant(BitWidth::W64, (-8i64) as u64),
        )
        .expect("pointer expression");
        machine.store_stack(0, &pointer).expect("store the pointer");
        let _ = store;
        // Popping restores vsp to the zeroed slot.
        let restored = machine.load_stack(0, BitWidth::W64).expect("pop");
        machine.vsp_offset = pointer_offset(&restored);
        machine.vsp = restored;
        assert_eq!(machine.vsp_offset, Some(-8));
        // Dereferencing it now reads the zero the handler wrote.
        let value = machine.load_stack(0, BitWidth::W64).expect("read the slot");
        assert_eq!(value, Expr::constant(BitWidth::W64, 0));
    }

    #[test]
    fn dispatch_is_program_semantic_noop() {
        let report = execute_symbolic_linear_block(
            &[instruction(
                VmHandlerSemantic::Dispatch,
                BitWidth::W64,
                None,
                Some(0),
            )],
            None,
            SymbolicExecutionLimits::default(),
        )
        .expect("internal dispatch must execute as a semantic no-op");

        assert_eq!(report.final_vsp_offset, Some(0));
        assert!(report.context_cells.is_empty());
        assert!(report.external_writes.is_empty());
    }

    #[test]
    fn continuation_preserves_vm_stack_and_memory_epoch() {
        let first = execute_symbolic_linear_block(
            &[instruction(
                VmHandlerSemantic::ConstantPush,
                BitWidth::W64,
                Some(0x1_4000_6000),
                Some(-8),
            )],
            None,
            SymbolicExecutionLimits::default(),
        )
        .expect("first block must execute");
        let continuation = first.continuation_state();
        assert_eq!(continuation.memory_epoch, first.memory_epoch);

        let second = execute_symbolic_linear_block_with_initial_state(
            &[],
            Some(VmHandlerSemantic::UnconditionalBranch),
            SymbolicExecutionLimits::default(),
            &continuation,
            None,
        )
        .expect("successor boundary must read the retained VM stack");

        assert_eq!(
            second.branch_target,
            Some(Expr::constant(BitWidth::W64, 0x1_4000_6000))
        );
        assert_eq!(second.memory_epoch, first.memory_epoch);
    }

    #[test]
    fn initial_state_preserves_nonzero_vsp_offset() {
        let initial = SymbolicVmInitialState {
            vsp: add_pointer_offset(Expr::register(BitWidth::W64, "vsp_initial"), -24)
                .expect("valid pointer expression"),
            ..SymbolicVmInitialState::default()
        };
        let report = execute_symbolic_linear_block_with_initial_state(
            &[],
            None,
            SymbolicExecutionLimits::default(),
            &initial,
            None,
        )
        .expect("nonzero continuation VSP must remain valid");
        assert_eq!(report.final_vsp_offset, Some(-24));
    }

    #[test]
    fn rejects_tainted_or_oversized_initial_state() {
        let tainted = SymbolicVmInitialState {
            memory_tainted: true,
            ..SymbolicVmInitialState::default()
        };
        let error = execute_symbolic_linear_block_with_initial_state(
            &[],
            None,
            SymbolicExecutionLimits::default(),
            &tainted,
            None,
        )
        .expect_err("tainted state must not be resumed with stale aliases");
        assert!(error.message.contains("cannot be resumed soundly"));

        let mut oversized = SymbolicVmInitialState::default();
        oversized
            .context_values
            .insert(0, Expr::constant(BitWidth::W64, 1));
        let error = execute_symbolic_linear_block_with_initial_state(
            &[],
            None,
            SymbolicExecutionLimits {
                max_memory_bytes: 7,
                ..SymbolicExecutionLimits::default()
            },
            &oversized,
            None,
        )
        .expect_err("initial state must obey the memory budget");
        assert!(error.message.contains("initial symbolic memory exceeds"));
    }

    // A store to a proven-region address resolves precisely and is recorded as
    // a side effect, leaving memory untainted and resumable (no regression of
    // the tracked-store path DllMain relies on).
    #[test]
    fn symbolic_store_to_tracked_region_resolves_precisely() {
        let mut machine = default_machine();
        machine
            .store_stack(0, &Expr::constant(BitWidth::W64, 9))
            .expect("seed precise VM stack");
        machine
            .store_external(
                0,
                stack_address(32),
                Expr::constant(BitWidth::W64, 0x1234),
            )
            .expect("tracked store resolves");
        assert!(!machine.memory_tainted);
        assert_eq!(machine.external_writes.len(), 1);
        let loaded = machine
            .load_memory(stack_address(32), BitWidth::W64)
            .expect("load");
        assert!(matches!(loaded.kind, ExprKind::Const { value: 0x1234 }));
        // The unrelated seeded cell is still precise.
        let seeded = machine
            .load_memory(stack_address(0), BitWidth::W64)
            .expect("load");
        assert!(matches!(seeded.kind, ExprKind::Const { value: 9 }));
    }

    // (a) An external store whose address is native-register-rooted or opaque
    // cannot reach the VM's private operand stack. Under the labeled
    // `AssumesExternalStoresDisjointFromVmStack` assumption the executor discards
    // only the non-vsp-rooted cells the store could alias, preserves the
    // vsp-rooted VM operand stack so the post-store VIP stays precise, records
    // the assumption, and continues without tainting (branch resolution and
    // resume remain possible). The write is still recorded for the lift.
    #[test]
    fn external_store_disjoint_from_vm_stack_preserves_stack_under_assumption() {
        let mut machine = default_machine();
        machine
            .store_memory(&stack_address(16), &Expr::constant(BitWidth::W64, 9))
            .expect("seed precise VM-stack cell");
        machine
            .store_memory(
                &Expr::constant(BitWidth::W64, 0x1_4000_1000),
                &Expr::constant(BitWidth::W64, 0x77),
            )
            .expect("seed precise non-vsp cell");
        let address = Expr::register(BitWidth::W64, "native_rcx_initial");
        machine
            .store_external(7, address.clone(), Expr::constant(BitWidth::W64, 0x55))
            .expect("disjoint external store is narrowed, not refused");
        assert!(!machine.memory_tainted);
        assert!(machine.image_reads_suppressed);
        assert!(machine.assumed_external_disjoint_from_vm_stack);
        assert_eq!(machine.memory_havocs.len(), 1);
        assert_eq!(machine.memory_havocs[0].instruction_index, 7);
        assert_eq!(machine.external_writes.len(), 1);
        assert_eq!(machine.external_writes[0].address, address);
        // The vsp-rooted VM operand-stack cell survives precisely.
        let preserved = machine
            .load_memory(stack_address(16), BitWidth::W64)
            .expect("load preserved VM-stack cell");
        assert!(matches!(preserved.kind, ExprKind::Const { value: 9 }));
        // The non-vsp cell the store could alias is discarded.
        let discarded = machine
            .load_memory(Expr::constant(BitWidth::W64, 0x1_4000_1000), BitWidth::W64)
            .expect("load discarded non-vsp cell");
        assert!(is_free_read(&discarded));
    }

    // The narrowed havoc preserves cells rooted at an allocation this frame
    // made, not only `vsp_initial`-rooted ones.
    //
    // Those discards are where the `memory_load_epoch` cascade starts: a read of
    // a discarded cell returns an opaque value, that value lands in the next
    // store's address, the address has no provable range, and the store havocs
    // again. The point of keeping them is to not start that chain.
    //
    // The concession is recorded separately, and only when a cell was actually
    // retained, so an entry with no allocation-rooted cells behaves exactly as
    // it did before and carries no new label.
    // `concat(c, vsp + k)` is still `vsp + constant`, so the bound must keep the
    // root -- that is the only reason this arm is worth having. A bound without
    // a root cannot reach `proven_store_byte_range`, which is why bounding an
    // unsupported node by its own width measured +0 before.
    #[test]
    fn a_constant_high_concat_keeps_the_root_and_shifts_the_offset() {
        let low = Expr::binary(
            BinaryOp::Add,
            Expr::register(BitWidth::W32, "vsp_initial"),
            Expr::constant(BitWidth::W32, 8),
        )
        .expect("same-width pointer expression");
        let address = Expr::concat(Expr::constant(BitWidth::W32, 3), low).expect("concat");
        let bounds = address_bounds(&address).expect("bounded");
        // The halves do not overlap, so the value is (3 << 32) + vsp + 8.
        let shifted = 3i64 << 32;
        assert_eq!(bounds.lo, shifted + 8);
        assert_eq!(bounds.hi, shifted + 8);
        assert!(bounds.root.is_some(), "the root must survive the concat");
    }

    // A non-constant high half is not a constant displacement, so it must still
    // refuse rather than invent a bound from the low half alone.
    #[test]
    fn a_symbolic_high_concat_is_not_bounded() {
        let address = Expr::concat(
            Expr::register(BitWidth::W32, "native_rax_initial"),
            Expr::register(BitWidth::W32, "vsp_initial"),
        )
        .expect("concat");
        assert!(address_bounds(&address).is_none());
    }

    #[test]
    fn narrowed_havoc_preserves_fresh_allocation_cells_under_its_own_label() {
        let mut machine = default_machine();
        let allocation = Expr::intrinsic(
            BitWidth::W64,
            "fresh_allocation_return_epoch_1".to_owned(),
            Vec::new(),
        );
        assert!(is_fresh_allocation_root(&allocation));
        let allocation_cell = Expr::binary(
            BinaryOp::Add,
            allocation,
            Expr::constant(BitWidth::W64, 24),
        )
        .expect("same-width pointer expression");
        assert_eq!(fresh_allocation_offset(&allocation_cell), Some(24));
        machine
            .store_memory(&allocation_cell, &Expr::constant(BitWidth::W64, 0x77))
            .expect("seed a cell in this frame's own allocation");
        machine
            .store_memory(&stack_address(16), &Expr::constant(BitWidth::W64, 9))
            .expect("seed a precise VM-stack cell");
        machine
            .store_memory(
                &Expr::constant(BitWidth::W64, 0x1_4000_1000),
                &Expr::constant(BitWidth::W64, 0x33),
            )
            .expect("seed a cell rooted at neither region");

        // An opaque, native-register-rooted address: no vsp dependency, so it
        // takes the narrowing rather than the broad havoc.
        let address = Expr::register(BitWidth::W64, "native_rbx_initial");
        machine
            .store_external(7, address, Expr::constant(BitWidth::W64, 0x55))
            .expect("unproven store is over-approximated, not refused");

        assert!(machine.assumed_external_disjoint_from_vm_stack);
        assert!(machine.assumed_external_disjoint_from_fresh_allocations);

        let kept_allocation = machine
            .load_memory(allocation_cell, BitWidth::W64)
            .expect("load the preserved allocation cell");
        assert!(
            matches!(kept_allocation.kind, ExprKind::Const { value: 0x77 }),
            "allocation-rooted cell must survive: {kept_allocation:?}"
        );
        let kept_stack = machine
            .load_memory(stack_address(16), BitWidth::W64)
            .expect("load the preserved VM-stack cell");
        assert!(matches!(kept_stack.kind, ExprKind::Const { value: 9 }));
        // A cell rooted at neither region is still discarded.
        let discarded = machine
            .load_memory(Expr::constant(BitWidth::W64, 0x1_4000_1000), BitWidth::W64)
            .expect("load the discarded cell");
        assert!(is_free_read(&discarded), "must not survive: {discarded:?}");
    }

    // With no allocation-rooted cell present, nothing changed and no new label
    // is taken -- the guard against a concession recorded wider than it is used.
    #[test]
    fn narrowed_havoc_takes_no_allocation_label_when_there_is_nothing_to_keep() {
        let mut machine = default_machine();
        machine
            .store_memory(&stack_address(16), &Expr::constant(BitWidth::W64, 9))
            .expect("seed a precise VM-stack cell");
        machine
            .store_external(
                3,
                Expr::register(BitWidth::W64, "native_rbx_initial"),
                Expr::constant(BitWidth::W64, 0x55),
            )
            .expect("unproven store is over-approximated, not refused");
        assert!(machine.assumed_external_disjoint_from_vm_stack);
        assert!(!machine.assumed_external_disjoint_from_fresh_allocations);
    }

    // (b) An external store whose address still references the VM stack pointer
    // could alias the VM's private operand stack, so no assumption applies: the
    // broad, assumption-free havoc discards all tracked cells and taints memory.
    // No false preservation — the previously-precise vsp cell reads back as a
    // fresh unknown, never the stale pre-store value.
    #[test]
    fn external_store_that_may_reach_vm_stack_broadly_havocs() {
        let mut machine = default_machine();
        machine
            .store_memory(&stack_address(16), &Expr::constant(BitWidth::W64, 9))
            .expect("seed precise VM-stack cell");
        // `vsp_initial + vsp_initial` references the VM stack but is not a proven
        // region pointer (coefficient 2), so it reaches the havoc still carrying
        // a vsp dependency.
        let address = Expr::binary(
            BinaryOp::Add,
            Expr::register(BitWidth::W64, "vsp_initial"),
            Expr::register(BitWidth::W64, "vsp_initial"),
        )
        .expect("same-width pointer expression");
        assert!(!is_proven_region_pointer(&address));
        machine
            .store_external(3, address.clone(), Expr::constant(BitWidth::W64, 0x55))
            .expect("unproven store is over-approximated, not refused");
        assert!(machine.memory_tainted);
        assert!(!machine.image_reads_suppressed);
        assert!(!machine.assumed_external_disjoint_from_vm_stack);
        assert_eq!(machine.memory_havocs.len(), 1);
        assert_eq!(machine.external_writes.len(), 1);
        assert_eq!(machine.external_writes[0].address, address);
        let reloaded = machine
            .load_memory(stack_address(16), BitWidth::W64)
            .expect("load");
        assert!(
            is_free_read(&reloaded),
            "stale pre-store value must not survive the havoc: {reloaded:?}"
        );
    }

    // A store whose address is data read from the VM frame (vsp_initial only
    // inside an untracked load's address) is a frame pointer only if one was lost
    // or escaped. Under the opt-in flag it takes the labeled narrowing instead of
    // tainting; without the flag, or once a frame pointer is lost, it still taints.
    #[test]
    fn loaded_data_store_address_takes_labeled_narrowing_only_when_opted_in() {
        let loaded = Expr::intrinsic(
            BitWidth::W64,
            "memory_load_epoch_3".to_owned(),
            vec![stack_address(192)],
        );
        let address = Expr::binary(
            BinaryOp::Add,
            loaded,
            Expr::register(BitWidth::W64, "native_rbx_initial"),
        )
        .expect("same-width pointer expression");
        assert!(!is_frame_pointer_value(&address));
        let limits = SymbolicExecutionLimits {
            assume_loaded_store_addresses_disjoint: true,
            ..SymbolicExecutionLimits::default()
        };
        let machine = |limits| {
            SymbolicMachine::new(&SymbolicVmInitialState::default(), None, limits)
                .expect("default state")
        };

        let mut off = machine(SymbolicExecutionLimits::default());
        off.store_external(1, address.clone(), Expr::constant(BitWidth::W8, 1))
            .expect("over-approximated");
        assert!(off.memory_tainted);
        assert!(!off.assumed_external_disjoint_from_vm_stack);

        let mut on = machine(limits);
        on.store_memory(&stack_address(16), &Expr::constant(BitWidth::W64, 9))
            .expect("seed");
        on.store_external(1, address.clone(), Expr::constant(BitWidth::W8, 1))
            .expect("narrowed");
        assert!(!on.memory_tainted);
        assert!(on.assumed_external_disjoint_from_vm_stack);
        let kept = on.load_memory(stack_address(16), BitWidth::W64).expect("load");
        assert!(matches!(kept.kind, ExprKind::Const { value: 9 }));

        let mut lost = machine(limits);
        lost.vm_stack_pointer_lost = true;
        lost.store_external(1, address, Expr::constant(BitWidth::W8, 1))
            .expect("over-approximated");
        assert!(lost.memory_tainted);
        assert!(!lost.assumed_external_disjoint_from_vm_stack);

        let indexed = Expr::binary(
            BinaryOp::Add,
            Expr::register(BitWidth::W64, "native_rbx_initial"),
            stack_address(216),
        )
        .expect("same-width pointer expression");
        let mut array = machine(limits);
        array
            .store_external(1, indexed, Expr::constant(BitWidth::W64, 1))
            .expect("over-approximated");
        assert!(array.memory_tainted, "a vsp-rooted indexed store stays refused");
    }

    // The frame address is non-escaping while it only flows into VM-stack cells
    // and the native register file; storing it to a non-frame address or handing
    // it to a native call marks it escaped, and in-frame writes report their top.
    #[test]
    fn vm_stack_address_escape_is_tracked_per_store() {
        let mut machine = default_machine();
        machine
            .store_stack(-8, &stack_address(24))
            .expect("frame pointer kept inside the frame");
        machine
            .store_external(
                0,
                Expr::register(BitWidth::W64, "native_rcx_initial"),
                Expr::constant(BitWidth::W64, 0x55),
            )
            .expect("benign external store");
        assert!(!machine.vm_stack_address_escapes);
        assert_eq!(machine.vm_stack_store_hi, Some(0));
        machine
            .store_external(
                1,
                Expr::constant(BitWidth::W64, 0x1_4000_2000),
                stack_address(24),
            )
            .expect("store of the frame pointer to a global");
        assert!(machine.vm_stack_address_escapes);
    }

    // Data read from the frame is not a frame pointer: re-storing it in the
    // frame or handing it out does not escape, while the frame pointer itself
    // (even after a round trip through a tracked cell) still does.
    #[test]
    fn frame_data_is_not_a_frame_pointer_but_the_pointer_still_escapes() {
        let global = || Expr::constant(BitWidth::W64, 0x1_4000_2000);
        let mut machine = default_machine();
        let data = machine.load_stack(16, BitWidth::W64).expect("untracked frame read");
        assert!(depends_on_vsp_root(&data));
        assert!(!is_frame_pointer_value(&data));
        machine.store_stack(-8, &data).expect("data re-stored in-frame");
        machine.store_external(0, global(), data.clone()).expect("data stored out");
        assert!(!machine.vm_stack_address_escapes);
        assert!(!machine.vm_stack_pointer_lost);
        assert!(machine.vm_stack_derived_escapes);

        let mut machine = default_machine();
        machine.store_stack(-8, &stack_address(24)).expect("pointer in-frame");
        assert!(machine.frame_pointer_in_memory && !machine.vm_stack_pointer_lost);
        let reloaded = machine.load_stack(-8, BitWidth::W64).expect("tracked reload");
        machine.store_external(0, global(), reloaded).expect("pointer stored out");
        assert!(machine.vm_stack_address_escapes);

        // Discarding a tracked cell that held a frame pointer is recorded, since
        // an untracked reload could then return that pointer as "data".
        let mut machine = default_machine();
        machine.store_stack(-8, &stack_address(24)).expect("pointer in-frame");
        machine
            .store_external(
                0,
                Expr::register(BitWidth::W64, "native_rcx_initial"),
                Expr::constant(BitWidth::W64, 1),
            )
            .expect("external store drops non-vsp cells only");
        assert!(!machine.vm_stack_pointer_lost);
        machine.retain_memory(|_| false);
        assert!(machine.vm_stack_pointer_lost);

        let mut machine = default_machine();
        let mixed = Expr::binary(BinaryOp::Add, data, stack_address(8)).expect("data + pointer");
        assert!(is_frame_pointer_value(&mixed), "uncertain values count as pointers");
        machine.store_external(0, global(), mixed).expect("store");
        assert!(machine.vm_stack_address_escapes);
    }

    #[test]
    fn concrete_null_memory_load_stops_before_inventing_a_value() {
        let initial = SymbolicVmInitialState {
            memory_values: vec![(stack_address(0), Expr::constant(BitWidth::W64, 0))],
            ..SymbolicVmInitialState::default()
        };
        let error = execute_symbolic_linear_block_with_initial_state(
            &[instruction(
                VmHandlerSemantic::MemoryLoad,
                BitWidth::W8,
                None,
                Some(7),
            )],
            None,
            SymbolicExecutionLimits::default(),
            &initial,
            None,
        )
        .expect_err("null dereference is an exception boundary, not an opaque load");

        assert_eq!(error.instruction_index, 0);
        assert_eq!(error.semantic, VmHandlerSemantic::MemoryLoad);
        assert!(error.message.contains("requires exception-flow analysis"));
    }

    /// A static image overlaid with an observed-runtime seed. `read_byte` is the
    /// sound static byte; `observed_byte` is the one-run seed and flips `seeded`
    /// when it is actually consulted (mirrors `ObservedImage`).
    #[derive(Default)]
    struct SeededMemorySource {
        image: BTreeMap<u64, u8>,
        observed: BTreeMap<u64, u8>,
        seeded: std::cell::Cell<bool>,
    }

    impl SeededMemorySource {
        fn image_write(&mut self, address: u64, bytes: &[u8]) {
            for (offset, byte) in bytes.iter().copied().enumerate() {
                self.image.insert(address + offset as u64, byte);
            }
        }

        fn observed_write(&mut self, address: u64, bytes: &[u8]) {
            for (offset, byte) in bytes.iter().copied().enumerate() {
                self.observed.insert(address + offset as u64, byte);
            }
        }
    }

    impl SymbolicMemorySource for SeededMemorySource {
        fn read_byte(&self, address: Va) -> Option<u8> {
            self.image.get(&address.get()).copied()
        }

        fn observed_byte(&self, address: Va) -> Option<u8> {
            let byte = self.observed.get(&address.get()).copied()?;
            self.seeded.set(true);
            Some(byte)
        }
    }

    const SWS_SLOT_VA: u64 = 0x1_8003_B900;
    const OBSERVED_PTR: u64 = 0x1_4000_9000;

    #[test]
    fn observed_seed_resolves_a_provably_null_slot_read_and_labels_it() {
        // The slot is zero in the static image (the documented null-read stall)
        // but runtime-initialized; the observed seed records its real value.
        let mut source = SeededMemorySource::default();
        source.image_write(SWS_SLOT_VA, &[0_u8; 8]);
        source.observed_write(SWS_SLOT_VA, &OBSERVED_PTR.to_le_bytes());

        let initial = SymbolicVmInitialState::default();
        let mut machine =
            SymbolicMachine::new(&initial, Some(&source), SymbolicExecutionLimits::default())
                .expect("machine");
        let value = machine
            .load_memory(Expr::constant(BitWidth::W64, SWS_SLOT_VA), BitWidth::W64)
            .expect("seeded read resolves");
        assert_eq!(value, Expr::constant(BitWidth::W64, OBSERVED_PTR));
        assert!(source.seeded.get(), "the observed seed must be labeled as used");
    }

    #[test]
    fn without_a_seed_the_null_slot_read_stays_zero_and_unlabeled() {
        // Same zero slot, no observed seed: the read stays the sound static zero
        // and nothing is labeled (so a later dereference still fails closed).
        let mut source = SeededMemorySource::default();
        source.image_write(SWS_SLOT_VA, &[0_u8; 8]);

        let initial = SymbolicVmInitialState::default();
        let mut machine =
            SymbolicMachine::new(&initial, Some(&source), SymbolicExecutionLimits::default())
                .expect("machine");
        let value = machine
            .load_memory(Expr::constant(BitWidth::W64, SWS_SLOT_VA), BitWidth::W64)
            .expect("zero image read");
        assert_eq!(value, Expr::constant(BitWidth::W64, 0));
        assert!(!source.seeded.get());
    }

    #[test]
    fn observed_seed_never_overrides_a_genuinely_resolvable_nonzero_read() {
        // A non-zero static byte is genuinely resolvable; the seed must not touch
        // it. This keeps the sound read unchanged even when a seed is present.
        let mut source = SeededMemorySource::default();
        source.image_write(SWS_SLOT_VA, &0xDEAD_BEEF_u32.to_le_bytes());
        source.observed_write(SWS_SLOT_VA, &0x1111_2222_u32.to_le_bytes());

        let initial = SymbolicVmInitialState::default();
        let mut machine =
            SymbolicMachine::new(&initial, Some(&source), SymbolicExecutionLimits::default())
                .expect("machine");
        let value = machine
            .load_memory(Expr::constant(BitWidth::W64, SWS_SLOT_VA), BitWidth::W32)
            .expect("image read");
        assert_eq!(value, Expr::constant(BitWidth::W32, 0xDEAD_BEEF));
        assert!(!source.seeded.get());
    }

    #[test]
    fn seed_lets_a_null_pointer_chase_continue_past_the_stall() {
        // End to end: load the slot pointer, then dereference it. The slot reads
        // zero from the static image, so the second load would dereference null.
        let instructions = [
            instruction(VmHandlerSemantic::MemoryLoad, BitWidth::W64, None, Some(0)),
            instruction(VmHandlerSemantic::MemoryLoad, BitWidth::W64, None, Some(0)),
        ];
        let initial = SymbolicVmInitialState {
            memory_values: vec![(
                stack_address(0),
                Expr::constant(BitWidth::W64, SWS_SLOT_VA),
            )],
            ..SymbolicVmInitialState::default()
        };

        // Fail-closed without a seed: the slot reads zero, the dereference nulls.
        let mut unseeded = SeededMemorySource::default();
        unseeded.image_write(SWS_SLOT_VA, &[0_u8; 8]);
        let error = execute_symbolic_linear_block_with_initial_state(
            &instructions,
            None,
            SymbolicExecutionLimits::default(),
            &initial,
            Some(&unseeded),
        )
        .expect_err("null pointer chase must fail closed");
        assert_eq!(error.instruction_index, 1);
        assert!(error.message.contains("requires exception-flow analysis"));
        assert!(!unseeded.seeded.get());

        // With the observed seed the slot yields a non-null pointer and recovery
        // continues through the dereference.
        let mut seeded = SeededMemorySource::default();
        seeded.image_write(SWS_SLOT_VA, &[0_u8; 8]);
        seeded.observed_write(SWS_SLOT_VA, &OBSERVED_PTR.to_le_bytes());
        let report = execute_symbolic_linear_block_with_initial_state(
            &instructions,
            None,
            SymbolicExecutionLimits::default(),
            &initial,
            Some(&seeded),
        )
        .expect("seeded pointer chase continues");
        assert_eq!(report.executed_instructions, 2);
        assert!(seeded.seeded.get(), "the resolved read must be labeled observed");
    }

    #[test]
    fn fresh_allocation_roots_form_isolated_writable_regions() {
        let root = Expr::intrinsic(
            BitWidth::W64,
            "fresh_allocation_return_epoch_7",
            vec![Expr::constant(BitWidth::W64, 0x150)],
        );
        let address = add_pointer_offset(root.clone(), 24).expect("valid fresh-region pointer");
        assert!(is_proven_region_pointer(&address));
        assert_eq!(
            canonicalize_pointer(address),
            add_pointer_offset(root, 24).expect("valid expected pointer")
        );

        let other = Expr::intrinsic(
            BitWidth::W64,
            "fresh_allocation_return_epoch_8",
            vec![Expr::constant(BitWidth::W64, 0x80)],
        );
        let mixed = Expr::binary(
            BinaryOp::Add,
            add_pointer_offset(
                Expr::intrinsic(BitWidth::W64, "fresh_allocation_return_epoch_7", Vec::new()),
                0,
            )
            .expect("valid first root"),
            other,
        )
        .expect("same-width pointer expression");
        assert!(!is_proven_region_pointer(&mixed));
    }

    #[test]
    fn canonicalizes_mba_obscured_stack_pointer() {
        let constant =
            |value: i64| Expr::constant(BitWidth::W64, u64::from_ne_bytes(value.to_ne_bytes()));
        let add = |left: Expr, right: Expr| {
            Expr::binary(BinaryOp::Add, left, right).expect("same-width addition")
        };
        let identical = |op: BinaryOp, value: Expr| {
            Expr::binary(op, value.clone(), value).expect("same-width idempotent expression")
        };

        let base = add(Expr::register(BitWidth::W64, "vsp_initial"), constant(-8));
        let first = add(
            identical(BinaryOp::And, Expr::unary(UnaryOp::Not, base)),
            constant(64),
        );
        let second = add(
            identical(BinaryOp::And, Expr::unary(UnaryOp::Not, first)),
            constant(-72),
        );
        let third = add(
            identical(BinaryOp::Or, Expr::unary(UnaryOp::Not, second)),
            constant(1_608),
        );
        let obscured = add(
            identical(BinaryOp::And, Expr::unary(UnaryOp::Not, third)),
            constant(1_688),
        );

        assert_eq!(pointer_offset(&obscured), Some(-64));
        assert_eq!(
            canonicalize_pointer(obscured),
            add(Expr::register(BitWidth::W64, "vsp_initial"), constant(-64))
        );
    }

    #[test]
    fn unknown_loads_are_versioned_across_writes() {
        let mut machine = SymbolicMachine::new(
            &SymbolicVmInitialState::default(),
            None,
            SymbolicExecutionLimits::default(),
        )
        .expect("default state");
        let address = Expr::register(BitWidth::W64, "unknown_pointer");
        let before = machine
            .load_memory(address.clone(), BitWidth::W64)
            .expect("symbolic load");
        machine
            .store_stack(0, &Expr::constant(BitWidth::W64, 1))
            .expect("proven stack write");
        let after = machine
            .load_memory(address, BitWidth::W64)
            .expect("versioned symbolic load");
        assert_ne!(before, after);
    }

    fn default_machine() -> SymbolicMachine<'static> {
        SymbolicMachine::new(
            &SymbolicVmInitialState::default(),
            None,
            SymbolicExecutionLimits::default(),
        )
        .expect("default state")
    }

    fn is_free_read(value: &Expr) -> bool {
        value
            .intrinsic_dependencies()
            .iter()
            .any(|name| name.starts_with("memory_load_epoch_"))
    }

    #[test]
    fn exact_and_narrower_reads_resolve_to_the_prior_store() {
        let mut machine = default_machine();
        let stored = Expr::constant(BitWidth::W64, 0x1122_3344_5566_7788);
        machine
            .store_memory(&stack_address(16), &stored)
            .expect("store");
        machine
            .store_memory(&stack_address(64), &Expr::constant(BitWidth::W64, 9))
            .expect("unrelated store");
        let exact = machine
            .load_memory(stack_address(16), BitWidth::W64)
            .expect("load");
        assert!(matches!(
            exact.kind,
            ExprKind::Const {
                value: 0x1122_3344_5566_7788
            }
        ));
        let high = machine
            .load_memory(stack_address(20), BitWidth::W32)
            .expect("load");
        assert!(matches!(high.kind, ExprKind::Const { value: 0x1122_3344 }));
    }

    #[test]
    fn non_matching_or_partially_overlapping_reads_stay_free() {
        let mut machine = default_machine();
        machine
            .store_memory(
                &stack_address(16),
                &Expr::constant(BitWidth::W32, 0xAABB_CCDD),
            )
            .expect("store");
        let other_base =
            add_pointer_offset(Expr::register(BitWidth::W64, "other"), 16).expect("add");
        assert!(is_free_read(
            &machine
                .load_memory(other_base, BitWidth::W32)
                .expect("load")
        ));
        assert!(is_free_read(
            &machine
                .load_memory(stack_address(24), BitWidth::W64)
                .expect("load")
        ));
        let straddle = machine
            .load_memory(stack_address(16), BitWidth::W64)
            .expect("load");
        assert!(is_free_read(&straddle));
        assert!(!matches!(straddle.kind, ExprKind::Const { .. }));
    }

    #[test]
    fn reads_after_a_havocing_native_call_do_not_see_prior_stores() {
        let initial = SymbolicVmInitialState {
            memory_values: vec![
                (stack_address(0), Expr::constant(BitWidth::W64, 0x1000)),
                (stack_address(8), Expr::constant(BitWidth::W64, 0x2000)),
                (stack_address(128), Expr::constant(BitWidth::W64, 0xDEAD)),
            ],
            ..SymbolicVmInitialState::default()
        };
        let report = execute_symbolic_native_call_boundary_with_initial_state(
            Va(0x1000),
            &NativeCallSelectorOutcome {
                selector: 2,
                vsp_delta: 8,
                target_stack_offset: 0,
                register_argument_offsets: vec![8],
                stack_argument_offsets: Vec::new(),
            },
            SymbolicExecutionLimits::default(),
            &initial,
            None,
        )
        .expect("native call");
        let mut machine = SymbolicMachine::new(
            &report.continuation_state(),
            None,
            SymbolicExecutionLimits::default(),
        )
        .expect("continuation");
        assert!(is_free_read(
            &machine
                .load_memory(stack_address(128), BitWidth::W64)
                .expect("load")
        ));
    }

    #[test]
    fn opaque_native_call_havocs_state_and_places_a_fresh_result() {
        let base = Expr::register(BitWidth::W64, "vsp_initial");
        let address =
            |offset| add_pointer_offset(base.clone(), offset).expect("bounded test stack address");
        let mut initial = SymbolicVmInitialState {
            memory_values: vec![
                (address(0), Expr::constant(BitWidth::W64, 0x1000)),
                (address(8), Expr::constant(BitWidth::W64, 0x2000)),
                (address(16), Expr::constant(BitWidth::W64, 0x3000)),
                (address(128), Expr::constant(BitWidth::W64, 0xDEAD)),
            ],
            ..SymbolicVmInitialState::default()
        };
        initial
            .context_values
            .insert(0, Expr::constant(BitWidth::W64, 0xBEEF));
        let report = execute_symbolic_native_call_boundary_with_initial_state(
            Va(0x1000),
            &NativeCallSelectorOutcome {
                selector: 2,
                vsp_delta: 16,
                target_stack_offset: 0,
                register_argument_offsets: vec![8, 16],
                stack_argument_offsets: Vec::new(),
            },
            SymbolicExecutionLimits::default(),
            &initial,
            None,
        )
        .expect("proven native-call outcome must transfer symbolically");

        assert_eq!(report.final_vsp_offset, Some(16));
        assert_eq!(report.memory_epoch, 2);
        assert!(!report.memory_tainted);
        assert_eq!(report.memory_havocs.len(), 1);
        assert_eq!(report.native_calls.len(), 1);
        assert_eq!(report.next_native_call_epoch, 1);
        assert_eq!(report.memory_values.len(), 8);
        assert!(
            report
                .memory_values
                .iter()
                .any(|(_, value)| format!("{value:?}").contains("native_call_return_epoch_0"))
        );
        assert!(format!("{:?}", report.context_cells[&0]).contains("context_after_native_call_0"));
        assert!(
            !report
                .memory_values
                .iter()
                .any(|(_, value)| format!("{value:?}").contains("57005"))
        );
    }

    fn run(instructions: &[DecodedVmInstruction]) -> Result<SymbolicVmExecutionReport, String> {
        execute_symbolic_linear_block(instructions, None, SymbolicExecutionLimits::default())
            .map_err(|error| error.message)
    }

    #[test]
    fn narrow_vsp_push_stores_only_the_low_bytes() {
        let report = run(&[instruction(
            VmHandlerSemantic::Push,
            BitWidth::W16,
            None,
            Some(-2),
        )])
        .expect("narrow push executes");
        assert_eq!(report.final_vsp_offset, Some(-2));
        assert_eq!(report.memory_values.len(), 2);

        let error = run(&[instruction(
            VmHandlerSemantic::Push,
            BitWidth::W16,
            None,
            Some(-8),
        )])
        .expect_err("a qword delta contradicts a word push");
        assert!(error.contains("does not match"), "{error}");
    }

    #[test]
    fn narrow_vsp_reload_is_not_executed_symbolically() {
        let error = run(&[instruction(
            VmHandlerSemantic::Pop,
            BitWidth::W32,
            None,
            None,
        )])
        .expect_err("partial VSP reload must not be guessed");
        assert!(error.contains("partial-width VSP reload"), "{error}");
    }

    #[test]
    fn byte_shift_in_word_slots_computes_a_zero_extended_result() {
        let mut shift = instruction(
            VmHandlerSemantic::ShiftRightLogical,
            BitWidth::W16,
            None,
            Some(-6),
        );
        shift.input_width = Some(BitWidth::W8);
        shift.resize = Some(ResizeOp::ZeroExtend);
        let report = run(&[
            instruction(
                VmHandlerSemantic::ConstantPush,
                BitWidth::W16,
                Some(1),
                Some(-2),
            ),
            instruction(
                VmHandlerSemantic::ConstantPush,
                BitWidth::W16,
                Some(0x1281),
                Some(-2),
            ),
            shift,
            instruction(
                VmHandlerSemantic::VirtualRegisterWrite,
                BitWidth::W64,
                Some(0x20),
                Some(8),
            ),
            instruction(
                VmHandlerSemantic::VirtualRegisterWrite,
                BitWidth::W16,
                Some(0x10),
                Some(2),
            ),
        ])
        .expect("byte shift executes");
        let cell = &report.context_cells[&0x10];
        let ExprKind::Concat { low, .. } = &cell.kind else {
            panic!("context cell must keep its untouched upper bytes: {cell:?}");
        };
        let ExprKind::Concat { low: word, .. } = &low.kind else {
            panic!("context cell must keep its untouched upper word: {cell:?}");
        };
        assert_eq!(**word, Expr::constant(BitWidth::W16, 0x40));
        assert_eq!(report.final_vsp_offset, Some(0));
    }

    #[test]
    fn wide_multiply_and_unflagged_divide_use_split_qword_results() {
        let pushes = |count: usize| {
            (0..count)
                .map(|index| {
                    instruction(
                        VmHandlerSemantic::ConstantPush,
                        BitWidth::W64,
                        Some(index as u64 + 3),
                        Some(-8),
                    )
                })
                .collect::<Vec<_>>()
        };
        let mut multiply = pushes(2);
        multiply.push(instruction(
            VmHandlerSemantic::SignedMultiply,
            BitWidth::W64,
            None,
            Some(-8),
        ));
        let report = run(&multiply).expect("qword signed multiply executes");
        assert_eq!(report.final_vsp_offset, Some(-24));

        let mut divide = pushes(3);
        divide.push(instruction(
            VmHandlerSemantic::UnsignedDivide,
            BitWidth::W64,
            None,
            Some(8),
        ));
        let report = run(&divide).expect("unflagged qword divide executes");
        assert_eq!(report.final_vsp_offset, Some(-16));

        let mut wrong = pushes(3);
        wrong.push(instruction(
            VmHandlerSemantic::UnsignedDivide,
            BitWidth::W64,
            None,
            Some(4),
        ));
        assert!(run(&wrong).is_err());
    }

    #[test]
    fn cpuid_and_popfq_follow_their_stack_layouts() {
        let mut block = vec![
            instruction(
                VmHandlerSemantic::ConstantPush,
                BitWidth::W32,
                Some(1),
                Some(-4),
            ),
            instruction(
                VmHandlerSemantic::ConstantPush,
                BitWidth::W32,
                Some(0),
                Some(-4),
            ),
            instruction(VmHandlerSemantic::Cpuid, BitWidth::W32, None, Some(-8)),
        ];
        let report = run(&block).expect("cpuid executes");
        assert_eq!(report.final_vsp_offset, Some(-16));

        block.truncate(0);
        block.push(instruction(
            VmHandlerSemantic::ConstantPush,
            BitWidth::W64,
            Some(0x246),
            Some(-8),
        ));
        block.push(instruction(
            VmHandlerSemantic::PopFlags,
            BitWidth::W64,
            None,
            Some(8),
        ));
        let report = run(&block).expect("popfq executes");
        assert_eq!(report.final_vsp_offset, Some(0));
        assert!(format!("{:?}", report.final_flags).contains("x86_popfq_1"));
    }

    #[test]
    fn leaf_only_cpuid_pushes_four_opaque_results_over_an_env_subleaf() {
        // A single leaf dword on the stack, net VSP delta -12: the executor reads
        // the leaf, synthesizes an environment subleaf, and writes four opaque
        // dword results. The subleaf being unknown keeps the results opaque.
        let block = vec![
            instruction(
                VmHandlerSemantic::ConstantPush,
                BitWidth::W32,
                Some(1),
                Some(-4),
            ),
            instruction(VmHandlerSemantic::Cpuid, BitWidth::W32, None, Some(-12)),
        ];
        let report = run(&block).expect("leaf-only cpuid executes");
        assert_eq!(report.final_vsp_offset, Some(-16));
        let dump = format!("{:?}", report.memory_values);
        assert!(dump.contains("x86_cpuid_eax_"), "{dump}");
        assert!(dump.contains("x86_cpuid_subleaf_"), "{dump}");
    }

    #[test]
    fn cpuid_with_unsupported_delta_fails_closed() {
        // Only the proven -8 and -12 layouts execute; any other recovered delta
        // must error rather than guess a stack layout.
        let block = vec![
            instruction(
                VmHandlerSemantic::ConstantPush,
                BitWidth::W32,
                Some(1),
                Some(-4),
            ),
            instruction(VmHandlerSemantic::Cpuid, BitWidth::W32, None, Some(-16)),
        ];
        assert!(run(&block).is_err());
    }

    fn read_only_image_machine(source: &dyn SymbolicMemorySource) -> SymbolicMachine<'_> {
        SymbolicMachine::new(
            &SymbolicVmInitialState::default(),
            Some(source),
            SymbolicExecutionLimits::default(),
        )
        .expect("default state")
    }

    #[test]
    fn constant_read_only_load_folds_to_image_bytes() {
        let bytes = crate::immutable_image::tests::two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let source = crate::ReadOnlyImage(&pe);
        let mut machine = read_only_image_machine(&source);
        let address = Expr::constant(BitWidth::W64, crate::immutable_image::tests::RDATA_VA);
        let value = machine.load_memory(address, BitWidth::W32).expect("load");
        assert!(matches!(value.kind, ExprKind::Const { value: 0xDEAD_BEEF }));
    }

    #[test]
    fn writable_section_and_unknown_address_loads_do_not_fold() {
        let bytes = crate::immutable_image::tests::two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let source = crate::ReadOnlyImage(&pe);
        let mut machine = read_only_image_machine(&source);
        for address in [
            Expr::constant(BitWidth::W64, crate::immutable_image::tests::DATA_VA),
            Expr::register(BitWidth::W64, "unknown_pointer"),
        ] {
            let value = machine.load_memory(address, BitWidth::W32).expect("load");
            assert!(!matches!(value.kind, ExprKind::Const { .. }), "{value:?}");
            assert!(
                value
                    .intrinsic_dependencies()
                    .iter()
                    .any(|name| name.starts_with("memory_load_epoch_"))
            );
        }
    }

    #[test]
    fn suppressed_read_only_page_is_served_only_when_enabled() {
        let bytes = crate::immutable_image::tests::two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let source = crate::ReadOnlyImage(&pe);
        let rdata = Expr::constant(BitWidth::W64, crate::immutable_image::tests::RDATA_VA);
        let data = Expr::constant(BitWidth::W64, crate::immutable_image::tests::DATA_VA);
        let suppressed = SymbolicVmInitialState {
            image_reads_suppressed: true,
            protection_may_have_changed: true,
            ..SymbolicVmInitialState::default()
        };
        let mut off = SymbolicMachine::new(
            &suppressed,
            Some(&source),
            SymbolicExecutionLimits::default(),
        )
        .expect("state");
        let value = off.load_memory(rdata.clone(), BitWidth::W32).expect("load");
        assert!(!matches!(value.kind, ExprKind::Const { .. }));
        assert!(!off.served_read_only_image);

        let limits = SymbolicExecutionLimits {
            serve_read_only_when_suppressed: true,
            ..SymbolicExecutionLimits::default()
        };
        let mut on =
            SymbolicMachine::new(&suppressed, Some(&source), limits).expect("state");
        let value = on.load_memory(rdata, BitWidth::W32).expect("load");
        assert!(matches!(value.kind, ExprKind::Const { value: 0xDEAD_BEEF }));
        assert!(on.served_read_only_image);

        let mut writable =
            SymbolicMachine::new(&suppressed, Some(&source), limits).expect("state");
        let value = writable.load_memory(data, BitWidth::W32).expect("load");
        assert!(!matches!(value.kind, ExprKind::Const { .. }));
        assert!(!writable.served_read_only_image);
    }

    // After an external store, a declared read-only page is served PROVEN (no
    // assumption flag) while no non-allocation call has run on the path; an
    // opaque native call (which may change protection) withdraws the proof.
    #[test]
    fn suppressed_read_only_page_is_proven_until_a_call_may_reprotect() {
        let bytes = crate::immutable_image::tests::two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let source = crate::ReadOnlyImage(&pe);
        let rdata = Expr::constant(BitWidth::W64, crate::immutable_image::tests::RDATA_VA);
        let data = Expr::constant(BitWidth::W64, crate::immutable_image::tests::DATA_VA);
        let mut machine = SymbolicMachine::new(
            &SymbolicVmInitialState::default(),
            Some(&source),
            SymbolicExecutionLimits::default(),
        )
        .expect("state");
        machine
            .store_external(
                1,
                Expr::register(BitWidth::W64, "native_rcx_initial"),
                Expr::constant(BitWidth::W64, 0x55),
            )
            .expect("external store");
        assert!(machine.image_reads_suppressed);
        let value = machine
            .load_memory(rdata.clone(), BitWidth::W32)
            .expect("load");
        assert!(matches!(value.kind, ExprKind::Const { value: 0xDEAD_BEEF }));
        assert!(!machine.served_read_only_image, "proven serve carries no assumption");
        let writable = machine.load_memory(data, BitWidth::W32).expect("load");
        assert!(!matches!(writable.kind, ExprKind::Const { .. }));

        machine.protection_may_have_changed = true;
        let value = machine.load_memory(rdata, BitWidth::W32).expect("load");
        assert!(!matches!(value.kind, ExprKind::Const { .. }), "{value:?}");
        assert!(machine.protection_may_have_changed && machine.image_reads_suppressed);
    }

    // Without a whole-image proof that protection cannot change, the in-path
    // proof alone is not enough: nothing is served unless the labeled flag is on.
    #[test]
    fn unwitnessed_image_is_never_served_as_proven() {
        struct Unwitnessed<'a, 'data>(crate::ReadOnlyImage<'a, 'data>);
        impl SymbolicMemorySource for Unwitnessed<'_, '_> {
            fn read_byte(&self, address: Va) -> Option<u8> {
                self.0.read_byte(address)
            }
            fn read_only_byte(&self, address: Va) -> Option<u8> {
                self.0.read_only_byte(address)
            }
        }
        let bytes = crate::immutable_image::tests::two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let source = Unwitnessed(crate::ReadOnlyImage(&pe));
        let rdata = Expr::constant(BitWidth::W64, crate::immutable_image::tests::RDATA_VA);
        let suppressed = SymbolicVmInitialState {
            image_reads_suppressed: true,
            ..SymbolicVmInitialState::default()
        };
        let mut machine = SymbolicMachine::new(
            &suppressed,
            Some(&source),
            SymbolicExecutionLimits::default(),
        )
        .expect("state");
        let value = machine.load_memory(rdata.clone(), BitWidth::W32).expect("load");
        assert!(!matches!(value.kind, ExprKind::Const { .. }), "{value:?}");
        let labeled = SymbolicExecutionLimits {
            serve_read_only_when_suppressed: true,
            ..SymbolicExecutionLimits::default()
        };
        let mut machine = SymbolicMachine::new(&suppressed, Some(&source), labeled).expect("state");
        let value = machine.load_memory(rdata, BitWidth::W32).expect("load");
        assert!(matches!(value.kind, ExprKind::Const { value: 0xDEAD_BEEF }));
        assert!(machine.served_read_only_image, "served only as a labeled assumption");
    }

    #[test]
    fn teb_relative_loads_stay_symbolic() {
        for offset in [0x30_u64, 0x60] {
            let initial = SymbolicVmInitialState {
                memory_values: vec![(stack_address(0), Expr::constant(BitWidth::W64, offset))],
                ..SymbolicVmInitialState::default()
            };
            let mut machine =
                SymbolicMachine::new(&initial, None, SymbolicExecutionLimits::default())
                    .expect("initial state");
            execute_instruction(
                &mut machine,
                7,
                &instruction(
                    VmHandlerSemantic::GsRelativeLoad,
                    BitWidth::W64,
                    None,
                    Some(0),
                ),
            )
            .expect("gs load");
            let value = machine.load_stack(0, BitWidth::W64).expect("result");
            assert!(
                value.intrinsic_dependencies().contains("x86_gs_load_7"),
                "gs:[{offset:#x}] must not fold: {value:?}"
            );
        }
    }
}
