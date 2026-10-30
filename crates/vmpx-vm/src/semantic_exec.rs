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
    VmSemanticMetadata, apply_native_call_effect, resolve_native_call_import,
    summarize_known_import,
};

const WIN64_ARGUMENT_REGISTERS: [&str; 4] = ["RCX", "RDX", "R8", "R9"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SymbolicExecutionLimits {
    pub max_instructions: usize,
    pub max_memory_bytes: usize,
    pub max_expression_nodes: usize,
}

impl Default for SymbolicExecutionLimits {
    fn default() -> Self {
        Self {
            max_instructions: 4_096,
            max_memory_bytes: 1_048_576,
            max_expression_nodes: 200_000,
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
    memory_tainted: bool,
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
            memory_tainted: initial.memory_tainted,
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
            {
                let byte_va = Va(value);
                if let Some(byte) = self
                    .memory_source
                    .and_then(|source| source.read_byte(byte_va))
                {
                    all_unknown = false;
                    self.resolved_image_addresses.insert(byte_va);
                    bytes.push(Expr::constant(BitWidth::W8, u64::from(byte)));
                } else {
                    bytes.push(Expr::intrinsic(
                        BitWidth::W8,
                        format!("memory_load_epoch_{}", self.memory_epoch),
                        vec![byte_address],
                    ));
                }
            } else {
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
            self.memory.insert(
                add_pointer_offset(address.clone(), i64::from(byte))
                    .map_err(|error| error.to_string())?,
                Expr::extract(BitWidth::W8, byte * 8, value.clone())
                    .map_err(|error| error.to_string())?,
            );
        }
        Ok(())
    }

    fn store_memory(&mut self, address: &Expr, value: &Expr) -> Result<(), String> {
        let address = canonicalize_pointer(address.clone());
        self.advance_memory_epoch()?;
        self.write_memory_bytes(&address, value)
    }

    fn load_external(&mut self, address: Expr, width: BitWidth) -> Result<Expr, String> {
        let address = canonicalize_pointer(address);
        if matches!(address.kind, ExprKind::Const { value: 0 }) {
            return Err(format!(
                "proven null memory read of {} byte(s) requires exception-flow analysis",
                width.bytes()
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
        if !is_proven_region_pointer(&address) {
            return Err(
                "external store address is not in a proven symbolic memory region".to_owned(),
            );
        }
        self.store_memory(&address, &value)?;
        self.external_writes.push(SymbolicMemoryWrite {
            instruction_index,
            address,
            value,
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
        let argument_capacity = outcome
            .register_argument_offsets
            .len()
            .saturating_add(outcome.stack_argument_offsets.len());
        let mut arguments = Vec::with_capacity(argument_capacity);
        let mut descriptors = Vec::with_capacity(argument_capacity);
        for (index, offset) in outcome.register_argument_offsets.iter().enumerate() {
            let value = self.load_stack(i64::from(*offset), BitWidth::W64)?;
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
                self.advance_memory_epoch()?;
                self.memory.clear();
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
            NativeCallMemoryDecision::PreserveTracked => {}
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
        let mut peak = 0_usize;
        let mut aggregate = 0_usize;
        for expression in std::iter::once(&self.vsp)
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
            )
        {
            let nodes = expression.node_count_bounded(limits.max_expression_nodes);
            peak = peak.max(nodes);
            aggregate = aggregate.saturating_add(nodes);
            if nodes > limits.max_expression_nodes || aggregate > limits.max_expression_nodes {
                return Err(format!(
                    "aggregate expressions exceed {} nodes",
                    limits.max_expression_nodes
                ));
            }
        }
        self.peak_expression_nodes = self.peak_expression_nodes.max(peak);
        Ok(())
    }
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
            let packed_result = Expr::intrinsic(
                BitWidth::W64,
                format!("x86_unsigned_divide_{width:?}_{instruction_index}").to_lowercase(),
                vec![high.clone(), low.clone(), divisor.clone()],
            );
            let remainder = Expr::extract(width, width.bits(), packed_result.clone())
                .map_err(|error| error.to_string())?;
            let quotient =
                Expr::extract(width, 0, packed_result).map_err(|error| error.to_string())?;
            let delta = i32::try_from(bytes)
                .ok()
                .and_then(|bytes| bytes.checked_sub(8))
                .ok_or_else(|| "division width exceeds VSP range".to_owned())?;
            expected_delta(instruction, delta)?;
            machine.adjust_vsp(delta)?;
            machine.store_stack(8, &remainder)?;
            machine.store_stack(i64::from(8_u32.saturating_add(bytes)), &quotient)?;
            machine.flags = Expr::intrinsic(
                BitWidth::W64,
                format!("vm_flags_unsigned_divide_{width:?}_{instruction_index}").to_lowercase(),
                vec![high, low, divisor],
            );
            let flags = machine.flags.clone();
            machine.store_stack(0, &flags)?;
        }
        VmHandlerSemantic::Mul => {
            let bytes = width.bytes();
            let multiplier = machine.load_stack(0, width)?;
            let accumulator = machine.load_stack(i64::from(bytes), width)?;
            let packed_result = Expr::intrinsic(
                BitWidth::W64,
                format!("x86_unsigned_multiply_{width:?}_{instruction_index}").to_lowercase(),
                vec![multiplier.clone(), accumulator.clone()],
            );
            let high = Expr::extract(width, width.bits(), packed_result.clone())
                .map_err(|error| error.to_string())?;
            let low = Expr::extract(width, 0, packed_result).map_err(|error| error.to_string())?;
            expected_delta(instruction, -8)?;
            machine.adjust_vsp(-8)?;
            machine.store_stack(8, &high)?;
            machine.store_stack(i64::from(8_u32.saturating_add(bytes)), &low)?;
            machine.flags = Expr::intrinsic(
                BitWidth::W64,
                format!("vm_flags_unsigned_multiply_{width:?}_{instruction_index}").to_lowercase(),
                vec![multiplier, accumulator],
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
            let value = machine.load_stack(0, width)?;
            let count = resized_to(
                width,
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
            machine.adjust_vsp(delta)?;
            machine.store_stack(8, &result)?;
            machine.flags = Expr::intrinsic(
                BitWidth::W64,
                format!("vm_flags_{:?}_{:?}", instruction.semantic, width).to_lowercase(),
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
            let delta = i32::try_from(8_u32.saturating_add(width.bytes()))
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
            expected_delta(instruction, -8)?;
            let prior_vsp = machine.vsp.clone();
            machine.adjust_vsp(-8)?;
            machine.store_stack(0, &prior_vsp)?;
        }
        VmHandlerSemantic::Pop => {
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
    let mut memory_values = machine.memory.into_iter().collect::<Vec<_>>();
    memory_values.sort_by_cached_key(|(address, _)| format!("{address:?}"));
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
    let mut memory_values = machine.memory.into_iter().collect::<Vec<_>>();
    memory_values.sort_by_cached_key(|(address, _)| format!("{address:?}"));
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

    #[test]
    fn symbolic_store_rejects_unbounded_aliases() {
        let mut machine = SymbolicMachine::new(
            &SymbolicVmInitialState::default(),
            None,
            SymbolicExecutionLimits::default(),
        )
        .expect("default state");
        machine
            .store_stack(0, &Expr::constant(BitWidth::W64, 9))
            .expect("seed precise VM stack");
        let address = Expr::register(BitWidth::W64, "unproven_pointer");
        let error = machine
            .store_external(0, address, Expr::constant(BitWidth::W64, 7))
            .expect_err("unproven alias must stop sound symbolic execution");
        assert!(error.contains("not in a proven symbolic memory region"));
        assert!(!machine.memory_tainted);
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
}
