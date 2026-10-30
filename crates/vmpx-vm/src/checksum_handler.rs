use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register};
use serde::Serialize;
use vmpx_disasm::{
    DirectJumpTraceLimits, DirectJumpTraceStopKind, DisasmError, RegisterDispatchProofLimits,
    trace_direct_jump_fragments,
};
use vmpx_pe::{PeImage, Rva, Va};

use crate::{
    BitWidth, BytecodeDirection, ChecksumAccumulatorSource, ChecksumCombine, ChecksumShift,
    ChecksumTableIndex, DispatchExtractionReport, HandlerRegisterRoles, HandlerSliceConfig,
    RecoveryStatus, RelativeDispatchCodec, TableChecksumMetadata, backward_slice_handler_trace,
    extract_relative_dispatch_codecs_with_postlude_direction,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TableChecksumHandlerReport {
    pub schema_version: u32,
    pub entry: Rva,
    pub status: RecoveryStatus,
    pub pointer_load: Option<Rva>,
    pub length_load: Option<Rva>,
    pub loop_branch: Option<Rva>,
    pub output_store: Option<Rva>,
    pub common_tail: Option<Rva>,
    pub dispatch_analysis: Option<DispatchExtractionReport>,
    pub metadata: Option<TableChecksumMetadata>,
    pub confidence: u8,
    pub evidence: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecoveredTableChecksumHandler {
    pub report: TableChecksumHandlerReport,
    pub metadata: TableChecksumMetadata,
    pub dispatch: RelativeDispatchCodec,
    pub dispatch_vip_delta: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TableChecksumHandlerAnalysis {
    pub report: TableChecksumHandlerReport,
    pub recovered: Option<RecoveredTableChecksumHandler>,
}

fn canonical(register: Register) -> Register {
    register.full_register()
}

fn register_operand(instruction: &Instruction, operand: u32) -> Option<Register> {
    (instruction.op_kind(operand) == OpKind::Register).then(|| instruction.op_register(operand))
}

fn immediate(instruction: &Instruction, operand: u32) -> Option<u64> {
    instruction.try_immediate(operand).ok()
}

fn signed_immediate(instruction: &Instruction, operand: u32) -> Option<i64> {
    immediate(instruction, operand).map(|value| i64::from_ne_bytes(value.to_ne_bytes()))
}

fn memory_displacement(instruction: &Instruction) -> i64 {
    i64::from_ne_bytes(instruction.memory_displacement64().to_ne_bytes())
}

fn memory_width(instruction: &Instruction) -> usize {
    instruction.memory_size().size()
}

fn is_memory_operand(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Memory
            | OpKind::MemorySegSI
            | OpKind::MemorySegESI
            | OpKind::MemorySegRSI
            | OpKind::MemorySegDI
            | OpKind::MemorySegEDI
            | OpKind::MemorySegRDI
    )
}

fn direct_load_from(
    instruction: &Instruction,
    base: Register,
    displacement: i64,
    width: usize,
) -> Option<Register> {
    (instruction.mnemonic() == Mnemonic::Mov
        && register_operand(instruction, 0).is_some()
        && is_memory_operand(instruction.op1_kind())
        && canonical(instruction.memory_base()) == canonical(base)
        && instruction.memory_index() == Register::None
        && memory_displacement(instruction) == displacement
        && memory_width(instruction) == width)
        .then(|| register_operand(instruction, 0))
        .flatten()
}

fn writes_register(
    factory: &mut InstructionInfoFactory,
    instruction: &Instruction,
    register: Register,
) -> bool {
    factory
        .info(instruction)
        .used_registers()
        .iter()
        .any(|used| {
            canonical(used.register()) == canonical(register)
                && matches!(
                    used.access(),
                    OpAccess::Write
                        | OpAccess::CondWrite
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                        | OpAccess::NoMemAccess
                )
        })
}

fn flags_defined(instruction: &Instruction) -> bool {
    instruction.rflags_written()
        | instruction.rflags_cleared()
        | instruction.rflags_set()
        | instruction.rflags_undefined()
        != 0
}

fn static_lea_address(pe: &PeImage<'_>, instruction: &Instruction) -> Option<(Register, Va)> {
    if instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && !is_memory_operand(instruction.op1_kind())
        && let Some(value) = immediate(instruction, 1)
    {
        let address = Va(value);
        pe.optional.image_base.checked_rva_of(address)?;
        return Some((instruction.op0_register(), address));
    }
    if instruction.mnemonic() != Mnemonic::Lea || !is_memory_operand(instruction.op1_kind()) {
        return None;
    }
    let destination = register_operand(instruction, 0)?;
    let address = if instruction.is_ip_rel_memory_operand() {
        Va(instruction.ip_rel_memory_address())
    } else if instruction.memory_base() == Register::None
        && instruction.memory_index() == Register::None
    {
        Va(instruction.memory_displacement64())
    } else {
        return None;
    };
    pe.optional.image_base.checked_rva_of(address)?;
    Some((destination, address))
}

fn table_is_file_backed(pe: &PeImage<'_>, address: Va) -> bool {
    let Some(start) = pe.optional.image_base.checked_rva_of(address) else {
        return false;
    };
    let Some(end) = start.checked_add(1023) else {
        return false;
    };
    pe.rva_to_file_offset(start).is_ok() && pe.rva_to_file_offset(end).is_ok()
}

fn report_failure(
    entry: Rva,
    pointer_load: Option<Rva>,
    length_load: Option<Rva>,
    diagnostics: Vec<String>,
) -> TableChecksumHandlerAnalysis {
    TableChecksumHandlerAnalysis {
        report: TableChecksumHandlerReport {
            schema_version: 1,
            entry,
            status: RecoveryStatus::Failed,
            pointer_load,
            length_load,
            loop_branch: None,
            output_store: None,
            common_tail: None,
            dispatch_analysis: None,
            metadata: None,
            confidence: 0,
            evidence: Vec::new(),
            diagnostics,
        },
        recovered: None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role {
    Acc,
    OrigPtr,
    OrigLen,
    StackPtr,
    Byte,
    WideIdx,
    Idx,
    Entry,
    Table(Va),
    Shifted,
    Combined,
    Keyed,
}

fn is_low_byte(register: Register) -> bool {
    register.size() == 1
        && !matches!(
            register,
            Register::AH | Register::BH | Register::CH | Register::DH
        )
}

fn written_registers(
    factory: &mut InstructionInfoFactory,
    instruction: &Instruction,
) -> Vec<Register> {
    factory
        .info(instruction)
        .used_registers()
        .iter()
        .filter(|used| {
            matches!(
                used.access(),
                OpAccess::Write
                    | OpAccess::CondWrite
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
            )
        })
        .map(|used| canonical(used.register()))
        .collect()
}

fn writes_memory(factory: &mut InstructionInfoFactory, instruction: &Instruction) -> bool {
    factory.info(instruction).used_memory().iter().any(|used| {
        matches!(
            used.access(),
            OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
        )
    })
}

fn is_rsp_slot(instruction: &Instruction, displacement: i64) -> bool {
    is_memory_operand(instruction.op0_kind())
        && canonical(instruction.memory_base()) == Register::RSP
        && instruction.memory_index() == Register::None
        && memory_displacement(instruction) == displacement
        && memory_width(instruction) == 8
}

/// Register-role tracker used to prove each native instruction on the
/// checksum paths is either part of the recurrence or a side-effect-free
/// write to a register nothing downstream depends on.
struct Scan<'a> {
    factory: InstructionInfoFactory,
    roles: BTreeMap<Register, Role>,
    zero: BTreeSet<Register>,
    protected: &'a BTreeSet<Register>,
    accumulator: Register,
}

#[derive(Default)]
struct IterationFacts {
    byte_loaded: bool,
    advanced: bool,
    counted: bool,
    table: Option<(Register, Va)>,
    iteration_xor: Option<u32>,
    byte_load: Option<Rva>,
}

impl<'a> Scan<'a> {
    fn new(protected: &'a BTreeSet<Register>, accumulator: Register) -> Self {
        Self {
            factory: InstructionInfoFactory::new(),
            roles: BTreeMap::new(),
            zero: BTreeSet::new(),
            protected,
            accumulator,
        }
    }

    fn role(&self, register: Register) -> Option<Role> {
        self.roles.get(&canonical(register)).copied()
    }

    fn set(&mut self, register: Register, role: Role) {
        self.zero.remove(&canonical(register));
        self.roles.insert(canonical(register), role);
    }

    fn clobber(&mut self, written: &[Register]) {
        for register in written {
            self.roles.remove(register);
            self.zero.remove(register);
        }
    }

    fn unique_role(&self, role: Role) -> Option<Register> {
        let mut found = self
            .roles
            .iter()
            .filter(|(_, candidate)| **candidate == role)
            .map(|(register, _)| *register);
        let first = found.next()?;
        found.next().is_none().then_some(first)
    }

    /// Validates an instruction that is not part of any recognized pattern and
    /// returns the registers it writes.
    fn generic(
        &mut self,
        instruction: &Instruction,
        accumulator_locked: bool,
    ) -> Result<Vec<Register>, String> {
        use iced_x86::FlowControl;
        if !matches!(
            instruction.flow_control(),
            FlowControl::Next | FlowControl::UnconditionalBranch
        ) {
            return Err(format!(
                "unexpected control flow `{:?}` on a checksum path",
                instruction.mnemonic()
            ));
        }
        if matches!(instruction.mnemonic(), Mnemonic::Push | Mnemonic::Pop) {
            return Err("unexpected stack operation on a checksum path".to_owned());
        }
        if writes_memory(&mut self.factory, instruction) {
            return Err(format!(
                "`{:?}` writes memory outside the recognized counters",
                instruction.mnemonic()
            ));
        }
        let written = written_registers(&mut self.factory, instruction);
        if written.contains(&Register::RSP) {
            return Err("native stack pointer is modified directly".to_owned());
        }
        if accumulator_locked && written.contains(&self.accumulator) {
            return Err("accumulator is written by an unrecognized instruction".to_owned());
        }
        self.clobber(&written);
        Ok(written)
    }

    fn violates_protection(&self, written: &[Register], exempt: &BTreeSet<Register>) -> bool {
        written
            .iter()
            .any(|register| self.protected.contains(register) && !exempt.contains(register))
    }

    /// One in-loop instruction. Returns the written registers.
    #[allow(clippy::too_many_lines)]
    fn loop_step(
        &mut self,
        instruction: &Instruction,
        rva: Rva,
        stack_mode: bool,
        pointer: Register,
        length: Register,
        facts: &mut IterationFacts,
    ) -> Result<Vec<Register>, String> {
        let acc = self.accumulator;
        let mnemonic = instruction.mnemonic();
        let op0 = register_operand(instruction, 0);
        let op1 = register_operand(instruction, 1);

        if stack_mode
            && mnemonic == Mnemonic::Mov
            && let Some(target) = op0
            && target.size() == 8
            && is_memory_operand(instruction.op1_kind())
            && canonical(instruction.memory_base()) == Register::RSP
            && instruction.memory_index() == Register::None
            && memory_displacement(instruction) == 0
            && memory_width(instruction) == 8
        {
            let written = written_registers(&mut self.factory, instruction);
            self.clobber(&written);
            self.set(target, Role::StackPtr);
            return Ok(written);
        }
        if mnemonic == Mnemonic::Movzx
            && let Some(target) = op0
            && matches!(target.size(), 4 | 8)
            && is_memory_operand(instruction.op1_kind())
            && instruction.memory_index() == Register::None
            && memory_displacement(instruction) == 0
            && memory_width(instruction) == 1
        {
            let base = canonical(instruction.memory_base());
            let wanted = if stack_mode {
                Role::StackPtr
            } else {
                Role::OrigPtr
            };
            if self.role(base) != Some(wanted) || (!stack_mode && base != pointer) {
                return Err("input byte is not loaded through the recovered pointer".to_owned());
            }
            if facts.byte_loaded {
                return Err("more than one input byte load per iteration".to_owned());
            }
            let written = written_registers(&mut self.factory, instruction);
            self.clobber(&written);
            self.set(target, Role::Byte);
            facts.byte_loaded = true;
            facts.byte_load = Some(rva);
            return Ok(written);
        }
        if mnemonic == Mnemonic::Xor
            && let (Some(left), Some(right)) = (op0, op1)
            && canonical(right) == acc
            && self.role(acc) == Some(Role::Acc)
        {
            if is_low_byte(left) && is_low_byte(right) && self.role(left) == Some(Role::Byte) {
                let written = written_registers(&mut self.factory, instruction);
                self.clobber(&written);
                self.set(left, Role::Idx);
                return Ok(written);
            }
            if left.size() == 4 && right.size() == 4 && self.role(left) == Some(Role::Byte) {
                let written = written_registers(&mut self.factory, instruction);
                self.clobber(&written);
                self.set(left, Role::WideIdx);
                return Ok(written);
            }
        }
        if mnemonic == Mnemonic::And
            && let Some(target) = op0
            && matches!(target.size(), 4 | 8)
            && immediate(instruction, 1) == Some(0xff)
            && self.role(target) == Some(Role::WideIdx)
        {
            let written = written_registers(&mut self.factory, instruction);
            self.clobber(&written);
            self.set(target, Role::Idx);
            return Ok(written);
        }
        if mnemonic == Mnemonic::Mov
            && let Some(target) = op0
            && target.size() == 4
            && is_memory_operand(instruction.op1_kind())
            && instruction.memory_index_scale() == 4
            && memory_displacement(instruction) == 0
            && memory_width(instruction) == 4
            && let Some(Role::Table(address)) = self.role(instruction.memory_base())
            && self.role(instruction.memory_index()) == Some(Role::Idx)
        {
            if facts.table.is_some() {
                return Err("more than one table lookup per iteration".to_owned());
            }
            facts.table = Some((canonical(instruction.memory_base()), address));
            let written = written_registers(&mut self.factory, instruction);
            self.clobber(&written);
            self.set(target, Role::Entry);
            return Ok(written);
        }
        if mnemonic == Mnemonic::Shr
            && let Some(target) = op0
            && canonical(target) == acc
            && target.size() == 4
            && immediate(instruction, 1) == Some(8)
            && self.role(acc) == Some(Role::Acc)
        {
            self.roles.insert(acc, Role::Shifted);
            return Ok(vec![acc]);
        }
        if mnemonic == Mnemonic::Xor
            && let (Some(left), Some(right)) = (op0, op1)
            && canonical(left) == acc
            && left.size() == 4
            && right.size() == 4
            && self.role(acc) == Some(Role::Shifted)
            && self.role(right) == Some(Role::Entry)
        {
            self.roles.insert(acc, Role::Combined);
            return Ok(vec![acc]);
        }
        if mnemonic == Mnemonic::Xor
            && let Some(left) = op0
            && canonical(left) == acc
            && left.size() == 4
            && instruction.op1_kind() != OpKind::Register
            && !is_memory_operand(instruction.op1_kind())
            && self.role(acc) == Some(Role::Combined)
            && let Some(value) = immediate(instruction, 1).and_then(|v| u32::try_from(v).ok())
        {
            self.roles.insert(acc, Role::Keyed);
            facts.iteration_xor = Some(value);
            return Ok(vec![acc]);
        }
        if stack_mode && mnemonic == Mnemonic::Inc && is_rsp_slot(instruction, 0) {
            if !facts.byte_loaded || facts.advanced {
                return Err("pointer advance is not exactly once after the byte load".to_owned());
            }
            facts.advanced = true;
            return Ok(Vec::new());
        }
        if stack_mode && mnemonic == Mnemonic::Dec && is_rsp_slot(instruction, 8) {
            if facts.counted {
                return Err("loop count decremented more than once".to_owned());
            }
            facts.counted = true;
            return Ok(Vec::new());
        }
        if !stack_mode
            && let Some(target) = op0
            && canonical(target) == pointer
            && target.size() == 8
            && (mnemonic == Mnemonic::Inc
                || (mnemonic == Mnemonic::Add && signed_immediate(instruction, 1) == Some(1)))
        {
            if !facts.byte_loaded || facts.advanced {
                return Err("pointer advance is not exactly once after the byte load".to_owned());
            }
            facts.advanced = true;
            return Ok(Vec::new());
        }
        if !stack_mode
            && mnemonic == Mnemonic::Dec
            && let Some(target) = op0
            && canonical(target) == length
            && target.size() == 8
        {
            if facts.counted {
                return Err("loop count decremented more than once".to_owned());
            }
            facts.counted = true;
            return Ok(Vec::new());
        }
        self.generic(instruction, true)
    }
}

/// Recognizes a counted W32 byte-table checksum loop and proves that both the
/// zero-length and loop-exit paths converge on the same ordinary dispatch
/// tail. Constants and register choices are recovered from data flow; no
/// handler addresses, table contents, or mutation bytes are signatured. Loop
/// state may live in registers or in two native-stack slots; every other
/// instruction on the prefix, loop and exit paths must be proven free of
/// effects on the dispatch tail's live inputs.
///
/// # Errors
///
/// Native decode/mapping failures are returned. Pattern ambiguity and semantic
/// rejection remain structured diagnostics in the report.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn analyze_table_checksum_handler(
    pe: &PeImage<'_>,
    entry: Rva,
    roles: HandlerRegisterRoles,
    vsp: Register,
    bytecode_direction: BytecodeDirection,
    live_out_registers: &[Register],
    trace_limits: DirectJumpTraceLimits,
    proof_limits: RegisterDispatchProofLimits,
) -> Result<TableChecksumHandlerAnalysis, DisasmError> {
    let prefix = trace_direct_jump_fragments(pe, entry, trace_limits)?;
    if prefix.stop.kind != DirectJumpTraceStopKind::ConditionalBranch {
        return Ok(report_failure(
            entry,
            None,
            None,
            vec!["handler prefix does not end in a conditional zero-length split".to_owned()],
        ));
    }
    let prefix_instructions = prefix.instructions().collect::<Vec<_>>();
    let Some(branch_instruction) = prefix_instructions.last() else {
        return Ok(report_failure(
            entry,
            None,
            None,
            vec!["conditional prefix contains no instructions".to_owned()],
        ));
    };
    let zero_successor = match branch_instruction.instruction.mnemonic() {
        Mnemonic::Je => prefix.stop.direct_target,
        Mnemonic::Jne => prefix.stop.fallthrough,
        mnemonic => {
            return Ok(report_failure(
                entry,
                None,
                None,
                vec![format!(
                    "length split uses {mnemonic:?}, expected equality/inequality"
                )],
            ));
        }
    };
    let nonzero_successor = match branch_instruction.instruction.mnemonic() {
        Mnemonic::Je => prefix.stop.fallthrough,
        _ => prefix.stop.direct_target,
    };
    let (Some(zero_successor), Some(nonzero_successor)) = (zero_successor, nonzero_successor)
    else {
        return Ok(report_failure(
            entry,
            None,
            None,
            vec!["length split does not have two executable successors".to_owned()],
        ));
    };

    let zero_trace = trace_direct_jump_fragments(pe, zero_successor, trace_limits)?;
    let zero_instructions = zero_trace.instructions().collect::<Vec<_>>();
    let output = zero_instructions
        .iter()
        .enumerate()
        .find_map(|(not_index, decoded)| {
            let instruction = &decoded.instruction;
            if instruction.mnemonic() != Mnemonic::Not {
                return None;
            }
            let accumulator = register_operand(instruction, 0)?;
            if accumulator.size() != 4 {
                return None;
            }
            let mut factory = InstructionInfoFactory::new();
            let mut intervening = zero_instructions[not_index + 1..].iter();
            let store = intervening.find(|candidate| {
                let store = &candidate.instruction;
                store.mnemonic() == Mnemonic::Mov
                    && is_memory_operand(store.op0_kind())
                    && canonical(store.memory_base()) == canonical(vsp)
                    && store.memory_index() == Register::None
                    && memory_displacement(store) == 0
                    && memory_width(store) == 4
                    && register_operand(store, 1)
                        .is_some_and(|source| canonical(source) == canonical(accumulator))
            })?;
            let clean = zero_instructions[not_index + 1..]
                .iter()
                .take_while(|candidate| candidate.rva != store.rva)
                .all(|candidate| {
                    !writes_register(&mut factory, &candidate.instruction, accumulator)
                        && !writes_register(&mut factory, &candidate.instruction, vsp)
                        && !writes_memory(&mut factory, &candidate.instruction)
                });
            clean.then_some((canonical(accumulator), store.rva))
        });
    let Some((accumulator, output_store)) = output else {
        return Ok(report_failure(
            entry,
            None,
            None,
            vec!["zero-length tail does not finalize and store a W32 accumulator".to_owned()],
        ));
    };

    let zero_slice = backward_slice_handler_trace(
        &zero_trace,
        &HandlerSliceConfig {
            live_out_registers: live_out_registers.to_vec(),
            live_out_rflags: 0,
        },
    );
    let dispatch_analysis = extract_relative_dispatch_codecs_with_postlude_direction(
        pe,
        &zero_trace,
        &zero_slice,
        roles,
        bytecode_direction,
        proof_limits,
    )?;
    let [dispatch_candidate] = dispatch_analysis.candidates.as_slice() else {
        return Ok(TableChecksumHandlerAnalysis {
            report: TableChecksumHandlerReport {
                schema_version: 1,
                entry,
                status: RecoveryStatus::Failed,
                pointer_load: None,
                length_load: None,
                loop_branch: None,
                output_store: Some(output_store),
                common_tail: Some(zero_successor),
                dispatch_analysis: Some(dispatch_analysis),
                metadata: None,
                confidence: 0,
                evidence: Vec::new(),
                diagnostics: vec![
                    "zero-length common tail did not yield exactly one threaded dispatch codec"
                        .to_owned(),
                ],
            },
            recovered: None,
        });
    };
    let dispatch_candidate = dispatch_candidate.clone();
    let dispatch_vip_delta = i32::try_from(dispatch_candidate.vip_delta).unwrap_or(i32::MAX);
    if dispatch_vip_delta == i32::MAX {
        return Ok(report_failure(
            entry,
            None,
            None,
            vec!["threaded dispatch VIP delta exceeds i32".to_owned()],
        ));
    }

    // Registers the dispatch tail genuinely consumes: keyless table dispatch
    // never reads the rolling-key or handler registers.
    let keyless = dispatch_candidate.codec.table.is_some();
    let tail_live_out = live_out_registers
        .iter()
        .copied()
        .filter(|register| {
            !keyless
                || (canonical(*register) != canonical(roles.rolling_key)
                    && canonical(*register) != canonical(roles.handler))
        })
        .collect::<Vec<_>>();
    let tail_slice = backward_slice_handler_trace(
        &zero_trace,
        &HandlerSliceConfig {
            live_out_registers: tail_live_out,
            live_out_rflags: 0,
        },
    );
    let mut protected = BTreeSet::new();
    for register in GENERAL_REGISTERS {
        if tail_slice
            .live_in_registers
            .iter()
            .any(|name| *name == format!("{register:?}"))
        {
            protected.insert(register);
        }
    }
    protected.insert(canonical(vsp));
    protected.insert(canonical(roles.vip));
    if let Some(table_register) = dispatch_candidate.table_register {
        protected.insert(canonical(table_register));
    }
    protected.remove(&accumulator);
    protected.remove(&Register::RSP);

    // Prefix: loads, packed VSP adjustment, zero accumulator, zero-length split.
    let fail = |pointer_load: Option<Rva>, length_load: Option<Rva>, message: String| {
        report_failure(entry, pointer_load, length_load, vec![message])
    };
    let mut scan = Scan::new(&protected, accumulator);
    let mut pointer_load = None;
    let mut length_load = None;
    let mut vsp_add = None;
    let mut split_ok = false;
    let no_exempt = BTreeSet::new();
    for decoded in &prefix_instructions[..prefix_instructions.len() - 1] {
        let instruction = &decoded.instruction;
        let result: Result<Vec<Register>, String> = (|| {
            if vsp_add.is_none() {
                if let Some(register) = direct_load_from(instruction, vsp, 0, 8) {
                    let written = written_registers(&mut scan.factory, instruction);
                    scan.clobber(&written);
                    scan.set(register, Role::OrigPtr);
                    pointer_load = Some(decoded.rva);
                    return Ok(written);
                }
                if let Some(register) = direct_load_from(instruction, vsp, 8, 8) {
                    let written = written_registers(&mut scan.factory, instruction);
                    scan.clobber(&written);
                    scan.set(register, Role::OrigLen);
                    length_load = Some(decoded.rva);
                    return Ok(written);
                }
            }
            if instruction.mnemonic() == Mnemonic::Add
                && register_operand(instruction, 0)
                    .is_some_and(|register| canonical(register) == canonical(vsp))
                && signed_immediate(instruction, 1) == Some(12)
            {
                if vsp_add.is_some() {
                    return Err("more than one packed VSP adjustment".to_owned());
                }
                vsp_add = Some(decoded.rva);
                return Ok(Vec::new());
            }
            if matches!(instruction.mnemonic(), Mnemonic::Xor | Mnemonic::Sub)
                && let Some(register) = register_operand(instruction, 0)
                && register.size() >= 4
                && register_operand(instruction, 1)
                    .is_some_and(|other| canonical(other) == canonical(register))
            {
                let written = scan.generic(instruction, false)?;
                scan.zero.insert(canonical(register));
                return Ok(written);
            }
            scan.generic(instruction, false)
        })();
        let written = match result {
            Ok(written) => written,
            Err(message) => return Ok(fail(pointer_load, length_load, message)),
        };
        if scan.violates_protection(&written, &no_exempt) {
            return Ok(fail(
                pointer_load,
                length_load,
                "prefix clobbers a register the dispatch tail depends on".to_owned(),
            ));
        }
        if flags_defined(instruction) {
            split_ok = match instruction.mnemonic() {
                Mnemonic::Test => register_operand(instruction, 0)
                    .zip(register_operand(instruction, 1))
                    .is_some_and(|(left, right)| {
                        canonical(left) == canonical(right)
                            && left.size() == 8
                            && right.size() == 8
                            && scan.role(left) == Some(Role::OrigLen)
                    }),
                Mnemonic::Cmp => register_operand(instruction, 0)
                    .zip(register_operand(instruction, 1))
                    .is_some_and(|(left, right)| {
                        left.size() == 8
                            && right.size() == 8
                            && ((scan.role(left) == Some(Role::OrigLen)
                                && scan.zero.contains(&canonical(right)))
                                || (scan.role(right) == Some(Role::OrigLen)
                                    && scan.zero.contains(&canonical(left))))
                    }),
                _ => false,
            };
        }
    }
    let (Some(pointer_load), Some(length_load)) = (pointer_load, length_load) else {
        return Ok(fail(
            pointer_load,
            length_load,
            "no W64 pointer/length load from VSP+0/VSP+8 was recovered".to_owned(),
        ));
    };
    if vsp_add.is_none() {
        return Ok(fail(
            Some(pointer_load),
            Some(length_load),
            "expected one packed pointer/length VSP adjustment of 12".to_owned(),
        ));
    }
    if !split_ok {
        return Ok(fail(
            Some(pointer_load),
            Some(length_load),
            "conditional split is not a W64 length-versus-zero comparison on the loaded length"
                .to_owned(),
        ));
    }
    if !scan.zero.contains(&accumulator) {
        return Ok(fail(
            Some(pointer_load),
            Some(length_load),
            "checksum accumulator is not zero at the length split".to_owned(),
        ));
    }
    let (Some(pointer_register), Some(length_register)) = (
        scan.unique_role(Role::OrigPtr),
        scan.unique_role(Role::OrigLen),
    ) else {
        return Ok(fail(
            Some(pointer_load),
            Some(length_load),
            "loaded pointer/length are not preserved to the split".to_owned(),
        ));
    };
    let fail = |message: String| fail(Some(pointer_load), Some(length_load), message);

    let loop_trace = trace_direct_jump_fragments(pe, nonzero_successor, trace_limits)?;
    if loop_trace.stop.kind != DirectJumpTraceStopKind::ConditionalBranch {
        return Ok(fail(
            "nonzero path does not end in a counted loop branch".to_owned(),
        ));
    }
    let loop_instructions = loop_trace.instructions().collect::<Vec<_>>();
    let successors = [loop_trace.stop.direct_target, loop_trace.stop.fallthrough];
    let mut loop_entry = None;
    let mut loop_exit = None;
    for successor in successors.into_iter().flatten() {
        if loop_instructions
            .iter()
            .any(|decoded| decoded.rva == successor)
        {
            loop_entry = Some(successor);
        } else {
            loop_exit = Some(successor);
        }
    }
    let (Some(loop_entry), Some(loop_exit)) = (loop_entry, loop_exit) else {
        return Ok(fail(
            "counted loop does not have one back-edge and one exit".to_owned(),
        ));
    };
    let Some(loop_branch) = loop_instructions.last() else {
        return Ok(fail("counted loop is empty".to_owned()));
    };
    if loop_branch.instruction.mnemonic() != Mnemonic::Jne
        || loop_trace.stop.direct_target != Some(loop_entry)
    {
        return Ok(fail(
            "counted loop back-edge is not JNE after decrement".to_owned(),
        ));
    }
    let entry_index = loop_instructions
        .iter()
        .position(|decoded| decoded.rva == loop_entry)
        .unwrap_or(0);
    let body_end = loop_instructions.len() - 1;
    let dec_index = loop_instructions[entry_index..body_end]
        .iter()
        .rposition(|decoded| decoded.instruction.mnemonic() == Mnemonic::Dec)
        .map(|index| index + entry_index);
    let Some(dec_index) = dec_index else {
        return Ok(fail("counted loop has no decrement".to_owned()));
    };
    let dec_instruction = &loop_instructions[dec_index].instruction;
    let stack_mode = is_rsp_slot(dec_instruction, 8);
    if !stack_mode
        && register_operand(dec_instruction, 0)
            .is_none_or(|register| canonical(register) != length_register)
    {
        return Ok(fail(
            "counted loop does not decrement the recovered length".to_owned(),
        ));
    }
    if loop_instructions[dec_index + 1..body_end]
        .iter()
        .any(|decoded| flags_defined(&decoded.instruction))
    {
        return Ok(fail("loop-count flags are clobbered before JNE".to_owned()));
    }

    // Nonzero path before the loop label: stack saves and table setup.
    let mut pushed: Vec<(Register, Option<Role>)> = Vec::new();
    let mut pushed_set = BTreeSet::new();
    let mut lea_rva = None;
    for decoded in &loop_instructions[..entry_index] {
        let instruction = &decoded.instruction;
        if instruction.mnemonic() == Mnemonic::Push {
            let Some(register) = register_operand(instruction, 0).filter(|r| r.size() == 8) else {
                return Ok(fail("loop setup pushes a non-register value".to_owned()));
            };
            pushed.push((canonical(register), scan.role(register)));
            pushed_set.insert(canonical(register));
            continue;
        }
        let exempt = if stack_mode { &no_exempt } else { &pushed_set };
        let table_setup = static_lea_address(pe, instruction)
            .filter(|(_, address)| table_is_file_backed(pe, *address));
        let written = if let Some((target, address)) = table_setup {
            let written = match scan.generic(instruction, true) {
                Ok(written) => written,
                Err(message) => return Ok(fail(message)),
            };
            scan.set(target, Role::Table(address));
            lea_rva = Some(decoded.rva);
            written
        } else {
            match scan.generic(instruction, true) {
                Ok(written) => written,
                Err(message) => return Ok(fail(message)),
            }
        };
        if scan.violates_protection(&written, exempt) {
            return Ok(fail(
                "loop setup clobbers a register the dispatch tail depends on".to_owned(),
            ));
        }
    }
    if stack_mode {
        let roles_pushed = pushed.iter().map(|(_, role)| *role).collect::<Vec<_>>();
        if roles_pushed != [Some(Role::OrigLen), Some(Role::OrigPtr)] {
            return Ok(fail(
                "stack-resident loop state is not the saved original length then pointer"
                    .to_owned(),
            ));
        }
    }

    // One iteration of the loop body.
    scan.roles.insert(accumulator, Role::Acc);
    let mut facts = IterationFacts::default();
    for decoded in &loop_instructions[entry_index..body_end] {
        let exempt = if stack_mode { &no_exempt } else { &pushed_set };
        let written = match scan.loop_step(
            &decoded.instruction,
            decoded.rva,
            stack_mode,
            pointer_register,
            length_register,
            &mut facts,
        ) {
            Ok(written) => written,
            Err(message) => {
                return Ok(fail(format!(
                    "nonzero loop does not prove the supported W32 byte-table recurrence: {message}"
                )));
            }
        };
        let written = written
            .into_iter()
            .filter(|register| *register != accumulator)
            .collect::<Vec<_>>();
        if scan.violates_protection(&written, exempt) {
            return Ok(fail(
                "loop clobbers a register the dispatch tail depends on".to_owned(),
            ));
        }
    }
    let Some((table_register, table_address)) = facts.table else {
        return Ok(fail(
            "nonzero loop does not prove the supported W32 byte-table recurrence: no table lookup"
                .to_owned(),
        ));
    };
    let carried_ok = matches!(scan.role(accumulator), Some(Role::Combined | Role::Keyed))
        && scan.role(table_register) == Some(Role::Table(table_address))
        && facts.byte_loaded
        && facts.advanced
        && facts.counted
        && (stack_mode
            || (scan.role(pointer_register) == Some(Role::OrigPtr)
                && scan.role(length_register) == Some(Role::OrigLen)));
    if !carried_ok {
        return Ok(fail(
            "nonzero loop does not prove the supported W32 byte-table recurrence: iteration state is not loop-carried"
                .to_owned(),
        ));
    }
    let iteration_xor = facts.iteration_xor.unwrap_or(0);
    let byte_load = facts.byte_load.unwrap_or(loop_entry);

    // Loop exit: restore saved state and converge on the zero-length tail.
    let exit_trace = trace_direct_jump_fragments(pe, loop_exit, trace_limits)?;
    let exit_instructions = exit_trace.instructions().collect::<Vec<_>>();
    let Some(join_index) = exit_instructions
        .iter()
        .position(|decoded| decoded.rva == zero_successor)
    else {
        return Ok(fail(
            "counted loop exit does not converge on the zero-length tail".to_owned(),
        ));
    };
    let mut popped = Vec::new();
    for decoded in &exit_instructions[..join_index] {
        let instruction = &decoded.instruction;
        if instruction.mnemonic() == Mnemonic::Pop {
            let Some(register) = register_operand(instruction, 0).filter(|r| r.size() == 8) else {
                return Ok(fail("loop exit pops a non-register value".to_owned()));
            };
            popped.push(canonical(register));
            continue;
        }
        let written = match scan.generic(instruction, true) {
            Ok(written) => written,
            Err(message) => return Ok(fail(message)),
        };
        if scan.violates_protection(&written, &no_exempt) {
            return Ok(fail(
                "loop exit clobbers a register the dispatch tail depends on".to_owned(),
            ));
        }
    }
    let balanced = if stack_mode {
        popped.len() == 2 && popped.iter().all(|register| !protected.contains(register))
    } else {
        let expected = pushed.iter().rev().map(|(register, _)| *register);
        popped.iter().copied().eq(expected)
    };
    if !balanced {
        return Ok(fail(
            "loop exit does not restore the saved native stack state".to_owned(),
        ));
    }
    let metadata = TableChecksumMetadata {
        checksum_width: BitWidth::W32,
        accumulator_source: ChecksumAccumulatorSource::Constant { value: 0 },
        initial_xor: 0,
        pointer_stack_offset: 0,
        length_stack_offset: 8,
        length_width: BitWidth::W64,
        pointer_stride: 1,
        table_address,
        table_entries: 256,
        table_entry_width: BitWidth::W32,
        table_index: ChecksumTableIndex::AccumulatorLowByteXorInputByte,
        shift: ChecksumShift::LogicalRight,
        shift_bits: 8,
        combine: ChecksumCombine::ShiftedAccumulatorXorTableEntry,
        iteration_xor,
        final_not: true,
        final_xor: 0,
        result_stack_offset: 0,
        vsp_delta: 12,
    };
    debug_assert!(metadata.validate().is_ok());
    let lea_rva = lea_rva.unwrap_or(loop_entry);
    let evidence = vec![
        "loads a W64 pointer and packed zero-extended length from VSP+0/VSP+8".to_owned(),
        "zero length and counted-loop exit converge on one W32 finalization/store tail".to_owned(),
        format!(
            "byte loop at {byte_load} indexes a file-backed 256xW32 table established at {lea_rva}"
        ),
        "each iteration combines SHR32(accumulator, 8) with the selected table entry".to_owned(),
        format!("each iteration advances the pointer and XORs {iteration_xor:#010X}"),
        format!(
            "loop state is held in {}; every other native instruction on the prefix, loop and exit paths leaves the dispatch tail's live inputs untouched",
            if stack_mode {
                "two saved native-stack slots"
            } else {
                "registers"
            }
        ),
        "common tail yields exactly one ordinary relative-dispatch codec".to_owned(),
    ];
    let report = TableChecksumHandlerReport {
        schema_version: 1,
        entry,
        status: RecoveryStatus::Recovered,
        pointer_load: Some(pointer_load),
        length_load: Some(length_load),
        loop_branch: Some(loop_branch.rva),
        output_store: Some(output_store),
        common_tail: Some(zero_successor),
        dispatch_analysis: Some(dispatch_analysis),
        metadata: Some(metadata.clone()),
        confidence: 100,
        evidence,
        diagnostics: Vec::new(),
    };
    Ok(TableChecksumHandlerAnalysis {
        report: report.clone(),
        recovered: Some(RecoveredTableChecksumHandler {
            report,
            metadata,
            dispatch: dispatch_candidate.codec,
            dispatch_vip_delta,
        }),
    })
}

const GENERAL_REGISTERS: [Register; 16] = [
    Register::RAX,
    Register::RBX,
    Register::RCX,
    Register::RDX,
    Register::RSI,
    Register::RDI,
    Register::RBP,
    Register::RSP,
    Register::R8,
    Register::R9,
    Register::R10,
    Register::R11,
    Register::R12,
    Register::R13,
    Register::R14,
    Register::R15,
];

#[cfg(test)]
mod tests {
    use iced_x86::code_asm::*;

    use super::*;

    fn executable_pe64(code: &[u8]) -> Vec<u8> {
        let mut data = vec![0_u8; 0x1600];
        data[0..2].copy_from_slice(&0x5A4D_u16.to_le_bytes());
        data[0x3C..0x40].copy_from_slice(&0x80_u32.to_le_bytes());
        data[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
        let coff = 0x84;
        data[coff..coff + 2].copy_from_slice(&0x8664_u16.to_le_bytes());
        data[coff + 2..coff + 4].copy_from_slice(&1_u16.to_le_bytes());
        data[coff + 16..coff + 18].copy_from_slice(&0xF0_u16.to_le_bytes());
        data[coff + 18..coff + 20].copy_from_slice(&0x2022_u16.to_le_bytes());
        let optional = coff + 20;
        data[optional..optional + 2].copy_from_slice(&0x20B_u16.to_le_bytes());
        data[optional + 16..optional + 20].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 20..optional + 24].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&0x1_4000_0000_u64.to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 56..optional + 60].copy_from_slice(&0x3000_u32.to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&3_u16.to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&16_u32.to_le_bytes());
        let section = optional + 0xF0;
        data[section..section + 5].copy_from_slice(b".text");
        data[section + 8..section + 12].copy_from_slice(&0x1400_u32.to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&0x1400_u32.to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&0x200_u32.to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&0x6000_0020_u32.to_le_bytes());
        data[0x200..0x200 + code.len()].copy_from_slice(code);
        for index in 0..256_u32 {
            let offset = 0x1200 + usize::try_from(index * 4).expect("small table offset");
            data[offset..offset + 4].copy_from_slice(&(index ^ 0xA5A5_5A5A).to_le_bytes());
        }
        data
    }

    #[derive(Clone, Copy, Default)]
    struct Shape {
        invalid_combine: bool,
        subtract_to_zero: bool,
        stack_state: bool,
        compare_zero_register: bool,
        compare_nonzero_register: bool,
        clobber_vip_unsaved: bool,
        stray_memory_write: bool,
        advance_twice: bool,
        shift_before_index: bool,
        unbalanced_exit: bool,
    }

    fn checksum_code(shape: Shape) -> Vec<u8> {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        let mut loop_body = assembler.create_label();
        let mut common_tail = assembler.create_label();
        assembler.mov(r12, qword_ptr(rbp)).expect("pointer load");
        assembler.mov(r10, qword_ptr(rbp + 8)).expect("length load");
        assembler.add(rbp, 12).expect("consume packed inputs");
        if shape.subtract_to_zero {
            assembler.sub(r9d, r9d).expect("zero accumulator");
        } else {
            assembler.xor(r9d, r9d).expect("zero accumulator");
        }
        if shape.compare_zero_register {
            assembler.cmp(r10, r9).expect("compare length with zero");
        } else if shape.compare_nonzero_register {
            assembler.cmp(r10, r11).expect("compare length with r11");
        } else {
            assembler.test(r10, r10).expect("test length");
        }
        assembler.jz(common_tail).expect("zero-length tail");
        assembler
            .mov(rdx, 0x1_4000_2000_u64)
            .expect("table address");
        if shape.stack_state {
            assembler.push(r10).expect("save length");
            assembler.push(r12).expect("save pointer");
            assembler.set_label(&mut loop_body).expect("loop label");
            assembler.mov(rax, qword_ptr(rsp)).expect("load pointer");
            assembler.movzx(eax, byte_ptr(rax)).expect("input byte");
            if shape.shift_before_index {
                assembler.shr(r9d, 8).expect("shift accumulator");
            }
            assembler.xor(al, r9b).expect("form table index");
            assembler
                .mov(eax, dword_ptr(rdx + rax * 4))
                .expect("table lookup");
            if !shape.shift_before_index {
                assembler.shr(r9d, 8).expect("shift accumulator");
            }
            assembler.xor(r9d, eax).expect("combine table entry");
            assembler.xor(r9d, 0x1234_5678_i32).expect("iteration xor");
            assembler.inc(qword_ptr(rsp)).expect("advance pointer");
            if shape.advance_twice {
                assembler.inc(qword_ptr(rsp)).expect("advance pointer");
            }
            if shape.stray_memory_write {
                assembler
                    .mov(qword_ptr(rbp - 0x40), rax)
                    .expect("stray VM-stack write");
            }
            assembler.dec(qword_ptr(rsp + 8)).expect("decrement length");
            assembler.jne(loop_body).expect("loop back-edge");
            assembler.pop(r12).expect("drop pointer");
            if !shape.unbalanced_exit {
                assembler.pop(r10).expect("drop length");
            }
        } else {
            if !shape.clobber_vip_unsaved {
                assembler.push(rsi).expect("save aliased VIP");
            }
            assembler.set_label(&mut loop_body).expect("loop label");
            assembler.movzx(esi, byte_ptr(r12)).expect("input byte");
            assembler.xor(esi, r9d).expect("form table index");
            assembler.and(esi, 0xff).expect("mask table index");
            assembler
                .mov(esi, dword_ptr(rdx + rsi * 4))
                .expect("table lookup");
            assembler.shr(r9d, 8).expect("shift accumulator");
            if shape.invalid_combine {
                assembler.add(r9d, esi).expect("invalid table combine");
            } else {
                assembler.xor(r9d, esi).expect("combine table entry");
            }
            assembler.inc(r12).expect("advance pointer");
            assembler.xor(r9d, 0x1234_5678_i32).expect("iteration xor");
            assembler.dec(r10).expect("decrement length");
            assembler.jne(loop_body).expect("loop back-edge");
            if !shape.clobber_vip_unsaved {
                assembler.pop(rsi).expect("restore VIP");
            }
        }
        assembler.set_label(&mut common_tail).expect("tail label");
        assembler.not(r9d).expect("finalize checksum");
        assembler.mov(dword_ptr(rbp), r9d).expect("store checksum");
        assembler.mov(eax, dword_ptr(rsi)).expect("dispatch token");
        assembler.add(rsi, 4).expect("advance VIP");
        assembler.xor(eax, r8d).expect("decode dispatch");
        assembler.push(r8).expect("save rolling key");
        assembler
            .xor(dword_ptr(rsp), eax)
            .expect("update low rolling key");
        assembler.pop(r8).expect("restore updated key");
        assembler.movsxd(rax, eax).expect("extend displacement");
        assembler.add(rdi, rax).expect("update handler");
        assembler.jmp(rdi).expect("threaded dispatch");
        assembler.assemble(0x1_4000_1000).expect("assemble handler")
    }

    fn analyze(code: &[u8]) -> TableChecksumHandlerAnalysis {
        let bytes = executable_pe64(code);
        let pe = PeImage::parse(&bytes).expect("valid synthetic PE");
        analyze_table_checksum_handler(
            &pe,
            Rva(0x1000),
            HandlerRegisterRoles {
                vip: Register::RSI,
                rolling_key: Register::R8,
                handler: Register::RDI,
            },
            Register::RBP,
            BytecodeDirection::Ascending,
            &[Register::RBP, Register::RSI, Register::RDI, Register::R8],
            DirectJumpTraceLimits::default(),
            RegisterDispatchProofLimits::default(),
        )
        .expect("synthetic handler must decode")
    }

    fn assert_rejected(shape: Shape, needle: &str) {
        let analysis = analyze(&checksum_code(shape));
        assert!(
            analysis.recovered.is_none(),
            "near-miss must stay Failed: {:#?}",
            analysis.report
        );
        assert!(
            analysis
                .report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains(needle)),
            "expected `{needle}` in {:#?}",
            analysis.report.diagnostics
        );
    }

    #[test]
    fn recognizes_mutation_independent_table_checksum_contract() {
        let analysis = analyze(&checksum_code(Shape::default()));
        let recovered = analysis
            .recovered
            .unwrap_or_else(|| panic!("checksum must recover: {:#?}", analysis.report));
        assert_eq!(recovered.metadata.iteration_xor, 0x1234_5678);
        assert_eq!(recovered.metadata.table_address, Va(0x1_4000_2000));
        assert_eq!(recovered.metadata.vsp_delta, 12);
        assert_eq!(recovered.dispatch_vip_delta, 4);
    }

    #[test]
    fn recognizes_subtract_to_zero_accumulator_initialization() {
        let analysis = analyze(&checksum_code(Shape {
            subtract_to_zero: true,
            ..Shape::default()
        }));
        assert!(analysis.recovered.is_some(), "{:#?}", analysis.report);
    }

    #[test]
    fn rejects_non_xor_table_combination() {
        assert_rejected(
            Shape {
                invalid_combine: true,
                ..Shape::default()
            },
            "recurrence",
        );
    }

    #[test]
    fn recognizes_stack_resident_loop_state_with_zero_register_compare() {
        let analysis = analyze(&checksum_code(Shape {
            stack_state: true,
            compare_zero_register: true,
            ..Shape::default()
        }));
        let recovered = analysis
            .recovered
            .unwrap_or_else(|| panic!("stack-state checksum must recover: {:#?}", analysis.report));
        assert_eq!(recovered.metadata.iteration_xor, 0x1234_5678);
        assert_eq!(recovered.metadata.vsp_delta, 12);
    }

    #[test]
    fn recognizes_register_state_with_zero_register_compare() {
        let analysis = analyze(&checksum_code(Shape {
            compare_zero_register: true,
            ..Shape::default()
        }));
        assert!(analysis.recovered.is_some(), "{:#?}", analysis.report);
    }

    #[test]
    fn rejects_compare_against_unproven_register() {
        assert_rejected(
            Shape {
                compare_nonzero_register: true,
                ..Shape::default()
            },
            "length-versus-zero",
        );
    }

    #[test]
    fn rejects_vip_clobber_without_save_restore() {
        assert_rejected(
            Shape {
                clobber_vip_unsaved: true,
                ..Shape::default()
            },
            "dispatch tail depends on",
        );
    }

    #[test]
    fn rejects_stray_memory_write_in_loop() {
        assert_rejected(
            Shape {
                stack_state: true,
                stray_memory_write: true,
                ..Shape::default()
            },
            "writes memory",
        );
    }

    #[test]
    fn rejects_double_pointer_advance() {
        assert_rejected(
            Shape {
                stack_state: true,
                advance_twice: true,
                ..Shape::default()
            },
            "recurrence",
        );
    }

    #[test]
    fn rejects_shift_before_index_formation() {
        assert_rejected(
            Shape {
                stack_state: true,
                shift_before_index: true,
                ..Shape::default()
            },
            "recurrence",
        );
    }

    #[test]
    fn rejects_unbalanced_stack_exit() {
        assert_rejected(
            Shape {
                stack_state: true,
                unbalanced_exit: true,
                ..Shape::default()
            },
            "restore",
        );
    }
}
