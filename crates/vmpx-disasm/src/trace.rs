use std::collections::BTreeSet;

use iced_x86::{
    ConditionCode, FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess,
    OpKind, Register, RflagsBits,
};
use serde::Serialize;
use vmpx_pe::{PeImage, Rva};

use crate::{DecodedInstruction, DisasmError, decode_one, is_executable_rva};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectJumpTraceLimits {
    pub max_fragments: usize,
    pub max_instructions: usize,
    pub max_instructions_per_fragment: usize,
    /// Continue through a conditional branch along its FALLTHROUGH instead of
    /// stopping there, recording each site in
    /// [`DirectJumpTrace::conditional_continuations`].
    ///
    /// Off by default, because choosing an arm is not a proof: the other arm may
    /// do something different, so anything recovered downstream of one is
    /// conditional on that arm being the real path. A caller that enables this
    /// must label whatever it recovers.
    ///
    /// It exists because VMProtect splits a handler across fragments joined by
    /// junk predicates, and a trace that stops at the first of them never
    /// reaches the handler's dispatch tail -- measured on systems.xpl, this is
    /// what leaves ~150 handler-spec refusals and blocks 48 entries.
    pub follow_conditional_fallthrough: bool,
    /// With `follow_conditional_fallthrough`, continue along the branch's TAKEN
    /// target instead of its fallthrough. The two together give the only two
    /// paths through a single conditional, which is what makes agreement
    /// between them a proof rather than a guess.
    pub follow_conditional_target: bool,
    /// How many conditional branches may be walked through. Beyond one, the two
    /// arm-choices stop covering every path -- with two branches the pair
    /// all-fallthrough / all-taken is 2 of 4 -- so agreement proves nothing and
    /// the default of 1 keeps the guarantee.
    pub max_conditional_continuations: usize,
}

impl Default for DirectJumpTraceLimits {
    fn default() -> Self {
        Self {
            max_fragments: 1_024,
            max_instructions: 50_000,
            max_instructions_per_fragment: 10_000,
            follow_conditional_fallthrough: false,
            follow_conditional_target: false,
            max_conditional_continuations: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectJumpTraceStopKind {
    IndirectBranch,
    ConditionalBranch,
    Return,
    InterruptOrException,
    RepeatedInstruction,
    ResourceLimit,
    DecodeError,
    UnmappedFallthrough,
    NonExecutableDirectTarget,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DirectJumpTraceStop {
    pub kind: DirectJumpTraceStopKind,
    pub at: Rva,
    pub direct_target: Option<Rva>,
    pub fallthrough: Option<Rva>,
    pub detail: Option<String>,
}

#[derive(Clone, Debug)]
pub struct DirectJumpFragment {
    pub start: Rva,
    pub end: Rva,
    pub instructions: Vec<DecodedInstruction>,
    pub next_fragment: Option<Rva>,
}

#[derive(Clone, Debug)]
pub struct DirectJumpTrace {
    pub entry: Rva,
    pub fragments: Vec<DirectJumpFragment>,
    pub stop: DirectJumpTraceStop,
    pub decoded_instructions: usize,
    /// Conditional branches the trace walked THROUGH, by taking the fallthrough
    /// (see `DirectJumpTraceLimits::follow_conditional_fallthrough`). Empty
    /// unless that was enabled. Non-empty means everything recovered from this
    /// trace holds only on the fallthrough arm of each site listed.
    pub conditional_continuations: Vec<Rva>,
}

impl DirectJumpTrace {
    pub fn instructions(&self) -> impl Iterator<Item = &DecodedInstruction> {
        self.fragments
            .iter()
            .flat_map(|fragment| fragment.instructions.iter())
    }

    #[must_use]
    pub fn ends_in_indirect_branch(&self) -> bool {
        self.stop.kind == DirectJumpTraceStopKind::IndirectBranch
    }
}

fn fragment_end(instructions: &[DecodedInstruction], fallback: Rva) -> Rva {
    instructions
        .last()
        .and_then(DecodedInstruction::next_rva)
        .unwrap_or(fallback)
}

fn push_fragment(
    fragments: &mut Vec<DirectJumpFragment>,
    start: Rva,
    instructions: Vec<DecodedInstruction>,
    next_fragment: Option<Rva>,
) {
    let end = fragment_end(&instructions, start);
    fragments.push(DirectJumpFragment {
        start,
        end,
        instructions,
        next_fragment,
    });
}

fn stop(
    kind: DirectJumpTraceStopKind,
    at: Rva,
    direct_target: Option<Rva>,
    fallthrough: Option<Rva>,
    detail: Option<String>,
) -> DirectJumpTraceStop {
    DirectJumpTraceStop {
        kind,
        at,
        direct_target,
        fallthrough,
        detail,
    }
}

/// Linearizes a bounded instruction chain whose fragments are connected by
/// direct unconditional jumps.
///
/// Calls stay in the trace and execution continues at their fallthrough. A
/// conditional branch stops the trace because choosing a path requires state
/// analysis. An indirect branch is retained as the terminal instruction and is
/// commonly the VM dispatch boundary consumed by higher analysis layers.
///
/// # Errors
///
/// The entry must be executable and contain a valid file-backed instruction.
/// Later decode and mapping failures are represented by the returned stop
/// reason so partial evidence remains available.
#[allow(clippy::too_many_lines)]


/// The x86 status flags this tracks, each known or not.
///
/// MEASURED +0 on systems.xpl: it decides none of the 162 conditionals that stop
/// a handler trace there, because VMProtect does not build those predicates out
/// of constant flags. The flag write and the branch are separated by real
/// arithmetic on symbolic registers:
///
/// ```text
///     clc                  ; carry cleared
///     add  r11,r8          ; and immediately clobbered
///     jmp  r11
///     ja   <target>        ; needs CF=0 and ZF=0, both unknown
/// ```
///
/// So the refusal is correct and deciding these needs the register VALUES, i.e.
/// native symbolic execution of the handler, not local flag tracking. Kept
/// because it is a proof rather than an assumption and costs one byte of state
/// per flag, and because a predicate that IS constant must not stop a trace;
/// documented so the next reader does not re-derive the measurement.
///
/// A fourth approach was tried and reverted too: resume at the point where the
/// two arms REJOIN, taking no instruction from either, which is sound for a
/// recogniser reading the stream provided both arms change nothing but flags.
/// They do not. The fallthrough of the conditional that stops these traces is
///
/// ```text
///     ja   <target>
///     mov  rax,rsp        ; fallthrough does real work
/// ```
///
/// so the arms are not junk and skipping them would drop effects. Which settles
/// the class: these handlers genuinely BRANCH, and codec recovery models a
/// linear handler via a backward slice of a linear trace. No way of tracing
/// further fixes that -- it needs a handler model that admits control flow.
/// How many register constants the trace remembers. VMProtect sets up one or two
/// per predicate; a small fixed set keeps [`FlagState`] `Copy`.
const TRACKED_CONSTANTS: usize = 4;

/// The 64-bit container, so `r8w` and `r8` share a slot.
fn container(register: Register) -> Register {
    register.full_register()
}

/// Low `bytes * 8` bits of `value`.
fn masked(value: u64, bytes: u32) -> u64 {
    if bytes >= 8 {
        value
    } else {
        value & ((1u64 << (bytes * 8)) - 1)
    }
}

/// The flags a `cmp a, b` at `bytes` width produces, by the architecture's own
/// definitions. `a` and `b` are the full container values; only the low
/// `bytes` participate, which is what makes `cmp r8w, imm` exact given `r8`.
fn compare_flags(a: u64, b: u64, bytes: u32) -> (bool, bool, bool, bool) {
    let width = bytes * 8;
    let (a, b) = (masked(a, bytes), masked(b, bytes));
    let result = masked(a.wrapping_sub(b), bytes);
    let sign = 1u64 << (width - 1);
    let carry = a < b;
    let zero = result == 0;
    let negative = result & sign != 0;
    // Signed overflow: the operands differed in sign and the result took the
    // subtrahend's sign.
    let overflow = ((a ^ b) & (a ^ result) & sign) != 0;
    (carry, zero, negative, overflow)
}

/// The flags a logical `and` of `a` and `b` at `bytes` width produces. `test`
/// is that without the write-back, and CF and OF are cleared.
fn logical_flags(a: u64, b: u64, bytes: u32) -> (bool, bool, bool, bool) {
    let width = bytes * 8;
    let result = masked(masked(a, bytes) & masked(b, bytes), bytes);
    (false, result == 0, result & (1u64 << (width - 1)) != 0, false)
}

/// Whether any operand is a memory reference.
///
/// A logical op reading memory still clears CF and OF, but the trace treats a
/// memory operand as a reason to learn nothing: the same conservatism the rest
/// of this module applies, so a faulting or protector-rewritten operand cannot
/// decide a branch.
fn touches_memory(instruction: &Instruction) -> bool {
    (0..instruction.op_count()).any(|op| instruction.op_kind(op) == OpKind::Memory)
}

#[derive(Clone, Copy, Default)]
struct FlagState {
    cf: Option<bool>,
    zf: Option<bool>,
    sf: Option<bool>,
    of: Option<bool>,
    /// Registers whose exact value is known, by 64-bit container.
    ///
    /// Only an immediate load records one, and any other write clears it, so a
    /// slot is either the value the register provably holds or absent. This is
    /// what lets a real comparison -- `cmp r8w, 98EAh`, the branch that stops
    /// the dominant blocked handler -- be decided instead of stopping the trace.
    constants: [Option<(Register, u64)>; TRACKED_CONSTANTS],
}

impl FlagState {
    /// The value `register` provably holds, if known.
    fn constant(&self, register: Register) -> Option<u64> {
        let want = container(register);
        self.constants
            .iter()
            .flatten()
            .find(|(slot, _)| *slot == want)
            .map(|(_, value)| *value)
    }

    /// Records that `register` provably holds `value`.
    fn set_constant(&mut self, register: Register, value: u64) {
        let want = container(register);
        self.forget(register);
        for slot in &mut self.constants {
            if slot.is_none() {
                *slot = Some((want, value));
                return;
            }
        }
        // Full: drop the oldest rather than keep a stale set. Losing a constant
        // only costs a decision; keeping a wrong one would cost soundness.
        self.constants.rotate_left(1);
        self.constants[TRACKED_CONSTANTS - 1] = Some((want, value));
    }

    /// Forgets `register`'s value.
    fn forget(&mut self, register: Register) {
        let want = container(register);
        for slot in &mut self.constants {
            if slot.is_some_and(|(held, _)| held == want) {
                *slot = None;
            }
        }
    }

    /// An operand's provable value: an immediate, or a register we track.
    fn operand_value(&self, instruction: &Instruction, operand: u32) -> Option<u64> {
        match instruction.op_kind(operand) {
            OpKind::Register => self.constant(instruction.op_register(operand)),
            OpKind::Immediate8 => Some(u64::from(instruction.immediate8())),
            OpKind::Immediate16 => Some(u64::from(instruction.immediate16())),
            OpKind::Immediate32 => Some(u64::from(instruction.immediate32())),
            OpKind::Immediate64 => Some(instruction.immediate64()),
            OpKind::Immediate8to16 => Some(u64::from(instruction.immediate8to16() as u16)),
            OpKind::Immediate8to32 => Some(u64::from(instruction.immediate8to32() as u32)),
            OpKind::Immediate8to64 => Some(instruction.immediate8to64() as u64),
            OpKind::Immediate32to64 => Some(instruction.immediate32to64() as u64),
            _ => None,
        }
    }

    /// Folds an exactly-evaluated comparison or test, when both operands are
    /// known. Returns whether it applied.
    fn fold_known_compare(&mut self, instruction: &Instruction) -> bool {
        if touches_memory(instruction) || instruction.op_count() != 2 {
            return false;
        }
        let logical = match instruction.mnemonic() {
            Mnemonic::Cmp => false,
            Mnemonic::Test => true,
            _ => return false,
        };
        if instruction.op0_kind() != OpKind::Register {
            return false;
        }
        let Ok(bytes) = u32::try_from(instruction.op0_register().size()) else {
            return false;
        };
        if !matches!(bytes, 1 | 2 | 4 | 8) {
            return false;
        }
        let (Some(left), Some(right)) = (
            self.operand_value(instruction, 0),
            self.operand_value(instruction, 1),
        ) else {
            return false;
        };
        let (cf, zf, sf, of) = if logical {
            logical_flags(left, right, bytes)
        } else {
            compare_flags(left, right, bytes)
        };
        self.cf = Some(cf);
        self.zf = Some(zf);
        self.sf = Some(sf);
        self.of = Some(of);
        true
    }

    /// Updates the tracked constants for one instruction.
    fn track_constants(&mut self, instruction: &Instruction) {
        // An immediate load into a 32- or 64-bit register fully defines the
        // container (a 32-bit write zero-extends); a narrower one leaves the
        // upper bits, so the value is no longer known.
        if instruction.mnemonic() == Mnemonic::Mov
            && instruction.op0_kind() == OpKind::Register
            && !touches_memory(instruction)
        {
            let destination = instruction.op0_register();
            let wide = matches!(destination.size(), 4 | 8);
            let value = self.operand_value(instruction, 1);
            match (wide, value) {
                (true, Some(value)) => {
                    let bytes = u32::try_from(destination.size()).unwrap_or(8);
                    self.set_constant(destination, masked(value, bytes));
                    return;
                }
                _ => {
                    self.forget(destination);
                    return;
                }
            }
        }
        // `xor r,r` / `sub r,r` on a 32- or 64-bit register is a proven zero.
        if matches!(instruction.mnemonic(), Mnemonic::Xor | Mnemonic::Sub)
            && instruction.op0_kind() == OpKind::Register
            && instruction.op1_kind() == OpKind::Register
            && instruction.op0_register() == instruction.op1_register()
            && matches!(instruction.op0_register().size(), 4 | 8)
        {
            self.set_constant(instruction.op0_register(), 0);
            return;
        }
        // Anything else that writes a tracked register invalidates it. Read off
        // iced's access info so an implicit destination counts.
        let mut factory = InstructionInfoFactory::new();
        let written: Vec<Register> = factory
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
            .map(|used| used.register())
            .collect();
        for register in written {
            self.forget(register);
        }
    }
}

impl FlagState {
    /// Folds one instruction's effect on the flags.
    ///
    /// Only the forms VMProtect uses to build an opaque predicate are tracked
    /// exactly; every other flag write makes the flags it touches unknown, which
    /// is what keeps this from deciding a branch it has not actually proven.
    fn apply(&mut self, instruction: &Instruction) {
        // An exactly-evaluated comparison beats every rule below it: it gives
        // all four flags rather than the two a logical op clears.
        if self.fold_known_compare(instruction) {
            self.track_constants(instruction);
            return;
        }
        self.track_constants(instruction);
        match instruction.mnemonic() {
            Mnemonic::Clc => {
                self.cf = Some(false);
                return;
            }
            Mnemonic::Stc => {
                self.cf = Some(true);
                return;
            }
            Mnemonic::Cmc => {
                self.cf = self.cf.map(|carry| !carry);
                return;
            }
            // `cmp r,r` / `sub r,r` / `xor r,r` on the SAME register: the result
            // is zero whatever the register holds.
            Mnemonic::Cmp | Mnemonic::Sub | Mnemonic::Xor
                if instruction.op0_kind() == OpKind::Register
                    && instruction.op1_kind() == OpKind::Register
                    && instruction.op0_register() == instruction.op1_register() =>
            {
                self.cf = Some(false);
                self.zf = Some(true);
                self.sf = Some(false);
                self.of = Some(false);
                return;
            }
            // A logical operation CLEARS CF and OF, by definition, whatever its
            // operands hold -- the architecture says so for `and`, `or`, `xor`
            // and `test`. So an unsigned-carry or overflow predicate after one
            // is decided even though the value tested is entirely unknown, and
            // VMProtect's padding is full of them.
            //
            // This is reached only by the forms the exact arm above did not
            // already settle, so `xor r,r` keeps its stronger result. ZF and SF
            // genuinely depend on the value and stay unknown, which is what
            // leaves a real `jz`/`js` guard still stopping the trace.
            Mnemonic::And | Mnemonic::Or | Mnemonic::Xor | Mnemonic::Test
                if !touches_memory(instruction) =>
            {
                self.cf = Some(false);
                self.of = Some(false);
                self.zf = None;
                self.sf = None;
                return;
            }
            _ => {}
        }
        let modified = instruction.rflags_modified();
        if modified & RflagsBits::CF != 0 {
            self.cf = None;
        }
        if modified & RflagsBits::ZF != 0 {
            self.zf = None;
        }
        if modified & RflagsBits::SF != 0 {
            self.sf = None;
        }
        if modified & RflagsBits::OF != 0 {
            self.of = None;
        }
    }

    /// Whether this conditional branch is taken, when every flag it reads is
    /// known. `None` means undecided, and the caller must not cross it.
    fn decides(self, condition: ConditionCode) -> Option<bool> {
        let (cf, zf, sf, of) = (self.cf, self.zf, self.sf, self.of);
        match condition {
            ConditionCode::b => cf,
            ConditionCode::ae => cf.map(|c| !c),
            ConditionCode::e => zf,
            ConditionCode::ne => zf.map(|z| !z),
            ConditionCode::be => Some(cf? || zf?),
            ConditionCode::a => Some(!cf? && !zf?),
            ConditionCode::s => sf,
            ConditionCode::ns => sf.map(|s| !s),
            ConditionCode::o => of,
            ConditionCode::no => of.map(|o| !o),
            ConditionCode::l => Some(sf? != of?),
            ConditionCode::ge => Some(sf? == of?),
            ConditionCode::le => Some(zf? || sf? != of?),
            ConditionCode::g => Some(!zf? && sf? == of?),
            // Parity is not tracked; nothing VMProtect builds here needs it.
            _ => None,
        }
    }
}

pub fn trace_direct_jump_fragments(
    pe: &PeImage<'_>,
    entry: Rva,
    limits: DirectJumpTraceLimits,
) -> Result<DirectJumpTrace, DisasmError> {
    if !is_executable_rva(pe, entry) {
        return Err(DisasmError::Unmapped {
            rva: entry,
            message: "trace entry is not executable".to_owned(),
        });
    }
    decode_one(pe, entry)?;

    let mut fragments = Vec::new();
    // Flags carry ACROSS fragments, not per fragment. Fragments are joined by
    // unconditional jumps, which do not touch the flags, so resetting at each
    // boundary would throw away the `clc`/`stc` that sets up an opaque
    // predicate -- and VMProtect routinely puts the flag write and the branch
    // that reads it in different fragments. Resetting measured +0 for exactly
    // that reason.
    let mut flags = FlagState::default();
    let mut conditional_continuations = Vec::new();
    let mut visited = BTreeSet::new();
    let mut decoded_instructions = 0_usize;
    let mut fragment_start = entry;

    let terminal = 'trace: loop {
        if fragments.len() >= limits.max_fragments {
            break stop(
                DirectJumpTraceStopKind::ResourceLimit,
                fragment_start,
                None,
                None,
                Some("maximum fragment count reached".to_owned()),
            );
        }

        let mut current = fragment_start;
        let mut instructions = Vec::new();
        loop {
            if decoded_instructions >= limits.max_instructions
                || instructions.len() >= limits.max_instructions_per_fragment
            {
                push_fragment(&mut fragments, fragment_start, instructions, None);
                break 'trace stop(
                    DirectJumpTraceStopKind::ResourceLimit,
                    current,
                    None,
                    None,
                    Some("maximum instruction count reached".to_owned()),
                );
            }
            if !visited.insert(current) {
                if !instructions.is_empty() {
                    push_fragment(&mut fragments, fragment_start, instructions, None);
                }
                break 'trace stop(
                    DirectJumpTraceStopKind::RepeatedInstruction,
                    current,
                    None,
                    None,
                    None,
                );
            }

            let decoded = match decode_one(pe, current) {
                Ok(decoded) => decoded,
                Err(error) => {
                    if !instructions.is_empty() {
                        push_fragment(&mut fragments, fragment_start, instructions, None);
                    }
                    break 'trace stop(
                        DirectJumpTraceStopKind::DecodeError,
                        current,
                        None,
                        None,
                        Some(error.to_string()),
                    );
                }
            };
            decoded_instructions = decoded_instructions.saturating_add(1);
            let flow = decoded.instruction.flow_control();
            let decided = (flow == FlowControl::ConditionalBranch)
                .then(|| flags.decides(decoded.instruction.condition_code()))
                .flatten();
            flags.apply(&decoded.instruction);
            let next = decoded.next_rva();
            let direct_target = decoded.direct_target_rva(pe);
            let instruction_rva = decoded.rva;
            instructions.push(decoded);

            match flow {
                FlowControl::UnconditionalBranch => {
                    let executable_target =
                        direct_target.filter(|target| is_executable_rva(pe, *target));
                    if let Some(target) = executable_target {
                        push_fragment(&mut fragments, fragment_start, instructions, Some(target));
                        fragment_start = target;
                        continue 'trace;
                    }
                    push_fragment(&mut fragments, fragment_start, instructions, None);
                    break 'trace stop(
                        DirectJumpTraceStopKind::NonExecutableDirectTarget,
                        instruction_rva,
                        direct_target,
                        None,
                        None,
                    );
                }
                FlowControl::ConditionalBranch => {
                    // A predicate whose flags are all known is decided here, and
                    // the arm it selects is the ONLY reachable one -- so taking
                    // it is a proof and the other arm is dead. VMProtect builds
                    // these out of `stc`/`clc`/`cmc` and `cmp r,r` to split a
                    // handler across fragments; a real guard (the stack-check
                    // postlude's `ja` after `cmp vsp,limit`) has unknown flags
                    // and still stops the trace.
                    if let Some(taken) = decided {
                        let arm = if taken { direct_target } else { next };
                        if let Some(arm) = arm.filter(|rva| is_executable_rva(pe, *rva)) {
                            push_fragment(
                                &mut fragments,
                                fragment_start,
                                instructions,
                                Some(arm),
                            );
                            fragment_start = arm;
                            continue 'trace;
                        }
                    }
                    // Walking on means picking an arm, which is not a proof; the
                    // site is recorded so the caller can label what it recovers.
                    let arm = if limits.follow_conditional_target {
                        direct_target
                    } else {
                        next
                    };
                    if limits.follow_conditional_fallthrough
                        && conditional_continuations.len()
                            < limits.max_conditional_continuations
                        && let Some(fallthrough) = arm.filter(|rva| is_executable_rva(pe, *rva))
                    {
                        conditional_continuations.push(instruction_rva);
                        push_fragment(
                            &mut fragments,
                            fragment_start,
                            instructions,
                            Some(fallthrough),
                        );
                        fragment_start = fallthrough;
                        continue 'trace;
                    }
                    push_fragment(&mut fragments, fragment_start, instructions, None);
                    break 'trace stop(
                        DirectJumpTraceStopKind::ConditionalBranch,
                        instruction_rva,
                        direct_target,
                        next,
                        None,
                    );
                }
                FlowControl::IndirectBranch => {
                    push_fragment(&mut fragments, fragment_start, instructions, None);
                    break 'trace stop(
                        DirectJumpTraceStopKind::IndirectBranch,
                        instruction_rva,
                        None,
                        None,
                        None,
                    );
                }
                FlowControl::Return => {
                    push_fragment(&mut fragments, fragment_start, instructions, None);
                    break 'trace stop(
                        DirectJumpTraceStopKind::Return,
                        instruction_rva,
                        None,
                        None,
                        None,
                    );
                }
                FlowControl::Interrupt | FlowControl::Exception | FlowControl::XbeginXabortXend => {
                    push_fragment(&mut fragments, fragment_start, instructions, None);
                    break 'trace stop(
                        DirectJumpTraceStopKind::InterruptOrException,
                        instruction_rva,
                        None,
                        None,
                        Some(format!("{flow:?}")),
                    );
                }
                FlowControl::Call | FlowControl::IndirectCall | FlowControl::Next => {}
            }

            let Some(fallthrough) = next.filter(|rva| is_executable_rva(pe, *rva)) else {
                push_fragment(&mut fragments, fragment_start, instructions, None);
                break 'trace stop(
                    DirectJumpTraceStopKind::UnmappedFallthrough,
                    instruction_rva,
                    None,
                    next,
                    None,
                );
            };
            current = fallthrough;
        }
    };

    Ok(DirectJumpTrace {
        entry,
        fragments,
        stop: terminal,
        decoded_instructions,
        conditional_continuations,
    })
}

#[cfg(test)]
mod tests {
    /// Decode one assembled instruction, for the flag tests below.
    fn one(build: impl FnOnce(&mut iced_x86::code_asm::CodeAssembler)) -> Instruction {
        let mut assembler =
            iced_x86::code_asm::CodeAssembler::new(64).expect("assembler");
        build(&mut assembler);
        let bytes = assembler.assemble(0x1000).expect("assemble");
        let mut decoder =
            iced_x86::Decoder::with_ip(64, &bytes, 0x1000, iced_x86::DecoderOptions::NONE);
        decoder.decode()
    }

    /// `compare_flags` must agree with the architecture's definition of `cmp`,
    /// computed independently here in signed/unsigned arithmetic.
    ///
    /// This is the test that matters most in this module: every other rule only
    /// ever gives up, while a wrong flag here would DECIDE a branch wrongly and
    /// put a path in a CFG that the program never takes.
    #[test]
    fn compare_flags_match_an_independent_reference() {
        for bytes in [1u32, 2, 4, 8] {
            let width = bytes * 8;
            let interesting: Vec<u64> = vec![
                0,
                1,
                2,
                0x7f,
                0x80,
                0xff,
                0x7fff,
                0x8000,
                0xffff,
                0x98ea,
                0x7fff_ffff,
                0x8000_0000,
                0xffff_ffff,
                u64::MAX,
                u64::MAX - 1,
            ];
            for &a in &interesting {
                for &b in &interesting {
                    let (cf, zf, sf, of) = compare_flags(a, b, bytes);
                    let (ma, mb) = (masked(a, bytes), masked(b, bytes));
                    // Unsigned borrow.
                    assert_eq!(cf, ma < mb, "CF {a:#x} {b:#x} w{width}");
                    assert_eq!(zf, ma == mb, "ZF {a:#x} {b:#x} w{width}");
                    // Sign of the truncated difference.
                    let result = masked(ma.wrapping_sub(mb), bytes);
                    assert_eq!(sf, result >> (width - 1) == 1, "SF {a:#x} {b:#x} w{width}");
                    // Signed overflow, via i128 so it cannot itself wrap.
                    let sign = 1i128 << (width - 1);
                    let to_signed = |v: u64| {
                        let v = i128::from(v);
                        if v >= sign { v - (sign << 1) } else { v }
                    };
                    let exact = to_signed(ma) - to_signed(mb);
                    assert_eq!(
                        of,
                        exact < -sign || exact >= sign,
                        "OF {a:#x} {b:#x} w{width}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_tracked_constant_decides_the_comparison_that_blocks_the_sample() {
        use iced_x86::code_asm::{r8d, r8w};
        // The shape at 0xf0aeec4: the register is loaded from an immediate and
        // then compared at 16 bits. 0x1234 <= 0x98ea unsigned, so `jbe` is taken.
        let mut flags = FlagState::default();
        flags.apply(&one(|a| {
            a.mov(r8d, 0x1234u32).expect("mov");
        }));
        flags.apply(&one(|a| {
            a.cmp(r8w, 0x98eai32).expect("cmp");
        }));
        assert_eq!(flags.decides(ConditionCode::be), Some(true));
        assert_eq!(flags.decides(ConditionCode::a), Some(false));
        assert_eq!(flags.decides(ConditionCode::e), Some(false));
    }

    #[test]
    fn an_untracked_write_forgets_the_constant_and_the_branch_stops() {
        use iced_x86::code_asm::{ecx, r8d, r8w};
        let mut flags = FlagState::default();
        flags.apply(&one(|a| {
            a.mov(r8d, 0x1234u32).expect("mov");
        }));
        // An opaque write must invalidate the slot, or the comparison after it
        // would be decided from a value the register no longer holds.
        flags.apply(&one(|a| {
            a.add(r8d, ecx).expect("add");
        }));
        flags.apply(&one(|a| {
            a.cmp(r8w, 0x98eai32).expect("cmp");
        }));
        assert_eq!(flags.decides(ConditionCode::be), None);
    }

    #[test]
    fn a_narrow_immediate_load_does_not_define_the_container() {
        use iced_x86::code_asm::{r8w};
        let mut flags = FlagState::default();
        // `mov r8w, 5` leaves the upper 48 bits as they were, so the container
        // value is NOT known and a later 16-bit compare must not be decided
        // from it -- even though the low half would have been enough.
        flags.apply(&one(|a| {
            a.mov(r8w, 5i32).expect("mov");
        }));
        flags.apply(&one(|a| {
            a.cmp(r8w, 5i32).expect("cmp");
        }));
        assert_eq!(flags.decides(ConditionCode::e), None);
    }

    #[test]
    fn a_zeroing_xor_is_a_tracked_constant() {
        use iced_x86::code_asm::{eax};
        let mut flags = FlagState::default();
        flags.apply(&one(|a| {
            a.xor(eax, eax).expect("xor");
        }));
        assert_eq!(flags.constant(iced_x86::Register::RAX), Some(0));
        flags.apply(&one(|a| {
            a.cmp(eax, 0i32).expect("cmp");
        }));
        assert_eq!(flags.decides(ConditionCode::e), Some(true));
    }

    #[test]
    fn a_known_test_decides_zero_and_sign_too() {
        use iced_x86::code_asm::{eax};
        let mut flags = FlagState::default();
        flags.apply(&one(|a| {
            a.mov(eax, 0x8000_0000u32).expect("mov");
        }));
        flags.apply(&one(|a| {
            a.test(eax, eax).expect("test");
        }));
        // The value is known, so unlike the flag-only rule this gives all four.
        assert_eq!(flags.decides(ConditionCode::e), Some(false));
        assert_eq!(flags.decides(ConditionCode::s), Some(true));
        assert_eq!(flags.decides(ConditionCode::ae), Some(true));
        assert_eq!(flags.decides(ConditionCode::no), Some(true));
    }

    #[test]
    fn a_logical_operation_decides_carry_and_overflow_predicates() {
        use iced_x86::code_asm::{CodeAssembler, eax, ecx};
        // `test eax,ecx` tells us nothing about the value, but CF and OF are
        // cleared by definition, so `jae` and `jno` are decided and `je` is not.
        let mut a = CodeAssembler::new(64).expect("asm");
        a.test(eax, ecx).expect("test");
        let bytes = a.assemble(0x1000).expect("assemble");
        let mut decoder = iced_x86::Decoder::with_ip(64, &bytes, 0x1000, iced_x86::DecoderOptions::NONE);
        let instruction = decoder.decode();

        let mut flags = FlagState::default();
        flags.apply(&instruction);
        assert_eq!(flags.decides(ConditionCode::ae), Some(true));
        assert_eq!(flags.decides(ConditionCode::b), Some(false));
        assert_eq!(flags.decides(ConditionCode::no), Some(true));
        assert_eq!(flags.decides(ConditionCode::o), Some(false));
        // The value is still unknown, so a real zero/sign guard must not decide.
        assert_eq!(flags.decides(ConditionCode::e), None);
        assert_eq!(flags.decides(ConditionCode::s), None);
        // And a signed comparison needs OF *and* SF, so it stays undecided.
        assert_eq!(flags.decides(ConditionCode::l), None);
    }

    #[test]
    fn a_logical_operation_on_memory_decides_nothing() {
        use iced_x86::code_asm::{CodeAssembler, dword_ptr, eax, rbx};
        let mut a = CodeAssembler::new(64).expect("asm");
        a.test(dword_ptr(rbx), eax).expect("test mem");
        let bytes = a.assemble(0x1000).expect("assemble");
        let mut decoder = iced_x86::Decoder::with_ip(64, &bytes, 0x1000, iced_x86::DecoderOptions::NONE);
        let instruction = decoder.decode();
        let mut flags = FlagState::default();
        flags.apply(&instruction);
        assert_eq!(flags.decides(ConditionCode::ae), None);
        assert_eq!(flags.decides(ConditionCode::no), None);
    }

    #[test]
    fn xor_on_one_register_still_gives_the_stronger_result() {
        use iced_x86::code_asm::{CodeAssembler, eax};
        let mut a = CodeAssembler::new(64).expect("asm");
        a.xor(eax, eax).expect("xor");
        let bytes = a.assemble(0x1000).expect("assemble");
        let mut decoder = iced_x86::Decoder::with_ip(64, &bytes, 0x1000, iced_x86::DecoderOptions::NONE);
        let instruction = decoder.decode();
        let mut flags = FlagState::default();
        flags.apply(&instruction);
        // The general logical rule would leave ZF unknown; the exact arm must win.
        assert_eq!(flags.decides(ConditionCode::e), Some(true));
        assert_eq!(flags.decides(ConditionCode::ae), Some(true));
    }

    use super::*;

    fn executable_pe64(code: &[u8]) -> Vec<u8> {
        let mut data = vec![0_u8; 0x400];
        data[0..2].copy_from_slice(&0x5A4D_u16.to_le_bytes());
        data[0x3C..0x40].copy_from_slice(&(0x80_u32).to_le_bytes());
        data[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
        let coff = 0x84;
        data[coff..coff + 2].copy_from_slice(&(0x8664_u16).to_le_bytes());
        data[coff + 2..coff + 4].copy_from_slice(&(1_u16).to_le_bytes());
        data[coff + 16..coff + 18].copy_from_slice(&(0xF0_u16).to_le_bytes());
        data[coff + 18..coff + 20].copy_from_slice(&(0x2022_u16).to_le_bytes());
        let optional = coff + 20;
        data[optional..optional + 2].copy_from_slice(&0x20B_u16.to_le_bytes());
        data[optional + 16..optional + 20].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 20..optional + 24].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&(0x1_4000_0000_u64).to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[optional + 56..optional + 60].copy_from_slice(&(0x2000_u32).to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&(3_u16).to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&(16_u32).to_le_bytes());
        let section = optional + 0xF0;
        data[section..section + 5].copy_from_slice(b".text");
        data[section + 8..section + 12].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&(0x6000_0020_u32).to_le_bytes());
        data[0x200..0x200 + code.len()].copy_from_slice(code);
        data
    }

    #[test]
    fn linearizes_direct_jump_fragments_until_indirect_dispatch() {
        let mut code = vec![0x90_u8; 0x20];
        code[0..8].copy_from_slice(&[0x48, 0x31, 0xC0, 0xE9, 0x08, 0x00, 0x00, 0x00]);
        code[0x10..0x16].copy_from_slice(&[0x48, 0x83, 0xC0, 0x01, 0xFF, 0xE0]);
        let data = executable_pe64(&code);
        let pe = PeImage::parse(&data).expect("valid synthetic PE");
        let trace = trace_direct_jump_fragments(&pe, Rva(0x1000), DirectJumpTraceLimits::default())
            .expect("trace must decode");

        assert_eq!(trace.fragments.len(), 2);
        assert_eq!(trace.fragments[0].next_fragment, Some(Rva(0x1010)));
        assert_eq!(trace.fragments[1].next_fragment, None);
        assert_eq!(trace.decoded_instructions, 4);
        assert_eq!(trace.stop.kind, DirectJumpTraceStopKind::IndirectBranch);
        assert_eq!(trace.stop.at, Rva(0x1014));
        assert!(trace.ends_in_indirect_branch());
    }

    #[test]
    fn detects_direct_jump_cycles_without_looping() {
        let data = executable_pe64(&[0xE9, 0xFB, 0xFF, 0xFF, 0xFF]);
        let pe = PeImage::parse(&data).expect("valid synthetic PE");
        let trace = trace_direct_jump_fragments(&pe, Rva(0x1000), DirectJumpTraceLimits::default())
            .expect("trace must decode");

        assert_eq!(trace.fragments.len(), 1);
        assert_eq!(trace.decoded_instructions, 1);
        assert_eq!(
            trace.stop.kind,
            DirectJumpTraceStopKind::RepeatedInstruction
        );
        assert_eq!(trace.stop.at, Rva(0x1000));
    }

    #[test]
    fn enforces_zero_resource_limits_without_decoding() {
        let data = executable_pe64(&[0x90]);
        let pe = PeImage::parse(&data).expect("valid synthetic PE");
        let trace = trace_direct_jump_fragments(
            &pe,
            Rva(0x1000),
            DirectJumpTraceLimits {
                follow_conditional_fallthrough: false,
                follow_conditional_target: false,
                max_conditional_continuations: 1,
                max_fragments: 0,
                max_instructions: 0,
                max_instructions_per_fragment: 0,
            },
        )
        .expect("entry validation is independent from resource traversal");

        assert!(trace.fragments.is_empty());
        assert_eq!(trace.decoded_instructions, 0);
        assert_eq!(trace.stop.kind, DirectJumpTraceStopKind::ResourceLimit);
    }
}
