use std::collections::BTreeSet;

use iced_x86::{ConditionCode, FlowControl, Instruction, Mnemonic, OpKind, RflagsBits};
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
#[derive(Clone, Copy, Default)]
struct FlagState {
    cf: Option<bool>,
    zf: Option<bool>,
    sf: Option<bool>,
    of: Option<bool>,
}

impl FlagState {
    /// Folds one instruction's effect on the flags.
    ///
    /// Only the forms VMProtect uses to build an opaque predicate are tracked
    /// exactly; every other flag write makes the flags it touches unknown, which
    /// is what keeps this from deciding a branch it has not actually proven.
    fn apply(&mut self, instruction: &Instruction) {
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
