use std::collections::BTreeSet;

use iced_x86::FlowControl;
use serde::Serialize;
use vmpx_pe::{PeImage, Rva};

use crate::{DecodedInstruction, DisasmError, decode_one, is_executable_rva};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectJumpTraceLimits {
    pub max_fragments: usize,
    pub max_instructions: usize,
    pub max_instructions_per_fragment: usize,
}

impl Default for DirectJumpTraceLimits {
    fn default() -> Self {
        Self {
            max_fragments: 1_024,
            max_instructions: 50_000,
            max_instructions_per_fragment: 10_000,
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
