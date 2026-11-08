//! Native execution of VMProtect's byte-at-a-time integrity-checksum loop.
//!
//! The protector verifies large regions of its own image with a table-driven
//! recurrence, one input byte per iteration. Interpreted, that costs ~17
//! instructions per byte (the recurrence plus `clc`/`stc`/`cmc`/`cmp` flag
//! noise), so on a 253MB image a detonation can spend billions of steps inside
//! it and never reach the code worth observing.
//!
//! [`Shape::recognize`] matches the loop structurally from the decoded stream;
//! on a match the caller runs the recurrence directly, which is one operation
//! per byte instead of seventeen.
//!
//! Soundness rests on three properties, each checked rather than assumed:
//!
//! 1. Flags are dead ACROSS iterations. Every flag the body writes is consumed
//!    by that same iteration's `jne`, so folding cannot lose a flag another
//!    iteration would have read. The caller still leaves the LAST iteration to
//!    the interpreter, so the exit state -- flags included -- is produced by the
//!    ordinary instruction semantics and never computed here.
//! 2. Every byte read is checked against guest memory through the normal read
//!    path, so an unmapped or unreadable address declines the fold instead of
//!    faulting or inventing a value.
//! 3. The recurrence only reads. It writes no guest memory, so no hook write
//!    event can be missed; the caller additionally declines when a hook wants
//!    read or flow events, which folding would elide.

use iced_x86::{Code, Instruction, Mnemonic, OpKind, Register};

/// A matched checksum loop: which registers carry what, and the constants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Shape {
    /// Holds the 32-bit accumulator.
    pub accumulator: Register,
    /// Walks the input bytes; advances by one per iteration.
    pub pointer: Register,
    /// Base of the 4-byte-entry lookup table.
    pub table: Register,
    /// Counts iterations down; the loop exits when it reaches zero.
    pub counter: Register,
    /// XORed into the accumulator once per iteration.
    pub iteration_xor: u32,
    /// Address the backward branch returns to (the loop head).
    pub head: u64,
    /// Scratch register the table entry lands in.
    pub scratch: Register,
    /// Control transfers inside the body, as `(from, to)`. A fold elides these,
    /// so the caller must check no hook wants them.
    pub flow_sites: [(u64, u64); 2],
    pub flow_site_count: u8,
}

/// Instructions that only touch flags, which the protector interleaves as noise.
/// Skipping them is safe because the body's flags are dead across iterations.
fn is_flag_noise(instruction: &Instruction) -> bool {
    // A memory operand is not noise: it could fault, or a hook could want the
    // read, and folding would elide both.
    let touches_memory = (0..instruction.op_count())
        .any(|op| instruction.op_kind(op) == OpKind::Memory);
    if touches_memory {
        return false;
    }
    matches!(
        instruction.mnemonic(),
        Mnemonic::Clc | Mnemonic::Stc | Mnemonic::Cmc | Mnemonic::Cmp | Mnemonic::Test
    )
}

/// `reg` as its 64-bit container, so `esi` and `rsi` compare equal.
fn full(register: Register) -> Register {
    register.full_register()
}

/// How many pads a walk follows before the first stage matches. Past this an
/// address is treated as not being this loop at all.
const PRE_MATCH_STEPS: u32 = 8;

impl Shape {
    /// Matches the recurrence starting at `head`, following unconditional jumps
    /// and ignoring flag noise. `decode` yields the instruction at an address.
    ///
    /// Returns `None` on anything that is not exactly this loop, so an
    /// unrecognized shape simply runs interpreted.
    pub(crate) fn recognize(
        head: u64,
        mut decode: impl FnMut(u64) -> Option<Instruction>,
    ) -> Option<Self> {
        // Step 0: `movzx r32, byte [pointer]`.
        let mut address = head;
        let mut steps = 0u32;
        let mut shape = Self {
            accumulator: Register::None,
            pointer: Register::None,
            table: Register::None,
            counter: Register::None,
            iteration_xor: 0,
            head,
            scratch: Register::None,
            flow_sites: [(0, 0); 2],
            flow_site_count: 0,
        };
        let mut stage = 0u8;
        // A generous cap: the real body is ~10 operations plus noise, and the
        // bound keeps a mismatched address from walking the whole section.
        while steps < 64 {
            steps += 1;
            // Give up early while nothing has matched yet.
            //
            // `run_loop` probes a shape at every block start, and the protector
            // pads its code with `jmp +0` and flag-only junk, both of which this
            // walk follows. At a non-matching address that meant chasing the
            // padding to the step cap before failing, so a MISS cost about as
            // much as a hit: measured on systems.xpl's init, the probes were
            // taking the emulator from 192M to 110M steps/s. A real loop reaches
            // its first operation within a few pads, so capping the chase before
            // the first match keeps every other address cheap.
            if stage == 0 && steps > PRE_MATCH_STEPS {
                return None;
            }
            let instruction = decode(address)?;
            let length = u64::from(u32::try_from(instruction.len()).ok()?);
            let next = address.wrapping_add(length);
            if is_flag_noise(&instruction) {
                address = next;
                continue;
            }
            // Follow the unconditional jump that splits the body into blocks.
            if instruction.mnemonic() == Mnemonic::Jmp
                && instruction.op0_kind() == OpKind::NearBranch64
            {
                let target = instruction.near_branch64();
                let slot = usize::from(shape.flow_site_count);
                if slot >= shape.flow_sites.len() {
                    return None;
                }
                shape.flow_sites[slot] = (address, target);
                shape.flow_site_count += 1;
                address = target;
                continue;
            }
            match stage {
                // movzx scratch, byte [pointer]
                0 => {
                    if instruction.mnemonic() != Mnemonic::Movzx
                        || instruction.op0_kind() != OpKind::Register
                        || instruction.op1_kind() != OpKind::Memory
                        || instruction.memory_displ_size() != 0
                        || instruction.memory_index() != Register::None
                        || instruction.memory_base() == Register::None
                        || instruction.memory_displacement64() != 0
                        || instruction.code() != Code::Movzx_r32_rm8
                    {
                        return None;
                    }
                    shape.scratch = instruction.op0_register();
                    shape.pointer = full(instruction.memory_base());
                }
                // xor scratch, accumulator
                1 => {
                    if instruction.mnemonic() != Mnemonic::Xor
                        || instruction.op0_kind() != OpKind::Register
                        || instruction.op1_kind() != OpKind::Register
                        || instruction.op0_register() != shape.scratch
                    {
                        return None;
                    }
                    shape.accumulator = instruction.op1_register();
                }
                // and scratch, 0xFF
                2 => {
                    if instruction.mnemonic() != Mnemonic::And
                        || instruction.op0_register() != shape.scratch
                        || !matches!(instruction.op1_kind(), OpKind::Immediate32)
                        || instruction.immediate32() != 0xFF
                    {
                        return None;
                    }
                }
                // mov scratch, [table + scratch*4]
                3 => {
                    if instruction.mnemonic() != Mnemonic::Mov
                        || instruction.op0_register() != shape.scratch
                        || instruction.op1_kind() != OpKind::Memory
                        || instruction.memory_index_scale() != 4
                        || full(instruction.memory_index()) != full(shape.scratch)
                        || instruction.memory_base() == Register::None
                        || instruction.memory_displacement64() != 0
                    {
                        return None;
                    }
                    shape.table = full(instruction.memory_base());
                }
                // shr accumulator, 8
                4 => {
                    if instruction.mnemonic() != Mnemonic::Shr
                        || instruction.op0_register() != shape.accumulator
                        || !matches!(instruction.op1_kind(), OpKind::Immediate8)
                        || instruction.immediate8() != 8
                    {
                        return None;
                    }
                }
                // xor accumulator, scratch
                5 => {
                    if instruction.mnemonic() != Mnemonic::Xor
                        || instruction.op0_register() != shape.accumulator
                        || instruction.op1_kind() != OpKind::Register
                        || instruction.op1_register() != shape.scratch
                    {
                        return None;
                    }
                }
                // inc pointer
                6 => {
                    if instruction.mnemonic() != Mnemonic::Inc
                        || full(instruction.op0_register()) != shape.pointer
                    {
                        return None;
                    }
                }
                // xor accumulator, imm32
                7 => {
                    if instruction.mnemonic() != Mnemonic::Xor
                        || instruction.op0_register() != shape.accumulator
                        || !matches!(instruction.op1_kind(), OpKind::Immediate32)
                    {
                        return None;
                    }
                    shape.iteration_xor = instruction.immediate32();
                }
                // dec counter
                8 => {
                    if instruction.mnemonic() != Mnemonic::Dec
                        || instruction.op0_kind() != OpKind::Register
                    {
                        return None;
                    }
                    shape.counter = full(instruction.op0_register());
                    // The counter must not alias anything the body also uses.
                    if [shape.pointer, shape.table, full(shape.accumulator)]
                        .contains(&shape.counter)
                    {
                        return None;
                    }
                }
                // jne head
                9 => {
                    if instruction.mnemonic() != Mnemonic::Jne
                        || instruction.op0_kind() != OpKind::NearBranch64
                        || instruction.near_branch64() != head
                    {
                        return None;
                    }
                    let slot = usize::from(shape.flow_site_count);
                    if slot >= shape.flow_sites.len() {
                        return None;
                    }
                    shape.flow_sites[slot] = (address, head);
                    shape.flow_site_count += 1;
                    // Distinct roles, or a fold would not model the loop.
                    if shape.pointer == shape.table
                        || shape.pointer == full(shape.accumulator)
                        || shape.table == full(shape.accumulator)
                    {
                        return None;
                    }
                    return Some(shape);
                }
                _ => return None,
            }
            stage += 1;
            address = next;
        }
        None
    }

    /// One iteration of the recurrence: the table index, then the accumulator.
    ///
    /// `index = (input ^ accumulator) & 0xFF`, then
    /// `accumulator = ((accumulator >> 8) ^ table[index]) ^ iteration_xor`.
    pub(crate) const fn step(&self, accumulator: u32, input: u8, entry: u32) -> u32 {
        let _ = input;
        ((accumulator >> 8) ^ entry) ^ self.iteration_xor
    }

    /// The table index this iteration reads.
    pub(crate) const fn index(accumulator: u32, input: u8) -> u8 {
        (accumulator as u8) ^ input
    }
}

#[cfg(test)]
mod tests {
    use iced_x86::{Decoder, DecoderOptions};

    use super::*;

    /// The exact body observed in systems.xpl at rva 0xf04f107, flag noise and
    /// the block-splitting `jmp` included.
    fn body(head: u64) -> Vec<u8> {
        let mut assembler = iced_x86::code_asm::CodeAssembler::new(64).expect("asm");
        use iced_x86::code_asm::*;
        assembler.movzx(esi, byte_ptr(rbx)).expect("movzx");
        assembler.xor(esi, r15d).expect("xor");
        assembler.clc().expect("clc");
        assembler.and(esi, 0xFF).expect("and");
        assembler.mov(esi, dword_ptr(rdx + rsi * 4)).expect("mov");
        assembler.stc().expect("stc");
        assembler.shr(r15d, 8u32).expect("shr");
        assembler.cmc().expect("cmc");
        assembler.xor(r15d, esi).expect("xor2");
        assembler.inc(rbx).expect("inc");
        assembler.xor(r15d, 0xFEFD_F493u32 as i32).expect("xor3");
        assembler.dec(r13).expect("dec");
        assembler.jne(head).expect("jne");
        assembler.assemble(head).expect("assemble")
    }

    fn decoder_over(bytes: Vec<u8>, head: u64) -> impl FnMut(u64) -> Option<Instruction> {
        move |address: u64| {
            let offset = usize::try_from(address.checked_sub(head)?).ok()?;
            if offset >= bytes.len() {
                return None;
            }
            let mut decoder =
                Decoder::with_ip(64, &bytes[offset..], address, DecoderOptions::NONE);
            let instruction = decoder.decode();
            (!instruction.is_invalid()).then_some(instruction)
        }
    }

    #[test]
    fn recognizes_the_observed_checksum_body() {
        const HEAD: u64 = 0x0F04_F107;
        let shape = Shape::recognize(HEAD, decoder_over(body(HEAD), HEAD)).expect("recognized");
        assert_eq!(shape.pointer, Register::RBX);
        assert_eq!(shape.table, Register::RDX);
        assert_eq!(shape.counter, Register::R13);
        assert_eq!(shape.accumulator, Register::R15D);
        assert_eq!(shape.scratch, Register::ESI);
        assert_eq!(shape.iteration_xor, 0xFEFD_F493);
        assert_eq!(shape.head, HEAD);
        // At least the backward branch; the assembler elides a `jmp` to the
        // next instruction, so the real sample's body-splitting `jmp` is a
        // second site there but not here. The last site must close the loop.
        assert!(shape.flow_site_count >= 1);
        let last = usize::from(shape.flow_site_count) - 1;
        assert_eq!(shape.flow_sites[last].1, HEAD);
    }

    #[test]
    fn a_body_that_branches_somewhere_else_is_not_this_loop() {
        const HEAD: u64 = 0x0F04_F107;
        // Same body, but the backward branch does not return to the head, so
        // folding would skip whatever the other target does.
        let mut bytes = body(HEAD);
        let last = bytes.len() - 4;
        bytes[last..].copy_from_slice(&0x1234_i32.to_le_bytes());
        assert!(Shape::recognize(HEAD, decoder_over(bytes, HEAD)).is_none());
    }

    #[test]
    fn the_recurrence_matches_a_byte_at_a_time_reference() {
        let shape = Shape {
            accumulator: Register::R15D,
            pointer: Register::RBX,
            table: Register::RDX,
            counter: Register::R13,
            iteration_xor: 0xFEFD_F493,
            head: 0,
            scratch: Register::ESI,
            flow_sites: [(0, 0); 2],
            flow_site_count: 0,
        };
        // A table standing in for the protector's; any values exercise the shape.
        let table: Vec<u32> = (0..256u32).map(|i| i.wrapping_mul(0x0100_1001) ^ 0xA5A5).collect();
        let input: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37).wrapping_add(11)).collect();

        let mut folded = 0x1234_5678u32;
        for byte in &input {
            let entry = table[usize::from(Shape::index(folded, *byte))];
            folded = shape.step(folded, *byte, entry);
        }

        // The same thing written out longhand, as the loop body does it.
        let mut reference = 0x1234_5678u32;
        for byte in &input {
            let mut scratch = u32::from(*byte);
            scratch ^= reference;
            scratch &= 0xFF;
            scratch = table[scratch as usize];
            reference >>= 8;
            reference ^= scratch;
            reference ^= 0xFEFD_F493;
        }
        assert_eq!(folded, reference);
    }
}
