//! Native execution of VMProtect's byte-at-a-time copy loop.
//!
//! The phase after integrity checking copies large regions one byte per
//! iteration, at ~11 interpreted instructions per byte. The shape, as observed
//! in systems.xpl at rva 0xefd3649:
//!
//! ```text
//!   head: mov  scratch8, [source]
//!         inc  destination
//!         inc  source
//!         sub  counter32, 1
//!         mov  [destination + other - 1], scratch8
//!         je   exit                 ; counter reached zero
//!         cmp  destination, -1
//!         jb   head
//! ```
//!
//! Unlike the checksum recurrence this loop WRITES guest memory, so a fold has
//! to go through the ordinary write path (page permissions, and the code
//! generation bump that revalidates cached blocks) and has to decline when a
//! hook wants the writes.
//!
//! `cmp` is NOT flag noise here: the `jb` reads its carry, so the comparison is
//! part of the loop's exit condition and skipping it would hide a way out. Only
//! the carry-only instructions are treated as noise.

use iced_x86::{Code, Instruction, Mnemonic, OpKind, Register};

/// A matched copy loop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Shape {
    /// Reads the source bytes; advances by one per iteration.
    pub source: Register,
    /// Incremented each iteration and used to form the store address.
    pub destination: Register,
    /// The other register in the store's effective address; constant.
    pub other: Register,
    /// Displacement in the store's effective address.
    pub displacement: i64,
    /// 32-bit countdown; the loop leaves when it reaches zero.
    pub counter: Register,
    /// Byte register the value passes through.
    pub scratch: Register,
    /// Loop head.
    pub head: u64,
    /// Control transfers inside the body, as `(from, to)`.
    pub flow_sites: [(u64, u64); 8],
    pub flow_site_count: u8,
}

/// Only carry-only instructions are noise. `cmp`/`test` are load-bearing here.
fn is_flag_noise(instruction: &Instruction) -> bool {
    matches!(instruction.mnemonic(), Mnemonic::Clc | Mnemonic::Stc | Mnemonic::Cmc)
}

fn full(register: Register) -> Register {
    register.full_register()
}

impl Shape {
    /// Matches the copy loop at `head`, or `None` for anything else.
    pub(crate) fn recognize(
        head: u64,
        mut decode: impl FnMut(u64) -> Option<Instruction>,
    ) -> Option<Self> {
        let mut shape = Self {
            source: Register::None,
            destination: Register::None,
            other: Register::None,
            displacement: 0,
            counter: Register::None,
            scratch: Register::None,
            head,
            flow_sites: [(0, 0); 8],
            flow_site_count: 0,
        };
        let mut address = head;
        let mut stage = 0u8;
        let mut steps = 0u32;
        let mut record = |shape: &mut Self, from: u64, to: u64| -> bool {
            let slot = usize::from(shape.flow_site_count);
            if slot >= shape.flow_sites.len() {
                return false;
            }
            shape.flow_sites[slot] = (from, to);
            shape.flow_site_count += 1;
            true
        };
        while steps < 64 {
            steps += 1;
            let instruction = decode(address)?;
            let length = u64::from(u32::try_from(instruction.len()).ok()?);
            let next = address.wrapping_add(length);
            if is_flag_noise(&instruction) {
                address = next;
                continue;
            }
            if instruction.mnemonic() == Mnemonic::Jmp
                && instruction.op0_kind() == OpKind::NearBranch64
            {
                let target = instruction.near_branch64();
                if !record(&mut shape, address, target) {
                    return None;
                }
                address = target;
                continue;
            }
            match stage {
                // mov scratch8, [source]
                0 => {
                    if instruction.code() != Code::Mov_r8_rm8
                        || instruction.op1_kind() != OpKind::Memory
                        || instruction.memory_index() != Register::None
                        || instruction.memory_base() == Register::None
                        || instruction.memory_displacement64() != 0
                    {
                        return None;
                    }
                    shape.scratch = instruction.op0_register();
                    shape.source = full(instruction.memory_base());
                }
                // inc destination
                1 => {
                    if instruction.mnemonic() != Mnemonic::Inc
                        || instruction.op0_kind() != OpKind::Register
                    {
                        return None;
                    }
                    shape.destination = full(instruction.op0_register());
                }
                // inc source
                2 => {
                    if instruction.mnemonic() != Mnemonic::Inc
                        || instruction.op0_kind() != OpKind::Register
                        || full(instruction.op0_register()) != shape.source
                    {
                        return None;
                    }
                }
                // sub counter32, 1
                3 => {
                    if instruction.mnemonic() != Mnemonic::Sub
                        || instruction.op0_kind() != OpKind::Register
                        || instruction.op0_register().size() != 4
                        || instruction.try_immediate(1).ok() != Some(1)
                    {
                        return None;
                    }
                    shape.counter = instruction.op0_register();
                }
                // mov [destination + other + displacement], scratch8
                4 => {
                    if instruction.code() != Code::Mov_rm8_r8
                        || instruction.op0_kind() != OpKind::Memory
                        || instruction.op1_register() != shape.scratch
                        || instruction.memory_index_scale() != 1
                        || instruction.memory_base() == Register::None
                        || instruction.memory_index() == Register::None
                    {
                        return None;
                    }
                    // iced may put the incremented register in either slot.
                    let base = full(instruction.memory_base());
                    let index = full(instruction.memory_index());
                    shape.other = if base == shape.destination {
                        index
                    } else if index == shape.destination {
                        base
                    } else {
                        return None;
                    };
                    #[allow(clippy::cast_possible_wrap)]
                    {
                        shape.displacement = instruction.memory_displacement64() as i64;
                    }
                }
                // je exit -- taken exactly when the counter hit zero
                5 => {
                    if instruction.mnemonic() != Mnemonic::Je
                        || instruction.op0_kind() != OpKind::NearBranch64
                    {
                        return None;
                    }
                    if !record(&mut shape, address, instruction.near_branch64()) {
                        return None;
                    }
                }
                // cmp destination, -1
                6 => {
                    if instruction.mnemonic() != Mnemonic::Cmp
                        || instruction.op0_kind() != OpKind::Register
                        || full(instruction.op0_register()) != shape.destination
                        || instruction.try_immediate(1).ok() != Some(u64::MAX)
                    {
                        return None;
                    }
                }
                // jb head -- taken while the destination is below u64::MAX
                7 => {
                    if instruction.mnemonic() != Mnemonic::Jb
                        || instruction.op0_kind() != OpKind::NearBranch64
                        || instruction.near_branch64() != head
                    {
                        return None;
                    }
                    if !record(&mut shape, address, head) {
                        return None;
                    }
                    // Roles must be distinct, and neither pointer may alias the
                    // counter, or the fold would not model the loop.
                    let counter = full(shape.counter);
                    if shape.source == shape.destination
                        || shape.source == shape.other
                        || counter == shape.source
                        || counter == shape.destination
                        || counter == shape.other
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

    /// The store address for the iteration that leaves `destination` at
    /// `destination_after`, written exactly as the instruction forms it.
    pub(crate) fn store_address(&self, destination_after: u64, other: u64) -> u64 {
        #[allow(clippy::cast_sign_loss)]
        destination_after.wrapping_add(other).wrapping_add(self.displacement as u64)
    }
}

#[cfg(test)]
mod tests {
    use iced_x86::code_asm::*;
    use iced_x86::{Decoder, DecoderOptions};

    use super::*;

    /// The body observed at rva 0xefd3649.
    pub(crate) fn body(head: u64) -> Vec<u8> {
        let mut a = CodeAssembler::new(64).expect("asm");
        let mut loop_head = a.create_label();
        let mut exit = a.create_label();
        a.set_label(&mut loop_head).expect("label");
        a.mov(r11b, byte_ptr(rax)).expect("load");
        a.inc(r14).expect("inc dest");
        a.inc(rax).expect("inc src");
        a.sub(ebp, 1).expect("sub");
        a.mov(byte_ptr(r14 + rdx - 1), r11b).expect("store");
        a.je(exit).expect("je");
        a.cmp(r14, -1).expect("cmp");
        a.jb(loop_head).expect("jb");
        a.set_label(&mut exit).expect("exit");
        a.hlt().expect("hlt");
        a.assemble(head).expect("assemble")
    }

    fn decoder_over(bytes: Vec<u8>, head: u64) -> impl FnMut(u64) -> Option<Instruction> {
        move |address: u64| {
            let offset = usize::try_from(address.checked_sub(head)?).ok()?;
            if offset >= bytes.len() {
                return None;
            }
            let mut decoder = Decoder::with_ip(64, &bytes[offset..], address, DecoderOptions::NONE);
            let instruction = decoder.decode();
            (!instruction.is_invalid()).then_some(instruction)
        }
    }

    #[test]
    fn recognizes_the_observed_copy_body() {
        const HEAD: u64 = 0x0EFD_3649;
        let shape = Shape::recognize(HEAD, decoder_over(body(HEAD), HEAD)).expect("recognized");
        assert_eq!(shape.source, Register::RAX);
        assert_eq!(shape.destination, Register::R14);
        assert_eq!(shape.other, Register::RDX);
        assert_eq!(shape.displacement, -1);
        assert_eq!(shape.counter, Register::EBP);
        assert_eq!(shape.scratch, Register::R11L);
        let last = usize::from(shape.flow_site_count) - 1;
        assert_eq!(shape.flow_sites[last].1, HEAD);
    }

    #[test]
    fn a_backward_branch_elsewhere_is_not_this_loop() {
        const HEAD: u64 = 0x0EFD_3649;
        let mut bytes = body(HEAD);
        let last = bytes.len() - 5;
        bytes[last..last + 4].copy_from_slice(&0x4321_i32.to_le_bytes());
        assert!(Shape::recognize(HEAD, decoder_over(bytes, HEAD)).is_none());
    }

    #[test]
    fn the_store_address_matches_the_instruction_form() {
        let shape = Shape {
            source: Register::RAX,
            destination: Register::R14,
            other: Register::RDX,
            displacement: -1,
            counter: Register::EBP,
            scratch: Register::R11L,
            head: 0,
            flow_sites: [(0, 0); 8],
            flow_site_count: 0,
        };
        // `inc r14` then `mov [r14+rdx-1]`: with r14 starting at D, iteration k
        // stores at D + k + rdx.
        let (start, other) = (0x4000u64, 0x1_0000u64);
        for k in 0..8u64 {
            assert_eq!(shape.store_address(start + k + 1, other), start + k + other);
        }
    }
}
