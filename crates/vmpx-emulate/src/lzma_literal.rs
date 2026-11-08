//! Native execution of the LZMA literal decoder the protector's init runs.
//!
//! VMProtect decompresses its packed sections with LZMA, and the literal
//! decoder is where the init actually goes: profiling the first TLS callback
//! puts almost every sample in one ~150-byte loop, and 120 billion interpreted
//! steps were not enough to finish it. Each output byte costs eight bit
//! decodes, and each bit decode is ~30 interpreted instructions once the
//! protector's junk is counted.
//!
//! The loop is a textbook range-coder bit tree, scattered across fragments
//! joined by direct jumps and padded with instructions whose results die before
//! anything reads them:
//!
//! ```text
//!   head:  movsxd rdx,r10d              index = symbol
//!          cmp   r8d,1000000h           range < kTopValue ?
//!          jae   skip_normalise
//!          cmp   r9,r13                 input exhausted ?
//!          je    exhausted
//!          movzx eax,byte ptr [r9]
//!          shl   edi,8                  code  <<= 8
//!          shl   r8d,8                  range <<= 8
//!          or    edi,eax                code |= byte
//!          inc   r9
//!   skip:  movzx ecx,word ptr [rbx+rdx*2]   prob
//!          mov   eax,r8d
//!          shr   eax,0Bh                range >> 11   kNumBitModelTotalBits
//!          imul  eax,ecx                bound
//!          cmp   edi,eax
//!          jae   one
//!          mov   r8d,eax                range = bound
//!          mov   eax,800h               kBitModelTotal
//!          sub   eax,ecx
//!          sar   eax,5                  kNumMoveBits
//!          add   ax,cx                  prob += (2048-prob)>>5
//!          add   r10d,r10d              symbol <<= 1
//!          mov   [rbx+rdx*2],ax
//!          jmp   tail
//!   one:   sub   r8d,eax                range -= bound
//!          sub   edi,eax                code  -= bound
//!          movzx eax,cx
//!          shr   ax,5
//!          lea   r10d,[r10+r10+1]       symbol = symbol*2 + 1
//!          sub   cx,ax                  prob -= prob>>5
//!          mov   [rbx+rdx*2],cx
//!   tail:  cmp   r10d,100h
//!          jl    head
//! ```
//!
//! Soundness follows the integrity-checksum fold, with one addition because
//! this loop WRITES guest memory:
//!
//! 1. The junk writes registers, not just flags, so skipping it needs more than
//!    the checksum loop's flag-noise rule. Every skipped write must land in a
//!    register the body itself overwrites before the next iteration reads it
//!    (`rax` by `mov eax,r8d`, `rcx` by `movzx ecx,[rbx+rdx*2]`), and the
//!    matcher requires both of those rewrites to be present.
//! 2. Every byte read -- input and probability -- goes through the normal guest
//!    read path, so an unmapped address declines the fold instead of faulting
//!    or inventing a value.
//! 3. The probability store is a real guest write, so the caller must decline
//!    when a hook wants writes, reads or the elided control transfers.
//!
//! The caller leaves the LAST bit to the interpreter, so the exit state --
//! flags included -- is produced by ordinary instruction semantics and never
//! computed here.

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

/// `kNumBitModelTotalBits`: the shift applied to `range` before scaling.
const MODEL_TOTAL_BITS: u32 = 11;
/// `kBitModelTotal`: probabilities live in `[0, 2048)`.
const BIT_MODEL_TOTAL: u32 = 1 << MODEL_TOTAL_BITS;
/// `kNumMoveBits`: how fast a probability adapts.
const MOVE_BITS: u32 = 5;
/// `kTopValue`: `range` below this pulls another input byte.
const TOP_VALUE: u32 = 1 << 24;
/// The bit tree ends when the symbol reaches this, having accumulated 8 bits.
const SYMBOL_DONE: u32 = 0x100;

/// A matched literal decoder: which register carries what.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Shape {
    /// Accumulates the symbol; starts at 1, ends at or past [`SYMBOL_DONE`].
    pub symbol: Register,
    /// Base of the 16-bit probability array, indexed by the symbol.
    pub probs: Register,
    /// Range-coder range.
    pub range: Register,
    /// Range-coder code.
    pub code: Register,
    /// Walks the compressed input.
    pub input: Register,
    /// One past the last readable input byte.
    pub input_end: Register,
    /// The loop head, which the back edge returns to.
    pub head: u64,
    /// Control transfers inside the body that a fold elides, as `(from, to)`.
    pub flow_sites: [(u64, u64); 4],
    pub flow_site_count: u8,
}

/// One decoded bit's effect on the range coder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Step {
    pub range: u32,
    pub code: u32,
    pub symbol: u32,
    pub probability: u16,
    /// Where the probability belongs, as a byte offset from `probs`.
    pub probability_offset: u64,
}

impl Shape {
    /// One bit of the tree, given the current coder state and the probability
    /// just read. Pure: the caller performs the guest reads and the store.
    #[must_use]
    pub(crate) fn step(range: u32, code: u32, symbol: u32, probability: u16) -> Step {
        let probability_offset = u64::from(symbol) * 2;
        let bound = (range >> MODEL_TOTAL_BITS).wrapping_mul(u32::from(probability));
        if code < bound {
            // The model moves toward zero and the symbol takes a 0 bit.
            let moved = u32::from(probability) + ((BIT_MODEL_TOTAL - u32::from(probability)) >> MOVE_BITS);
            Step {
                range: bound,
                code,
                symbol: symbol << 1,
                probability: moved as u16,
                probability_offset,
            }
        } else {
            let moved = u32::from(probability) - (u32::from(probability) >> MOVE_BITS);
            Step {
                range: range - bound,
                code: code - bound,
                symbol: (symbol << 1) | 1,
                probability: moved as u16,
                probability_offset,
            }
        }
    }

    /// Whether `range` is small enough that the coder pulls another input byte.
    #[must_use]
    pub(crate) fn needs_normalise(range: u32) -> bool {
        range < TOP_VALUE
    }

    /// `range` and `code` after folding one input byte in.
    #[must_use]
    pub(crate) fn normalise(range: u32, code: u32, byte: u8) -> (u32, u32) {
        (range << 8, (code << 8) | u32::from(byte))
    }

    /// Whether the tree has accumulated a whole byte.
    #[must_use]
    pub(crate) fn complete(symbol: u32) -> bool {
        symbol >= SYMBOL_DONE
    }
}

/// Registers the body is proven to overwrite before the next iteration reads
/// them, so junk writing them can be skipped.
///
/// Both rewrites are required by the matcher, so this is a checked property and
/// not an assumption about what the protector happens to emit.
fn is_dead_scratch(register: Register, scratch: [Register; 2]) -> bool {
    scratch.iter().any(|dead| dead.full_register() == register.full_register())
}

/// Instructions the protector interleaves that cannot affect the fold.
///
/// Flag-only noise, or a write to a register the body rewrites. A memory
/// operand is never noise: it could fault, or a hook could want the access, and
/// folding would elide both.
pub(crate) fn is_skippable(instruction: &Instruction, scratch: [Register; 2]) -> bool {
    let touches_memory =
        (0..instruction.op_count()).any(|op| instruction.op_kind(op) == OpKind::Memory);
    if touches_memory {
        return false;
    }
    if matches!(
        instruction.mnemonic(),
        Mnemonic::Clc | Mnemonic::Stc | Mnemonic::Cmc | Mnemonic::Cmp | Mnemonic::Test
    ) {
        return true;
    }
    instruction.op0_kind() == OpKind::Register
        && is_dead_scratch(instruction.op0_register(), scratch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_bit_narrows_the_range_and_shifts_the_symbol_left() {
        // code below bound selects the 0 branch.
        let step = Shape::step(0x0100_0000, 0, 1, 1024);
        assert_eq!(step.symbol, 2);
        assert_eq!(step.range, (0x0100_0000u32 >> 11) * 1024);
        assert_eq!(step.code, 0);
        // The model moves toward certainty by (2048 - p) >> 5.
        assert_eq!(step.probability, 1024 + ((2048 - 1024) >> 5));
        assert_eq!(step.probability_offset, 2);
    }

    #[test]
    fn a_one_bit_subtracts_the_bound_and_sets_the_low_symbol_bit() {
        let range = 0x0100_0000u32;
        let bound = (range >> 11) * 1024;
        let step = Shape::step(range, bound + 7, 1, 1024);
        assert_eq!(step.symbol, 3);
        assert_eq!(step.range, range - bound);
        assert_eq!(step.code, 7);
        assert_eq!(step.probability, 1024 - (1024 >> 5));
    }

    #[test]
    fn normalisation_pulls_a_byte_only_below_the_top_value() {
        assert!(Shape::needs_normalise(TOP_VALUE - 1));
        assert!(!Shape::needs_normalise(TOP_VALUE));
        let (range, code) = Shape::normalise(0x0001_0000, 0x0000_00ab, 0xcd);
        assert_eq!(range, 0x0100_0000);
        assert_eq!(code, 0x0000_abcd);
    }

    #[test]
    fn the_tree_ends_after_eight_bits() {
        let mut symbol = 1u32;
        for _ in 0..8 {
            assert!(!Shape::complete(symbol), "ended early at {symbol:#x}");
            symbol <<= 1;
        }
        assert!(Shape::complete(symbol));
        assert_eq!(symbol, SYMBOL_DONE);
    }

    #[test]
    fn eight_bits_reconstruct_the_byte_the_encoder_meant() {
        // Drive the tree with a fixed probability so the decoded bits follow
        // directly from where `code` sits relative to each bound.
        let mut symbol = 1u32;
        let (mut range, mut code) = (0xffff_ffffu32, 0u32);
        for _ in 0..8 {
            let step = Shape::step(range, code, symbol, 1024);
            range = step.range;
            code = step.code;
            symbol = step.symbol;
            if range < TOP_VALUE {
                let (r, c) = Shape::normalise(range, code, 0);
                range = r;
                code = c;
            }
        }
        assert!(Shape::complete(symbol));
        // code stayed at 0, so every bit took the 0 branch: the byte is 0.
        assert_eq!((symbol & 0xff) as u8, 0);
    }
}
