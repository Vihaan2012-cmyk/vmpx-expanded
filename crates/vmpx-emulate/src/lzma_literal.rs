//! Native execution of the LZMA literal decoder the protector's init runs.
//!
//! `VMProtect` decompresses its packed sections with LZMA, and the literal
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
//!          jae   skip
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
//! Soundness follows the integrity-checksum fold, with two additions, because
//! this loop both branches and writes.
//!
//! 1. The junk writes registers, not just flags, so skipping it needs more than
//!    the checksum loop's flag-noise rule. A skipped instruction may write only
//!    registers that are dead at exactly that point, which [`Cursor::next`]
//!    tracks stage by stage -- `rax` and `rcx` are dead before the body defines
//!    them and again after the probability store, and live in between. Deadness
//!    is read off iced's register-access info, so an implicit write counts.
//! 2. A skipped instruction may not clobber flags a matched `cmp` is holding for
//!    its `jcc`; otherwise the fold would branch on the `cmp` where the hardware
//!    branches on the junk. Each `jcc` is therefore fetched with flags declared
//!    live, which bars any flag-writing skip between the two.
//! 3. Both arms converge: the taken side of the normalisation branch must land
//!    on the path already walked, and the one-arm must rejoin it after the
//!    store. Neither is assumed -- a shape whose arms go elsewhere is declined,
//!    because folding it would skip whatever else they do.
//!
//! Every byte read -- input and probability -- goes through the normal guest
//! read path, so an unmapped address declines the fold instead of faulting or
//! inventing a value. The probability store is a real guest write, so the caller
//! must decline when a hook wants writes, reads or the elided control transfers.
//!
//! The caller leaves the LAST bit to the interpreter, so the exit state --
//! flags included -- is produced by ordinary instruction semantics and never
//! computed here.

use iced_x86::{
    Code, FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register,
};

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

/// How far a walk follows the body before giving up. The real body is ~33
/// operations plus junk and zero-displacement jump padding; the bound keeps a
/// mismatch from walking a whole section.
const WALK_CAP: usize = 256;

/// A matched literal decoder: which register carries what.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Shape {
    /// Accumulates the symbol; starts at 1, ends at or past [`SYMBOL_DONE`].
    pub symbol: Register,
    /// Holds the symbol sign-extended, as the probability array's index.
    pub index: Register,
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
    /// Scratch holding the scaled bound; dead at the head by construction.
    pub scratch_a: Register,
    /// Scratch holding the probability; dead at the head by construction.
    pub scratch_c: Register,
    /// The loop head, which the back edge returns to.
    pub head: u64,
    /// The address range a fold stays inside, as `[start, end)`.
    ///
    /// This replaces a list of elided `(from, to)` transfers. `VMProtect` splits
    /// the body with `jmp +0` padding -- six of them in the observed loop, more
    /// in the arms -- so an exact list is both large and no more useful than the
    /// range containing it: the only consumer asks whether a hook wants flow
    /// anywhere a fold would elide, and every such address is in here. The
    /// range also covers branch targets the fold decides without taking, so the
    /// input-exhausted exit is accounted for even though a fold stops short of
    /// it.
    pub body: (u64, u64),
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
    /// What the two scratch registers hold once this bit has retired.
    ///
    /// Both are dead for the loop's own purposes -- the body redefines them
    /// before reading them, which is what licensed skipping junk that writes
    /// them. They are modelled anyway because the input-exhausted branch LEAVES
    /// the loop, and what it does with them is outside the matched shape.
    pub scratch_a: u32,
    pub scratch_c: u32,
}

/// `reg` as its 64-bit container, so `eax` and `rax` compare equal.
fn full(register: Register) -> Register {
    register.full_register()
}

/// Whether `register` is the low 16 bits of the same container as `of`.
fn is_low16(register: Register, of: Register) -> bool {
    register.size() == 2 && full(register) == full(of)
}

/// Whether every register this instruction writes is in `dead`.
///
/// Read off iced's access info rather than the operand list, so an implicit
/// destination -- the kind a hand-rolled operand check misses -- counts. An
/// instruction that writes no register at all passes: the protector's junk is
/// mostly flag-only (`cmp bl,r12b`, `test dl,dil`, `clc`, `cmc`), and those
/// flags are dead wherever `flags_live` says they are.
fn writes_only_dead(
    instruction: &Instruction,
    factory: &mut InstructionInfoFactory,
    dead: &[Register],
) -> bool {
    factory.info(instruction).used_registers().iter().all(|used| {
        let writes = matches!(
            used.access(),
            OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
        );
        !writes || dead.iter().any(|slot| full(*slot) == full(used.register()))
    })
}

/// Whether this instruction cannot affect the fold and may be stepped over.
///
/// A memory operand is never skippable: it could fault, or a hook could want
/// the access, and folding would elide both. Neither is a control transfer.
fn is_skippable(
    instruction: &Instruction,
    factory: &mut InstructionInfoFactory,
    dead: &[Register],
    flags_live: bool,
) -> bool {
    if (0..instruction.op_count()).any(|op| instruction.op_kind(op) == OpKind::Memory) {
        return false;
    }
    if instruction.flow_control() != FlowControl::Next {
        return false;
    }
    if flags_live && instruction.rflags_modified() != 0 {
        return false;
    }
    writes_only_dead(instruction, factory, dead)
}

/// Collects the straight-line path from `start`, following direct jumps.
///
/// Stops on reaching an address in `stop`, after a backward conditional branch,
/// or at [`WALK_CAP`]. `extent` grows to cover every instruction walked and
/// every branch target seen, which is what a fold must not elide events inside.
fn walk(
    start: u64,
    stop: &[u64],
    decode: &mut dyn FnMut(u64) -> Option<Instruction>,
    extent: &mut (u64, u64),
) -> Option<(Vec<Instruction>, u64)> {
    let mut out = Vec::new();
    let mut address = start;
    while out.len() < WALK_CAP {
        if stop.contains(&address) {
            return Some((out, address));
        }
        let instruction = decode(address)?;
        let length = u64::from(u32::try_from(instruction.len()).ok()?);
        extent.0 = extent.0.min(address);
        extent.1 = extent.1.max(address.checked_add(length)?);
        if instruction.op0_kind() == OpKind::NearBranch64 {
            let target = instruction.near_branch64();
            extent.0 = extent.0.min(target);
            extent.1 = extent.1.max(target.checked_add(1)?);
        }
        if instruction.mnemonic() == Mnemonic::Jmp && instruction.op0_kind() == OpKind::NearBranch64
        {
            address = instruction.near_branch64();
            continue;
        }
        let backward = instruction.op0_kind() == OpKind::NearBranch64
            && instruction.near_branch64() <= start
            && instruction.flow_control() == FlowControl::ConditionalBranch;
        out.push(instruction);
        if backward {
            return Some((out, address));
        }
        address = address.checked_add(length)?;
    }
    None
}

/// Walks a collected path, matching each stage against the instruction in front
/// of it and stepping over only what no stage claims.
///
/// Giving the stage first refusal is what makes this work on the real loop. A
/// skip rule strong enough to step over `cmp bl,r12b` also steps over
/// `cmp range,kTopValue`, and one weak enough to keep the second also stalls on
/// the first; there is no predicate on an instruction alone that separates them.
/// What separates them is that one is the instruction the shape is looking for.
struct Cursor<'a> {
    items: &'a [Instruction],
    at: usize,
    factory: InstructionInfoFactory,
}

impl<'a> Cursor<'a> {
    fn new(items: &'a [Instruction]) -> Self {
        Self { items, at: 0, factory: InstructionInfoFactory::new() }
    }

    /// The next instruction satisfying `want`, skipping what provably cannot
    /// matter. `dead` lists the registers a skipped write may target here, and
    /// `flags_live` bars a flag-clobbering skip between a matched `cmp` and the
    /// `jcc` reading it.
    fn take(
        &mut self,
        dead: &[Register],
        flags_live: bool,
        want: &dyn Fn(&Instruction) -> bool,
    ) -> Option<Instruction> {
        while let Some(instruction) = self.items.get(self.at) {
            self.at += 1;
            if want(instruction) {
                return Some(*instruction);
            }
            if !is_skippable(instruction, &mut self.factory, dead, flags_live) {
                return None;
            }
        }
        None
    }

    /// Where the path continues, without consuming anything.
    fn peek_ip(&self) -> Option<u64> {
        self.items.get(self.at).map(Instruction::ip)
    }

    /// The addresses still ahead: where a diverging arm is allowed to rejoin.
    fn remaining_ips(&self) -> Vec<u64> {
        self.items[self.at.min(self.items.len())..].iter().map(Instruction::ip).collect()
    }

    /// Whether only skippable instructions remain.
    fn exhausted(&mut self, dead: &[Register]) -> bool {
        self.take(dead, false, &|_| false).is_none() && self.at >= self.items.len()
    }
}

/// `<op> register, register`, with both registers pinned.
fn is_rr(instruction: &Instruction, mnemonic: Mnemonic, dst: Register, src: Register) -> bool {
    instruction.mnemonic() == mnemonic
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register() == dst
        && instruction.op1_kind() == OpKind::Register
        && instruction.op1_register() == src
}

/// `<op> register, imm8`, with the register pinned.
fn is_ri8(instruction: &Instruction, mnemonic: Mnemonic, dst: Register, value: u32) -> bool {
    instruction.mnemonic() == mnemonic
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register() == dst
        && instruction.op1_kind() == OpKind::Immediate8
        && u32::from(instruction.immediate8()) == value
}

/// `<op> dst16, src16`, where both name the low half of the given containers.
fn is_rr16(instruction: &Instruction, mnemonic: Mnemonic, dst: Register, src: Register) -> bool {
    instruction.mnemonic() == mnemonic
        && instruction.op0_kind() == OpKind::Register
        && is_low16(instruction.op0_register(), dst)
        && instruction.op1_kind() == OpKind::Register
        && is_low16(instruction.op1_register(), src)
}

/// The `[probs + index*2]` operand, with no displacement, both bases pinned.
fn is_prob_slot(instruction: &Instruction, operand: u32, probs: Register, index: Register) -> bool {
    instruction.op_kind(operand) == OpKind::Memory
        && full(instruction.memory_base()) == full(probs)
        && full(instruction.memory_index()) == full(index)
        && instruction.memory_index_scale() == 2
        && instruction.memory_displacement64() == 0
        && instruction.segment_prefix() == Register::None
}

/// A store of a 16-bit register into the probability slot.
fn is_prob_store(
    instruction: &Instruction,
    probs: Register,
    index: Register,
    value: Register,
) -> bool {
    instruction.mnemonic() == Mnemonic::Mov
        && is_prob_slot(instruction, 0, probs, index)
        && instruction.op1_kind() == OpKind::Register
        && is_low16(instruction.op1_register(), value)
}

/// A conditional branch of the given kind.
fn is_jcc(instruction: &Instruction, mnemonic: Mnemonic) -> bool {
    instruction.mnemonic() == mnemonic && instruction.op0_kind() == OpKind::NearBranch64
}

impl Shape {
    /// Matches the literal decoder starting at `head`. `decode` yields the
    /// instruction at an address.
    ///
    /// Returns `None` on anything that is not exactly this loop, so an
    /// unrecognized shape simply runs interpreted.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    pub(crate) fn recognize(
        head: u64,
        mut decode: impl FnMut(u64) -> Option<Instruction>,
    ) -> Option<Self> {
        let mut extent = (u64::MAX, 0u64);
        let (main, _) = walk(head, &[], &mut decode, &mut extent)?;

        // Identify the two scratch registers up front, so the stage walk knows
        // from the first instruction which writes are dead. Both are pinned by
        // their memory form; the register-form `movzx` in the one-arm and the
        // protector's `movzx eax,si` junk are not candidates.
        let scratch_a = main
            .iter()
            .find(|i| i.code() == Code::Movzx_r32_rm8 && i.op1_kind() == OpKind::Memory)?
            .op0_register();
        let scratch_c = main
            .iter()
            .find(|i| {
                i.code() == Code::Movzx_r32_rm16
                    && i.op1_kind() == OpKind::Memory
                    && i.memory_index_scale() == 2
            })?
            .op0_register();
        if full(scratch_a) == full(scratch_c) {
            return None;
        }
        // Deadness by region: both scratches are dead before the body defines
        // them, the probability register stays dead until it is read, and both
        // are dead again once the probability store has consumed them.
        let both = [scratch_a, scratch_c];
        let only_c = [scratch_c];
        let live: [Register; 0] = [];

        let mut cursor = Cursor::new(&main);

        // movsxd index, symbol
        let instruction = cursor.take(&both, false, &|i| {
            i.code() == Code::Movsxd_r64_rm32 && i.op1_kind() == OpKind::Register
        })?;
        let index = instruction.op0_register();
        let symbol = instruction.op1_register();

        // cmp range, kTopValue / jae skip
        let instruction = cursor.take(&both, false, &|i| {
            i.mnemonic() == Mnemonic::Cmp
                && i.op0_kind() == OpKind::Register
                && i.op1_kind() == OpKind::Immediate32
                && i.immediate32() == TOP_VALUE
        })?;
        let range = instruction.op0_register();
        let skip = cursor.take(&both, true, &|i| is_jcc(i, Mnemonic::Jae))?.near_branch64();

        // cmp input, input_end / je exhausted
        let instruction = cursor.take(&both, false, &|i| {
            i.mnemonic() == Mnemonic::Cmp
                && i.op0_kind() == OpKind::Register
                && i.op0_register().size() == 8
                && i.op1_kind() == OpKind::Register
                && i.op1_register().size() == 8
        })?;
        let input = full(instruction.op0_register());
        let input_end = full(instruction.op1_register());
        cursor.take(&both, true, &|i| is_jcc(i, Mnemonic::Je))?;

        // movzx scratch_a, byte [input]
        cursor.take(&both, false, &|i| {
            i.code() == Code::Movzx_r32_rm8
                && i.op0_register() == scratch_a
                && i.op1_kind() == OpKind::Memory
                && full(i.memory_base()) == input
                && i.memory_index() == Register::None
                && i.memory_displacement64() == 0
                && i.segment_prefix() == Register::None
        })?;

        // shl code, 8 / shl range, 8 / or code, scratch_a / inc input
        let instruction = cursor.take(&only_c, false, &|i| {
            i.mnemonic() == Mnemonic::Shl
                && i.op0_kind() == OpKind::Register
                && i.op0_register() != range
                && i.op1_kind() == OpKind::Immediate8
                && i.immediate8() == 8
        })?;
        let code = instruction.op0_register();
        cursor.take(&only_c, false, &|i| is_ri8(i, Mnemonic::Shl, range, 8))?;
        cursor.take(&only_c, false, &|i| is_rr(i, Mnemonic::Or, code, scratch_a))?;
        cursor.take(&both, false, &|i| {
            i.mnemonic() == Mnemonic::Inc
                && i.op0_kind() == OpKind::Register
                && full(i.op0_register()) == input
        })?;

        // The normalisation branch must rejoin the path walked here.
        if cursor.peek_ip()? != skip {
            return None;
        }

        // movzx scratch_c, word [probs + index*2]
        let instruction = cursor.take(&both, false, &|i| {
            i.code() == Code::Movzx_r32_rm16
                && i.op0_register() == scratch_c
                && i.op1_kind() == OpKind::Memory
                && i.memory_base() != Register::None
                && i.memory_index_scale() == 2
                && full(i.memory_index()) == full(index)
                && i.memory_displacement64() == 0
                && i.segment_prefix() == Register::None
        })?;
        let probs = full(instruction.memory_base());

        // mov scratch_a, range / shr scratch_a, 11 / imul scratch_a, scratch_c
        cursor.take(&live, false, &|i| is_rr(i, Mnemonic::Mov, scratch_a, range))?;
        cursor.take(&live, false, &|i| is_ri8(i, Mnemonic::Shr, scratch_a, MODEL_TOTAL_BITS))?;
        cursor.take(&live, false, &|i| is_rr(i, Mnemonic::Imul, scratch_a, scratch_c))?;

        // cmp code, bound / jae one
        cursor.take(&live, false, &|i| is_rr(i, Mnemonic::Cmp, code, scratch_a))?;
        let one = cursor.take(&live, true, &|i| is_jcc(i, Mnemonic::Jae))?.near_branch64();

        // The zero arm: range = bound, prob += (kBitModelTotal - prob) >> 5,
        // symbol <<= 1, store.
        cursor.take(&live, false, &|i| is_rr(i, Mnemonic::Mov, range, scratch_a))?;
        cursor.take(&live, false, &|i| {
            i.mnemonic() == Mnemonic::Mov
                && i.op0_kind() == OpKind::Register
                && i.op0_register() == scratch_a
                && i.op1_kind() == OpKind::Immediate32
                && i.immediate32() == BIT_MODEL_TOTAL
        })?;
        cursor.take(&live, false, &|i| is_rr(i, Mnemonic::Sub, scratch_a, scratch_c))?;
        cursor.take(&live, false, &|i| is_ri8(i, Mnemonic::Sar, scratch_a, MOVE_BITS))?;
        cursor.take(&live, false, &|i| is_rr16(i, Mnemonic::Add, scratch_a, scratch_c))?;
        cursor.take(&live, false, &|i| is_rr(i, Mnemonic::Add, symbol, symbol))?;
        cursor.take(&live, false, &|i| is_prob_store(i, probs, index, scratch_a))?;

        // Past the store both scratches are dead again, and the one-arm must
        // rejoin somewhere ahead of here rather than running off elsewhere.
        let rejoin_points = cursor.remaining_ips();
        let (arm, rejoin) = walk(one, &rejoin_points, &mut decode, &mut extent)?;
        let mut arm_cursor = Cursor::new(&arm);

        // sub range, bound / sub code, bound
        arm_cursor.take(&live, false, &|i| is_rr(i, Mnemonic::Sub, range, scratch_a))?;
        arm_cursor.take(&live, false, &|i| is_rr(i, Mnemonic::Sub, code, scratch_a))?;
        // movzx scratch_a, prob16 / shr scratch_a16, 5
        //
        // The bound dies at the `sub` above, so junk may write that register
        // here -- the sample does exactly that, with an `inc rax` one pad before
        // the `movzx` that redefines it.
        arm_cursor.take(&[scratch_a], false, &|i| {
            i.code() == Code::Movzx_r32_rm16
                && i.op0_register() == scratch_a
                && i.op1_kind() == OpKind::Register
                && is_low16(i.op1_register(), scratch_c)
        })?;
        arm_cursor.take(&live, false, &|i| {
            i.mnemonic() == Mnemonic::Shr
                && i.op0_kind() == OpKind::Register
                && is_low16(i.op0_register(), scratch_a)
                && i.op1_kind() == OpKind::Immediate8
                && u32::from(i.immediate8()) == MOVE_BITS
        })?;
        // lea symbol, [symbol + symbol + 1]
        arm_cursor.take(&live, false, &|i| {
            i.mnemonic() == Mnemonic::Lea
                && i.op0_kind() == OpKind::Register
                && i.op0_register() == symbol
                && full(i.memory_base()) == full(symbol)
                && full(i.memory_index()) == full(symbol)
                && i.memory_index_scale() == 1
                && i.memory_displacement64() == 1
        })?;
        // sub prob16, scratch_a16 / store
        arm_cursor.take(&live, false, &|i| is_rr16(i, Mnemonic::Sub, scratch_c, scratch_a))?;
        arm_cursor.take(&live, false, &|i| is_prob_store(i, probs, index, scratch_c))?;
        if !arm_cursor.exhausted(&both) {
            return None;
        }

        // Advance the main path to where the arm rejoined it, then match the
        // tail test that both arms share.
        while cursor.peek_ip().is_some_and(|ip| ip != rejoin) {
            cursor.take(&both, false, &|_| false)?;
        }
        if cursor.peek_ip()? != rejoin {
            return None;
        }
        cursor.take(&both, false, &|i| {
            i.mnemonic() == Mnemonic::Cmp
                && i.op0_kind() == OpKind::Register
                && i.op0_register() == symbol
                && i.op1_kind() == OpKind::Immediate32
                && i.immediate32() == SYMBOL_DONE
        })?;
        let back = cursor.take(&both, true, &|i| is_jcc(i, Mnemonic::Jl))?;
        if back.near_branch64() != head {
            return None;
        }

        // Nine distinct roles, or the fold would not model the loop.
        let roles = [
            full(symbol),
            full(index),
            probs,
            full(range),
            full(code),
            input,
            input_end,
            full(scratch_a),
            full(scratch_c),
        ];
        for (position, role) in roles.iter().enumerate() {
            if *role == Register::None || roles[position + 1..].contains(role) {
                return None;
            }
        }
        if extent.0 > extent.1 {
            return None;
        }

        Some(Self {
            symbol,
            index,
            probs,
            range,
            code,
            input,
            input_end,
            scratch_a,
            scratch_c,
            head,
            body: extent,
        })
    }

    /// One bit of the tree, given the current coder state and the probability
    /// just read. Pure: the caller performs the guest reads and the store.
    #[must_use]
    pub(crate) fn step(range: u32, code: u32, symbol: u32, probability: u16) -> Step {
        let probability_offset = u64::from(symbol) * 2;
        let bound = (range >> MODEL_TOTAL_BITS).wrapping_mul(u32::from(probability));
        if code < bound {
            // The model moves toward zero and the symbol takes a 0 bit.
            let moved =
                u32::from(probability) + ((BIT_MODEL_TOTAL - u32::from(probability)) >> MOVE_BITS);
            Step {
                range: bound,
                code,
                symbol: symbol << 1,
                probability: moved as u16,
                probability_offset,
                // `mov eax,800h / sub eax,ecx / sar eax,5` leaves at most 0x40
                // in the register, so the following `add ax,cx` cannot carry out
                // of 16 bits: it ends holding exactly the moved probability.
                // `ecx` is untouched on this arm.
                scratch_a: moved,
                scratch_c: u32::from(probability),
            }
        } else {
            let moved = u32::from(probability) - (u32::from(probability) >> MOVE_BITS);
            Step {
                range: range - bound,
                code: code - bound,
                symbol: (symbol << 1) | 1,
                probability: moved as u16,
                probability_offset,
                // `movzx eax,cx` zero-extends, so `shr ax,5` leaves the whole
                // register holding the shifted probability, and `sub cx,ax`
                // leaves `ecx` holding the moved one.
                scratch_a: u32::from(probability) >> MOVE_BITS,
                scratch_c: moved,
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

#[cfg(test)]
mod tests {
    use iced_x86::{Decoder, DecoderOptions};

    use super::*;

    /// The literal decoder as it actually appears in systems.xpl, taken
    /// verbatim from the mapped image at rva `0xefd2849`.
    ///
    /// A hand-written version of this loop is what the matcher was first built
    /// against, and it passed while the real thing did not match at all: the
    /// sample interleaves flag-only junk (`cmp bl,r12b`, `cmp r13b,0x4b`,
    /// `test dl,dil`, `clc`, `cmc`), register junk whose results die
    /// (`btr cx,r11w`, `adc cx,0x361`, `shr cl,cl`, `shrd rcx,rbx,0xb3`,
    /// `movsx ax,r14b`, `movzx eax,si`) and six zero-displacement `jmp +0`
    /// pads. Only the real bytes exercise any of that.
    const REAL: &[u8] = &[
        0x49, 0x63, 0xd2, 0x41, 0x3a, 0xdc, 0xe9, 0x00, 0x00, 0x00, 0x00, 0x41,
        0x81, 0xf8, 0x00, 0x00, 0x00, 0x01, 0xe9, 0x00, 0x00, 0x00, 0x00, 0x0f,
        0x83, 0x36, 0x00, 0x00, 0x00, 0x4d, 0x3b, 0xcd, 0x0f, 0x84, 0xf4, 0x0e,
        0x00, 0x00, 0x41, 0x0f, 0xb6, 0x01, 0x66, 0x44, 0x0f, 0xb3, 0xd9, 0x66,
        0x81, 0xd1, 0x61, 0x03, 0xd2, 0xe9, 0xc1, 0xe7, 0x08, 0x41, 0x80, 0xfd,
        0x4b, 0x41, 0xc1, 0xe0, 0x08, 0x48, 0x0f, 0xac, 0xd9, 0xb3, 0x0b, 0xf8,
        0x66, 0x41, 0x0f, 0xbe, 0xc6, 0x0f, 0xb7, 0xc6, 0x49, 0xff, 0xc1, 0x0f,
        0xb7, 0x0c, 0x53, 0x41, 0x8b, 0xc0, 0xc1, 0xe8, 0x0b, 0xe9, 0x00, 0x00,
        0x00, 0x00, 0x0f, 0xaf, 0xc1, 0x40, 0x84, 0xfa, 0x3b, 0xf8, 0xe9, 0x00,
        0x00, 0x00, 0x00, 0x0f, 0x83, 0x80, 0x00, 0x00, 0x00, 0x44, 0x8b, 0xc0,
        0xb8, 0x00, 0x08, 0x00, 0x00, 0x2b, 0xc1, 0xf5, 0x40, 0x3a, 0xc4, 0xc1,
        0xf8, 0x05, 0xf8, 0x66, 0x44, 0x85, 0xdd, 0x66, 0x03, 0xc1, 0x45, 0x03,
        0xd2, 0xe9, 0x00, 0x00, 0x00, 0x00, 0x66, 0x89, 0x04, 0x53, 0xe9, 0x00,
        0x00, 0x00, 0x00, 0xe9, 0x7f, 0x00, 0x00, 0x00, 0x44, 0x2b, 0xc0, 0x2b,
        0xf8, 0x80, 0xc4, 0x74, 0x66, 0x0f, 0xbc, 0xc4, 0xf6, 0xd0, 0x0f, 0xb7,
        0xc1, 0x41, 0xf6, 0xc0, 0x81, 0x66, 0xc1, 0xe8, 0x05, 0x66, 0x41, 0x3b,
        0xfa, 0x47, 0x8d, 0x54, 0x12, 0x01, 0x66, 0x2b, 0xc8, 0x66, 0x89, 0x8c,
        0x53, 0x00, 0x02, 0x00, 0x00, 0x41, 0xf7, 0xc1, 0x69, 0xa4, 0xf1, 0xfa,
        0xf5, 0xf9, 0x45, 0x85, 0xdb, 0x0f, 0x84, 0x0b, 0xff, 0xff, 0xff, 0x41,
        0x81, 0xfa, 0x00, 0x01, 0x00, 0x00, 0x0f, 0x8c, 0x11, 0xfe, 0xff, 0xff,
        0xe9, 0x40, 0x00, 0x00, 0x00, 0x44, 0x2b, 0xc0, 0x45, 0x3a, 0xc1, 0x2b,
        0xf8, 0x48, 0xff, 0xc0, 0x0f, 0xb7, 0xc1, 0x40, 0x3a, 0xff, 0x66, 0xc1,
        0xe8, 0x05, 0x41, 0x3a, 0xec, 0xe9, 0x01, 0x00, 0x00, 0x00, 0xf5, 0x47,
        0x8d, 0x54, 0x12, 0x01, 0xf8, 0xf9, 0x66, 0x2b, 0xc8, 0xf8, 0x66, 0x89,
        0x0c, 0x53, 0xf8, 0x41, 0x81, 0xfa, 0x00, 0x01, 0x00, 0x00, 0xe9, 0x00,
        0x00, 0x00, 0x00, 0x0f, 0x8c, 0xcb, 0xfe, 0xff, 0xff, 0x48, 0x8b, 0x84,
        0x24, 0x90, 0x00, 0x00, 0x00, 0x66, 0x44, 0x0f, 0xbe, 0xdf, 0x49, 0xff,
        0xc6, 0x44, 0x1a, 0xdf, 0x66, 0x81, 0xfc, 0x36, 0x2e, 0x66, 0x45, 0x2b,
        0xde, 0x83, 0xfd, 0x04, 0x44, 0x0f, 0xb7, 0xde, 0x45, 0x0f, 0xbf, 0xda,
        0x41, 0x0f, 0x99, 0xc3, 0x45, 0x88, 0x54, 0x06, 0xff, 0x45, 0x0f, 0xb6,
        0xda, 0xe9, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x8d, 0x17, 0x00, 0x00, 0x00,
        0x2b, 0xc0, 0x48, 0x0f, 0xb7, 0xec, 0x66, 0x41, 0x0f, 0xbe, 0xee, 0x8b,
        0xe8, 0xe9, 0x00, 0x00, 0x00, 0x00, 0xe9, 0x68, 0x0d, 0x00, 0x00, 0x83,
        0xfd, 0x0a, 0xe9, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x8d, 0x0d, 0x00, 0x00,
        0x00, 0x83, 0xed, 0x03, 0xe9, 0x00, 0x00, 0x00, 0x00, 0xe9, 0x4d, 0x0d,
        0x00, 0x00, 0x83, 0xed, 0x06, 0xe9, 0x00, 0x00, 0x00, 0x00, 0xe9, 0x40,
        0x0d, 0x00, 0x00, 0x44, 0x2b, 0xc0, 0x2b, 0xf8, 0x0f, 0xb7, 0xc1, 0x66,
        0xc1, 0xe8, 0x05, 0x48, 0x81, 0xea, 0x4a, 0x05, 0x79, 0x3f, 0x66, 0x41,
        0x03, 0xd7, 0x48, 0x63, 0xd5, 0xf8, 0x66, 0x2b, 0xc8, 0x49, 0xf7, 0xc2,
        0x2a, 0x41, 0xee, 0xbe, 0x66, 0x41, 0x89, 0x0c, 0x5f, 0xf8, 0x41, 0x81,
        0xf8, 0x00, 0x00, 0x00, 0x01, 0xe9, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x83,
        0x33, 0x00, 0x00, 0x00, 0x4d, 0x3b, 0xcd, 0xe9, 0x00, 0x00, 0x00, 0x00,
        0x0f, 0x84, 0x1c, 0x0d, 0x00, 0x00, 0x41, 0x0f, 0xb6, 0x01, 0xc1, 0xe7,
        0x08, 0x66, 0xd3, 0xd9, 0x66, 0x40, 0x0f, 0xb6, 0xcf, 0x41, 0xc1, 0xe0,
        0x08, 0x0b, 0xf8, 0x66, 0x87, 0xc9, 0x0f, 0xb7, 0xc6, 0x48, 0x98, 0x49,
        0xff, 0xc1, 0xc1, 0xf8, 0xbc, 0x66, 0x98, 0x41, 0x0f, 0xb7, 0x8c, 0x57,
        0x80, 0x01, 0x00, 0x00, 0x41, 0xf6, 0xc5, 0xc1, 0x41, 0x8b, 0xc0, 0xc1,
        0xe8, 0x0b, 0x49, 0x85, 0xd7, 0x0f, 0xaf, 0xc1, 0x45, 0x84, 0xfb, 0xf8,
        0x3b, 0xf8, 0xe9, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x83, 0xa4, 0x00, 0x00,
        0x00, 0x44, 0x8b, 0xc0, 0x66, 0xd3, 0xc8, 0x48, 0x0f, 0xb3, 0xd0, 0x66,
        0x41, 0xf7, 0xdb, 0x41, 0xbb, 0x00, 0x00, 0x00, 0x00, 0x49, 0x63, 0xc5,
        0xf8, 0xb8, 0x00, 0x08, 0x00, 0x00, 0xf7, 0xc3, 0x97, 0xdc, 0xfe, 0xbe,
        0x2b, 0xc1, 0xf9, 0x44, 0x89, 0x64, 0x24, 0x70, 0xc1, 0xf8, 0x05, 0x66,
        0x03, 0xc1, 0x40, 0x80, 0xff, 0x25, 0xf5, 0x83, 0xfd, 0x07, 0x66, 0x0f,
        0xcd, 0x48, 0x0f, 0xb7, 0xed, 0x41, 0x8d, 0x6b, 0x08, 0x66, 0x41, 0x89,
        0x84, 0x57, 0x80, 0x01, 0x00, 0x00, 0x0f, 0xb7, 0xc0, 0x66, 0xf7, 0xd0,
        0x48, 0x99, 0x8b, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00, 0x41, 0x0f, 0xbf,
        0xd2, 0x44, 0x89, 0x94, 0x24, 0x80, 0x00, 0x00, 0x00, 0x0f, 0xb7, 0xd4,
        0xe9, 0x00, 0x00, 0x00, 0x00, 0x89, 0x44, 0x24, 0x04, 0xb8, 0x03, 0x00,
        0x00, 0x00, 0x0f, 0xca, 0xb2, 0x41, 0x66, 0x41, 0x0f, 0x44, 0xd7, 0x49,
        0x8d, 0x97, 0x64, 0x06, 0x00, 0x00, 0xe9, 0x00, 0x00, 0x00, 0x00, 0x41,
        0x0f, 0x4c, 0xc3, 0xe9, 0x00, 0x00, 0x00, 0x00, 0x89, 0x04, 0x24, 0xe9,
        0x00, 0x00, 0x00, 0x00, 0xe9, 0x70, 0x04, 0x00, 0x00, 0x44, 0x2b, 0xc0,
        0xf8, 0xf9, 0x2b, 0xf8, 0xc0, 0xf8, 0xe7, 0x0f, 0xb7, 0xc1, 0x66, 0xc1,
    ];

    fn decoder_over(bytes: &[u8], head: u64) -> impl FnMut(u64) -> Option<Instruction> + use<'_> {
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

    /// The loop head's real address, so the fixture's relative branches resolve
    /// exactly as they do in the image.
    const HEAD: u64 = 0x0003_7A97_2849;

    #[test]
    fn recognizes_the_decoder_in_the_sample() {
        let shape = Shape::recognize(HEAD, decoder_over(REAL, HEAD)).expect("recognized");
        assert_eq!(shape.symbol, Register::R10D);
        assert_eq!(shape.index, Register::RDX);
        assert_eq!(shape.probs, Register::RBX);
        assert_eq!(shape.range, Register::R8D);
        assert_eq!(shape.code, Register::EDI);
        assert_eq!(shape.input, Register::R9);
        assert_eq!(shape.input_end, Register::R13);
        assert_eq!(shape.scratch_a, Register::EAX);
        assert_eq!(shape.scratch_c, Register::ECX);
        assert_eq!(shape.head, HEAD);
        // The head is the lowest address a fold touches: the only backward
        // branch on the walked path is the back edge to it.
        assert_eq!(shape.body.0, HEAD);
        // The body reaches past the loop's own code, because it also has to
        // cover the input-exhausted exit ~0xf00 bytes away.
        assert!(shape.body.1 > HEAD + 0x123);
    }

    #[test]
    fn the_body_covers_the_input_exhausted_exit() {
        // That exit leaves the loop, and a fold decides it without taking it,
        // so a hook wanting flow there must still decline the fold.
        let shape = Shape::recognize(HEAD, decoder_over(REAL, HEAD)).expect("recognized");
        // `je` at head+0x20 with displacement 0xef4 targets head+0xf1c.
        let exhausted = HEAD + 0x26 + 0x0ef4;
        assert!(
            shape.body.0 <= exhausted && exhausted < shape.body.1,
            "exit {exhausted:#x} outside body {:#x}..{:#x}",
            shape.body.0,
            shape.body.1
        );
    }

    #[test]
    fn a_back_edge_that_leaves_the_loop_is_not_this_shape() {
        // Rewriting the tail's branch target means folding would skip whatever
        // lives at the other target, so the shape must be declined.
        let mut bytes = REAL.to_vec();
        let head = Shape::recognize(HEAD, decoder_over(REAL, HEAD)).expect("recognized").head;
        assert_eq!(head, HEAD);
        // Find the `jl` whose target is the head and bend it elsewhere.
        let mut at = None;
        for offset in 0..bytes.len() - 6 {
            if bytes[offset] == 0x0f && bytes[offset + 1] == 0x8c {
                let displacement =
                    i32::from_le_bytes(bytes[offset + 2..offset + 6].try_into().unwrap());
                let target = HEAD
                    .wrapping_add(offset as u64 + 6)
                    .wrapping_add(i64::from(displacement) as u64);
                if target == HEAD {
                    at = Some(offset);
                    break;
                }
            }
        }
        let at = at.expect("the back edge is a 6-byte jl");
        bytes[at + 2..at + 6].copy_from_slice(&0x20_i32.to_le_bytes());
        assert!(Shape::recognize(HEAD, decoder_over(&bytes, HEAD)).is_none());
    }

    #[test]
    fn junk_that_writes_a_live_register_declines_the_fold() {
        // The skip rule stands on deadness. Turn one pad into a write of the
        // register holding `code`, which the body reads, and the shape must go
        // away rather than be folded with that write elided.
        let mut bytes = REAL.to_vec();
        // The `jmp +0` pad at head+0x06 becomes `inc edi` plus shorter pads.
        bytes[6] = 0xff;
        bytes[7] = 0xc7;
        bytes[8] = 0x90;
        bytes[9] = 0x90;
        bytes[10] = 0x90;
        assert!(Shape::recognize(HEAD, decoder_over(&bytes, HEAD)).is_none());
    }

    /// Where the reachable tail's back edge sits, and what follows it.
    ///
    /// The fixture holds two copies of the one-arm and two tails; the dead copy
    /// branches to `head - 0xff`. The reachable back edge is the 6-byte `jl`
    /// whose target IS the head, so the instruction after it is where the loop
    /// leaves -- which is the only address a differential can stop both runs at.
    fn loop_exit(bytes: &[u8], head: u64) -> u64 {
        for offset in 0..bytes.len() - 6 {
            if bytes[offset] != 0x0f || bytes[offset + 1] != 0x8c {
                continue;
            }
            let displacement =
                i32::from_le_bytes(bytes[offset + 2..offset + 6].try_into().unwrap());
            let after = head.wrapping_add(offset as u64 + 6);
            if after.wrapping_add(i64::from(displacement) as u64) == head {
                return after;
            }
        }
        panic!("no back edge to the head");
    }

    const GUEST_CODE: u64 = 0x0001_0000;
    const GUEST_PROBS: u64 = 0x0002_0000;
    const GUEST_INPUT: u64 = 0x0003_0000;

    /// The sample's decoder in a contained guest, with `seed` choosing the coder
    /// state so different runs take different arms.
    fn guest(seed: u64, accelerate: bool) -> crate::emu::Emulator {
        let mut mem = crate::memory::Memory::new();
        mem.map(GUEST_CODE, crate::memory::PAGE_SIZE, crate::memory::Perm::RX).unwrap();
        mem.map(GUEST_PROBS, crate::memory::PAGE_SIZE, crate::memory::Perm::RW).unwrap();
        mem.map(GUEST_INPUT, crate::memory::PAGE_SIZE, crate::memory::Perm::RW).unwrap();
        mem.poke(GUEST_CODE, REAL).unwrap();

        // Probabilities spread across the legal range, so the comparison against
        // `bound` lands on both sides and both arms get exercised.
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut probabilities = Vec::new();
        for _ in 0..0x100u32 {
            let value = 1 + (next() % 2046) as u16;
            probabilities.extend_from_slice(&value.to_le_bytes());
        }
        mem.poke(GUEST_PROBS, &probabilities).unwrap();
        let input: Vec<u8> = (0..0x200u32).map(|_| next() as u8).collect();
        mem.poke(GUEST_INPUT, &input).unwrap();

        let mut cpu = crate::cpu::Cpu::new(GUEST_CODE);
        // Junk in the body reads rbp, rsi, r11, r12, r14 and the stack pointer's
        // low byte, so everything is given a defined value.
        for register in &mut cpu.regs {
            *register = next() & 0xFFFF;
        }
        cpu.regs[3] = GUEST_PROBS; // rbx: probability array
        cpu.regs[5] = u64::from(1u32); // rbp, kept small
        cpu.regs[7] = next() & 0xFFFF_FFFF; // rdi: code
        cpu.regs[8] = if seed.is_multiple_of(3) { 0x00FF_FFFF } else { 0xFFFF_FFFF }; // r8: range
        cpu.regs[9] = GUEST_INPUT; // r9: input
        cpu.regs[10] = 1; // r10: symbol starts at 1
        cpu.regs[13] = GUEST_INPUT + 0x200; // r13: one past the last input byte
        cpu.rflags = 2;
        let mut emulator = crate::emu::Emulator::new(cpu, mem);
        emulator.accelerate_checksums = accelerate;
        emulator.resolve_undefined_flags = true;
        emulator.record_trace = false;
        emulator
    }

    /// Folding the sample's own decoder must leave exactly the state the
    /// interpreter leaves.
    ///
    /// This is the check the unit tests above cannot make. They prove the
    /// arithmetic and prove the matcher accepts the right bytes; neither proves
    /// that executing the fold and executing the instructions agree -- on the
    /// registers the fold does not model, on the probabilities it writes, or on
    /// the flags it deliberately leaves alone. Two of the three bugs in building
    /// this fold would have survived the unit tests and died here.
    #[test]
    fn folding_the_sample_matches_interpreting_it() {
        let exit = loop_exit(REAL, GUEST_CODE);
        let mut folded_runs = 0u64;
        let mut arms = (0u64, 0u64);
        let mut normalised = 0u64;
        for seed in 1..96u64 {
            let mut interpreted = guest(seed, false);
            let mut accelerated = guest(seed, true);
            let want = interpreted.run_until(exit, 100_000);
            let got = accelerated.run_until(exit, 100_000);
            assert_eq!(want, got, "seed {seed} stop");
            assert_eq!(
                interpreted.cpu, accelerated.cpu,
                "seed {seed} cpu: folded {:?}",
                accelerated.prof.literals_folded
            );
            assert!(
                interpreted.mem.pages().eq(accelerated.mem.pages()),
                "seed {seed} memory"
            );
            folded_runs += accelerated.prof.literals_folded;
            // Normalisation is a separate path through the fold, including the
            // guest read it performs, so it has to be shown to happen.
            normalised += u64::from(interpreted.cpu.regs[9] > GUEST_INPUT);
            // The decoded byte tells which way the bits went, so a run of
            // all-zero or all-one arms would show up as a degenerate spread.
            let symbol = interpreted.cpu.regs[10] & 0xFF;
            arms.0 += u64::from(symbol.count_zeros() - 56);
            arms.1 += u64::from(symbol.count_ones());
        }
        assert!(folded_runs > 0, "nothing folded, so nothing was compared");
        assert!(arms.0 > 0 && arms.1 > 0, "only one arm exercised: {arms:?}");
        assert!(normalised > 0, "no run pulled an input byte");
    }

    /// A read-only probability array must decline rather than fault or skip.
    #[test]
    fn an_unwritable_probability_array_declines_the_fold() {
        let exit = loop_exit(REAL, GUEST_CODE);
        let mut interpreted = guest(7, false);
        let mut accelerated = guest(7, true);
        for emulator in [&mut interpreted, &mut accelerated] {
            emulator
                .mem
                .protect(GUEST_PROBS, crate::memory::PAGE_SIZE, crate::memory::Perm::R)
                .unwrap();
        }
        let want = interpreted.run_until(exit, 100_000);
        let got = accelerated.run_until(exit, 100_000);
        assert_eq!(want, got, "stop");
        assert_eq!(interpreted.cpu, accelerated.cpu, "cpu");
        assert_eq!(accelerated.prof.literals_folded, 0, "folded an unwritable array");
    }

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
        // The zero arm leaves the moved probability in one scratch and the old
        // one in the other, exactly as `add ax,cx` and an untouched `ecx` do.
        assert_eq!(step.scratch_a, u32::from(step.probability));
        assert_eq!(step.scratch_c, 1024);
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
        assert_eq!(step.scratch_a, 1024 >> 5);
        assert_eq!(step.scratch_c, u32::from(step.probability));
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
