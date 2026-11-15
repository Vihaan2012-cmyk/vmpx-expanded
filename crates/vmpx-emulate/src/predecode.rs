//! Decode-time resolution of an iced [`Instruction`] into a compact [`Pre`]
//! the interpreter hot loop dispatches on directly.
//!
//! The resolved form is computed once when a decode-cache slot is filled and is
//! reused only while the guest bytes still match (see `Emulator::decode`), so a
//! self-modifying write invalidates it exactly like the raw decode. Anything
//! this module does not cover (or that the generic path would stop on) resolves
//! to [`K::Slow`] and runs through the original `exec`, so behaviour — including
//! every stop reason — is unchanged.

use iced_x86::{Code, ConditionCode, FlowControl, Instruction, Mnemonic, OpKind, Register};

use crate::cpu::{locate_reg, mask};

pub(crate) const T_REG: u8 = 1;
pub(crate) const T_MEM: u8 = 2;
pub(crate) const T_IMM: u8 = 3;

/// A pre-resolved operand. `tag` selects which fields are meaningful:
/// - `T_REG`: `r` = regs index, `bytes` = width, `sh` = bit shift (8 for AH..BH),
///   `v` = value mask (`mask(bytes)`).
/// - `T_MEM`: `bytes` = access width, `r` = base reg, `x` = index reg, `sc` =
///   index scale (0 = no index), `fl` bit0 = base present, bit1 = gs-relative,
///   `v` = displacement (absolute address for rip-relative).
/// - `T_IMM`: `v` = immediate, already masked to the consumer's width.
#[derive(Clone, Copy)]
pub(crate) struct Opnd {
    pub tag: u8,
    pub bytes: u8,
    pub r: u8,
    pub x: u8,
    pub sh: u8,
    pub sc: u8,
    pub fl: u8,
    pub v: u64,
}

const NONE: Opnd = Opnd { tag: 0, bytes: 0, r: 0, x: 0, sh: 0, sc: 0, fl: 0, v: 0 };

/// Execution kind: operation x operand form, so the hot loop is one flat jump
/// table with operands already resolved.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum K {
    Slow,
    Nop,
    MovRR,
    MovRI,
    MovRM,
    MovMR,
    MovMI,
    ZxRR,
    ZxRM,
    SxRR,
    SxRM,
    Lea,
    AddRR,
    AddRI,
    AddRM,
    AddMR,
    AddMI,
    SubRR,
    SubRI,
    SubRM,
    SubMR,
    SubMI,
    AndRR,
    AndRI,
    AndRM,
    AndMR,
    AndMI,
    OrRR,
    OrRI,
    OrRM,
    OrMR,
    OrMI,
    XorRR,
    XorRI,
    XorRM,
    XorMR,
    XorMI,
    CmpRR,
    CmpRI,
    CmpRM,
    CmpMR,
    CmpMI,
    TestRR,
    TestRI,
    TestRM,
    TestMR,
    TestMI,
    IncR,
    DecR,
    NegR,
    NotR,
    IncM,
    DecM,
    NegM,
    NotM,
    ShR,
    ShM,
    RotR,
    RotM,
    PushR,
    PushI,
    PopR,
    Jmp,
    Call,
    JmpD,
    CallD,
    Ret,
    Jcc,
    Clc,
    Stc,
    Cmc,
    Cld,
    Std,
    AdcSbb,
    AdcSbbRR,
    AdcSbbRI,
    Xadd,
    Bswap,
    Cwde,
    Cdqe,
    Cdq,
    Cqo,
    Pushfq,
    Setcc,
    Cmov,
    /// Two-operand `imul r, r` / `imul r, m`. The LZMA range decoder the
    /// protector unpacks with runs one of these per decoded bit, which made
    /// `Imul` 38% of all generic-path steps on systems.xpl.
    ImulRR,
    ImulRM,
}

/// Pre-resolved instruction: kind, up to two operands and a few scalars whose
/// meaning is per-kind (see the constructors in [`resolve`]).
#[derive(Clone, Copy)]
pub(crate) struct Pre {
    pub kind: K,
    pub mn: Mnemonic,
    pub len: u8,
    /// Guest address of this instruction (set when placed in a block).
    pub rip: u64,
    /// Operand width in bytes (or the `inv` bit for `Jcc`/`Setcc`/`Cmov`).
    pub n: u8,
    /// Per-kind selector (shift/rotate kind, sign-extend shift, condition base).
    pub sub: u8,
    /// May leave the straight-line path (set when placed in a block).
    pub flow: bool,
    /// The next op in its block is a direct `jmp` run inline after this one.
    pub fj: bool,
    /// The flags this op produces are overwritten, unobserved, by a later op in
    /// its block (set when placed in a block).
    pub nf: bool,
    /// This `Jcc`'s flags come from the op immediately before it in the block,
    /// so the pair could be evaluated as one.
    pub pairable: bool,
    /// May write memory or run a hook (so guest code may have changed after it).
    pub chk: bool,
    pub a: Opnd,
    pub b: Opnd,
    /// Branch target / `ret` pop extra.
    pub t: u64,
    /// Flag bits a `Jcc`/`Setcc`/`Cmov` reads (undefined-flag guard).
    pub m: u16,
}

impl Pre {
    pub(crate) const INVALID: Pre = Pre {
        kind: K::Slow,
        mn: Mnemonic::INVALID,
        len: 0,
        rip: 0,
        n: 0,
        sub: 0,
        flow: false,
        fj: false,
        nf: false,
        pairable: false,
        chk: true,
        a: NONE,
        b: NONE,
        t: 0,
        m: 0,
    };
}

const ALU_KINDS: [[K; 5]; 7] = [
    [K::AddRR, K::AddRI, K::AddRM, K::AddMR, K::AddMI],
    [K::SubRR, K::SubRI, K::SubRM, K::SubMR, K::SubMI],
    [K::AndRR, K::AndRI, K::AndRM, K::AndMR, K::AndMI],
    [K::OrRR, K::OrRI, K::OrRM, K::OrMR, K::OrMI],
    [K::XorRR, K::XorRI, K::XorRM, K::XorMR, K::XorMI],
    [K::CmpRR, K::CmpRI, K::CmpRM, K::CmpMR, K::CmpMI],
    [K::TestRR, K::TestRI, K::TestRM, K::TestMR, K::TestMI],
];

/// Operand width exactly as the generic path's `op_bytes` computes it.
fn op_bytes(ins: &Instruction, i: u32) -> Option<u8> {
    let n = match ins.op_kind(i) {
        OpKind::Register => ins.op_register(i).size(),
        OpKind::Memory => ins.memory_size().size(),
        _ => 0,
    };
    matches!(n, 1 | 2 | 4 | 8).then_some(n as u8)
}

fn reg_opnd(r: Register) -> Option<Opnd> {
    let (idx, bytes, sh) = locate_reg(r)?;
    Some(Opnd { tag: T_REG, bytes, r: idx, x: 0, sh, sc: 0, fl: 0, v: mask(u32::from(bytes)) })
}

/// The instruction's memory operand, mirroring the generic path's `ea`. `None`
/// for anything `ea` would stop on (fs: segment, non-64-bit base/index).
fn mem_opnd(ins: &Instruction, n: u8) -> Option<Opnd> {
    let seg = ins.memory_segment();
    if seg == Register::FS {
        return None;
    }
    let base = ins.memory_base();
    let disp = ins.memory_displacement64();
    if base == Register::RIP {
        return Some(Opnd { tag: T_MEM, bytes: n, r: 0, x: 0, sh: 0, sc: 0, fl: 0, v: disp });
    }
    let mut o = Opnd { tag: T_MEM, bytes: n, r: 0, x: 0, sh: 0, sc: 0, fl: 0, v: disp };
    if base != Register::None {
        if !base.is_gpr64() {
            return None;
        }
        o.r = base.number() as u8;
        o.fl |= 1;
    }
    let idx = ins.memory_index();
    if idx != Register::None {
        if !idx.is_gpr64() {
            return None;
        }
        o.x = idx.number() as u8;
        o.sc = ins.memory_index_scale() as u8;
    }
    if seg == Register::GS {
        o.fl |= 2;
    }
    Some(o)
}

/// Operand `i` as the generic `read_op`/`write_op` would address it with width `n`.
fn opnd(ins: &Instruction, i: u32, n: u8) -> Option<Opnd> {
    match ins.op_kind(i) {
        OpKind::Register => reg_opnd(ins.op_register(i)),
        OpKind::Memory => mem_opnd(ins, n),
        OpKind::Immediate8
        | OpKind::Immediate16
        | OpKind::Immediate32
        | OpKind::Immediate64
        | OpKind::Immediate8to16
        | OpKind::Immediate8to32
        | OpKind::Immediate8to64
        | OpKind::Immediate32to64 => Some(Opnd {
            tag: T_IMM,
            bytes: n,
            r: 0,
            x: 0,
            sh: 0,
            sc: 0,
            fl: 0,
            v: ins.immediate(i) & mask(u32::from(n)),
        }),
        _ => None,
    }
}

/// `(base index, inverted)` of a condition code (o/b/e/be/s/p/l/le pairs).
fn cc_parts(cc: ConditionCode) -> Option<(u8, u8)> {
    use ConditionCode as C;
    Some(match cc {
        C::o => (0, 0),
        C::no => (0, 1),
        C::b => (1, 0),
        C::ae => (1, 1),
        C::e => (2, 0),
        C::ne => (2, 1),
        C::be => (3, 0),
        C::a => (3, 1),
        C::s => (4, 0),
        C::ns => (4, 1),
        C::p => (5, 0),
        C::np => (5, 1),
        C::l => (6, 0),
        C::ge => (6, 1),
        C::le => (7, 0),
        C::g => (7, 1),
        C::None => return None,
    })
}

/// Flags a condition-base reads (matches the generic `cond` guard).
pub(crate) fn cc_need(base: u8) -> u64 {
    use crate::cpu::{CF, OF, PF, SF, ZF};
    match base {
        0 => OF,
        1 => CF,
        2 => ZF,
        3 => CF | ZF,
        4 => SF,
        5 => PF,
        6 => SF | OF,
        _ => ZF | SF | OF,
    }
}

fn form(dst: &Opnd, src: &Opnd) -> Option<usize> {
    match (dst.tag, src.tag) {
        (T_REG, T_REG) => Some(0),
        (T_REG, T_IMM) => Some(1),
        (T_REG, T_MEM) => Some(2),
        (T_MEM, T_REG) => Some(3),
        (T_MEM, T_IMM) => Some(4),
        _ => None,
    }
}

/// Resolve `ins` (valid) into its fast form, or [`K::Slow`] if the generic
/// path must run it.
pub(crate) fn resolve(ins: &Instruction) -> Pre {
    let mut p = Pre::INVALID;
    p.mn = ins.mnemonic();
    p.len = ins.len() as u8;
    // rep/repne are only meaningful on string ops / `ret`; the generic path
    // decides what to do with them.
    if ins.has_rep_prefix() || ins.has_repne_prefix() {
        return p;
    }
    if let Some(k) = resolve_kind(ins, &mut p) {
        p.kind = k;
        // Register-form ops are specialized per width and assume no high-byte
        // (AH..BH) operand; those take the generic path.
        let reg_form = matches!(
            k,
            K::MovRR
                | K::MovRI
                | K::AddRR
                | K::AddRI
                | K::SubRR
                | K::SubRI
                | K::AndRR
                | K::AndRI
                | K::OrRR
                | K::OrRI
                | K::XorRR
                | K::XorRI
                | K::CmpRR
                | K::CmpRI
                | K::TestRR
                | K::TestRI
                | K::ImulRR
                | K::AdcSbbRR
                | K::AdcSbbRI
        );
        let dst_form = matches!(
            k,
            K::MovRM | K::ZxRR | K::ZxRM | K::SxRR | K::SxRM | K::Lea | K::ShR
        );
        if (reg_form && (p.a.sh != 0 || p.b.sh != 0)) || (dst_form && p.a.sh != 0) {
            p.kind = K::Slow;
        }
    }
    p
}

fn resolve_kind(ins: &Instruction, p: &mut Pre) -> Option<K> {
    use Mnemonic as M;
    let mn = ins.mnemonic();
    Some(match mn {
        M::Nop => K::Nop,
        // Only the two-operand forms: one-operand `imul` writes RDX:RAX and the
        // three-operand form needs a third slot, so both stay generic.
        M::Imul if ins.op_count() == 2 => {
            let n = op_bytes(ins, 0)?;
            let a = opnd(ins, 0, n)?;
            let b = opnd(ins, 1, n)?;
            if a.tag != T_REG {
                return None;
            }
            p.n = n;
            p.a = a;
            p.b = b;
            match b.tag {
                T_REG => K::ImulRR,
                T_MEM => K::ImulRM,
                _ => return None,
            }
        }
        M::Mov => {
            let n = op_bytes(ins, 0)?;
            let a = opnd(ins, 0, n)?;
            let b = opnd(ins, 1, n)?;
            p.n = n;
            p.a = a;
            p.b = b;
            [K::MovRR, K::MovRI, K::MovRM, K::MovMR, K::MovMI][form(&a, &b)?]
        }
        M::Movzx | M::Movsx | M::Movsxd => {
            let (dn, sn) = (op_bytes(ins, 0)?, op_bytes(ins, 1)?);
            if sn >= dn {
                return None;
            }
            let a = opnd(ins, 0, dn)?;
            let b = opnd(ins, 1, sn)?;
            if a.tag != T_REG {
                return None;
            }
            p.n = dn;
            p.sub = 64 - sn * 8;
            p.a = a;
            p.b = b;
            match (mn == M::Movzx, b.tag) {
                (true, T_REG) => K::ZxRR,
                (true, T_MEM) => K::ZxRM,
                (false, T_REG) => K::SxRR,
                (false, T_MEM) => K::SxRM,
                _ => return None,
            }
        }
        M::Lea => {
            let n = op_bytes(ins, 0)?;
            if ins.op_kind(1) != OpKind::Memory || ins.op_kind(0) != OpKind::Register {
                return None;
            }
            p.n = n;
            p.a = opnd(ins, 0, n)?;
            p.b = mem_opnd(ins, n)?;
            K::Lea
        }
        M::Add | M::Sub | M::And | M::Or | M::Xor | M::Cmp | M::Test => {
            let n = op_bytes(ins, 0)?;
            let a = opnd(ins, 0, n)?;
            let b = opnd(ins, 1, n)?;
            let op = match mn {
                M::Add => 0,
                M::Sub => 1,
                M::And => 2,
                M::Or => 3,
                M::Xor => 4,
                M::Cmp => 5,
                _ => 6,
            };
            p.n = n;
            p.a = a;
            p.b = b;
            ALU_KINDS[op][form(&a, &b)?]
        }
        M::Inc | M::Dec | M::Neg | M::Not => {
            let n = op_bytes(ins, 0)?;
            let a = opnd(ins, 0, n)?;
            p.n = n;
            p.a = a;
            match (a.tag, mn) {
                (T_REG, M::Inc) => K::IncR,
                (T_REG, M::Dec) => K::DecR,
                (T_REG, M::Neg) => K::NegR,
                (T_REG, _) => K::NotR,
                (T_MEM, M::Inc) => K::IncM,
                (T_MEM, M::Dec) => K::DecM,
                (T_MEM, M::Neg) => K::NegM,
                (T_MEM, _) => K::NotM,
                _ => return None,
            }
        }
        M::Shl | M::Sal | M::Shr | M::Sar | M::Rol | M::Ror | M::Rcl | M::Rcr => {
            let n = op_bytes(ins, 0)?;
            let a = opnd(ins, 0, n)?;
            let b = if ins.op_count() == 1 {
                Opnd { tag: T_IMM, bytes: 1, v: 1, ..NONE }
            } else {
                opnd(ins, 1, 1)?
            };
            if b.tag == T_MEM || a.tag == T_IMM {
                return None;
            }
            p.n = n;
            p.a = a;
            p.b = b;
            let rot = matches!(mn, M::Rol | M::Ror | M::Rcl | M::Rcr);
            p.sub = match mn {
                M::Shl | M::Sal | M::Rol => 0,
                M::Shr | M::Ror => 1,
                M::Sar | M::Rcl => 2,
                _ => 3,
            };
            match (rot, a.tag) {
                (false, T_REG) => K::ShR,
                (false, _) => K::ShM,
                (true, T_REG) => K::RotR,
                (true, _) => K::RotM,
            }
        }
        M::Push => match ins.op0_kind() {
            OpKind::Register if ins.op0_register().is_gpr64() => {
                p.a = reg_opnd(ins.op0_register())?;
                K::PushR
            }
            OpKind::Immediate8to64 | OpKind::Immediate32to64 => {
                p.a = opnd(ins, 0, 8)?;
                K::PushI
            }
            _ => return None,
        },
        M::Pop => {
            if ins.op0_kind() == OpKind::Register && ins.op0_register().is_gpr64() {
                p.a = reg_opnd(ins.op0_register())?;
                K::PopR
            } else {
                return None;
            }
        }
        M::Jmp | M::Call => {
            p.a = match ins.op0_kind() {
                OpKind::NearBranch64 => Opnd { tag: T_IMM, bytes: 8, v: ins.near_branch_target(), ..NONE },
                OpKind::Register if ins.op0_register().is_gpr64() => reg_opnd(ins.op0_register())?,
                OpKind::Memory if ins.memory_size().size() == 8 => mem_opnd(ins, 8)?,
                _ => return None,
            };
            match (mn == M::Jmp, p.a.tag) {
                (true, T_IMM) => K::JmpD,
                (true, _) => K::Jmp,
                (false, T_IMM) => K::CallD,
                (false, _) => K::Call,
            }
        }
        M::Ret => {
            if !matches!(ins.code(), Code::Retnq | Code::Retnq_imm16) {
                return None;
            }
            p.t = if ins.op_count() == 1 { ins.immediate(0) } else { 0 };
            K::Ret
        }
        M::Adc | M::Sbb => {
            let n = op_bytes(ins, 0)?;
            let a = opnd(ins, 0, n)?;
            let b = opnd(ins, 1, n)?;
            if a.tag == T_IMM {
                return None;
            }
            p.n = n;
            p.sub = u8::from(mn == M::Sbb);
            p.a = a;
            p.b = b;
            match (a.tag, b.tag) {
                (T_REG, T_REG) => K::AdcSbbRR,
                (T_REG, T_IMM) => K::AdcSbbRI,
                _ => K::AdcSbb,
            }
        }
        M::Xadd => {
            let n = op_bytes(ins, 0)?;
            let a = opnd(ins, 0, n)?;
            let b = opnd(ins, 1, n)?;
            if a.tag == T_IMM || b.tag == T_IMM {
                return None;
            }
            p.n = n;
            p.a = a;
            p.b = b;
            K::Xadd
        }
        M::Bswap => {
            let n = op_bytes(ins, 0)?;
            if !matches!(n, 4 | 8) {
                return None;
            }
            let a = opnd(ins, 0, n)?;
            if a.tag != T_REG {
                return None;
            }
            p.n = n;
            p.a = a;
            K::Bswap
        }
        M::Cwde => K::Cwde,
        M::Cdqe => K::Cdqe,
        M::Cdq => K::Cdq,
        M::Cqo => K::Cqo,
        M::Pushfq => K::Pushfq,
        M::Clc => K::Clc,
        M::Stc => K::Stc,
        M::Cmc => K::Cmc,
        M::Cld => K::Cld,
        M::Std => K::Std,
        _ => return resolve_cc(ins, p),
    })
}

/// Condition-code instructions (`jcc`, `setcc`, `cmovcc`), in the generic
/// path's precedence order (it matches specific mnemonics first, which none of
/// these are).
fn resolve_cc(ins: &Instruction, p: &mut Pre) -> Option<K> {
    let (base, inv) = cc_parts(ins.condition_code())?;
    p.sub = base;
    p.m = cc_need(base) as u16;
    if ins.flow_control() == FlowControl::Next {
        if ins.op_count() == 1 {
            let a = opnd(ins, 0, 1)?;
            if a.tag == T_IMM {
                return None;
            }
            p.n = inv;
            p.a = a;
            return Some(K::Setcc);
        }
        let n = op_bytes(ins, 0)?;
        let a = opnd(ins, 0, n)?;
        let b = opnd(ins, 1, n)?;
        if a.tag != T_REG {
            return None;
        }
        p.n = n;
        p.sub = base | (inv << 4);
        p.a = a;
        p.b = b;
        return Some(K::Cmov);
    }
    if ins.flow_control() == FlowControl::ConditionalBranch && ins.op0_kind() == OpKind::NearBranch64 {
        p.n = inv;
        p.t = ins.near_branch_target();
        return Some(K::Jcc);
    }
    None
}
