//! Straight-line native x64 symbolic lifter.
//!
//! Executes a branch-free block over symbolic 64-bit GPR inputs (`rax_in`, ...)
//! and recovers each register as an expression of those inputs. The expression
//! domain is abstracted by [`Algebra`] (this crate cannot depend on `vmpx-vm`,
//! which depends on it); [`crate::NativeAlgebra`] is the built-in domain.
//!
//! Anything not explicitly modeled fails closed with [`LiftError`].

use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{
    Decoder, DecoderOptions, Formatter, Instruction, IntelFormatter, Mnemonic, OpKind, Register,
};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Width {
    W8,
    W16,
    W32,
    W64,
}

impl Width {
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::W8 => 8,
            Self::W16 => 16,
            Self::W32 => 32,
            Self::W64 => 64,
        }
    }

    #[must_use]
    pub const fn bytes(self) -> u32 {
        self.bits() / 8
    }

    #[must_use]
    pub const fn mask(self) -> u64 {
        match self {
            Self::W64 => u64::MAX,
            _ => (1u64 << self.bits()) - 1,
        }
    }

    #[must_use]
    pub const fn truncate(self, value: u64) -> u64 {
        value & self.mask()
    }

    /// Sign-extends a width-bit value to 64 bits.
    #[must_use]
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub const fn sign_extend(self, value: u64) -> u64 {
        let shift = 64 - self.bits();
        (((value << shift) as i64) >> shift) as u64
    }

    #[must_use]
    pub const fn from_bytes(bytes: usize) -> Option<Self> {
        match bytes {
            1 => Some(Self::W8),
            2 => Some(Self::W16),
            4 => Some(Self::W32),
            8 => Some(Self::W64),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UnOp {
    Not,
    Neg,
}

/// Same-width binary operators. Shift/rotate counts are the same width as the
/// value and the lifter only ever passes counts strictly below the width.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Sar,
    Rol,
    Ror,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Resize {
    Truncate,
    ZeroExtend,
    SignExtend,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("{0}")]
pub struct AlgebraError(pub String);

/// Expression domain the lifter builds into.
pub trait Algebra {
    type Expr: Clone + Eq;

    fn constant(&self, width: Width, value: u64) -> Self::Expr;
    fn input(&self, width: Width, name: &str) -> Self::Expr;
    /// A read of block-entry memory at `address`.
    fn load(&self, width: Width, address: Self::Expr) -> Self::Expr;
    /// # Errors
    /// Domain-specific construction failure.
    fn unary(&self, op: UnOp, value: Self::Expr) -> Result<Self::Expr, AlgebraError>;
    /// Operands always have equal widths.
    /// # Errors
    /// Domain-specific construction failure.
    fn binary(
        &self,
        op: BinOp,
        left: Self::Expr,
        right: Self::Expr,
    ) -> Result<Self::Expr, AlgebraError>;
    /// # Errors
    /// Domain-specific construction failure.
    fn resize(
        &self,
        kind: Resize,
        target: Width,
        value: Self::Expr,
    ) -> Result<Self::Expr, AlgebraError>;
    fn intrinsic(&self, width: Width, name: &str, args: Vec<Self::Expr>) -> Self::Expr;
    fn width(&self, expr: &Self::Expr) -> Width;
    fn as_const(&self, expr: &Self::Expr) -> Option<u64>;
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Gpr {
    Rax,
    Rcx,
    Rdx,
    Rbx,
    Rsp,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

impl Gpr {
    pub const ALL: [Self; 16] = [
        Self::Rax,
        Self::Rcx,
        Self::Rdx,
        Self::Rbx,
        Self::Rsp,
        Self::Rbp,
        Self::Rsi,
        Self::Rdi,
        Self::R8,
        Self::R9,
        Self::R10,
        Self::R11,
        Self::R12,
        Self::R13,
        Self::R14,
        Self::R15,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rax => "rax",
            Self::Rcx => "rcx",
            Self::Rdx => "rdx",
            Self::Rbx => "rbx",
            Self::Rsp => "rsp",
            Self::Rbp => "rbp",
            Self::Rsi => "rsi",
            Self::Rdi => "rdi",
            Self::R8 => "r8",
            Self::R9 => "r9",
            Self::R10 => "r10",
            Self::R11 => "r11",
            Self::R12 => "r12",
            Self::R13 => "r13",
            Self::R14 => "r14",
            Self::R15 => "r15",
        }
    }

    /// Name of the symbolic block-entry value, e.g. `rax_in`.
    #[must_use]
    pub fn input_name(self) -> String {
        format!("{}_in", self.name())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Flag {
    Cf,
    Pf,
    Af,
    Zf,
    Sf,
    Of,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FlagState<E> {
    /// 0/1 value (8-bit). Built from the intrinsics `zf`, `sf`, `pf` (of the
    /// result) and `cf_add`/`of_add`/`af_add`/`cf_sub`/`of_sub`/`af_sub` (of the operands).
    Value(E),
    /// Possibly changed by the block; value not modeled (undefined or not tracked).
    Clobbered,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryWrite<E> {
    pub address: E,
    pub value: E,
    pub width: Width,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryRead<E> {
    pub address: E,
    pub width: Width,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockSemantics<E> {
    /// Final value of all 16 GPRs as full 64-bit expressions of the `*_in` inputs.
    pub regs: BTreeMap<Gpr, E>,
    /// GPRs (including partial writes) the block wrote.
    pub written: BTreeSet<Gpr>,
    /// Only flags the block may have changed; absent flags keep their entry value.
    pub flags: BTreeMap<Flag, FlagState<E>>,
    /// Stores in program order.
    pub writes: Vec<MemoryWrite<E>>,
    /// Distinct reads of block-entry memory (`Algebra::load` leaves).
    pub reads: Vec<MemoryRead<E>>,
    pub instruction_count: usize,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum LiftError {
    #[error("unsupported instruction #{index} `{instruction}`: {reason}")]
    Unsupported {
        index: usize,
        instruction: String,
        reason: String,
    },
    #[error("memory aliasing not provable at instruction #{index} `{instruction}`: {reason}")]
    MemoryAliasing {
        index: usize,
        instruction: String,
        reason: String,
    },
    #[error("expression construction failed at instruction #{index} `{instruction}`: {reason}")]
    Algebra {
        index: usize,
        instruction: String,
        reason: String,
    },
    #[error("invalid instruction encoding at byte offset {offset}")]
    Decode { offset: usize },
}

enum Reason {
    Unsupported(String),
    Aliasing(String),
    Algebra(String),
}

type R<T> = Result<T, Reason>;

fn unsupported<T>(message: impl Into<String>) -> R<T> {
    Err(Reason::Unsupported(message.into()))
}

/// Lifts a branch-free sequence of decoded instructions.
///
/// # Errors
/// Fails closed on any instruction, operand form or memory-aliasing situation
/// that is not modeled.
pub fn lift_block<A: Algebra>(
    algebra: &A,
    instructions: &[Instruction],
) -> Result<BlockSemantics<A::Expr>, LiftError> {
    let mut lifter = Lifter::new(algebra);
    for (index, instruction) in instructions.iter().enumerate() {
        lifter.exec(instruction).map_err(|reason| {
            let instruction = format_instruction(instruction);
            match reason {
                Reason::Unsupported(reason) => LiftError::Unsupported {
                    index,
                    instruction,
                    reason,
                },
                Reason::Aliasing(reason) => LiftError::MemoryAliasing {
                    index,
                    instruction,
                    reason,
                },
                Reason::Algebra(reason) => LiftError::Algebra {
                    index,
                    instruction,
                    reason,
                },
            }
        })?;
    }
    Ok(lifter.finish(instructions.len()))
}

/// Decodes `bytes` as 64-bit code at `ip` and lifts the whole buffer as one block.
///
/// # Errors
/// [`LiftError::Decode`] on an invalid encoding, otherwise as [`lift_block`].
pub fn lift_bytes<A: Algebra>(
    algebra: &A,
    bytes: &[u8],
    ip: u64,
) -> Result<BlockSemantics<A::Expr>, LiftError> {
    let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
    let mut instructions = Vec::new();
    while decoder.can_decode() {
        let offset = decoder.position();
        let instruction = decoder.decode();
        if instruction.is_invalid() {
            return Err(LiftError::Decode { offset });
        }
        instructions.push(instruction);
    }
    lift_block(algebra, &instructions)
}

fn format_instruction(instruction: &Instruction) -> String {
    let mut text = String::new();
    IntelFormatter::new().format(instruction, &mut text);
    text
}

struct MemKey<E> {
    sym: Option<E>,
    disp: i64,
}

struct StoreRec<E> {
    key: MemKey<E>,
    write: MemoryWrite<E>,
}

enum Rel {
    Disjoint,
    Exact,
    Unknown,
}

fn relate<E: Eq>(a: &MemKey<E>, aw: Width, b: &MemKey<E>, bw: Width) -> Rel {
    if a.sym != b.sym {
        return Rel::Unknown;
    }
    let Some(diff) = a.disp.checked_sub(b.disp) else {
        return Rel::Unknown;
    };
    let Some(neg) = diff.checked_neg() else {
        return Rel::Unknown;
    };
    if diff >= i64::from(bw.bytes()) || neg >= i64::from(aw.bytes()) {
        Rel::Disjoint
    } else if diff == 0 && aw == bw {
        Rel::Exact
    } else {
        Rel::Unknown
    }
}

fn gpr_info(reg: Register) -> R<(Gpr, Width, bool)> {
    let (width, high) = match reg {
        Register::AH | Register::CH | Register::DH | Register::BH => (Width::W8, true),
        r if r.is_gpr8() => (Width::W8, false),
        r if r.is_gpr16() => (Width::W16, false),
        r if r.is_gpr32() => (Width::W32, false),
        r if r.is_gpr64() => (Width::W64, false),
        _ => return unsupported(format!("register {reg:?} is not a general-purpose register")),
    };
    let gpr = match reg.full_register() {
        Register::RAX => Gpr::Rax,
        Register::RCX => Gpr::Rcx,
        Register::RDX => Gpr::Rdx,
        Register::RBX => Gpr::Rbx,
        Register::RSP => Gpr::Rsp,
        Register::RBP => Gpr::Rbp,
        Register::RSI => Gpr::Rsi,
        Register::RDI => Gpr::Rdi,
        Register::R8 => Gpr::R8,
        Register::R9 => Gpr::R9,
        Register::R10 => Gpr::R10,
        Register::R11 => Gpr::R11,
        Register::R12 => Gpr::R12,
        Register::R13 => Gpr::R13,
        Register::R14 => Gpr::R14,
        Register::R15 => Gpr::R15,
        other => return unsupported(format!("register {other:?}")),
    };
    Ok((gpr, width, high))
}

#[derive(Clone, Copy, PartialEq)]
enum Arith {
    Add,
    Sub,
    Logic,
}

struct Lifter<'a, A: Algebra> {
    alg: &'a A,
    regs: Vec<A::Expr>,
    written: BTreeSet<Gpr>,
    flags: BTreeMap<Flag, FlagState<A::Expr>>,
    stores: Vec<StoreRec<A::Expr>>,
    reads: Vec<MemoryRead<A::Expr>>,
}

impl<'a, A: Algebra> Lifter<'a, A> {
    fn new(alg: &'a A) -> Self {
        let regs = Gpr::ALL
            .iter()
            .map(|g| alg.input(Width::W64, &g.input_name()))
            .collect();
        Self {
            alg,
            regs,
            written: BTreeSet::new(),
            flags: BTreeMap::new(),
            stores: Vec::new(),
            reads: Vec::new(),
        }
    }

    fn finish(self, instruction_count: usize) -> BlockSemantics<A::Expr> {
        BlockSemantics {
            regs: Gpr::ALL.into_iter().zip(self.regs).collect(),
            written: self.written,
            flags: self.flags,
            writes: self.stores.into_iter().map(|s| s.write).collect(),
            reads: self.reads,
            instruction_count,
        }
    }

    fn k(&self, width: Width, value: u64) -> A::Expr {
        self.alg.constant(width, value)
    }

    fn bin(&self, op: BinOp, left: A::Expr, right: A::Expr) -> R<A::Expr> {
        self.alg
            .binary(op, left, right)
            .map_err(|e| Reason::Algebra(e.0))
    }

    fn un(&self, op: UnOp, value: A::Expr) -> R<A::Expr> {
        self.alg.unary(op, value).map_err(|e| Reason::Algebra(e.0))
    }

    fn resize(&self, kind: Resize, target: Width, value: A::Expr) -> R<A::Expr> {
        if self.alg.width(&value) == target {
            return Ok(value);
        }
        self.alg
            .resize(kind, target, value)
            .map_err(|e| Reason::Algebra(e.0))
    }

    fn read_reg(&self, reg: Register) -> R<A::Expr> {
        let (gpr, width, high) = gpr_info(reg)?;
        let full = self.regs[gpr as usize].clone();
        if high {
            let shifted = self.bin(BinOp::Shr, full, self.k(Width::W64, 8))?;
            return self.resize(Resize::Truncate, Width::W8, shifted);
        }
        self.resize(Resize::Truncate, width, full)
    }

    fn write_reg(&mut self, reg: Register, value: A::Expr) -> R<()> {
        let (gpr, width, high) = gpr_info(reg)?;
        if self.alg.width(&value) != width {
            return Err(Reason::Algebra(format!(
                "write width {:?} to {reg:?}",
                self.alg.width(&value)
            )));
        }
        let new = match width {
            Width::W64 => value,
            Width::W32 => self.resize(Resize::ZeroExtend, Width::W64, value)?,
            Width::W16 | Width::W8 => {
                let keep = if high { !0xFF00 } else { !width.mask() };
                let old = self.regs[gpr as usize].clone();
                let kept = self.bin(BinOp::And, old, self.k(Width::W64, keep))?;
                let mut insert = self.resize(Resize::ZeroExtend, Width::W64, value)?;
                if high {
                    insert = self.bin(BinOp::Shl, insert, self.k(Width::W64, 8))?;
                }
                self.bin(BinOp::Or, kept, insert)?
            }
        };
        self.regs[gpr as usize] = new;
        self.written.insert(gpr);
        Ok(())
    }

    fn operand_width(inst: &Instruction, index: u32) -> R<Width> {
        match inst.op_kind(index) {
            OpKind::Register => Ok(gpr_info(inst.op_register(index))?.1),
            OpKind::Memory => {
                let size = inst.memory_size().size();
                Width::from_bytes(size).map_or_else(
                    || unsupported(format!("memory operand size {size} bytes")),
                    Ok,
                )
            }
            kind => unsupported(format!("operand {index} kind {kind:?} has no width")),
        }
    }

    /// Returns the symbolic address plus the (symbolic base, displacement) key.
    fn mem_address(&self, inst: &Instruction, deref: bool) -> R<(A::Expr, MemKey<A::Expr>)> {
        if deref && matches!(inst.segment_prefix(), Register::FS | Register::GS) {
            return unsupported("FS/GS segment-relative access");
        }
        let mut sym: Option<A::Expr> = None;
        let mut disp = inst.memory_displacement64().cast_signed();
        match inst.memory_base() {
            Register::None | Register::RIP => {}
            base if base.is_gpr64() => sym = Some(self.read_reg(base)?),
            other => return unsupported(format!("address base register {other:?}")),
        }
        match inst.memory_index() {
            Register::None => {}
            index if index.is_gpr64() => {
                let mut value = self.read_reg(index)?;
                let scale = inst.memory_index_scale();
                if scale != 1 {
                    value = self.bin(BinOp::Mul, value, self.k(Width::W64, u64::from(scale)))?;
                }
                sym = Some(match sym {
                    Some(base) => self.bin(BinOp::Add, base, value)?,
                    None => value,
                });
            }
            other => return unsupported(format!("address index register {other:?}")),
        }
        if let Some(constant) = sym.as_ref().and_then(|s| self.alg.as_const(s)) {
            disp = disp.wrapping_add(constant.cast_signed());
            sym = None;
        }
        let address = match &sym {
            Some(base) if disp == 0 => base.clone(),
            Some(base) => self.bin(
                BinOp::Add,
                base.clone(),
                self.k(Width::W64, disp.cast_unsigned()),
            )?,
            None => self.k(Width::W64, disp.cast_unsigned()),
        };
        Ok((address, MemKey { sym, disp }))
    }

    fn load_mem(&mut self, inst: &Instruction, width: Width) -> R<A::Expr> {
        let (address, key) = self.mem_address(inst, true)?;
        for store in self.stores.iter().rev() {
            match relate(&key, width, &store.key, store.write.width) {
                Rel::Disjoint => {}
                Rel::Exact => return Ok(store.write.value.clone()),
                Rel::Unknown => {
                    return Err(Reason::Aliasing(
                        "load may alias or partially overlap an earlier store".into(),
                    ));
                }
            }
        }
        let read = MemoryRead {
            address: address.clone(),
            width,
        };
        if !self.reads.contains(&read) {
            self.reads.push(read);
        }
        Ok(self.alg.load(width, address))
    }

    fn store_mem(&mut self, inst: &Instruction, value: A::Expr) -> R<()> {
        let width = self.alg.width(&value);
        let (address, key) = self.mem_address(inst, true)?;
        self.stores.push(StoreRec {
            key,
            write: MemoryWrite {
                address,
                value,
                width,
            },
        });
        Ok(())
    }

    fn read_operand(&mut self, inst: &Instruction, index: u32, width: Width) -> R<A::Expr> {
        match inst.op_kind(index) {
            OpKind::Register => {
                let value = self.read_reg(inst.op_register(index))?;
                if self.alg.width(&value) != width {
                    return unsupported("operand width mismatch");
                }
                Ok(value)
            }
            OpKind::Memory => self.load_mem(inst, width),
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => Ok(self.k(width, inst.immediate(index))),
            kind => unsupported(format!("operand {index} kind {kind:?}")),
        }
    }

    fn write_operand(&mut self, inst: &Instruction, index: u32, value: A::Expr) -> R<()> {
        match inst.op_kind(index) {
            OpKind::Register => self.write_reg(inst.op_register(index), value),
            OpKind::Memory => self.store_mem(inst, value),
            kind => unsupported(format!("destination operand {index} kind {kind:?}")),
        }
    }

    fn set_flag(&mut self, flag: Flag, value: A::Expr) {
        self.flags.insert(flag, FlagState::Value(value));
    }

    fn clobber(&mut self, flags: &[Flag]) {
        for flag in flags {
            self.flags.insert(*flag, FlagState::Clobbered);
        }
    }

    fn set_flags(&mut self, kind: Arith, a: &A::Expr, b: &A::Expr, result: &A::Expr, keep_cf: bool) {
        let w8 = Width::W8;
        let res = || vec![result.clone()];
        let flag = |name: &str, args: Vec<A::Expr>| self.alg.intrinsic(w8, name, args);
        let zf = flag("zf", res());
        let sf = flag("sf", res());
        let pf = flag("pf", res());
        let ops = || vec![a.clone(), b.clone()];
        let (cf, of, af) = match kind {
            Arith::Add => (
                Some(flag("cf_add", ops())),
                flag("of_add", ops()),
                Some(flag("af_add", ops())),
            ),
            Arith::Sub => (
                Some(flag("cf_sub", ops())),
                flag("of_sub", ops()),
                Some(flag("af_sub", ops())),
            ),
            Arith::Logic => (Some(self.k(w8, 0)), self.k(w8, 0), None),
        };
        self.set_flag(Flag::Zf, zf);
        self.set_flag(Flag::Sf, sf);
        self.set_flag(Flag::Pf, pf);
        self.set_flag(Flag::Of, of);
        if !keep_cf && let Some(cf) = cf {
            self.set_flag(Flag::Cf, cf);
        }
        match af {
            Some(af) => self.set_flag(Flag::Af, af),
            None => self.clobber(&[Flag::Af]),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn exec(&mut self, inst: &Instruction) -> R<()> {
        if inst.has_lock_prefix() || inst.has_rep_prefix() || inst.has_repne_prefix() {
            return unsupported("lock/rep prefix");
        }
        let mnemonic = inst.mnemonic();
        match mnemonic {
            Mnemonic::Nop => Ok(()),
            Mnemonic::Mov => {
                Self::expect_ops(inst, 2)?;
                let width = Self::operand_width(inst, 0)?;
                let value = self.read_operand(inst, 1, width)?;
                self.write_operand(inst, 0, value)
            }
            Mnemonic::Movzx | Mnemonic::Movsx | Mnemonic::Movsxd => {
                Self::expect_ops(inst, 2)?;
                let dst = Self::operand_width(inst, 0)?;
                let src = Self::operand_width(inst, 1)?;
                if src.bits() > dst.bits() {
                    return unsupported("source wider than destination");
                }
                let value = self.read_operand(inst, 1, src)?;
                let kind = if mnemonic == Mnemonic::Movzx {
                    Resize::ZeroExtend
                } else {
                    Resize::SignExtend
                };
                let value = self.resize(kind, dst, value)?;
                self.write_operand(inst, 0, value)
            }
            Mnemonic::Lea => {
                Self::expect_ops(inst, 2)?;
                if inst.op_kind(1) != OpKind::Memory {
                    return unsupported("lea without memory operand");
                }
                let dst = Self::operand_width(inst, 0)?;
                let (address, _) = self.mem_address(inst, false)?;
                let value = self.resize(Resize::Truncate, dst, address)?;
                self.write_operand(inst, 0, value)
            }
            Mnemonic::Add
            | Mnemonic::Sub
            | Mnemonic::And
            | Mnemonic::Or
            | Mnemonic::Xor
            | Mnemonic::Cmp
            | Mnemonic::Test => {
                Self::expect_ops(inst, 2)?;
                let width = Self::operand_width(inst, 0)?;
                let a = self.read_operand(inst, 0, width)?;
                let b = self.read_operand(inst, 1, width)?;
                let (op, kind) = match mnemonic {
                    Mnemonic::Add => (BinOp::Add, Arith::Add),
                    Mnemonic::Sub | Mnemonic::Cmp => (BinOp::Sub, Arith::Sub),
                    Mnemonic::And | Mnemonic::Test => (BinOp::And, Arith::Logic),
                    Mnemonic::Or => (BinOp::Or, Arith::Logic),
                    _ => (BinOp::Xor, Arith::Logic),
                };
                let result = self.bin(op, a.clone(), b.clone())?;
                self.set_flags(kind, &a, &b, &result, false);
                if matches!(mnemonic, Mnemonic::Cmp | Mnemonic::Test) {
                    Ok(())
                } else {
                    self.write_operand(inst, 0, result)
                }
            }
            Mnemonic::Imul => {
                let (a, b) = match inst.op_count() {
                    2 => {
                        let width = Self::operand_width(inst, 0)?;
                        (
                            self.read_operand(inst, 0, width)?,
                            self.read_operand(inst, 1, width)?,
                        )
                    }
                    3 => {
                        let width = Self::operand_width(inst, 0)?;
                        (
                            self.read_operand(inst, 1, width)?,
                            self.read_operand(inst, 2, width)?,
                        )
                    }
                    _ => return unsupported("one-operand widening imul"),
                };
                let result = self.bin(BinOp::Mul, a, b)?;
                self.clobber(&[Flag::Cf, Flag::Pf, Flag::Af, Flag::Zf, Flag::Sf, Flag::Of]);
                self.write_operand(inst, 0, result)
            }
            Mnemonic::Not => {
                Self::expect_ops(inst, 1)?;
                let width = Self::operand_width(inst, 0)?;
                let value = self.read_operand(inst, 0, width)?;
                let result = self.un(UnOp::Not, value)?;
                self.write_operand(inst, 0, result)
            }
            Mnemonic::Neg => {
                Self::expect_ops(inst, 1)?;
                let width = Self::operand_width(inst, 0)?;
                let value = self.read_operand(inst, 0, width)?;
                let result = self.un(UnOp::Neg, value.clone())?;
                let zero = self.k(width, 0);
                self.set_flags(Arith::Sub, &zero, &value, &result, false);
                self.write_operand(inst, 0, result)
            }
            Mnemonic::Inc | Mnemonic::Dec => {
                Self::expect_ops(inst, 1)?;
                let width = Self::operand_width(inst, 0)?;
                let value = self.read_operand(inst, 0, width)?;
                let one = self.k(width, 1);
                let (op, kind) = if mnemonic == Mnemonic::Inc {
                    (BinOp::Add, Arith::Add)
                } else {
                    (BinOp::Sub, Arith::Sub)
                };
                let result = self.bin(op, value.clone(), one.clone())?;
                self.set_flags(kind, &value, &one, &result, true);
                self.write_operand(inst, 0, result)
            }
            Mnemonic::Shl | Mnemonic::Shr | Mnemonic::Sar | Mnemonic::Rol | Mnemonic::Ror => {
                self.exec_shift(inst, mnemonic)
            }
            Mnemonic::Xchg => {
                Self::expect_ops(inst, 2)?;
                let width = Self::operand_width(inst, 0)?;
                let a = self.read_operand(inst, 0, width)?;
                let b = self.read_operand(inst, 1, width)?;
                if inst.op_kind(0) == OpKind::Memory {
                    self.write_operand(inst, 0, b)?;
                    self.write_operand(inst, 1, a)
                } else {
                    self.write_operand(inst, 1, a)?;
                    self.write_operand(inst, 0, b)
                }
            }
            other => unsupported(format!("mnemonic {other:?} is not modeled")),
        }
    }

    fn expect_ops(inst: &Instruction, count: u32) -> R<()> {
        if inst.op_count() == count {
            Ok(())
        } else {
            unsupported(format!(
                "expected {count} operands, found {}",
                inst.op_count()
            ))
        }
    }

    fn exec_shift(&mut self, inst: &Instruction, mnemonic: Mnemonic) -> R<()> {
        Self::expect_ops(inst, 2)?;
        let width = Self::operand_width(inst, 0)?;
        let value = self.read_operand(inst, 0, width)?;
        let count = self.read_operand(inst, 1, Width::W8)?;
        let Some(raw) = self.alg.as_const(&count) else {
            return unsupported("variable shift/rotate count");
        };
        let bits = u64::from(width.bits());
        let masked = raw & if width == Width::W64 { 63 } else { 31 };
        if masked == 0 {
            if width == Width::W32 && inst.op_kind(0) == OpKind::Register {
                return unsupported("zero shift count on a 32-bit register");
            }
            return Ok(());
        }
        let rotate = matches!(mnemonic, Mnemonic::Rol | Mnemonic::Ror);
        let result = match mnemonic {
            Mnemonic::Shl | Mnemonic::Shr if masked >= bits => self.k(width, 0),
            Mnemonic::Shl => self.bin(BinOp::Shl, value, self.k(width, masked))?,
            Mnemonic::Shr => self.bin(BinOp::Shr, value, self.k(width, masked))?,
            Mnemonic::Sar => self.bin(BinOp::Sar, value, self.k(width, masked.min(bits - 1)))?,
            _ => {
                let eff = masked % bits;
                if eff == 0 {
                    value
                } else {
                    let op = if mnemonic == Mnemonic::Rol {
                        BinOp::Rol
                    } else {
                        BinOp::Ror
                    };
                    self.bin(op, value, self.k(width, eff))?
                }
            }
        };
        if rotate {
            self.clobber(&[Flag::Cf, Flag::Of]);
        } else {
            self.clobber(&[Flag::Cf, Flag::Pf, Flag::Af, Flag::Zf, Flag::Sf, Flag::Of]);
        }
        self.write_operand(inst, 0, result)
    }
}
