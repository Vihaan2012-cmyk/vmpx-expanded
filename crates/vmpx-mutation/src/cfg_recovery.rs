//! Native CFG recovery — the substrate the mutation passes operate on.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use iced_x86::{
    Decoder, DecoderOptions, FlowControl, Formatter, Instruction, IntelFormatter, Mnemonic,
};
use vmpx_disasm::{
    Algebra, AlgebraError, BinOp, BlockSemantics, LiftError, LiftOptions, Resize, UnOp, Width,
    lift_block_with, lift_bytes_with,
};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, ResizeOp, UnaryOp};

use crate::Problem;

/// Builds native lifts directly in [`vmpx_vm::Expr`] form.
#[derive(Clone, Copy, Debug, Default)]
pub struct VmAlgebra;

fn bw(width: Width) -> BitWidth {
    match width {
        Width::W8 => BitWidth::W8,
        Width::W16 => BitWidth::W16,
        Width::W32 => BitWidth::W32,
        Width::W64 => BitWidth::W64,
    }
}

fn nw(width: BitWidth) -> Width {
    match width {
        BitWidth::W8 => Width::W8,
        BitWidth::W16 => Width::W16,
        BitWidth::W32 => Width::W32,
        BitWidth::W64 => Width::W64,
    }
}

impl Algebra for VmAlgebra {
    type Expr = Expr;

    fn constant(&self, width: Width, value: u64) -> Expr {
        Expr::constant(bw(width), value)
    }

    fn input(&self, width: Width, name: &str) -> Expr {
        Expr::register(bw(width), name)
    }

    fn load(&self, width: Width, address: Expr) -> Expr {
        Expr::load(bw(width), address)
    }

    fn unary(&self, op: UnOp, value: Expr) -> Result<Expr, AlgebraError> {
        let op = match op {
            UnOp::Not => UnaryOp::Not,
            UnOp::Neg => UnaryOp::Neg,
        };
        Ok(Expr::unary(op, value))
    }

    fn binary(&self, op: BinOp, left: Expr, right: Expr) -> Result<Expr, AlgebraError> {
        let op = match op {
            BinOp::Add => BinaryOp::Add,
            BinOp::Sub => BinaryOp::Sub,
            BinOp::Mul => BinaryOp::Mul,
            BinOp::And => BinaryOp::And,
            BinOp::Or => BinaryOp::Or,
            BinOp::Xor => BinaryOp::Xor,
            BinOp::Shl => BinaryOp::Shl,
            BinOp::Shr => BinaryOp::Shr,
            BinOp::Sar => BinaryOp::Sar,
            BinOp::Rol => BinaryOp::Rol,
            BinOp::Ror => BinaryOp::Ror,
        };
        Expr::binary(op, left, right).map_err(|e| AlgebraError(e.to_string()))
    }

    fn resize(&self, kind: Resize, target: Width, value: Expr) -> Result<Expr, AlgebraError> {
        let op = match kind {
            Resize::Truncate => ResizeOp::Truncate,
            Resize::ZeroExtend => ResizeOp::ZeroExtend,
            Resize::SignExtend => ResizeOp::SignExtend,
        };
        Expr::resize(op, bw(target), value).map_err(|e| AlgebraError(e.to_string()))
    }

    fn intrinsic(&self, width: Width, name: &str, args: Vec<Expr>) -> Expr {
        Expr::intrinsic(bw(width), name, args)
    }

    fn width(&self, expr: &Expr) -> Width {
        nw(expr.width)
    }

    fn as_const(&self, expr: &Expr) -> Option<u64> {
        match expr.kind {
            ExprKind::Const { value } => Some(value),
            _ => None,
        }
    }
}

/// Modeling assumptions for native lifting and CFG recovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CfgOptions {
    /// Allow the lifter's labeled stack-disjoint alias relaxation (see
    /// [`vmpx_disasm::LiftOptions::assume_stack_disjoint`]). Blocks that used it carry
    /// [`NativeBlock::assumes_stack_disjoint`].
    pub assume_stack_disjoint: bool,
}

impl CfgOptions {
    /// Only proven lifts; no modeling assumption.
    pub const PROVEN: Self = Self {
        assume_stack_disjoint: false,
    };
    /// Stack-disjoint relaxation on (the default of the recovery entry points).
    pub const ASSUME_STACK_DISJOINT: Self = Self {
        assume_stack_disjoint: true,
    };
}

impl Default for CfgOptions {
    fn default() -> Self {
        Self::ASSUME_STACK_DISJOINT
    }
}

impl From<CfgOptions> for LiftOptions {
    fn from(o: CfgOptions) -> Self {
        Self {
            assume_stack_disjoint: o.assume_stack_disjoint,
        }
    }
}

/// Lifts a straight-line native x64 block to [`Expr`] semantics. Registers are
/// expressed over `<reg>_in` inputs (e.g. `rcx_in`). Enables the labeled
/// stack-disjoint assumption; check [`BlockSemantics::assumes_stack_disjoint`].
///
/// # Errors
/// See [`vmpx_disasm::lift_bytes`]; fails closed on anything unmodeled.
pub fn lift_native_block(bytes: &[u8], ip: u64) -> Result<BlockSemantics<Expr>, LiftError> {
    lift_native_block_with(bytes, ip, CfgOptions::default())
}

/// [`lift_native_block`] with explicit [`CfgOptions`].
///
/// # Errors
/// As [`lift_native_block`].
pub fn lift_native_block_with(
    bytes: &[u8],
    ip: u64,
    options: CfgOptions,
) -> Result<BlockSemantics<Expr>, LiftError> {
    lift_bytes_with(&VmAlgebra, bytes, ip, options.into())
}

/// How a native basic block ends. Only proven direct targets produce edges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Terminator {
    /// Body runs into the next leader with no branch instruction.
    FallThrough(u64),
    /// `jmp rel` to a constant target.
    Jump(u64),
    /// `jcc rel`: `cc` is the condition code; `not_taken` is the next instruction.
    CondJump {
        cc: String,
        taken: u64,
        not_taken: u64,
    },
    Return,
    /// Indirect/computed branch, call, out-of-range target, undecodable bytes or a
    /// body the lifter refuses. No successor edges are recorded.
    Unsupported(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EdgeKind {
    FallThrough,
    Jump,
    Taken,
    NotTaken,
}

#[derive(Clone, Debug)]
pub struct NativeBlock {
    pub start: u64,
    /// One past the last byte of the terminator (or body, if it has none).
    pub end: u64,
    /// Lift of the straight-line body (terminator excluded); `None` if the lifter refused it.
    pub semantics: Option<BlockSemantics<Expr>>,
    pub terminator: Terminator,
    /// The lift relied on the stack-disjoint assumption (mirrors
    /// `semantics.assumes_stack_disjoint`; `false` when not lifted).
    pub assumes_stack_disjoint: bool,
}

#[derive(Clone, Debug)]
pub struct NativeCfg {
    pub blocks: BTreeMap<u64, NativeBlock>,
    pub edges: Vec<(u64, u64, EdgeKind)>,
    pub entry: u64,
    /// OR of the blocks' [`NativeBlock::assumes_stack_disjoint`]: some block's lift is
    /// only valid if the function's frame is not reachable via non-`rsp`-derived pointers.
    pub assumes_stack_disjoint: bool,
}

impl NativeCfg {
    /// True when no block ended in [`Terminator::Unsupported`].
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.blocks
            .values()
            .all(|b| !matches!(b.terminator, Terminator::Unsupported(_)))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CfgLimits {
    pub max_blocks: usize,
    pub max_instructions: usize,
}

impl Default for CfgLimits {
    fn default() -> Self {
        Self {
            max_blocks: 4096,
            max_instructions: 65_536,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CfgError {
    EntryOutOfRange(u64),
    /// Retained for API stability; no longer produced. Overlapping instructions
    /// (a branch into the middle of a previously-decoded instruction) are now
    /// resolved by decoding each proven-reachable offset on its own.
    OverlappingInstructions(u64),
    LimitExceeded,
    Pe(String),
}

impl fmt::Display for CfgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntryOutOfRange(ip) => write!(f, "entry {ip:#x} is outside the code window"),
            Self::OverlappingInstructions(ip) => {
                write!(f, "overlapping instruction decode at {ip:#x}")
            }
            Self::LimitExceeded => write!(f, "CFG size limit exceeded"),
            Self::Pe(m) => write!(f, "PE error: {m}"),
        }
    }
}

impl std::error::Error for CfgError {}

enum Flow {
    Next,
    Jmp(u64),
    Jcc(String, u64, u64),
    Ret,
    Other(String),
}

fn classify(ins: &Instruction) -> Flow {
    match ins.flow_control() {
        FlowControl::Next => Flow::Next,
        FlowControl::UnconditionalBranch if ins.is_jmp_short_or_near() => {
            Flow::Jmp(ins.near_branch_target())
        }
        FlowControl::ConditionalBranch if ins.is_jcc_short_or_near() => Flow::Jcc(
            format!("{:?}", ins.condition_code()),
            ins.near_branch_target(),
            ins.next_ip(),
        ),
        FlowControl::Return if ins.mnemonic() == Mnemonic::Ret => Flow::Ret,
        fc => {
            let mut text = String::new();
            IntelFormatter::new().format(ins, &mut text);
            Flow::Other(format!("{fc:?} `{text}`"))
        }
    }
}

/// Recovers the CFG of the native function at `entry` in `code` (mapped at `base_ip`),
/// with the labeled stack-disjoint assumption on (see [`CfgOptions`]); the label is
/// surfaced as [`NativeCfg::assumes_stack_disjoint`].
///
/// # Errors
/// See [`recover_native_cfg_with`].
pub fn recover_native_cfg(code: &[u8], base_ip: u64, entry: u64) -> Result<NativeCfg, CfgError> {
    recover_native_cfg_with(code, base_ip, entry, CfgLimits::default())
}

/// [`recover_native_cfg`] with no modeling assumption: only proven lifts.
///
/// # Errors
/// See [`recover_native_cfg_with`].
pub fn recover_native_cfg_proven(
    code: &[u8],
    base_ip: u64,
    entry: u64,
) -> Result<NativeCfg, CfgError> {
    recover_native_cfg_opts(
        code,
        base_ip,
        entry,
        CfgLimits::default(),
        CfgOptions::PROVEN,
    )
}

/// Like [`recover_native_cfg`] with explicit size limits. Only constant `jmp`/`jcc`
/// targets are followed; indirect branches, calls and unliftable blocks end in
/// [`Terminator::Unsupported`] with no outgoing edges. A branch into the middle of a
/// previously-decoded instruction re-decodes at that proven-reachable offset, so
/// overlapping-instruction anti-disassembly resolves instead of failing.
///
/// # Errors
/// Entry outside `code`, or exceeding `limits`.
pub fn recover_native_cfg_with(
    code: &[u8],
    base_ip: u64,
    entry: u64,
    limits: CfgLimits,
) -> Result<NativeCfg, CfgError> {
    recover_native_cfg_opts(code, base_ip, entry, limits, CfgOptions::default())
}

/// [`recover_native_cfg_with`] with explicit [`CfgOptions`].
///
/// # Errors
/// As [`recover_native_cfg_with`].
#[allow(clippy::too_many_lines)]
pub fn recover_native_cfg_opts(
    code: &[u8],
    base_ip: u64,
    entry: u64,
    limits: CfgLimits,
    options: CfgOptions,
) -> Result<NativeCfg, CfgError> {
    let end_ip = base_ip.saturating_add(code.len() as u64);
    let in_range = |ip: u64| ip >= base_ip && ip < end_ip;
    if !in_range(entry) {
        return Err(CfgError::EntryOutOfRange(entry));
    }

    let mut instrs: BTreeMap<u64, Instruction> = BTreeMap::new();
    let mut bad: BTreeMap<u64, String> = BTreeMap::new();
    let mut leaders: BTreeSet<u64> = BTreeSet::new();
    let mut work = vec![entry];
    leaders.insert(entry);

    while let Some(leader) = work.pop() {
        let mut ip = leader;
        loop {
            if instrs.contains_key(&ip) || bad.contains_key(&ip) {
                leaders.insert(ip);
                break;
            }
            if !in_range(ip) {
                bad.insert(ip, "runs past the end of the code window".to_owned());
                break;
            }
            // `ip` is reached by proven control flow (entry, a constant direct branch
            // target, or a linear fall-through), so the bytes there are a real
            // instruction even when they overlap one decoded on another path:
            // VMProtect mutation branches into the middle of a "cover" instruction, so
            // two reachable streams legally share bytes. We decode each reachable offset
            // on its own and let the entries overlap, rather than treating the clash as
            // undecodable. No speculative offset is ever decoded, so this stays sound.
            let off = usize::try_from(ip - base_ip).map_err(|_| CfgError::LimitExceeded)?;
            let ins = Decoder::with_ip(64, &code[off..], ip, DecoderOptions::NONE).decode();
            if ins.is_invalid() {
                bad.insert(ip, "invalid instruction encoding".to_owned());
                break;
            }
            instrs.insert(ip, ins);
            if instrs.len() > limits.max_instructions {
                return Err(CfgError::LimitExceeded);
            }
            let mut targets = [None, None];
            match classify(&ins) {
                Flow::Next => {
                    ip = ins.next_ip();
                    continue;
                }
                Flow::Jmp(t) => targets[0] = Some(t),
                Flow::Jcc(_, t, f) => targets = [Some(t), Some(f)],
                Flow::Ret | Flow::Other(_) => {}
            }
            for t in targets.into_iter().flatten() {
                if in_range(t) && leaders.insert(t) {
                    if leaders.len() > limits.max_blocks {
                        return Err(CfgError::LimitExceeded);
                    }
                    work.push(t);
                }
            }
            break;
        }
    }

    let mut blocks = BTreeMap::new();
    let mut edges = Vec::new();
    for &start in &leaders {
        let mut ip = start;
        let mut body = Vec::new();
        let mut terminator = loop {
            let Some(ins) = instrs.get(&ip) else {
                let why = bad.get(&ip).map_or("undecoded bytes", String::as_str);
                break Terminator::Unsupported(why.to_owned());
            };
            match classify(ins) {
                Flow::Next => {
                    body.push(*ins);
                    ip = ins.next_ip();
                    if leaders.contains(&ip) {
                        break Terminator::FallThrough(ip);
                    }
                }
                Flow::Jmp(t) if leaders.contains(&t) => {
                    ip = ins.next_ip();
                    break Terminator::Jump(t);
                }
                Flow::Jcc(cc, t, f) if leaders.contains(&t) && leaders.contains(&f) => {
                    ip = ins.next_ip();
                    break Terminator::CondJump {
                        cc,
                        taken: t,
                        not_taken: f,
                    };
                }
                Flow::Ret => {
                    ip = ins.next_ip();
                    break Terminator::Return;
                }
                Flow::Jmp(_) | Flow::Jcc(..) => {
                    ip = ins.next_ip();
                    break Terminator::Unsupported("direct branch target outside code".to_owned());
                }
                Flow::Other(why) => {
                    ip = ins.next_ip();
                    break Terminator::Unsupported(why);
                }
            }
        };
        let semantics = match lift_block_with(&VmAlgebra, &body, options.into()) {
            Ok(sem) => Some(sem),
            Err(e) => {
                terminator = Terminator::Unsupported(e.to_string());
                None
            }
        };
        match &terminator {
            Terminator::FallThrough(t) => edges.push((start, *t, EdgeKind::FallThrough)),
            Terminator::Jump(t) => edges.push((start, *t, EdgeKind::Jump)),
            Terminator::CondJump {
                taken, not_taken, ..
            } => {
                edges.push((start, *taken, EdgeKind::Taken));
                edges.push((start, *not_taken, EdgeKind::NotTaken));
            }
            Terminator::Return | Terminator::Unsupported(_) => {}
        }
        blocks.insert(
            start,
            NativeBlock {
                start,
                end: ip,
                assumes_stack_disjoint: semantics
                    .as_ref()
                    .is_some_and(|s| s.assumes_stack_disjoint),
                semantics,
                terminator,
            },
        );
    }
    let assumes_stack_disjoint = blocks.values().any(|b| b.assumes_stack_disjoint);
    Ok(NativeCfg {
        blocks,
        edges,
        entry,
        assumes_stack_disjoint,
    })
}

/// PE variant: reads the executable section holding `entry` and recovers its CFG.
/// Block addresses are virtual addresses (`image_base + rva`).
///
/// # Errors
/// [`CfgError::Pe`] if `entry` is not in a file-backed executable section, else as
/// [`recover_native_cfg_with`].
pub fn recover_native_cfg_pe(pe: &PeImage<'_>, entry: Rva) -> Result<NativeCfg, CfgError> {
    recover_native_cfg_pe_with(pe, entry, CfgOptions::default())
}

/// [`recover_native_cfg_pe`] with explicit [`CfgOptions`].
///
/// # Errors
/// As [`recover_native_cfg_pe`].
pub fn recover_native_cfg_pe_with(
    pe: &PeImage<'_>,
    entry: Rva,
    options: CfgOptions,
) -> Result<NativeCfg, CfgError> {
    if !vmpx_disasm::is_executable_rva(pe, entry) {
        return Err(CfgError::Pe(format!(
            "{entry} is not in an executable section"
        )));
    }
    let section = pe
        .sections
        .iter()
        .find(|s| s.contains_mapped_rva(entry))
        .ok_or_else(|| CfgError::Pe(format!("{entry} is not in a section")))?;
    pe.rva_to_file_offset(entry)
        .map_err(|e| CfgError::Pe(e.to_string()))?;
    let start = usize::try_from(section.pointer_to_raw_data.0)
        .map_err(|_| CfgError::Pe("raw pointer too large".to_owned()))?;
    let len = if section.virtual_size == 0 {
        section.size_of_raw_data
    } else {
        section.virtual_size.min(section.size_of_raw_data)
    } as usize;
    let data = pe.data();
    let code = data
        .get(start..start.saturating_add(len).min(data.len()))
        .ok_or_else(|| CfgError::Pe("section bytes outside file".to_owned()))?;
    let base = pe
        .optional
        .image_base
        .checked_add_rva(section.virtual_address)
        .ok_or_else(|| CfgError::Pe("section VA overflows".to_owned()))?;
    let entry_va = pe
        .optional
        .image_base
        .checked_add_rva(entry)
        .ok_or_else(|| CfgError::Pe("entry VA overflows".to_owned()))?;
    recover_native_cfg_opts(code, base.0, entry_va.0, CfgLimits::default(), options)
}

// WAVE2: WP5 — decode a mutation-protected function to a native CFG (reuse
// vmpx-disasm decode/CFG). Intended signature once the CFG type exists:
//   pub fn recover_cfg(image: &ReadOnlyImage, entry: u64) -> Result<NativeCfg, CfgError>
// Stub keeps the module declared so WP5 fills only this file.
#[must_use]
pub fn recover_cfg(problem: &Problem) -> Option<()> {
    let _ = problem;
    None
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation)]

    use super::*;
    use vmpx_disasm::Gpr;

    fn eval(e: &Expr, env: &dyn Fn(&str) -> u64) -> u64 {
        let w = e.width;
        match &e.kind {
            ExprKind::Const { value } => *value,
            ExprKind::Register { name } => w.truncate(env(name)),
            ExprKind::Unary { op, value } => {
                let v = eval(value, env);
                w.truncate(match op {
                    UnaryOp::Not => !v,
                    UnaryOp::Neg => v.wrapping_neg(),
                })
            }
            ExprKind::Binary { op, left, right } => {
                let (l, r) = (eval(left, env), eval(right, env));
                w.truncate(match op {
                    BinaryOp::Add => l.wrapping_add(r),
                    BinaryOp::Sub => l.wrapping_sub(r),
                    BinaryOp::Mul => l.wrapping_mul(r),
                    BinaryOp::Xor => l ^ r,
                    BinaryOp::And => l & r,
                    BinaryOp::Or => l | r,
                    _ => panic!("unexpected op in test"),
                })
            }
            ExprKind::Resize { op, value } => {
                let v = eval(value, env);
                match op {
                    ResizeOp::Truncate | ResizeOp::ZeroExtend => w.truncate(v),
                    ResizeOp::SignExtend => panic!("unexpected sext in test"),
                }
            }
            _ => panic!("unexpected node in test"),
        }
    }

    const SAMPLES: [(u64, u64); 4] = [
        (0, 0),
        (0xDEAD_BEEF_1234_5678, 0xFFFF_FFFF_FFFF_FFFF),
        (0x1_0000_0001, 0x8000_0000),
        (u64::MAX, 1),
    ];

    fn env_for(c: u64, d: u64) -> impl Fn(&str) -> u64 {
        move |n: &str| match n {
            "rcx_in" => c,
            "rdx_in" => d,
            other => panic!("unexpected input {other}"),
        }
    }

    #[test]
    fn mov_xor_32bit_zero_extends() {
        // mov eax, ecx; xor eax, edx
        let sem = lift_native_block(&[0x89, 0xC8, 0x31, 0xD0], 0x1000).expect("lift");
        let rax = &sem.regs[&Gpr::Rax];
        assert_eq!(rax.width, BitWidth::W64);
        for (c, d) in SAMPLES {
            let want = u64::from((c as u32) ^ (d as u32));
            assert_eq!(eval(rax, &env_for(c, d)), want);
        }
    }

    #[test]
    fn lea_add_truncates() {
        // lea eax, [rcx+rdx]
        let sem = lift_native_block(&[0x8D, 0x04, 0x11], 0x1000).expect("lift");
        let rax = &sem.regs[&Gpr::Rax];
        for (c, d) in SAMPLES {
            let want = u64::from((c.wrapping_add(d)) as u32);
            assert_eq!(eval(rax, &env_for(c, d)), want);
        }
    }

    // ip = 0x1000: test ecx,ecx; je 0x1009; mov eax,1; ret
    const IF_CODE: &[u8] = &[0x85, 0xC9, 0x74, 0x05, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];

    #[test]
    fn straight_line_is_one_block() {
        let cfg = recover_native_cfg(&[0x89, 0xC8, 0x31, 0xD0, 0xC3], 0x1000, 0x1000).expect("cfg");
        assert_eq!(cfg.blocks.len(), 1);
        assert!(cfg.edges.is_empty());
        assert!(cfg.is_complete());
        let b = &cfg.blocks[&0x1000];
        assert_eq!(b.terminator, Terminator::Return);
        assert_eq!(b.semantics.as_ref().expect("sem").instruction_count, 2);
    }

    #[test]
    fn conditional_branch_splits_three_blocks() {
        let cfg = recover_native_cfg(IF_CODE, 0x1000, 0x1000).expect("cfg");
        assert_eq!(cfg.blocks.len(), 3);
        assert!(cfg.is_complete());
        assert_eq!(
            cfg.blocks[&0x1000].terminator,
            Terminator::CondJump {
                cc: "e".to_owned(),
                taken: 0x1009,
                not_taken: 0x1004
            }
        );
        assert_eq!(
            cfg.blocks[&0x1004].terminator,
            Terminator::FallThrough(0x1009)
        );
        assert_eq!(cfg.blocks[&0x1009].terminator, Terminator::Return);
        let mut edges = cfg.edges.clone();
        edges.sort_by_key(|e| (e.0, e.1));
        assert_eq!(
            edges,
            vec![
                (0x1000, 0x1004, EdgeKind::NotTaken),
                (0x1000, 0x1009, EdgeKind::Taken),
                (0x1004, 0x1009, EdgeKind::FallThrough),
            ]
        );
    }

    #[test]
    fn backward_jump_splits_existing_block() {
        // inc eax; dec ecx; jne 0x1002; ret
        let code = [0xFF, 0xC0, 0xFF, 0xC9, 0x75, 0xFC, 0xC3];
        let cfg = recover_native_cfg(&code, 0x1000, 0x1000).expect("cfg");
        assert_eq!(cfg.blocks.len(), 3);
        assert_eq!(
            cfg.blocks[&0x1000].terminator,
            Terminator::FallThrough(0x1002)
        );
        assert!(cfg.edges.contains(&(0x1002, 0x1002, EdgeKind::Taken)));
    }

    // sub rsp,0x28; mov [rsp+0x20],rcx; mov rax,[rdx]; mov rbx,[rsp+0x20]; ret
    const FRAME: &[u8] = &[
        0x48, 0x83, 0xEC, 0x28, 0x48, 0x89, 0x4C, 0x24, 0x20, 0x48, 0x8B, 0x02, 0x48, 0x8B, 0x5C,
        0x24, 0x20, 0xC3,
    ];
    // mov [rcx],rbx; mov rax,[rdx]; ret
    const ALIAS: &[u8] = &[0x48, 0x89, 0x19, 0x48, 0x8B, 0x02, 0xC3];

    #[test]
    fn stack_disjoint_load_lifts_with_label_and_proven_mode_refuses() {
        let cfg = recover_native_cfg(FRAME, 0x1000, 0x1000).expect("cfg");
        let b = &cfg.blocks[&0x1000];
        assert!(b.semantics.is_some() && b.assumes_stack_disjoint);
        assert!(cfg.assumes_stack_disjoint && cfg.is_complete());

        let cfg = recover_native_cfg_proven(FRAME, 0x1000, 0x1000).expect("cfg");
        let b = &cfg.blocks[&0x1000];
        assert!(b.semantics.is_none() && !b.assumes_stack_disjoint);
        assert!(!cfg.assumes_stack_disjoint && !cfg.is_complete());

        assert!(
            lift_native_block(&FRAME[..FRAME.len() - 1], 0x1000)
                .is_ok_and(|s| s.assumes_stack_disjoint)
        );
        assert!(
            lift_native_block_with(&FRAME[..FRAME.len() - 1], 0x1000, CfgOptions::PROVEN).is_err()
        );
    }

    #[test]
    fn genuine_alias_fails_closed_even_with_assumption() {
        let cfg = recover_native_cfg(ALIAS, 0x1000, 0x1000).expect("cfg");
        assert!(cfg.blocks[&0x1000].semantics.is_none());
        assert!(!cfg.assumes_stack_disjoint && !cfg.is_complete());
    }

    #[test]
    fn overlap_resolution_decodes_mid_instruction_target() {
        // 0x1000 jne 0x1005; 0x1002 mov eax,0x90C39090 (covers 0x1002..0x1007);
        // the taken target 0x1005 lands *inside* that mov, where the bytes are `C3` (ret).
        // A linear/one-pass decode would call this an overlap; decoding at the
        // proven-reached offset recovers the `ret` block.
        let code = [0x75, 0x03, 0xB8, 0x90, 0x90, 0xC3, 0x90, 0xC3];
        let cfg = recover_native_cfg(&code, 0x1000, 0x1000).expect("overlap resolves");
        assert!(cfg.is_complete());
        assert_eq!(cfg.blocks.len(), 3);
        // The mid-instruction target decoded to its own `ret` block.
        let tgt = &cfg.blocks[&0x1005];
        assert_eq!(tgt.terminator, Terminator::Return);
        assert!(tgt.semantics.is_some());
        // The covering instruction kept its own (overlapping) decode and lifted.
        let cover = &cfg.blocks[&0x1002];
        assert_eq!(cover.terminator, Terminator::Return);
        assert_eq!(cover.semantics.as_ref().expect("sem").instruction_count, 1);
        assert!(cfg.edges.contains(&(0x1000, 0x1005, EdgeKind::Taken)));
        assert!(cfg.edges.contains(&(0x1000, 0x1002, EdgeKind::NotTaken)));
    }

    #[test]
    fn computed_target_still_fails_closed() {
        // Overlap tolerance must not license speculative decode: an unresolved
        // computed branch stays refused with no edge and no invented successor block.
        // mov eax,ecx; jmp rax
        let code = [0x89, 0xC8, 0xFF, 0xE0];
        let cfg = recover_native_cfg(&code, 0x1000, 0x1000).expect("cfg");
        assert_eq!(cfg.blocks.len(), 1);
        assert!(cfg.edges.is_empty());
        assert!(!cfg.is_complete());
        assert!(matches!(
            cfg.blocks[&0x1000].terminator,
            Terminator::Unsupported(_)
        ));
    }

    #[test]
    fn indirect_jump_is_unsupported_without_edge() {
        // jmp rax
        let cfg = recover_native_cfg(&[0xFF, 0xE0], 0x1000, 0x1000).expect("cfg");
        assert_eq!(cfg.blocks.len(), 1);
        assert!(cfg.edges.is_empty());
        assert!(!cfg.is_complete());
        assert!(matches!(
            cfg.blocks[&0x1000].terminator,
            Terminator::Unsupported(_)
        ));
    }

    #[test]
    fn call_and_out_of_range_jump_fail_closed() {
        let cfg = recover_native_cfg(&[0xE8, 0, 0, 0, 0, 0xC3], 0x1000, 0x1000).expect("cfg");
        assert!(matches!(
            cfg.blocks[&0x1000].terminator,
            Terminator::Unsupported(_)
        ));
        assert!(cfg.edges.is_empty());
        let cfg = recover_native_cfg(&[0xE9, 0, 1, 0, 0], 0x1000, 0x1000).expect("cfg");
        assert!(matches!(
            cfg.blocks[&0x1000].terminator,
            Terminator::Unsupported(_)
        ));
        assert!(cfg.edges.is_empty());
    }

    #[test]
    fn corpus_unitops_exports_recover() {
        let Ok(bytes) = std::fs::read("E:/VMPX/testcorpus/corpus/unitops.dll") else {
            return;
        };
        let pe = PeImage::parse(&bytes).expect("pe");
        let report = pe.report();
        let exports = report.exports.expect("exports");
        let (mut blocks, mut edges, mut funcs, mut unsup) = (0, 0, 0, 0);
        for sym in exports.symbols.iter().filter(|s| s.forwarder.is_none()) {
            let cfg = recover_native_cfg_pe(&pe, sym.rva).expect("cfg");
            assert!(!cfg.blocks.is_empty());
            funcs += 1;
            blocks += cfg.blocks.len();
            edges += cfg.edges.len();
            unsup += cfg
                .blocks
                .values()
                .filter(|b| matches!(b.terminator, Terminator::Unsupported(_)))
                .count();
        }
        eprintln!("unitops: {funcs} funcs, {blocks} blocks, {edges} edges, {unsup} unsupported");
        assert!(funcs > 0);
    }

    #[test]
    fn limits_and_bad_entry() {
        assert_eq!(
            recover_native_cfg(IF_CODE, 0x1000, 0x2000).unwrap_err(),
            CfgError::EntryOutOfRange(0x2000)
        );
        let lim = CfgLimits {
            max_blocks: 2,
            max_instructions: 100,
        };
        assert_eq!(
            recover_native_cfg_with(IF_CODE, 0x1000, 0x1000, lim).unwrap_err(),
            CfgError::LimitExceeded
        );
    }

    #[test]
    fn algebra_round_trips_constant_and_binary() {
        let a = VmAlgebra;
        let c = a.constant(Width::W32, 0x1_0000_0005);
        assert_eq!(c, Expr::constant(BitWidth::W32, 5));
        assert_eq!(a.as_const(&c), Some(5));
        assert_eq!(a.width(&c), Width::W32);
        let x = a.input(Width::W32, "x");
        assert_eq!(a.as_const(&x), None);
        let sum = a.binary(BinOp::Add, x.clone(), c).expect("add");
        assert_eq!(
            sum,
            Expr::binary(
                BinaryOp::Add,
                Expr::register(BitWidth::W32, "x"),
                Expr::constant(BitWidth::W32, 5)
            )
            .expect("add")
        );
        assert!(a.binary(BinOp::Add, x, a.constant(Width::W8, 1)).is_err());
    }
}
