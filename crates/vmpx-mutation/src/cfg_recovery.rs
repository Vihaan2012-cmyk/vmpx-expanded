//! Native CFG recovery — the substrate the mutation passes operate on.

use vmpx_disasm::{
    Algebra, AlgebraError, BinOp, BlockSemantics, LiftError, Resize, UnOp, Width, lift_bytes,
};
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

/// Lifts a straight-line native x64 block to [`Expr`] semantics. Registers are
/// expressed over `<reg>_in` inputs (e.g. `rcx_in`).
///
/// # Errors
/// See [`vmpx_disasm::lift_bytes`]; fails closed on anything unmodeled.
pub fn lift_native_block(bytes: &[u8], ip: u64) -> Result<BlockSemantics<Expr>, LiftError> {
    lift_bytes(&VmAlgebra, bytes, ip)
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
