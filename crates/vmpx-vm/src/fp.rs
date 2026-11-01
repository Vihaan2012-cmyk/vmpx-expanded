//! IEEE-754 floating-point vocabulary for the expression IR.
//!
//! `Expr` is an integer bit-vector IR; a float is its bit pattern (`W64` for
//! f64, `W32` for f32), exactly how `movq`/`movsd` see it. FP operations are
//! `Intrinsic` nodes with reserved `fp_*` names, so the type tag is the name:
//!
//! - `fp_{add,sub,mul,div,min,max}_{f32,f64}` (a, b)
//! - `fp_{sqrt,neg,abs}_{f32,f64}` (a)
//! - `fp_cvt_{f32,f64}_to_{f64,f32}` (a)
//! - `fp_from_{i32,i64}_to_{f32,f64}` (signed integer)
//! - `fp_to_{i32,i64}_from_{f32,f64}` (truncating, x86 "integer indefinite")
//! - `fp_cmp_{eq,ne,lt,le,gt,ge,unord,ord}_{f32,f64}` (a, b) -> `W8` 0/1
//! - `fp_call_<target>_{f32,f64}` (args...) a pure external (libm) function
//!
//! Semantics are SSE scalar with the default MXCSR (round-to-nearest-even,
//! all exceptions masked, no FTZ/DAZ). Constant operands fold bit-exactly,
//! including x86 NaN propagation (first NaN operand, quieted; invalid -> the
//! default QNaN), independent of the host. A call is never evaluated.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::mutable_key_type
)]

use crate::{BitWidth, Expr};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FpFormat {
    F32,
    F64,
}

impl FpFormat {
    #[must_use]
    pub const fn width(self) -> BitWidth {
        match self {
            Self::F32 => BitWidth::W32,
            Self::F64 => BitWidth::W64,
        }
    }

    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }

    const fn quiet_bit(self) -> u64 {
        match self {
            Self::F32 => 1 << 22,
            Self::F64 => 1 << 51,
        }
    }

    const fn default_nan(self) -> u64 {
        match self {
            Self::F32 => 0xFFC0_0000,
            Self::F64 => 0xFFF8_0000_0000_0000,
        }
    }

    const fn sign_bit(self) -> u64 {
        match self {
            Self::F32 => 1 << 31,
            Self::F64 => 1 << 63,
        }
    }

    fn is_nan(self, bits: u64) -> bool {
        match self {
            Self::F32 => f32::from_bits(bits as u32).is_nan(),
            Self::F64 => f64::from_bits(bits).is_nan(),
        }
    }

    fn value(self, bits: u64) -> f64 {
        match self {
            Self::F32 => f64::from(f32::from_bits(bits as u32)),
            Self::F64 => f64::from_bits(bits),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FpBinary {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FpUnary {
    Sqrt,
    Neg,
    Abs,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FpPredicate {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Unord,
    Ord,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum FpIntrinsic {
    Binary(FpBinary, FpFormat),
    Unary(FpUnary, FpFormat),
    Convert { from: FpFormat, to: FpFormat },
    FromInt { int_bits: u32, to: FpFormat },
    ToInt { from: FpFormat, int_bits: u32 },
    Compare(FpPredicate, FpFormat),
    Call { target: String, format: FpFormat },
}

const BINARY: [(&str, FpBinary); 6] = [
    ("add", FpBinary::Add),
    ("sub", FpBinary::Sub),
    ("mul", FpBinary::Mul),
    ("div", FpBinary::Div),
    ("min", FpBinary::Min),
    ("max", FpBinary::Max),
];
const UNARY: [(&str, FpUnary); 3] = [
    ("sqrt", FpUnary::Sqrt),
    ("neg", FpUnary::Neg),
    ("abs", FpUnary::Abs),
];
const PREDICATE: [(&str, FpPredicate); 8] = [
    ("eq", FpPredicate::Eq),
    ("ne", FpPredicate::Ne),
    ("lt", FpPredicate::Lt),
    ("le", FpPredicate::Le),
    ("gt", FpPredicate::Gt),
    ("ge", FpPredicate::Ge),
    ("unord", FpPredicate::Unord),
    ("ord", FpPredicate::Ord),
];

fn int_bits(s: &str) -> Option<u32> {
    match s {
        "i32" => Some(32),
        "i64" => Some(64),
        _ => None,
    }
}

impl FpIntrinsic {
    /// Recognizes a reserved `fp_*` intrinsic name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        let rest = name.strip_prefix("fp_")?;
        if let Some(call) = rest.strip_prefix("call_") {
            let (target, suffix) = call.rsplit_once('_')?;
            if target.is_empty() {
                return None;
            }
            return Some(Self::Call {
                target: target.to_owned(),
                format: FpFormat::parse(suffix)?,
            });
        }
        let parts: Vec<&str> = rest.split('_').collect();
        match parts.as_slice() {
            [op, fmt] => {
                let fmt = FpFormat::parse(fmt)?;
                if let Some((_, op)) = BINARY.iter().find(|(n, _)| n == op) {
                    return Some(Self::Binary(*op, fmt));
                }
                UNARY.iter().find(|(n, _)| n == op).map(|(_, op)| Self::Unary(*op, fmt))
            }
            ["cvt", from, "to", to] => Some(Self::Convert {
                from: FpFormat::parse(from)?,
                to: FpFormat::parse(to)?,
            }),
            ["from", int, "to", to] => Some(Self::FromInt {
                int_bits: int_bits(int)?,
                to: FpFormat::parse(to)?,
            }),
            ["to", int, "from", from] => Some(Self::ToInt {
                from: FpFormat::parse(from)?,
                int_bits: int_bits(int)?,
            }),
            ["cmp", pred, fmt] => {
                let (_, pred) = PREDICATE.iter().find(|(n, _)| n == pred)?;
                Some(Self::Compare(*pred, FpFormat::parse(fmt)?))
            }
            _ => None,
        }
    }

    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::Binary(op, f) => {
                format!("fp_{}_{}", BINARY.iter().find(|(_, o)| o == op).map_or("", |p| p.0), f.suffix())
            }
            Self::Unary(op, f) => {
                format!("fp_{}_{}", UNARY.iter().find(|(_, o)| o == op).map_or("", |p| p.0), f.suffix())
            }
            Self::Convert { from, to } => format!("fp_cvt_{}_to_{}", from.suffix(), to.suffix()),
            Self::FromInt { int_bits, to } => format!("fp_from_i{int_bits}_to_{}", to.suffix()),
            Self::ToInt { from, int_bits } => format!("fp_to_i{int_bits}_from_{}", from.suffix()),
            Self::Compare(p, f) => format!(
                "fp_cmp_{}_{}",
                PREDICATE.iter().find(|(_, o)| o == p).map_or("", |p| p.0),
                f.suffix()
            ),
            Self::Call { target, format } => format!("fp_call_{target}_{}", format.suffix()),
        }
    }

    /// True for the intrinsics whose result is itself a float bit pattern.
    #[must_use]
    pub const fn yields_float(&self) -> bool {
        matches!(
            self,
            Self::Binary(..)
                | Self::Unary(..)
                | Self::Convert { .. }
                | Self::FromInt { .. }
                | Self::Call { .. }
        )
    }

    /// Operand order does not change the value (modulo which NaN is chosen).
    #[must_use]
    pub const fn is_commutative(&self) -> bool {
        matches!(
            self,
            Self::Binary(FpBinary::Add | FpBinary::Mul, _)
                | Self::Compare(
                    FpPredicate::Eq | FpPredicate::Ne | FpPredicate::Unord | FpPredicate::Ord,
                    _
                )
        )
    }

    #[must_use]
    pub const fn result_width(&self) -> BitWidth {
        match self {
            Self::Binary(_, f) | Self::Unary(_, f) | Self::Call { format: f, .. } => f.width(),
            Self::Convert { to, .. } | Self::FromInt { to, .. } => to.width(),
            Self::ToInt { int_bits, .. } => {
                if *int_bits == 32 {
                    BitWidth::W32
                } else {
                    BitWidth::W64
                }
            }
            Self::Compare(..) => BitWidth::W8,
        }
    }

    /// The operand widths this intrinsic requires (`None` for a variadic call).
    #[must_use]
    pub fn operand_widths(&self) -> Option<Vec<BitWidth>> {
        Some(match self {
            Self::Binary(_, f) | Self::Compare(_, f) => vec![f.width(); 2],
            Self::Unary(_, f) => vec![f.width()],
            Self::Convert { from, .. } | Self::ToInt { from, .. } => vec![from.width()],
            Self::FromInt { int_bits, .. } => {
                vec![if *int_bits == 32 { BitWidth::W32 } else { BitWidth::W64 }]
            }
            Self::Call { .. } => return None,
        })
    }

    /// Bit-exact result for constant operands under x86 SSE semantics; `None`
    /// for a call or ill-shaped operands.
    #[must_use]
    pub fn evaluate(&self, args: &[u64]) -> Option<u64> {
        match (self, args) {
            (Self::Binary(op, f), [a, b]) => Some(binary(*op, *f, *a, *b)),
            (Self::Unary(op, f), [a]) => Some(unary(*op, *f, *a)),
            (Self::Convert { from, to }, [a]) => Some(convert(*from, *to, *a)),
            (Self::FromInt { int_bits, to }, [a]) => Some(from_int(*int_bits, *to, *a)),
            (Self::ToInt { from, int_bits }, [a]) => Some(to_int(*from, *int_bits, *a)),
            (Self::Compare(p, f), [a, b]) => Some(u64::from(compare(*p, *f, *a, *b))),
            _ => None,
        }
    }

    /// Builds the intrinsic over `args`.
    ///
    /// # Errors
    /// An operand whose width does not match.
    pub fn build(&self, args: Vec<Expr>) -> Result<Expr, String> {
        if let Some(widths) = self.operand_widths()
            && (widths.len() != args.len() || widths.iter().zip(&args).any(|(w, a)| *w != a.width))
        {
            return Err(format!("operands of {} have the wrong shape", self.name()));
        }
        if matches!(self, Self::Call { .. })
            && args.iter().any(|a| !matches!(a.width, BitWidth::W32 | BitWidth::W64))
        {
            return Err("call operands must be 32/64-bit".to_owned());
        }
        Ok(Expr::intrinsic(self.result_width(), self.name(), args))
    }
}

fn propagate(f: FpFormat, a: u64, b: u64) -> Option<u64> {
    if f.is_nan(a) {
        Some(a | f.quiet_bit())
    } else if f.is_nan(b) {
        Some(b | f.quiet_bit())
    } else {
        None
    }
}

fn binary(op: FpBinary, f: FpFormat, a: u64, b: u64) -> u64 {
    if matches!(op, FpBinary::Min | FpBinary::Max) {
        if f.is_nan(a) || f.is_nan(b) {
            return b;
        }
        let (x, y) = (f.value(a), f.value(b));
        let pick_a = match op {
            FpBinary::Min => x < y,
            _ => x > y,
        };
        return if pick_a { a } else { b };
    }
    if let Some(nan) = propagate(f, a, b) {
        return nan;
    }
    match f {
        FpFormat::F32 => {
            let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
            let r = match op {
                FpBinary::Add => x + y,
                FpBinary::Sub => x - y,
                FpBinary::Mul => x * y,
                _ => x / y,
            };
            if r.is_nan() { f.default_nan() } else { u64::from(r.to_bits()) }
        }
        FpFormat::F64 => {
            let (x, y) = (f64::from_bits(a), f64::from_bits(b));
            let r = match op {
                FpBinary::Add => x + y,
                FpBinary::Sub => x - y,
                FpBinary::Mul => x * y,
                _ => x / y,
            };
            if r.is_nan() { f.default_nan() } else { r.to_bits() }
        }
    }
}

fn unary(op: FpUnary, f: FpFormat, a: u64) -> u64 {
    match op {
        FpUnary::Neg => a ^ f.sign_bit(),
        FpUnary::Abs => a & !f.sign_bit(),
        FpUnary::Sqrt => {
            if f.is_nan(a) {
                return a | f.quiet_bit();
            }
            match f {
                FpFormat::F32 => {
                    let r = f32::from_bits(a as u32).sqrt();
                    if r.is_nan() { f.default_nan() } else { u64::from(r.to_bits()) }
                }
                FpFormat::F64 => {
                    let r = f64::from_bits(a).sqrt();
                    if r.is_nan() { f.default_nan() } else { r.to_bits() }
                }
            }
        }
    }
}

fn convert(from: FpFormat, to: FpFormat, a: u64) -> u64 {
    if from == to {
        return a;
    }
    match (from, to) {
        (FpFormat::F32, FpFormat::F64) => {
            let bits = a as u32;
            if f32::from_bits(bits).is_nan() {
                let sign = u64::from(bits >> 31) << 63;
                let fraction = u64::from(bits & 0x007F_FFFF) << 29;
                sign | (0x7FF << 52) | fraction | to.quiet_bit()
            } else {
                f64::from(f32::from_bits(bits)).to_bits()
            }
        }
        _ => {
            if f64::from_bits(a).is_nan() {
                let sign = (a >> 63) << 31;
                let fraction = (a >> 29) & 0x007F_FFFF;
                sign | (0xFF << 23) | fraction | to.quiet_bit()
            } else {
                u64::from((f64::from_bits(a) as f32).to_bits())
            }
        }
    }
}

fn from_int(int_bits: u32, to: FpFormat, a: u64) -> u64 {
    let signed: i64 = if int_bits == 32 { i64::from(a as u32 as i32) } else { a as i64 };
    match to {
        FpFormat::F64 => (signed as f64).to_bits(),
        FpFormat::F32 => u64::from((signed as f32).to_bits()),
    }
}

fn to_int(from: FpFormat, int_bits: u32, a: u64) -> u64 {
    let indefinite: u64 = if int_bits == 32 { 0x8000_0000 } else { 1 << 63 };
    if from.is_nan(a) {
        return indefinite;
    }
    let t = from.value(a).trunc();
    let limit = if int_bits == 32 { 2_147_483_648.0 } else { 9_223_372_036_854_775_808.0 };
    if !(-limit <= t && t < limit) {
        return indefinite;
    }
    if int_bits == 32 { u64::from(t as i32 as u32) } else { t as i64 as u64 }
}

fn compare(p: FpPredicate, f: FpFormat, a: u64, b: u64) -> bool {
    let unordered = f.is_nan(a) || f.is_nan(b);
    let (x, y) = (f.value(a), f.value(b));
    match p {
        FpPredicate::Unord => unordered,
        FpPredicate::Ord => !unordered,
        FpPredicate::Ne => unordered || x != y,
        _ if unordered => false,
        FpPredicate::Eq => x == y,
        FpPredicate::Lt => x < y,
        FpPredicate::Le => x <= y,
        FpPredicate::Gt => x > y,
        _ => x >= y,
    }
}

/// Constant folding hook for `Expr::intrinsic`: `Some(bits)` when `name` is an
/// `fp_*` operation over constant operands of the right widths.
pub(crate) fn fold_constant(name: &str, width: BitWidth, arguments: &[Expr]) -> Option<u64> {
    let op = FpIntrinsic::parse(name)?;
    if op.result_width() != width {
        return None;
    }
    if let Some(widths) = op.operand_widths()
        && (widths.len() != arguments.len() || widths.iter().zip(arguments).any(|(w, a)| *w != a.width))
    {
        return None;
    }
    let constants: Option<Vec<u64>> = arguments
        .iter()
        .map(|a| match a.kind {
            crate::ExprKind::Const { value } => Some(value),
            _ => None,
        })
        .collect();
    op.evaluate(&constants?)
}

/// A float constant of `format` as a bit-pattern `Expr`.
#[must_use]
pub fn constant(format: FpFormat, value: f64) -> Expr {
    match format {
        FpFormat::F64 => Expr::constant(BitWidth::W64, value.to_bits()),
        FpFormat::F32 => Expr::constant(BitWidth::W32, u64::from((value as f32).to_bits())),
    }
}

/// `op(a, b)` over floats of `format`.
///
/// # Errors
/// An operand of the wrong width.
pub fn binary_expr(op: FpBinary, format: FpFormat, a: Expr, b: Expr) -> Result<Expr, String> {
    FpIntrinsic::Binary(op, format).build(vec![a, b])
}

/// `op(a)` over a float of `format`.
///
/// # Errors
/// An operand of the wrong width.
pub fn unary_expr(op: FpUnary, format: FpFormat, a: Expr) -> Result<Expr, String> {
    FpIntrinsic::Unary(op, format).build(vec![a])
}

/// A pure external (libm-style) function result; args are float/int bit patterns.
///
/// # Errors
/// An operand that is not 32/64-bit.
pub fn call_expr(target: &str, format: FpFormat, args: Vec<Expr>) -> Result<Expr, String> {
    FpIntrinsic::Call { target: target.to_owned(), format }.build(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k64(v: f64) -> Expr {
        constant(FpFormat::F64, v)
    }

    #[test]
    fn names_round_trip() {
        for n in [
            "fp_add_f64", "fp_mul_f32", "fp_sqrt_f64", "fp_neg_f32", "fp_cvt_f32_to_f64",
            "fp_from_i64_to_f64", "fp_to_i32_from_f64", "fp_cmp_lt_f64", "fp_call_powf_f64",
            "fp_call_iat_0x40_f32",
        ] {
            assert_eq!(FpIntrinsic::parse(n).expect(n).name(), n);
        }
        assert!(FpIntrinsic::parse("fp_add_f16").is_none());
        assert!(FpIntrinsic::parse("vm_flags_add_w64").is_none());
    }

    #[test]
    fn constants_fold_bit_exactly() {
        let e = binary_expr(FpBinary::Add, FpFormat::F64, k64(0.1), k64(0.2)).unwrap();
        assert_eq!(e, k64(0.1 + 0.2));
        let s = unary_expr(FpUnary::Sqrt, FpFormat::F64, k64(2.0)).unwrap();
        assert_eq!(s, k64(2.0_f64.sqrt()));
        let sym = Expr::register(BitWidth::W64, "x");
        assert!(matches!(
            binary_expr(FpBinary::Add, FpFormat::F64, sym, k64(1.0)).unwrap().kind,
            crate::ExprKind::Intrinsic { .. }
        ));
    }

    #[test]
    fn x86_nan_rules() {
        let snan = 0x7FF0_0000_0000_0001_u64;
        let qnan_other = 0x7FF8_0000_0000_00AB_u64;
        // first NaN operand wins, quieted
        assert_eq!(binary(FpBinary::Add, FpFormat::F64, snan, qnan_other), snan | (1 << 51));
        assert_eq!(binary(FpBinary::Add, FpFormat::F64, 1.0_f64.to_bits(), qnan_other), qnan_other);
        // invalid operation -> default QNaN
        assert_eq!(
            binary(FpBinary::Sub, FpFormat::F64, f64::INFINITY.to_bits(), f64::INFINITY.to_bits()),
            0xFFF8_0000_0000_0000
        );
        assert_eq!(unary(FpUnary::Sqrt, FpFormat::F64, (-1.0_f64).to_bits()), 0xFFF8_0000_0000_0000);
        // minsd: unordered returns the second operand
        assert_eq!(binary(FpBinary::Min, FpFormat::F64, qnan_other, 1.0_f64.to_bits()), 1.0_f64.to_bits());
    }

    #[test]
    fn conversions_and_compares() {
        assert_eq!(to_int(FpFormat::F64, 64, 1.0e30_f64.to_bits()), 1 << 63);
        assert_eq!(to_int(FpFormat::F64, 32, (-7.9_f64).to_bits()), u64::from((-7_i32) as u32));
        assert_eq!(from_int(64, FpFormat::F64, (-3_i64) as u64), (-3.0_f64).to_bits());
        assert!(compare(FpPredicate::Ne, FpFormat::F64, f64::NAN.to_bits(), 1.0_f64.to_bits()));
        assert!(!compare(FpPredicate::Lt, FpFormat::F64, f64::NAN.to_bits(), 1.0_f64.to_bits()));
        assert_eq!(convert(FpFormat::F64, FpFormat::F32, 0.1_f64.to_bits()), u64::from(0.1_f32.to_bits()));
    }
}
