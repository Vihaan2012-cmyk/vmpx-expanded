use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;

use crate::BitWidth;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOp {
    Not,
    Neg,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Xor,
    And,
    Or,
    Shl,
    Shr,
    Sar,
    Rol,
    Ror,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResizeOp {
    Truncate,
    ZeroExtend,
    SignExtend,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExprKind {
    Const {
        value: u64,
    },
    Register {
        name: String,
    },
    Unary {
        op: UnaryOp,
        value: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Load {
        address: Box<Expr>,
    },
    Intrinsic {
        name: String,
        arguments: Vec<Expr>,
    },
    Extract {
        value: Box<Expr>,
        lsb: u32,
    },
    Concat {
        high: Box<Expr>,
        low: Box<Expr>,
    },
    Resize {
        op: ResizeOp,
        value: Box<Expr>,
    },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct Expr {
    pub width: BitWidth,
    pub kind: ExprKind,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ExprError {
    #[error("expression width mismatch: left is {left:?}, right is {right:?}")]
    WidthMismatch { left: BitWidth, right: BitWidth },

    #[error("invalid {op:?} from {source_width:?} to {target_width:?}")]
    InvalidResize {
        op: ResizeOp,
        source_width: BitWidth,
        target_width: BitWidth,
    },

    #[error("invalid {target_width:?}-bit extraction at bit {lsb} from {source_width:?}")]
    InvalidExtract {
        target_width: BitWidth,
        source_width: BitWidth,
        lsb: u32,
    },

    #[error("cannot concatenate {high_width:?} and {low_width:?} into a supported width")]
    InvalidConcat {
        high_width: BitWidth,
        low_width: BitWidth,
    },
}

impl Expr {
    #[must_use]
    pub const fn constant(width: BitWidth, value: u64) -> Self {
        Self {
            width,
            kind: ExprKind::Const {
                value: width.truncate(value),
            },
        }
    }

    #[must_use]
    pub fn register(width: BitWidth, name: impl Into<String>) -> Self {
        Self {
            width,
            kind: ExprKind::Register { name: name.into() },
        }
    }

    #[must_use]
    pub fn load(width: BitWidth, address: Self) -> Self {
        Self {
            width,
            kind: ExprKind::Load {
                address: Box::new(address),
            },
        }
    }

    #[must_use]
    pub fn intrinsic(width: BitWidth, name: impl Into<String>, arguments: Vec<Self>) -> Self {
        simplify_intrinsic(width, name.into(), arguments)
    }

    /// Extracts a supported-width bit slice.
    ///
    /// # Errors
    ///
    /// Rejects slices that exceed the source or do not begin on a byte boundary.
    pub fn extract(target: BitWidth, lsb: u32, value: Self) -> Result<Self, ExprError> {
        let end = lsb.checked_add(target.bits());
        if !lsb.is_multiple_of(8) || end.is_none_or(|end| end > value.width.bits()) {
            return Err(ExprError::InvalidExtract {
                target_width: target,
                source_width: value.width,
                lsb,
            });
        }
        Ok(Self {
            width: target,
            kind: ExprKind::Extract {
                value: Box::new(value),
                lsb,
            },
        }
        .simplify())
    }

    /// Concatenates high and low bit-vectors into a supported width.
    ///
    /// # Errors
    ///
    /// Rejects combined widths outside 8/16/32/64 bits.
    pub fn concat(high: Self, low: Self) -> Result<Self, ExprError> {
        let Some(bits) = high.width.bits().checked_add(low.width.bits()) else {
            return Err(ExprError::InvalidConcat {
                high_width: high.width,
                low_width: low.width,
            });
        };
        let Some(width) = BitWidth::from_bits(bits) else {
            return Err(ExprError::InvalidConcat {
                high_width: high.width,
                low_width: low.width,
            });
        };
        Ok(Self {
            width,
            kind: ExprKind::Concat {
                high: Box::new(high),
                low: Box::new(low),
            },
        }
        .simplify())
    }

    #[must_use]
    pub fn unary(op: UnaryOp, value: Self) -> Self {
        let width = value.width;
        Self {
            width,
            kind: ExprKind::Unary {
                op,
                value: Box::new(value),
            },
        }
        .simplify()
    }

    /// Builds a same-width binary expression and applies local simplification.
    ///
    /// # Errors
    ///
    /// Returns [`ExprError::WidthMismatch`] when operand widths differ.
    pub fn binary(op: BinaryOp, left: Self, right: Self) -> Result<Self, ExprError> {
        if left.width != right.width {
            return Err(ExprError::WidthMismatch {
                left: left.width,
                right: right.width,
            });
        }
        let width = left.width;
        Ok(Self {
            width,
            kind: ExprKind::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
        }
        .simplify())
    }

    /// Builds an explicit truncation or extension.
    ///
    /// # Errors
    ///
    /// Rejects truncation to a wider width and extension to a narrower width.
    pub fn resize(op: ResizeOp, target: BitWidth, value: Self) -> Result<Self, ExprError> {
        let source = value.width;
        let valid = match op {
            ResizeOp::Truncate => target.bits() <= source.bits(),
            ResizeOp::ZeroExtend | ResizeOp::SignExtend => target.bits() >= source.bits(),
        };
        if !valid {
            return Err(ExprError::InvalidResize {
                op,
                source_width: source,
                target_width: target,
            });
        }
        Ok(Self {
            width: target,
            kind: ExprKind::Resize {
                op,
                value: Box::new(value),
            },
        }
        .simplify())
    }

    #[must_use]
    pub fn dependencies(&self) -> BTreeSet<String> {
        let mut dependencies = BTreeSet::new();
        self.collect_dependencies(&mut dependencies);
        dependencies
    }

    /// Returns every opaque intrinsic name reachable from this expression.
    /// Register dependencies alone are insufficient for determining whether an
    /// expression is directly compilable because intrinsics can represent
    /// memory epochs, native-call results, and processor operations.
    #[must_use]
    pub fn intrinsic_dependencies(&self) -> BTreeSet<String> {
        let mut intrinsics = BTreeSet::new();
        let mut pending = vec![self];
        while let Some(expression) = pending.pop() {
            match &expression.kind {
                ExprKind::Const { .. } | ExprKind::Register { .. } => {}
                ExprKind::Unary { value, .. }
                | ExprKind::Load { address: value }
                | ExprKind::Extract { value, .. }
                | ExprKind::Resize { value, .. } => pending.push(value),
                ExprKind::Binary { left, right, .. }
                | ExprKind::Concat {
                    high: left,
                    low: right,
                } => {
                    pending.push(right);
                    pending.push(left);
                }
                ExprKind::Intrinsic { name, arguments } => {
                    intrinsics.insert(name.clone());
                    pending.extend(arguments);
                }
            }
        }
        intrinsics
    }

    fn collect_dependencies(&self, output: &mut BTreeSet<String>) {
        match &self.kind {
            ExprKind::Const { .. } => {}
            ExprKind::Register { name } => {
                output.insert(name.clone());
            }
            ExprKind::Unary { value, .. }
            | ExprKind::Resize { value, .. }
            | ExprKind::Extract { value, .. } => {
                value.collect_dependencies(output);
            }
            ExprKind::Binary { left, right, .. }
            | ExprKind::Concat {
                high: left,
                low: right,
            } => {
                left.collect_dependencies(output);
                right.collect_dependencies(output);
            }
            ExprKind::Load { address } => address.collect_dependencies(output),
            ExprKind::Intrinsic { arguments, .. } => {
                for argument in arguments {
                    argument.collect_dependencies(output);
                }
            }
        }
    }

    /// Replaces named register leaves and resimplifies the expression.
    ///
    /// # Errors
    ///
    /// A replacement must have the same width as the register it replaces.
    pub fn substitute(&self, replacements: &BTreeMap<String, Self>) -> Result<Self, ExprError> {
        let result = match &self.kind {
            ExprKind::Const { .. } => self.clone(),
            ExprKind::Register { name } => {
                if let Some(replacement) = replacements.get(name) {
                    if replacement.width != self.width {
                        return Err(ExprError::WidthMismatch {
                            left: self.width,
                            right: replacement.width,
                        });
                    }
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            ExprKind::Unary { op, value } => Self::unary(*op, value.substitute(replacements)?),
            ExprKind::Binary { op, left, right } => Self::binary(
                *op,
                left.substitute(replacements)?,
                right.substitute(replacements)?,
            )?,
            ExprKind::Load { address } => Self::load(self.width, address.substitute(replacements)?),
            ExprKind::Intrinsic { name, arguments } => Self::intrinsic(
                self.width,
                name.clone(),
                arguments
                    .iter()
                    .map(|argument| argument.substitute(replacements))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            ExprKind::Extract { value, lsb } => {
                Self::extract(self.width, *lsb, value.substitute(replacements)?)?
            }
            ExprKind::Concat { high, low } => Self::concat(
                high.substitute(replacements)?,
                low.substitute(replacements)?,
            )?,
            ExprKind::Resize { op, value } => {
                Self::resize(*op, self.width, value.substitute(replacements)?)?
            }
        };
        Ok(result.simplify())
    }

    /// Rebuilds an expression bottom-up and permits a caller to replace opaque
    /// intrinsics after their arguments have themselves been rewritten. A
    /// replacement must retain the intrinsic result width.
    ///
    /// # Errors
    ///
    /// Returns a width/shape error if rebuilding a child expression fails or
    /// the callback returns a value of the wrong width.
    pub fn rewrite_intrinsics<F>(&self, rewrite: &mut F) -> Result<Self, ExprError>
    where
        F: FnMut(BitWidth, &str, &[Self]) -> Option<Self>,
    {
        let rebuilt = match &self.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => self.clone(),
            ExprKind::Unary { op, value } => Self::unary(*op, value.rewrite_intrinsics(rewrite)?),
            ExprKind::Binary { op, left, right } => Self::binary(
                *op,
                left.rewrite_intrinsics(rewrite)?,
                right.rewrite_intrinsics(rewrite)?,
            )?,
            ExprKind::Load { address } => {
                Self::load(self.width, address.rewrite_intrinsics(rewrite)?)
            }
            ExprKind::Intrinsic { name, arguments } => {
                let arguments = arguments
                    .iter()
                    .map(|argument| argument.rewrite_intrinsics(rewrite))
                    .collect::<Result<Vec<_>, _>>()?;
                if let Some(replacement) = rewrite(self.width, name, &arguments) {
                    if replacement.width != self.width {
                        return Err(ExprError::WidthMismatch {
                            left: self.width,
                            right: replacement.width,
                        });
                    }
                    replacement
                } else {
                    Self::intrinsic(self.width, name.clone(), arguments)
                }
            }
            ExprKind::Extract { value, lsb } => {
                Self::extract(self.width, *lsb, value.rewrite_intrinsics(rewrite)?)?
            }
            ExprKind::Concat { high, low } => Self::concat(
                high.rewrite_intrinsics(rewrite)?,
                low.rewrite_intrinsics(rewrite)?,
            )?,
            ExprKind::Resize { op, value } => {
                Self::resize(*op, self.width, value.rewrite_intrinsics(rewrite)?)?
            }
        };
        Ok(rebuilt.simplify())
    }

    #[must_use]
    pub fn simplify(&self) -> Self {
        match &self.kind {
            ExprKind::Const { value } => Self::constant(self.width, *value),
            ExprKind::Register { .. } => self.clone(),
            ExprKind::Load { address } => Self::load(self.width, address.simplify()),
            ExprKind::Intrinsic { name, arguments } => simplify_intrinsic(
                self.width,
                name.clone(),
                arguments.iter().map(Self::simplify).collect(),
            ),
            ExprKind::Extract { value, lsb } => {
                simplify_extract(self.width, *lsb, value.simplify())
            }
            ExprKind::Concat { high, low } => simplify_concat(high.simplify(), low.simplify()),
            ExprKind::Unary { op, value } => simplify_unary(*op, value.simplify()),
            ExprKind::Binary { op, left, right } => {
                simplify_binary(*op, left.simplify(), right.simplify())
            }
            ExprKind::Resize { op, value } => simplify_resize(*op, self.width, value.simplify()),
        }
    }

    /// Counts expression-tree nodes, saturating at `limit + 1`.
    #[must_use]
    pub fn node_count_bounded(&self, limit: usize) -> usize {
        let overflow = limit.saturating_add(1);
        let mut count = 0_usize;
        let mut pending = vec![self];
        while let Some(expression) = pending.pop() {
            if count >= limit {
                return overflow;
            }
            count += 1;
            match &expression.kind {
                ExprKind::Const { .. } | ExprKind::Register { .. } => {}
                ExprKind::Unary { value, .. }
                | ExprKind::Load { address: value }
                | ExprKind::Extract { value, .. }
                | ExprKind::Resize { value, .. } => pending.push(value),
                ExprKind::Binary { left, right, .. }
                | ExprKind::Concat {
                    high: left,
                    low: right,
                } => {
                    pending.push(right);
                    pending.push(left);
                }
                ExprKind::Intrinsic { arguments, .. } => pending.extend(arguments),
            }
        }
        count
    }
}

fn constant_value(expr: &Expr) -> Option<u64> {
    match expr.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

fn simplify_unary(op: UnaryOp, value: Expr) -> Expr {
    if let Some(constant) = constant_value(&value) {
        let result = match op {
            UnaryOp::Not => !constant,
            UnaryOp::Neg => constant.wrapping_neg(),
        };
        return Expr::constant(value.width, result);
    }
    if let ExprKind::Unary {
        op: inner_op,
        value: inner,
    } = &value.kind
        && *inner_op == op
    {
        return (**inner).clone();
    }
    Expr {
        width: value.width,
        kind: ExprKind::Unary {
            op,
            value: Box::new(value),
        },
    }
}

/// Recognizes `not(not(x) + a) + b` and returns `x + (b - a)`.
///
/// Complementing is an involution, so `not(not(x) + a) == x - a` in wrapping
/// arithmetic. VMProtect uses this pair with differing constants to express a
/// plain pointer adjustment, which must fold here or every stack-relative
/// address it builds stays opaque.
fn fold_double_complement_offset(complement: &Expr, offset: &Expr) -> Option<Expr> {
    let ExprKind::Unary {
        op: UnaryOp::Not,
        value: inner,
    } = &complement.kind
    else {
        return None;
    };
    let ExprKind::Binary {
        op: BinaryOp::Add,
        left,
        right,
    } = &inner.kind
    else {
        return None;
    };
    let (
        ExprKind::Unary {
            op: UnaryOp::Not,
            value: base,
        },
        inner_offset,
    ) = (&left.kind, right)
    else {
        return None;
    };
    let width = base.width;
    if offset.width != width || inner_offset.width != width {
        return None;
    }
    let (Some(outer), Some(inner)) = (constant_value(offset), constant_value(inner_offset)) else {
        return None;
    };
    if outer == inner {
        return Some((**base).clone());
    }
    let delta = outer.wrapping_sub(inner);
    Some(Expr {
        width,
        kind: ExprKind::Binary {
            op: BinaryOp::Add,
            left: base.clone(),
            right: Box::new(Expr::constant(width, delta)),
        },
    })
}

fn simplify_intrinsic(width: BitWidth, name: String, arguments: Vec<Expr>) -> Expr {
    let constants = arguments
        .iter()
        .map(constant_value)
        .collect::<Option<Vec<_>>>();
    if width == BitWidth::W64
        && let Some(constants) = constants.as_deref()
        && let Some(flags) = evaluate_vm_flags(&name, constants)
    {
        return Expr::constant(width, flags);
    }
    if width == BitWidth::W64
        && let Some(constants) = constants.as_deref()
        && let Some(value) = evaluate_x86_wide_integer_result(&name, &arguments, constants)
    {
        return Expr::constant(width, value);
    }
    Expr {
        width,
        kind: ExprKind::Intrinsic { name, arguments },
    }
}

fn evaluate_x86_wide_integer_result(
    name: &str,
    arguments: &[Expr],
    constants: &[u64],
) -> Option<u64> {
    if arguments.is_empty()
        || arguments
            .iter()
            .any(|argument| argument.width != arguments[0].width)
    {
        return None;
    }
    let operand_width = arguments[0].width;
    let bits = operand_width.bits();
    if bits > 32 {
        // The symbolic VM expression currently stores the packed low/high
        // result in a qword, so only operations whose complete pair fits are
        // representable without introducing a 128-bit expression width.
        return None;
    }
    let mask = operand_width.mask();
    if name.starts_with("x86_unsigned_multiply_") {
        let [left, right] = constants else {
            return None;
        };
        let product = (left & mask).wrapping_mul(right & mask);
        return Some(product);
    }
    if name.starts_with("x86_unsigned_divide_") {
        let [high, low, divisor] = constants else {
            return None;
        };
        let divisor = divisor & mask;
        if divisor == 0 {
            return None;
        }
        let dividend = ((high & mask) << bits) | (low & mask);
        let quotient = dividend / divisor;
        if quotient > mask {
            // Architectural DIV would raise #DE rather than produce a value.
            return None;
        }
        let remainder = dividend % divisor;
        return Some((remainder << bits) | quotient);
    }
    None
}

fn evaluate_vm_flags(name: &str, arguments: &[u64]) -> Option<u64> {
    let (semantic, width) = match name {
        "vm_flags_add_w8" => (BinaryOp::Add, BitWidth::W8),
        "vm_flags_add_w16" => (BinaryOp::Add, BitWidth::W16),
        "vm_flags_add_w32" => (BinaryOp::Add, BitWidth::W32),
        "vm_flags_add_w64" => (BinaryOp::Add, BitWidth::W64),
        "vm_flags_nor_w8" => (BinaryOp::And, BitWidth::W8),
        "vm_flags_nor_w16" => (BinaryOp::And, BitWidth::W16),
        "vm_flags_nor_w32" => (BinaryOp::And, BitWidth::W32),
        "vm_flags_nor_w64" => (BinaryOp::And, BitWidth::W64),
        "vm_flags_nand_w8" => (BinaryOp::Or, BitWidth::W8),
        "vm_flags_nand_w16" => (BinaryOp::Or, BitWidth::W16),
        "vm_flags_nand_w32" => (BinaryOp::Or, BitWidth::W32),
        "vm_flags_nand_w64" => (BinaryOp::Or, BitWidth::W64),
        _ => return None,
    };
    let [left, right] = arguments else {
        return None;
    };
    let left = width.truncate(*left);
    let right = width.truncate(*right);
    if semantic == BinaryOp::Add {
        Some(add_status_flags(width, left, right))
    } else {
        let result = match semantic {
            BinaryOp::And => (!left & !right) & width.mask(),
            BinaryOp::Or => (!left | !right) & width.mask(),
            _ => return None,
        };
        Some(logical_status_flags(width, result))
    }
}

fn common_status_flags(width: BitWidth, result: u64) -> u64 {
    const RESERVED_AND_INTERRUPT: u64 = 0x202;
    const PARITY: u64 = 1 << 2;
    const ZERO: u64 = 1 << 6;
    const SIGN: u64 = 1 << 7;

    let result = width.truncate(result);
    let mut flags = RESERVED_AND_INTERRUPT;
    if result.to_le_bytes()[0].count_ones().is_multiple_of(2) {
        flags |= PARITY;
    }
    if result == 0 {
        flags |= ZERO;
    }
    if result & (1_u64 << (width.bits() - 1)) != 0 {
        flags |= SIGN;
    }
    flags
}

fn logical_status_flags(width: BitWidth, result: u64) -> u64 {
    common_status_flags(width, result)
}

fn add_status_flags(width: BitWidth, left: u64, right: u64) -> u64 {
    const CARRY: u64 = 1;
    const AUXILIARY: u64 = 1 << 4;
    const OVERFLOW: u64 = 1 << 11;

    let result = width.truncate(left.wrapping_add(right));
    let mut flags = common_status_flags(width, result);
    if u128::from(left) + u128::from(right) > u128::from(width.mask()) {
        flags |= CARRY;
    }
    if (left ^ right ^ result) & 0x10 != 0 {
        flags |= AUXILIARY;
    }
    let sign = 1_u64 << (width.bits() - 1);
    if (!(left ^ right) & (left ^ result) & sign) != 0 {
        flags |= OVERFLOW;
    }
    flags
}

#[allow(clippy::too_many_lines)]
fn simplify_binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    let width = left.width;
    if let (Some(a), Some(b)) = (constant_value(&left), constant_value(&right)) {
        return Expr::constant(width, evaluate_binary(op, width, a, b));
    }
    let operands_are_complements = matches!(
        (&left.kind, &right.kind),
        (ExprKind::Unary { op: UnaryOp::Not, value }, _) if value.as_ref() == &right
    ) || matches!(
        (&left.kind, &right.kind),
        (_, ExprKind::Unary { op: UnaryOp::Not, value }) if value.as_ref() == &left
    );
    if operands_are_complements {
        match op {
            BinaryOp::And => return Expr::constant(width, 0),
            BinaryOp::Or | BinaryOp::Xor => return Expr::constant(width, width.mask()),
            _ => {}
        }
    }
    if op == BinaryOp::Add {
        // `not(not(x) + c) + c == x` for any constant `c`. VMProtect uses this
        // pair to express a plain pointer adjustment, so folding it here keeps
        // stack-relative addresses recognizable instead of opaque.
        if let Some(folded) = fold_double_complement_offset(&left, &right)
            .or_else(|| fold_double_complement_offset(&right, &left))
        {
            return folded;
        }
        return normalize_addition(width, left, right);
    }
    let zero_right = constant_value(&right) == Some(0);
    let all_ones_right = constant_value(&right) == Some(width.mask());
    if matches!(op, BinaryOp::And | BinaryOp::Or) && left == right {
        return left;
    }
    match op {
        BinaryOp::Sub
        | BinaryOp::Xor
        | BinaryOp::Or
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::Sar
        | BinaryOp::Rol
        | BinaryOp::Ror
            if zero_right =>
        {
            return left;
        }
        BinaryOp::Mul | BinaryOp::And if zero_right => return Expr::constant(width, 0),
        BinaryOp::Mul if constant_value(&right) == Some(1) => return left,
        BinaryOp::And if all_ones_right => return left,
        BinaryOp::Xor | BinaryOp::Sub if left == right => return Expr::constant(width, 0),
        _ => {}
    }
    Expr {
        width,
        kind: ExprKind::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        },
    }
}

fn normalize_addition(width: BitWidth, left: Expr, right: Expr) -> Expr {
    fn collect(expression: Expr, non_constants: &mut Vec<Expr>, constant: &mut u64) {
        match expression {
            Expr {
                kind:
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        left,
                        right,
                    },
                ..
            } => {
                collect(*left, non_constants, constant);
                collect(*right, non_constants, constant);
            }
            Expr {
                kind: ExprKind::Const { value },
                ..
            } => *constant = constant.wrapping_add(value),
            other => non_constants.push(other),
        }
    }

    let mut non_constants = Vec::new();
    let mut constant = 0_u64;
    collect(left, &mut non_constants, &mut constant);
    collect(right, &mut non_constants, &mut constant);
    constant = width.truncate(constant);

    let mut terms = non_constants.into_iter();
    let Some(mut result) = terms.next() else {
        return Expr::constant(width, constant);
    };
    for term in terms {
        result = Expr {
            width,
            kind: ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(result),
                right: Box::new(term),
            },
        };
    }
    if constant == 0 {
        result
    } else {
        Expr {
            width,
            kind: ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(result),
                right: Box::new(Expr::constant(width, constant)),
            },
        }
    }
}

fn simplify_resize(op: ResizeOp, target: BitWidth, value: Expr) -> Expr {
    if target == value.width {
        return value;
    }
    if let Some(constant) = constant_value(&value) {
        let resized = match op {
            ResizeOp::Truncate | ResizeOp::ZeroExtend => constant,
            ResizeOp::SignExtend => sign_extend(constant, value.width),
        };
        return Expr::constant(target, resized);
    }
    Expr {
        width: target,
        kind: ExprKind::Resize {
            op,
            value: Box::new(value),
        },
    }
}

fn simplify_extract(target: BitWidth, lsb: u32, value: Expr) -> Expr {
    if lsb == 0 && target == value.width {
        return value;
    }
    if let Some(constant) = constant_value(&value) {
        return Expr::constant(target, constant.wrapping_shr(lsb));
    }
    if let ExprKind::Extract {
        value: inner,
        lsb: inner_lsb,
    } = &value.kind
        && let Some(combined_lsb) = inner_lsb.checked_add(lsb)
    {
        return Expr {
            width: target,
            kind: ExprKind::Extract {
                value: inner.clone(),
                lsb: combined_lsb,
            },
        };
    }
    if let ExprKind::Concat { high, low } = &value.kind {
        let low_bits = low.width.bits();
        let end = lsb.saturating_add(target.bits());
        if end <= low_bits {
            return simplify_extract(target, lsb, (**low).clone());
        }
        if lsb >= low_bits {
            return simplify_extract(target, lsb - low_bits, (**high).clone());
        }
    }
    Expr {
        width: target,
        kind: ExprKind::Extract {
            value: Box::new(value),
            lsb,
        },
    }
}

fn simplify_concat(high: Expr, low: Expr) -> Expr {
    let width = BitWidth::from_bits(high.width.bits() + low.width.bits())
        .expect("concat constructor validates the combined width");
    if let (Some(high_value), Some(low_value)) = (constant_value(&high), constant_value(&low)) {
        return Expr::constant(width, high_value.wrapping_shl(low.width.bits()) | low_value);
    }
    if let (
        ExprKind::Extract {
            value: high_source,
            lsb: high_lsb,
        },
        ExprKind::Extract {
            value: low_source,
            lsb: low_lsb,
        },
    ) = (&high.kind, &low.kind)
        && high_source == low_source
        && low_lsb.checked_add(low.width.bits()) == Some(*high_lsb)
    {
        return simplify_extract(width, *low_lsb, (**high_source).clone());
    }
    Expr {
        width,
        kind: ExprKind::Concat {
            high: Box::new(high),
            low: Box::new(low),
        },
    }
}

fn evaluate_binary(op: BinaryOp, width: BitWidth, left: u64, right: u64) -> u64 {
    let shift_mask = if width == BitWidth::W64 { 0x3F } else { 0x1F };
    let shift_count = u32::try_from(right & shift_mask).unwrap_or(0);
    let rotate_count = shift_count % width.bits();
    let result = match op {
        BinaryOp::Add => left.wrapping_add(right),
        BinaryOp::Sub => left.wrapping_sub(right),
        BinaryOp::Mul => left.wrapping_mul(right),
        BinaryOp::Xor => left ^ right,
        BinaryOp::And => left & right,
        BinaryOp::Or => left | right,
        BinaryOp::Shl => left.wrapping_shl(shift_count),
        BinaryOp::Shr => left.wrapping_shr(shift_count),
        BinaryOp::Sar => arithmetic_shift_right(left, width, shift_count),
        BinaryOp::Rol => rotate_left(left, width, rotate_count),
        BinaryOp::Ror => rotate_right(left, width, rotate_count),
    };
    width.truncate(result)
}

fn sign_extend(value: u64, source: BitWidth) -> u64 {
    let value = source.truncate(value);
    let sign_bit = 1_u64 << (source.bits() - 1);
    if value & sign_bit == 0 {
        value
    } else {
        value | !source.mask()
    }
}

fn arithmetic_shift_right(value: u64, width: BitWidth, count: u32) -> u64 {
    let signed = i64::from_ne_bytes(sign_extend(value, width).to_ne_bytes());
    signed.wrapping_shr(count).cast_unsigned()
}

fn rotate_left(value: u64, width: BitWidth, count: u32) -> u64 {
    let bits = width.bits();
    let value = width.truncate(value);
    if count == 0 {
        value
    } else {
        width.truncate((value << count) | (value >> (bits - count)))
    }
}

fn rotate_right(value: u64, width: BitWidth, count: u32) -> u64 {
    let bits = width.bits();
    let value = width.truncate(value);
    if count == 0 {
        value
    } else {
        width.truncate((value >> count) | (value << (bits - count)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_double_complement_pointer_offset() {
        let base = Expr::register(BitWidth::W64, "vsp_initial");
        let offset = Expr::constant(BitWidth::W64, 64);
        let inner = Expr::binary(
            BinaryOp::Add,
            Expr::unary(UnaryOp::Not, base.clone()),
            offset.clone(),
        )
        .expect("add");
        let expression =
            Expr::binary(BinaryOp::Add, Expr::unary(UnaryOp::Not, inner), offset).expect("add");
        assert_eq!(expression.simplify(), base);
    }

    #[test]
    fn folds_double_complement_with_differing_offsets() {
        // not(not(x) + 64) + 72 == x + 8
        let base = Expr::register(BitWidth::W64, "vsp_initial");
        let inner = Expr::binary(
            BinaryOp::Add,
            Expr::unary(UnaryOp::Not, base.clone()),
            Expr::constant(BitWidth::W64, 64),
        )
        .expect("add");
        let expression = Expr::binary(
            BinaryOp::Add,
            Expr::unary(UnaryOp::Not, inner),
            Expr::constant(BitWidth::W64, 72),
        )
        .expect("add");
        let expected =
            Expr::binary(BinaryOp::Add, base, Expr::constant(BitWidth::W64, 8)).expect("add");
        assert_eq!(expression.simplify(), expected.simplify());
    }

    #[test]
    fn does_not_fold_double_complement_with_mismatched_offset() {
        let base = Expr::register(BitWidth::W64, "vsp_initial");
        let inner = Expr::binary(
            BinaryOp::Add,
            Expr::unary(UnaryOp::Not, base),
            Expr::constant(BitWidth::W64, 64),
        )
        .expect("add");
        let expression = Expr::binary(
            BinaryOp::Add,
            Expr::unary(UnaryOp::Not, inner),
            Expr::constant(BitWidth::W64, 72),
        )
        .expect("add");
        assert!(matches!(
            expression.simplify().kind,
            ExprKind::Binary { .. }
        ));
    }

    #[test]
    fn folds_width_truncated_arithmetic() {
        let expression = Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W8, 0xFF),
            Expr::constant(BitWidth::W8, 2),
        )
        .expect("matching widths");
        assert_eq!(expression, Expr::constant(BitWidth::W8, 1));
    }

    #[test]
    fn folds_complement_tautologies_used_by_nand_zero() {
        for width in [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64] {
            let value = Expr::register(width, format!("value_{width:?}"));
            let complement = Expr::unary(UnaryOp::Not, value.clone());

            let all_ones = Expr::binary(BinaryOp::Or, value.clone(), complement.clone())
                .expect("same-width complement");
            assert_eq!(all_ones, Expr::constant(width, width.mask()));
            assert_eq!(
                Expr::unary(UnaryOp::Not, all_ones),
                Expr::constant(width, 0)
            );
            assert_eq!(
                Expr::binary(BinaryOp::And, complement.clone(), value.clone())
                    .expect("same-width complement"),
                Expr::constant(width, 0)
            );
            assert_eq!(
                Expr::binary(BinaryOp::Xor, complement, value).expect("same-width complement"),
                Expr::constant(width, width.mask())
            );
        }
    }

    #[test]
    fn canonicalizes_nested_pointer_additions() {
        let pointer = Expr::register(BitWidth::W64, "pointer");
        let minus_32 = u64::from_ne_bytes((-32_i64).to_ne_bytes());
        let minus_8 = u64::from_ne_bytes((-8_i64).to_ne_bytes());
        let nested = Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W64, 56),
            Expr::binary(
                BinaryOp::Add,
                Expr::constant(BitWidth::W64, minus_32),
                Expr::binary(
                    BinaryOp::Add,
                    pointer.clone(),
                    Expr::constant(BitWidth::W64, minus_8),
                )
                .expect("matching widths"),
            )
            .expect("matching widths"),
        )
        .expect("matching widths");
        let expected = Expr::binary(BinaryOp::Add, pointer, Expr::constant(BitWidth::W64, 16))
            .expect("matching widths");
        assert_eq!(nested, expected);
    }

    #[test]
    fn folds_concrete_vm_status_flag_intrinsics() {
        const CARRY: u64 = 1;
        const RESERVED_AND_INTERRUPT: u64 = 0x202;
        const PARITY: u64 = 1 << 2;
        const AUXILIARY: u64 = 1 << 4;
        const ZERO: u64 = 1 << 6;

        let flags = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_add_w32",
            vec![
                Expr::constant(BitWidth::W32, u64::from(u32::MAX)),
                Expr::constant(BitWidth::W32, 1),
            ],
        );
        assert_eq!(
            flags,
            Expr::constant(
                BitWidth::W64,
                RESERVED_AND_INTERRUPT | CARRY | PARITY | AUXILIARY | ZERO,
            )
        );

        let nand_flags = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_nand_w32",
            vec![
                Expr::constant(BitWidth::W32, u64::from(u32::MAX)),
                Expr::constant(BitWidth::W32, u64::from(u32::MAX)),
            ],
        );
        assert_eq!(
            nand_flags,
            Expr::constant(BitWidth::W64, RESERVED_AND_INTERRUPT | PARITY | ZERO)
        );
    }

    #[test]
    fn substitutes_registers_and_tracks_dependencies() {
        let expression = Expr::binary(
            BinaryOp::Xor,
            Expr::register(BitWidth::W32, "key"),
            Expr::register(BitWidth::W32, "operand"),
        )
        .expect("matching widths");
        assert_eq!(
            expression.dependencies(),
            BTreeSet::from(["key".to_owned(), "operand".to_owned()])
        );
        let replacements = BTreeMap::from([
            ("key".to_owned(), Expr::constant(BitWidth::W32, 7)),
            ("operand".to_owned(), Expr::constant(BitWidth::W32, 3)),
        ]);
        assert_eq!(
            expression
                .substitute(&replacements)
                .expect("matching widths"),
            Expr::constant(BitWidth::W32, 4)
        );
    }

    #[test]
    fn tracks_nested_intrinsic_dependencies_separately_from_registers() {
        let expression = Expr::intrinsic(
            BitWidth::W64,
            "outer",
            vec![
                Expr::register(BitWidth::W64, "input"),
                Expr::intrinsic(BitWidth::W64, "inner", Vec::new()),
            ],
        );
        assert_eq!(
            expression.dependencies(),
            BTreeSet::from(["input".to_owned()])
        );
        assert_eq!(
            expression.intrinsic_dependencies(),
            BTreeSet::from(["inner".to_owned(), "outer".to_owned()])
        );
    }

    #[test]
    fn rewrites_intrinsics_bottom_up_and_checks_widths() {
        let expression = Expr::intrinsic(
            BitWidth::W64,
            "outer",
            vec![Expr::intrinsic(BitWidth::W64, "counter", Vec::new())],
        );
        let rewritten = expression
            .rewrite_intrinsics(&mut |width, name, arguments| match name {
                "counter" => Some(Expr::constant(width, 9)),
                "outer" if arguments == [Expr::constant(BitWidth::W64, 9)] => {
                    Some(Expr::constant(width, 17))
                }
                _ => None,
            })
            .expect("same-width rewrite");
        assert_eq!(rewritten, Expr::constant(BitWidth::W64, 17));

        let error = expression
            .rewrite_intrinsics(&mut |_, name, _| {
                (name == "counter").then(|| Expr::constant(BitWidth::W32, 0))
            })
            .expect_err("cross-width rewrite must fail");
        assert!(matches!(error, ExprError::WidthMismatch { .. }));
    }

    #[test]
    fn folds_representable_x86_unsigned_wide_results() {
        let divided = Expr::intrinsic(
            BitWidth::W64,
            "x86_unsigned_divide_w32_7",
            vec![
                Expr::constant(BitWidth::W32, 0),
                Expr::constant(BitWidth::W32, 17),
                Expr::constant(BitWidth::W32, 5),
            ],
        );
        assert_eq!(divided, Expr::constant(BitWidth::W64, (2_u64 << 32) | 3));
        let multiplied = Expr::intrinsic(
            BitWidth::W64,
            "x86_unsigned_multiply_w32_8",
            vec![
                Expr::constant(BitWidth::W32, u64::from(u32::MAX)),
                Expr::constant(BitWidth::W32, 2),
            ],
        );
        assert_eq!(multiplied, Expr::constant(BitWidth::W64, 0x1_FFFF_FFFE));
    }

    #[test]
    fn sign_extension_and_arithmetic_shift_are_width_aware() {
        let extended = Expr::resize(
            ResizeOp::SignExtend,
            BitWidth::W64,
            Expr::constant(BitWidth::W8, 0x80),
        )
        .expect("valid extension");
        assert_eq!(
            extended,
            Expr::constant(BitWidth::W64, 0xFFFF_FFFF_FFFF_FF80)
        );
        let shifted = Expr::binary(
            BinaryOp::Sar,
            Expr::constant(BitWidth::W8, 0x80),
            Expr::constant(BitWidth::W8, 1),
        )
        .expect("matching widths");
        assert_eq!(shifted, Expr::constant(BitWidth::W8, 0xC0));
    }

    #[test]
    fn x86_shift_counts_use_architectural_masks() {
        for (width, value, count, expected) in [
            (BitWidth::W8, 1, 8, 0),
            (BitWidth::W8, 1, 31, 0),
            (BitWidth::W16, 1, 16, 0),
            (BitWidth::W32, 1, 32, 1),
            (BitWidth::W64, 1, 64, 1),
        ] {
            let shifted = Expr::binary(
                BinaryOp::Shl,
                Expr::constant(width, value),
                Expr::constant(width, count),
            )
            .expect("matching widths");
            assert_eq!(shifted, Expr::constant(width, expected));
        }
        let arithmetic = Expr::binary(
            BinaryOp::Sar,
            Expr::constant(BitWidth::W16, 0x8000),
            Expr::constant(BitWidth::W16, 16),
        )
        .expect("matching widths");
        assert_eq!(arithmetic, Expr::constant(BitWidth::W16, 0xFFFF));
    }

    #[test]
    fn rejects_cross_width_binary_operations() {
        let error = Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W8, 1),
            Expr::constant(BitWidth::W16, 1),
        )
        .expect_err("width mismatch must fail");
        assert!(matches!(error, ExprError::WidthMismatch { .. }));
    }

    #[test]
    fn extracts_and_reassembles_partial_values() {
        let original = Expr::register(BitWidth::W64, "original");
        let low = Expr::extract(BitWidth::W32, 0, original.clone()).expect("low dword");
        let high = Expr::extract(BitWidth::W32, 32, original.clone()).expect("high dword");
        assert_eq!(Expr::concat(high, low).expect("qword concat"), original);
        assert_eq!(
            Expr::extract(BitWidth::W16, 8, Expr::constant(BitWidth::W32, 0xAABB_CCDD))
                .expect("constant extract"),
            Expr::constant(BitWidth::W16, 0xBBCC)
        );
    }
}
