use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Weak};

use serde::{Serialize, Serializer};
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
        value: ExprRef,
    },
    Binary {
        op: BinaryOp,
        left: ExprRef,
        right: ExprRef,
    },
    Load {
        address: ExprRef,
    },
    Intrinsic {
        name: String,
        arguments: Vec<Expr>,
    },
    Extract {
        value: ExprRef,
        lsb: u32,
    },
    Concat {
        high: ExprRef,
        low: ExprRef,
    },
    Resize {
        op: ResizeOp,
        value: ExprRef,
    },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct Expr {
    pub width: BitWidth,
    pub kind: ExprKind,
}

struct Node {
    expr: Expr,
    hash: u64,
    /// Set once one `simplify` pass over this node is known to be a no-op.
    stable: AtomicBool,
}

/// A hash-consed shared expression node. Every `ExprRef` comes from the global
/// interner, so structurally equal nodes are the same allocation: equality is a
/// pointer comparison, hashing is O(1), and cloning never copies a subtree.
#[derive(Clone)]
pub struct ExprRef(Arc<Node>);

type SubMemo = HashMap<*const Node, Expr>;
type RwMemo = HashMap<*const Node, Expr>;

#[derive(Default)]
struct Interner {
    buckets: HashMap<u64, Vec<Weak<Node>>>,
    inserted: usize,
    sweep_at: usize,
}

static INTERNER: LazyLock<Mutex<Interner>> = LazyLock::new(|| Mutex::new(Interner::default()));

impl ExprRef {
    #[must_use]
    pub fn new(expr: Expr) -> Self {
        let mut hasher = std::hash::DefaultHasher::new();
        expr.hash(&mut hasher);
        let hash = hasher.finish();
        let mut guard = INTERNER.lock().unwrap_or_else(|e| e.into_inner());
        let interner = &mut *guard;
        let bucket = interner.buckets.entry(hash).or_default();
        let mut index = 0;
        while index < bucket.len() {
            match bucket[index].upgrade() {
                Some(node) if node.expr == expr => return Self(node),
                Some(_) => index += 1,
                None => {
                    bucket.swap_remove(index);
                }
            }
        }
        let node = Arc::new(Node {
            expr,
            hash,
            stable: AtomicBool::new(false),
        });
        bucket.push(Arc::downgrade(&node));
        interner.inserted += 1;
        if interner.inserted >= interner.sweep_at.max(1 << 16) {
            interner.buckets.retain(|_, bucket| {
                bucket.retain(|weak| weak.strong_count() > 0);
                !bucket.is_empty()
            });
            interner.inserted = 0;
            interner.sweep_at = interner.buckets.len().saturating_mul(2);
        }
        Self(node)
    }

    fn address(&self) -> *const Node {
        Arc::as_ptr(&self.0)
    }

    fn sub_memo(
        &self,
        replacements: &BTreeMap<String, Expr>,
        memo: &mut SubMemo,
    ) -> Result<Expr, ExprError> {
        if let Some(done) = memo.get(&self.address()) {
            return Ok(done.clone());
        }
        let result = self.0.expr.sub_memo(replacements, memo)?;
        memo.insert(self.address(), result.clone());
        Ok(result)
    }

    fn rw_memo<F>(&self, rewrite: &mut F, memo: &mut RwMemo) -> Result<Expr, ExprError>
    where
        F: FnMut(BitWidth, &str, &[Expr]) -> Option<Expr>,
    {
        if let Some(done) = memo.get(&self.address()) {
            return Ok(done.clone());
        }
        let result = self.0.expr.rw_memo(rewrite, memo)?;
        memo.insert(self.address(), result.clone());
        Ok(result)
    }

    /// One bottom-up simplification pass, skipping subtrees already known stable.
    fn simplified(&self) -> Self {
        if self.0.stable.load(Ordering::Relaxed) {
            return self.clone();
        }
        let result = self.0.expr.simplify_pass();
        if result == self.0.expr {
            self.0.stable.store(true, Ordering::Relaxed);
            return self.clone();
        }
        Self::new(result)
    }
}

impl Deref for ExprRef {
    type Target = Expr;
    fn deref(&self) -> &Expr {
        &self.0.expr
    }
}

impl AsRef<Expr> for ExprRef {
    fn as_ref(&self) -> &Expr {
        &self.0.expr
    }
}

impl PartialEq for ExprRef {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ExprRef {}

impl Hash for ExprRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.hash);
    }
}

impl std::fmt::Debug for ExprRef {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.expr.fmt(formatter)
    }
}

impl Serialize for ExprRef {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.expr.serialize(serializer)
    }
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
                address: ExprRef::new(address),
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
                value: ExprRef::new(value),
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
                high: ExprRef::new(high),
                low: ExprRef::new(low),
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
                value: ExprRef::new(value),
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
                left: ExprRef::new(left),
                right: ExprRef::new(right),
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
                value: ExprRef::new(value),
            },
        }
        .simplify())
    }

    #[must_use]
    pub fn dependencies(&self) -> BTreeSet<String> {
        let mut dependencies = BTreeSet::new();
        self.walk_distinct(&mut |expression| {
            if let ExprKind::Register { name } = &expression.kind {
                dependencies.insert(name.clone());
            }
        });
        dependencies
    }

    /// Returns every opaque intrinsic name reachable from this expression.
    /// Register dependencies alone are insufficient for determining whether an
    /// expression is directly compilable because intrinsics can represent
    /// memory epochs, native-call results, and processor operations.
    #[must_use]
    pub fn intrinsic_dependencies(&self) -> BTreeSet<String> {
        let mut intrinsics = BTreeSet::new();
        self.walk_distinct(&mut |expression| {
            if let ExprKind::Intrinsic { name, .. } = &expression.kind {
                intrinsics.insert(name.clone());
            }
        });
        intrinsics
    }

    /// Visits this expression and every distinct shared descendant once.
    pub(crate) fn walk_distinct<'a>(&'a self, visit: &mut impl FnMut(&'a Self)) {
        let mut seen: HashSet<*const Node> = HashSet::new();
        let mut pending = vec![self];
        while let Some(expression) = pending.pop() {
            visit(expression);
            let mut push = |child: &'a ExprRef, pending: &mut Vec<&'a Self>| {
                if seen.insert(child.address()) {
                    pending.push(child.as_ref());
                }
            };
            match &expression.kind {
                ExprKind::Const { .. } | ExprKind::Register { .. } => {}
                ExprKind::Unary { value, .. }
                | ExprKind::Load { address: value }
                | ExprKind::Extract { value, .. }
                | ExprKind::Resize { value, .. } => push(value, &mut pending),
                ExprKind::Binary { left, right, .. }
                | ExprKind::Concat {
                    high: left,
                    low: right,
                } => {
                    push(right, &mut pending);
                    push(left, &mut pending);
                }
                ExprKind::Intrinsic { arguments, .. } => pending.extend(arguments),
            }
        }
    }

    /// Replaces named register leaves and resimplifies the expression.
    ///
    /// # Errors
    ///
    /// A replacement must have the same width as the register it replaces.
    pub fn substitute(&self, replacements: &BTreeMap<String, Self>) -> Result<Self, ExprError> {
        self.sub_memo(replacements, &mut HashMap::new())
    }

    /// Rebuilds an expression bottom-up and permits a caller to replace opaque
    /// intrinsics after their arguments have themselves been rewritten. A
    /// replacement must retain the intrinsic result width. The callback runs once
    /// per distinct shared subexpression.
    ///
    /// # Errors
    ///
    /// Returns a width/shape error if rebuilding a child expression fails or
    /// the callback returns a value of the wrong width.
    pub fn rewrite_intrinsics<F>(&self, rewrite: &mut F) -> Result<Self, ExprError>
    where
        F: FnMut(BitWidth, &str, &[Self]) -> Option<Self>,
    {
        self.rw_memo(rewrite, &mut HashMap::new())
    }

    fn sub_memo(
        &self,
        replacements: &BTreeMap<String, Self>,
        memo: &mut SubMemo,
    ) -> Result<Self, ExprError> {
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
            ExprKind::Unary { op, value } => Self::unary(*op, value.sub_memo(replacements, memo)?),
            ExprKind::Binary { op, left, right } => Self::binary(
                *op,
                left.sub_memo(replacements, memo)?,
                right.sub_memo(replacements, memo)?,
            )?,
            ExprKind::Load { address } => {
                Self::load(self.width, address.sub_memo(replacements, memo)?)
            }
            ExprKind::Intrinsic { name, arguments } => Self::intrinsic(
                self.width,
                name.clone(),
                arguments
                    .iter()
                    .map(|argument| argument.sub_memo(replacements, memo))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            ExprKind::Extract { value, lsb } => {
                Self::extract(self.width, *lsb, value.sub_memo(replacements, memo)?)?
            }
            ExprKind::Concat { high, low } => Self::concat(
                high.sub_memo(replacements, memo)?,
                low.sub_memo(replacements, memo)?,
            )?,
            ExprKind::Resize { op, value } => {
                Self::resize(*op, self.width, value.sub_memo(replacements, memo)?)?
            }
        };
        Ok(result.simplify())
    }

    fn rw_memo<F>(&self, rewrite: &mut F, memo: &mut RwMemo) -> Result<Self, ExprError>
    where
        F: FnMut(BitWidth, &str, &[Self]) -> Option<Self>,
    {
        let rebuilt = match &self.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => self.clone(),
            ExprKind::Unary { op, value } => Self::unary(*op, value.rw_memo(rewrite, memo)?),
            ExprKind::Binary { op, left, right } => Self::binary(
                *op,
                left.rw_memo(rewrite, memo)?,
                right.rw_memo(rewrite, memo)?,
            )?,
            ExprKind::Load { address } => Self::load(self.width, address.rw_memo(rewrite, memo)?),
            ExprKind::Intrinsic { name, arguments } => {
                let arguments = arguments
                    .iter()
                    .map(|argument| argument.rw_memo(rewrite, memo))
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
                Self::extract(self.width, *lsb, value.rw_memo(rewrite, memo)?)?
            }
            ExprKind::Concat { high, low } => {
                Self::concat(high.rw_memo(rewrite, memo)?, low.rw_memo(rewrite, memo)?)?
            }
            ExprKind::Resize { op, value } => {
                Self::resize(*op, self.width, value.rw_memo(rewrite, memo)?)?
            }
        };
        Ok(rebuilt.simplify())
    }

    #[must_use]
    pub fn simplify(&self) -> Self {
        self.simplify_pass()
    }

    fn simplify_pass(&self) -> Self {
        match &self.kind {
            ExprKind::Const { value } => Self::constant(self.width, *value),
            ExprKind::Register { .. } => self.clone(),
            ExprKind::Load { address } => {
                Self::load(self.width, address.simplified().deref().clone())
            }
            ExprKind::Intrinsic { name, arguments } => simplify_intrinsic(
                self.width,
                name.clone(),
                arguments.iter().map(Self::simplify_pass).collect(),
            ),
            ExprKind::Extract { value, lsb } => {
                simplify_extract(self.width, *lsb, value.simplified().deref().clone())
            }
            ExprKind::Concat { high, low } => simplify_concat(
                high.simplified().deref().clone(),
                low.simplified().deref().clone(),
            ),
            ExprKind::Unary { op, value } => {
                simplify_unary(*op, value.simplified().deref().clone())
            }
            ExprKind::Binary { op, left, right } => simplify_binary(
                *op,
                left.simplified().deref().clone(),
                right.simplified().deref().clone(),
            ),
            ExprKind::Resize { op, value } => {
                simplify_resize(*op, self.width, value.simplified().deref().clone())
            }
        }
    }

    /// Counts distinct expression nodes (shared subtrees once), saturating at
    /// `limit + 1`.
    #[must_use]
    pub fn node_count_bounded(&self, limit: usize) -> usize {
        distinct_node_count_bounded(std::iter::once(self), limit)
    }
}

/// Counts distinct nodes reachable from all `roots` together, so a subtree shared
/// between expressions is counted once. Saturates at `limit + 1`.
pub fn distinct_node_count_bounded<'a>(
    roots: impl IntoIterator<Item = &'a Expr>,
    limit: usize,
) -> usize {
    let overflow = limit.saturating_add(1);
    let mut seen: HashSet<*const Node> = HashSet::new();
    let mut count = 0_usize;
    let mut pending: Vec<&Expr> = Vec::new();
    let mut seen_roots: HashSet<&Expr> = HashSet::new();
    for root in roots {
        if !seen_roots.insert(root) {
            continue;
        }
        count += 1;
        if count > limit {
            return overflow;
        }
        pending.push(root);
        while let Some(expression) = pending.pop() {
            let mut visit = |child: &'a ExprRef, pending: &mut Vec<&'a Expr>| {
                if seen.insert(child.address()) {
                    count += 1;
                    pending.push(child.as_ref());
                }
            };
            match &expression.kind {
                ExprKind::Const { .. } | ExprKind::Register { .. } => {}
                ExprKind::Unary { value, .. }
                | ExprKind::Load { address: value }
                | ExprKind::Extract { value, .. }
                | ExprKind::Resize { value, .. } => visit(value, &mut pending),
                ExprKind::Binary { left, right, .. }
                | ExprKind::Concat {
                    high: left,
                    low: right,
                } => {
                    visit(right, &mut pending);
                    visit(left, &mut pending);
                }
                ExprKind::Intrinsic { arguments, .. } => {
                    for argument in arguments {
                        count += 1;
                        pending.push(argument);
                    }
                }
            }
            if count > limit {
                return overflow;
            }
        }
    }
    count
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
            value: ExprRef::new(value),
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
            right: ExprRef::new(Expr::constant(width, delta)),
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
    // `vm_flags_<op>_wN` names a plain arithmetic operation on a flag word. When
    // an argument is symbolic the constant evaluator above cannot fold it, but
    // the intrinsic is still just that operation, so rewriting it keeps pointer
    // arithmetic affine. Leaving it opaque would make every address derived from
    // it unanalyzable, which in turn turns a valid stack load into an
    // unresolvable one.
    if let Some((operation, operand_width)) = vm_flags_operation(&name)
        && operand_width == width
        && let [left, right] = arguments.as_slice()
        && let Ok(rewritten) = Expr::binary(operation, left.clone(), right.clone())
    {
        return rewritten.simplify();
    }
    if width == BitWidth::W64
        && let Some(constants) = constants.as_deref()
        && let Some(value) = evaluate_x86_wide_integer_result(&name, &arguments, constants)
    {
        return Expr::constant(width, value);
    }
    if let Some(constants) = constants.as_deref()
        && let Some(value) = evaluate_double_shift(&name, width, &arguments, constants)
    {
        return Expr::constant(width, value);
    }
    Expr {
        width,
        kind: ExprKind::Intrinsic { name, arguments },
    }
}

/// Exact SHLD/SHRD result for constant operands; `None` when the architecture
/// leaves the result undefined (16-bit counts above 16).
fn evaluate_double_shift(
    name: &str,
    width: BitWidth,
    arguments: &[Expr],
    constants: &[u64],
) -> Option<u64> {
    let left = if name.starts_with("x86_double_shift_left_") {
        true
    } else if name.starts_with("x86_double_shift_right_") {
        false
    } else {
        return None;
    };
    let ([destination, source, count], [d, s, c]) = (
        <[u64; 3]>::try_from(constants).ok()?,
        <[&Expr; 3]>::try_from(arguments.iter().collect::<Vec<_>>()).ok()?,
    );
    if d.width != width || s.width != width || c.width != BitWidth::W8 {
        return None;
    }
    let bits = width.bits();
    let count = u32::try_from(count & if bits == 64 { 0x3F } else { 0x1F }).ok()?;
    let (destination, source) = (width.truncate(destination), width.truncate(source));
    if count == 0 {
        return Some(destination);
    }
    if count > bits {
        return None;
    }
    let result = if left {
        (destination << count) | (source >> (bits - count))
    } else {
        (destination >> count) | (source << (bits - count))
    };
    Some(width.truncate(result))
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

/// The binary operation and width a `vm_flags_<op>_wN` intrinsic denotes.
fn vm_flags_operation(name: &str) -> Option<(BinaryOp, BitWidth)> {
    let (operation, suffix) = if let Some(suffix) = name.strip_prefix("vm_flags_add_w") {
        (BinaryOp::Add, suffix)
    } else if let Some(suffix) = name.strip_prefix("vm_flags_nor_w") {
        (BinaryOp::And, suffix)
    } else if let Some(suffix) = name.strip_prefix("vm_flags_nand_w") {
        (BinaryOp::Or, suffix)
    } else {
        return None;
    };
    let width = match suffix {
        "8" => BitWidth::W8,
        "16" => BitWidth::W16,
        "32" => BitWidth::W32,
        "64" => BitWidth::W64,
        _ => return None,
    };
    Some((operation, width))
}

pub(crate) fn evaluate_vm_flags(name: &str, arguments: &[u64]) -> Option<u64> {
    let (semantic, width) = match name {
        "vm_flags_add_w8" => (BinaryOp::Add, BitWidth::W8),
        "vm_flags_add_w16" => (BinaryOp::Add, BitWidth::W16),
        "vm_flags_add_w32" => (BinaryOp::Add, BitWidth::W32),
        "vm_flags_add_w64" => (BinaryOp::Add, BitWidth::W64),
        "vm_flags_sub_w8" => (BinaryOp::Sub, BitWidth::W8),
        "vm_flags_sub_w16" => (BinaryOp::Sub, BitWidth::W16),
        "vm_flags_sub_w32" => (BinaryOp::Sub, BitWidth::W32),
        "vm_flags_sub_w64" => (BinaryOp::Sub, BitWidth::W64),
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
    } else if semantic == BinaryOp::Sub {
        Some(sub_status_flags(width, left, right))
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

fn sub_status_flags(width: BitWidth, left: u64, right: u64) -> u64 {
    const CARRY: u64 = 1;
    const AUXILIARY: u64 = 1 << 4;
    const OVERFLOW: u64 = 1 << 11;

    let result = width.truncate(left.wrapping_sub(right));
    let mut flags = common_status_flags(width, result);
    if left < right {
        flags |= CARRY;
    }
    if (left ^ right ^ result) & 0x10 != 0 {
        flags |= AUXILIARY;
    }
    let sign = 1_u64 << (width.bits() - 1);
    if ((left ^ right) & (left ^ result) & sign) != 0 {
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
            left: ExprRef::new(left),
            right: ExprRef::new(right),
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
                collect((*left).clone(), non_constants, constant);
                collect((*right).clone(), non_constants, constant);
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
                left: ExprRef::new(result),
                right: ExprRef::new(term),
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
                left: ExprRef::new(result),
                right: ExprRef::new(Expr::constant(width, constant)),
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
            value: ExprRef::new(value),
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
            value: ExprRef::new(value),
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
            high: ExprRef::new(high),
            low: ExprRef::new(low),
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
    fn flag_intrinsics_fold_with_symbolic_operands() {
        // A flag intrinsic names a plain arithmetic operation. When one operand is
        // symbolic the constant evaluator cannot fold it, but leaving it opaque
        // would make every pointer derived from it non-affine, which turns a valid
        // stack load into an unresolvable one.
        let base = Expr::register(BitWidth::W64, "vsp_initial");
        let offset = Expr::constant(BitWidth::W64, (-80i64) as u64);
        let sum = Expr::binary(BinaryOp::Add, base, offset).expect("affine sum");
        let intrinsic = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_add_w64",
            vec![Expr::constant(BitWidth::W64, 8), sum],
        )
        .simplify();
        assert_eq!(
            crate::symbolic_pointer_offset(&intrinsic, "vsp_initial"),
            Some(-72),
            "a flag intrinsic over a symbolic pointer stays affine"
        );
    }

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
    fn eval(expression: &Expr, env: &BTreeMap<String, u64>) -> u64 {
        match &expression.kind {
            ExprKind::Const { value } => *value,
            ExprKind::Register { name } => expression.width.truncate(env[name]),
            ExprKind::Unary { op, value } => {
                let v = eval(value, env);
                expression.width.truncate(match op {
                    UnaryOp::Not => !v,
                    UnaryOp::Neg => v.wrapping_neg(),
                })
            }
            ExprKind::Binary { op, left, right } => {
                evaluate_binary(*op, expression.width, eval(left, env), eval(right, env))
            }
            ExprKind::Extract { value, lsb } => expression
                .width
                .truncate(eval(value, env).wrapping_shr(*lsb)),
            ExprKind::Concat { high, low } => expression
                .width
                .truncate(eval(high, env).wrapping_shl(low.width.bits()) | eval(low, env)),
            ExprKind::Resize { op, value } => {
                let v = eval(value, env);
                expression.width.truncate(match op {
                    ResizeOp::Truncate | ResizeOp::ZeroExtend => v,
                    ResizeOp::SignExtend => sign_extend(v, value.width),
                })
            }
            ExprKind::Load { .. } | ExprKind::Intrinsic { .. } => unreachable!("not generated"),
        }
    }

    /// Doubles the reference count of `value` at every step, so the tree form
    /// has 2^steps leaves while the shared form stays linear.
    fn doubling_chain(steps: u32) -> Expr {
        let mut value = Expr::register(BitWidth::W64, "x");
        for step in 0..steps {
            let count = Expr::constant(BitWidth::W64, u64::from(step % 31 + 1));
            let left = Expr::binary(BinaryOp::Rol, value.clone(), count.clone()).unwrap();
            let right = Expr::binary(BinaryOp::Ror, value, count).unwrap();
            value = Expr::binary(BinaryOp::Xor, left, right).unwrap();
        }
        value
    }

    #[test]
    fn shared_subexpressions_are_counted_once() {
        let chain = doubling_chain(48);
        // Naive tree counting would need ~3^48 nodes.
        let nodes = chain.node_count_bounded(1_000_000);
        assert!(nodes < 1_000, "shared chain has {nodes} distinct nodes");
        assert_eq!(chain.dependencies(), BTreeSet::from(["x".to_owned()]));
        assert_eq!(chain.simplify(), chain);
        assert_eq!(chain.clone(), chain);
    }

    #[test]
    fn substitution_over_shared_chain_stays_linear() {
        let rename = BTreeMap::from([("x".to_owned(), Expr::register(BitWidth::W64, "y"))]);
        let replaced = doubling_chain(48).substitute(&rename).unwrap();
        assert!(replaced.node_count_bounded(1_000_000) < 1_000);
        // Evaluation is tree-recursive, so compare on a chain it can afford.
        let small = doubling_chain(12);
        let renamed = small.substitute(&rename).unwrap();
        let value = 0x0123_4567_89AB_CDEF_u64;
        assert_eq!(
            eval(&renamed, &BTreeMap::from([("y".to_owned(), value)])),
            eval(&small, &BTreeMap::from([("x".to_owned(), value)]))
        );
    }

    #[test]
    fn distinct_count_dedups_across_roots() {
        let shared = doubling_chain(8);
        let other = Expr::binary(
            BinaryOp::Add,
            shared.clone(),
            Expr::register(BitWidth::W64, "k"),
        )
        .unwrap();
        let alone = distinct_node_count_bounded([&shared], usize::MAX - 1);
        let together = distinct_node_count_bounded([&shared, &other, &shared], usize::MAX - 1);
        assert!(together <= alone + 3, "{together} vs {alone}");
    }

    #[test]
    fn equal_expressions_share_one_node() {
        let build = || {
            Expr::binary(
                BinaryOp::Xor,
                Expr::register(BitWidth::W32, "a"),
                Expr::register(BitWidth::W32, "b"),
            )
            .unwrap()
        };
        let (first, second) = (build(), build());
        let (ExprKind::Binary { left: l1, .. }, ExprKind::Binary { left: l2, .. }) =
            (&first.kind, &second.kind)
        else {
            panic!("expected binary expressions");
        };
        assert!(l1 == l2 && Arc::ptr_eq(&l1.0, &l2.0));
    }

    #[test]
    fn simplification_preserves_evaluation() {
        let x = || Expr::register(BitWidth::W32, "x");
        let y = || Expr::register(BitWidth::W32, "y");
        let c = |value| Expr::constant(BitWidth::W32, value);
        let b = |op, l, r| Expr::binary(op, l, r).unwrap();
        let samples = [
            b(BinaryOp::Xor, x(), x()),
            b(BinaryOp::And, x(), x()),
            b(BinaryOp::Or, x(), c(0)),
            b(BinaryOp::Add, x(), c(0)),
            b(BinaryOp::Shl, x(), c(0)),
            b(BinaryOp::And, x(), Expr::unary(UnaryOp::Not, x())),
            b(BinaryOp::Or, Expr::unary(UnaryOp::Not, y()), y()),
            b(
                BinaryOp::Sub,
                b(BinaryOp::Add, x(), y()),
                b(BinaryOp::Add, y(), x()),
            ),
            b(
                BinaryOp::Add,
                b(BinaryOp::Add, x(), c(5)),
                b(BinaryOp::Add, y(), c(0xFFFF_FFFD)),
            ),
            b(BinaryOp::Mul, x(), c(1)),
            b(BinaryOp::And, x(), c(0xFFFF_FFFF)),
            Expr::resize(
                ResizeOp::SignExtend,
                BitWidth::W32,
                Expr::resize(ResizeOp::Truncate, BitWidth::W8, x()).unwrap(),
            )
            .unwrap(),
            Expr::concat(
                Expr::extract(BitWidth::W16, 16, x()).unwrap(),
                Expr::extract(BitWidth::W16, 0, x()).unwrap(),
            )
            .unwrap(),
        ];
        for sample in &samples {
            // A raw rebuild of the same shape without any constructor folding.
            let reference = sample.simplify();
            for (a, b) in [
                (0_u64, 0_u64),
                (1, 2),
                (0xFFFF_FFFF, 0x8000_0000),
                (0x1234_5678, 0x9ABC_DEF0),
            ] {
                let env = BTreeMap::from([("x".to_owned(), a), ("y".to_owned(), b)]);
                assert_eq!(eval(sample, &env), eval(&reference, &env), "{sample:?}");
                let folded = sample
                    .substitute(&BTreeMap::from([
                        ("x".to_owned(), c(a)),
                        ("y".to_owned(), c(b)),
                    ]))
                    .unwrap();
                assert_eq!(
                    folded,
                    c(eval(sample, &env)),
                    "full substitution must fold {sample:?}"
                );
            }
        }
    }

    #[test]
    fn double_shifts_fold_exactly() {
        let shift = |direction: &str, width, dst: u64, src: u64, count: u64| {
            Expr::intrinsic(
                width,
                format!("x86_double_shift_{direction}_{width:?}_7").to_lowercase(),
                vec![
                    Expr::constant(width, dst),
                    Expr::constant(width, src),
                    Expr::constant(BitWidth::W8, count),
                ],
            )
        };
        let w32 = BitWidth::W32;
        assert_eq!(
            shift("left", w32, 0x1234_5678, 0x9ABC_DEF0, 8),
            Expr::constant(w32, 0x3456_789A)
        );
        assert_eq!(
            shift("right", w32, 0x1234_5678, 0x9ABC_DEF0, 8),
            Expr::constant(w32, 0xF012_3456)
        );
        assert_eq!(
            shift("left", w32, 0x1234_5678, 0x9ABC_DEF0, 32),
            Expr::constant(w32, 0x1234_5678)
        );
        assert_eq!(
            shift("left", BitWidth::W64, 1, 1 << 63, 1),
            Expr::constant(BitWidth::W64, 3)
        );
        // 16-bit counts above 16 are architecturally undefined and stay opaque.
        assert!(matches!(
            shift("left", BitWidth::W16, 1, 1, 17).kind,
            ExprKind::Intrinsic { .. }
        ));
    }
}
