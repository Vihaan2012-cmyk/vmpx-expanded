//! Sound finite input-domain inference for a symbolic virtual-branch target.
//!
//! A branch target is an expression over the function's live-in registers. A
//! finite set of register values is a *proven exhaustive* domain only if the
//! target's image over that set equals its image over every possible value.
//! That is established structurally, never by sampling: each input register must
//! be observed only through
//!
//! * a low slice of at most [`InputDomainLimits::max_exhaustive_bits`] bits, which
//!   is covered by enumerating all of its values, or
//! * a bit field, or a constant mask, of at most that many observed bits, covered
//!   by enumerating every submask of the observed bits, or
//! * `vm_flags_add_wN(±reg + c1, c2)` status-flag words of one common width,
//!   which depend on the register only through the cell it falls in between a
//!   fixed set of cut points and its residue modulo 256 (see [`add_flags_cover`]).
//! * the CF/ZF/SF/OF bits of `vm_flags_sub_wN(x, y)` of two distinct free
//!   registers, which depend only on the operands' sign bits and the order of
//!   their other bits and are covered by `{0, 1, S, S + 1}` for each (see
//!   [`sign_order_cover`]). PF/AF, or any other use of either register, stay refused.
//! * the same flag cell/residue cover for `vm_flags_sub(c, x)`, `vm_flags_nor/nand(x, x)`
//!   and views `register >> k` of a narrow or truncated register, whose reachable
//!   range is restricted to `[0, 2^span)` (see [`flags_cover_in`]);
//! * a register observed through complements, constant shifts, masks and or-masks
//!   only, which depends on exactly the bits those select ([`demanded_bits`]).
//!
//! A `memory_load_epoch_N(address)` read that no tracked store covers is an
//! unconstrained value: it is replaced by a free register named after the read
//! (see [`abstract_epoch_loads`]) and then covered like any other register. The
//! address is dropped, so the domain over-approximates the reachable values and
//! can never omit a target.
//!
//! Logical `vm_flags_nor/nand_wN` words are pure functions of their operands and
//! add no observation of their own. Any other use (direct arithmetic, loads,
//! opaque intrinsics, flags of two input-dependent operands, mixed use kinds
//! beyond the exhaustive width) is refused: the
//! general case is undecidable here and stays a blocker.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;

use crate::{BinaryOp, BitWidth, Expr, ExprKind, ResizeOp, UnaryOp, expr::evaluate_vm_flags};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InputDomainLimits {
    pub max_values_per_variable: usize,
    pub max_exhaustive_bits: u32,
}

impl Default for InputDomainLimits {
    fn default() -> Self {
        Self {
            max_values_per_variable: 4_096,
            max_exhaustive_bits: 8,
        }
    }
}

/// One observation of a register through the add-status-flags of
/// `(±x + offset, constant)` at a fixed width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddFlagsObservation {
    pub negate: bool,
    pub offset: u64,
    pub constant: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum DomainCertificate {
    /// Every value of the observed low `bits` bits is enumerated.
    Exhaustive { bits: u32 },
    /// Every submask of `mask` is enumerated: the target depends on the register
    /// only through its bits in `mask`.
    MaskedExhaustive { mask: u64 },
    /// Cell/residue cover of add-flag observations of `observations` atoms.
    AddFlagsCover { bits: u32, observations: usize },
    /// `{0, 1, S, S + 1}` per operand of free `vm_flags_sub_wN` compares whose
    /// CF/ZF/SF/OF bits depend only on operand signs and order (see
    /// [`sign_order_cover`]).
    SignOrderCover { bits: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InferredInputDomains {
    pub domains: BTreeMap<String, Vec<Expr>>,
    pub certificates: BTreeMap<String, DomainCertificate>,
}

#[derive(Clone, Debug, Eq, Error, PartialEq, Serialize)]
pub enum InputDomainRefusal {
    #[error("branch target depends on memory")]
    MemoryDependent,
    #[error("{register}: {reason}")]
    Unsupported { register: String, reason: String },
    #[error("{register}: finite cover has more than {limit} values")]
    TooManyValues { register: String, limit: usize },
}

fn unsupported(register: &str, reason: &str) -> InputDomainRefusal {
    let register = match register.split_once('@') {
        Some((load, _)) => format!("{load} (untracked memory read)"),
        None => register.to_owned(),
    };
    InputDomainRefusal::Unsupported {
        register,
        reason: reason.to_owned(),
    }
}

const fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
}

/// Finite set of values of a `bits`-wide register that realizes every
/// combination of add-status-flags the observations can produce, or `None` when
/// `bits` is unsupported or the cover would exceed `max_values`.
///
/// For `bits` above the exhaustive bound the flags of `y + k` (`y = ±x + off`)
/// are determined by (a) which cell of the cut set `x` lies in (zero, sign, and
/// carry/overflow thresholds, each with its neighbour so a reflected `y` is also
/// split) and (b) `x mod 256` (parity and auxiliary carry). The first
/// `min(cell size, 256)` values of every cell therefore realize every attainable
/// flag tuple of every observation jointly; one value per distinct tuple is kept.
#[must_use]
pub fn add_flags_cover(
    bits: u32,
    observations: &[AddFlagsObservation],
    limits: InputDomainLimits,
) -> Option<Vec<u64>> {
    let atoms: Vec<_> = observations
        .iter()
        .copied()
        .map(FlagsAtom::AddConst)
        .collect();
    flags_cover(bits, &atoms, limits)
}

/// `+x + offset` or `-x + offset` for the register `x`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct View {
    negate: bool,
    offset: u64,
}

impl View {
    fn at(self, x: u64, bits: u32) -> u64 {
        let offset = self.offset & mask(bits);
        let value = if self.negate {
            offset.wrapping_sub(x)
        } else {
            x.wrapping_add(offset)
        };
        value & mask(bits)
    }

    /// The register value at which this view equals `threshold`.
    fn preimage(self, threshold: u128, bits: u32) -> u128 {
        let space = 1_u128 << bits;
        let offset = u128::from(self.offset & mask(bits));
        if self.negate {
            (offset + space - threshold) % space
        } else {
            (threshold + space - offset) % space
        }
    }
}

/// One status-flag word taken of the register at a fixed width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlagsAtom {
    /// `vm_flags_add(view, constant)`.
    AddConst(AddFlagsObservation),
    /// `vm_flags_sub(left, right)` of two same-direction views of the register:
    /// the difference is constant, so only the wrap and sign boundaries of the
    /// two operands and the low nibbles can change a flag.
    SubPair { left: View, right: View },
    /// `vm_flags_sub(constant, view)`: ZF/CF split at `view = constant`, the
    /// result sign at `view = constant - S`, the operand sign at `0` and `S`.
    SubConstLeft { view: View, constant: u64 },
    /// `vm_flags_nor/nand(view, view)`: the logical result is `!view`, so ZF
    /// splits at `view = all ones`, SF at `0` and `S`; PF reads the low byte.
    LogicalSelf { view: View, nand: bool },
}

impl FlagsAtom {
    fn tuple_word(self, bits: u32, x: u64) -> Option<u64> {
        match self {
            Self::AddConst(observation) => {
                let operand = View {
                    negate: observation.negate,
                    offset: observation.offset,
                }
                .at(x, bits);
                evaluate_vm_flags(
                    &format!("vm_flags_add_w{bits}"),
                    &[operand, observation.constant & mask(bits)],
                )
            }
            Self::SubPair { left, right } => evaluate_vm_flags(
                &format!("vm_flags_sub_w{bits}"),
                &[left.at(x, bits), right.at(x, bits)],
            ),
            Self::SubConstLeft { view, constant } => evaluate_vm_flags(
                &format!("vm_flags_sub_w{bits}"),
                &[constant & mask(bits), view.at(x, bits)],
            ),
            Self::LogicalSelf { view, nand } => {
                let operand = view.at(x, bits);
                evaluate_vm_flags(
                    &format!("vm_flags_{}_w{bits}", if nand { "nand" } else { "nor" }),
                    &[operand, operand],
                )
            }
        }
    }

    fn cut_points(self, bits: u32, cuts: &mut BTreeSet<u128>) {
        let space = 1_u128 << bits;
        let half = space >> 1;
        let mut insert = |x: u128| {
            cuts.insert(x);
            cuts.insert((x + 1) % space);
        };
        match self {
            Self::AddConst(observation) => {
                let constant = u128::from(observation.constant & mask(bits));
                let view = View {
                    negate: observation.negate,
                    offset: observation.offset,
                };
                // Thresholds of `y`: its own sign/zero boundary and where `y + k`
                // reaches zero, one (ZF cell), and the sign boundary.
                for threshold in [
                    0,
                    half,
                    (space - constant) % space,
                    (space + 1 - constant) % space,
                    (half + space - constant) % space,
                ] {
                    insert(view.preimage(threshold, bits));
                }
            }
            Self::SubPair { left, right } => {
                for view in [left, right] {
                    for threshold in [0, half] {
                        insert(view.preimage(threshold, bits));
                    }
                }
            }
            Self::LogicalSelf { view, .. } => {
                for threshold in [0, half, space - 1] {
                    insert(view.preimage(threshold, bits));
                }
            }
            Self::SubConstLeft { view, constant } => {
                let constant = u128::from(constant & mask(bits));
                for threshold in [0, half, constant, (constant + space - half) % space] {
                    insert(view.preimage(threshold, bits));
                }
            }
        }
    }
}

fn flags_cover(bits: u32, atoms: &[FlagsAtom], limits: InputDomainLimits) -> Option<Vec<u64>> {
    flags_cover_in(bits, atoms, bits, limits)
}

/// [`flags_cover`] for `x` restricted to `[0, 2^range_bits)`: each cell is cut
/// to that range, and its first `min(size, 256)` values still hit every residue
/// modulo 256 the cell holds.
fn flags_cover_in(
    bits: u32,
    atoms: &[FlagsAtom],
    range_bits: u32,
    limits: InputDomainLimits,
) -> Option<Vec<u64>> {
    if !matches!(bits, 8 | 16 | 32 | 64) || range_bits == 0 || range_bits > bits {
        return None;
    }
    let space = 1_u128 << range_bits;
    let max_values = limits.max_values_per_variable;
    if bits <= limits.max_exhaustive_bits {
        return exhaustive(range_bits, limits);
    }
    let mut cuts: BTreeSet<u128> = BTreeSet::from([0]);
    for atom in atoms {
        atom.cut_points(bits, &mut cuts);
    }
    let starts: Vec<u128> = cuts.into_iter().collect();
    // The target depends on `x` only through these flag words, so one value per
    // distinct tuple realizes the same image as the sampled cells.
    let mut seen: BTreeSet<Vec<u64>> = BTreeSet::new();
    let mut values = Vec::new();
    for (index, start) in starts.iter().enumerate() {
        if *start >= space {
            break;
        }
        let end = starts.get(index + 1).copied().unwrap_or(space).min(space);
        let take = (end - start).min(256);
        for value in *start..start + take {
            let value = u64::try_from(value).ok()?;
            let tuple = atoms
                .iter()
                .map(|atom| atom.tuple_word(bits, value))
                .collect::<Option<Vec<u64>>>()?;
            if seen.insert(tuple) {
                values.push(value);
            }
        }
        if values.len() > max_values {
            return None;
        }
    }
    Some(values)
}

fn exhaustive(bits: u32, limits: InputDomainLimits) -> Option<Vec<u64>> {
    let count = 1_u64.checked_shl(bits)?;
    (usize::try_from(count).ok()? <= limits.max_values_per_variable).then(|| (0..count).collect())
}

/// Every submask of `observed`, ascending; `None` above the value limit.
fn submasks(observed: u64, limits: InputDomainLimits) -> Option<Vec<u64>> {
    let count = 1_usize.checked_shl(observed.count_ones())?;
    if count > limits.max_values_per_variable {
        return None;
    }
    let mut values = Vec::with_capacity(count);
    let mut current = 0_u64;
    loop {
        values.push(current);
        if current == observed {
            return Some(values);
        }
        current = current.wrapping_sub(observed) & observed;
    }
}

/// Replaces every `memory_load_epoch_N(address)` with a free register named by
/// the read, so equal reads share one variable and the address no longer
/// constrains the expression. Reads of untracked memory can hold any value, so
/// the free variable's domain is a superset of the reachable ones.
///
/// # Errors
///
/// Propagates expression rebuild failures.
pub fn abstract_epoch_loads(expr: &Expr) -> Result<Expr, crate::ExprError> {
    expr.rewrite_intrinsics(&mut |width, name, arguments| {
        (name.starts_with("memory_load_epoch_") && arguments.len() == 1)
            .then(|| Expr::register(width, format!("{name}@{arguments:?}")))
    })
}

/// `expr == (±x + offset) mod 2^bits` for `x = register >> shift`, which ranges
/// over `[0, 2^span)`.
struct Affine {
    register: String,
    register_width: BitWidth,
    bits: u32,
    negate: bool,
    offset: u64,
    shift: u32,
    span: u32,
}

impl Affine {
    fn is_pure(&self) -> bool {
        !self.negate && self.offset == 0
    }
}

/// A zero extension: the bits above the wrapped value are constant zero.
fn zero_high(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Resize {
            op: ResizeOp::ZeroExtend,
            ..
        } => true,
        ExprKind::Concat { high, .. } => constant_of(high) == Some(0),
        _ => false,
    }
}

fn constant_of(expr: &Expr) -> Option<u64> {
    match expr.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

fn affine(expr: &Expr) -> Option<Affine> {
    let bits = expr.width.bits();
    let m = mask(bits);
    match &expr.kind {
        ExprKind::Register { name } => Some(Affine {
            register: name.clone(),
            register_width: expr.width,
            bits,
            negate: false,
            offset: 0,
            shift: 0,
            span: bits,
        }),
        ExprKind::Resize {
            op: ResizeOp::Truncate,
            value,
        }
        | ExprKind::Extract { value, lsb: 0 } => {
            let inner = affine(value)?;
            Some(Affine {
                bits,
                offset: inner.offset & m,
                span: inner.span.min(bits),
                ..inner
            })
        }
        ExprKind::Resize { value, .. }
        | ExprKind::Concat {
            high: _,
            low: value,
        } if zero_high(expr) => {
            let inner = affine(value)?;
            inner.is_pure().then_some(Affine { bits, ..inner })
        }
        ExprKind::Binary {
            op: BinaryOp::Shr,
            left,
            right,
        } => {
            let amount = u32::try_from(constant_of(right)?).ok()?;
            let inner = affine(left)?;
            (inner.is_pure() && inner.shift == 0 && amount < inner.span).then_some(Affine {
                shift: amount,
                span: inner.span - amount,
                ..inner
            })
        }
        ExprKind::Unary {
            op: UnaryOp::Not,
            value,
        } => {
            let inner = affine(value)?;
            Some(Affine {
                negate: !inner.negate,
                offset: inner.offset.wrapping_neg().wrapping_sub(1) & m,
                ..inner
            })
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } => {
            let inner = affine(value)?;
            Some(Affine {
                negate: !inner.negate,
                offset: inner.offset.wrapping_neg() & m,
                ..inner
            })
        }
        ExprKind::Binary { op, left, right } => match (op, constant_of(left), constant_of(right)) {
            (BinaryOp::Add, None, Some(c)) => {
                let inner = affine(left)?;
                Some(Affine {
                    offset: inner.offset.wrapping_add(c) & m,
                    ..inner
                })
            }
            (BinaryOp::Add, Some(c), None) => {
                let inner = affine(right)?;
                Some(Affine {
                    offset: inner.offset.wrapping_add(c) & m,
                    ..inner
                })
            }
            (BinaryOp::Sub, None, Some(c)) => {
                let inner = affine(left)?;
                Some(Affine {
                    offset: inner.offset.wrapping_sub(c) & m,
                    ..inner
                })
            }
            (BinaryOp::Sub, Some(c), None) => {
                let inner = affine(right)?;
                Some(Affine {
                    negate: !inner.negate,
                    offset: c.wrapping_sub(inner.offset) & m,
                    ..inner
                })
            }
            (BinaryOp::Xor, None, Some(c)) | (BinaryOp::Xor, Some(c), None) if c == m => {
                let inner = affine(if constant_of(left).is_some() {
                    right
                } else {
                    left
                })?;
                Some(Affine {
                    negate: !inner.negate,
                    offset: inner.offset.wrapping_neg().wrapping_sub(1) & m,
                    ..inner
                })
            }
            _ => None,
        },
        _ => None,
    }
}

/// Width of a `vm_flags_add_wN` or `vm_flags_sub_wN` intrinsic, and whether it
/// is the subtracting form.
fn flags_width(name: &str) -> Option<(u32, bool)> {
    if let Some(bits) = add_flags_width(name) {
        return Some((bits, false));
    }
    match name.strip_prefix("vm_flags_sub_w")? {
        "8" => Some((8, true)),
        "16" => Some((16, true)),
        "32" => Some((32, true)),
        "64" => Some((64, true)),
        _ => None,
    }
}

fn is_pure_logical_flags(name: &str) -> bool {
    ["vm_flags_nor_w", "vm_flags_nand_w"].iter().any(|prefix| {
        name.strip_prefix(prefix)
            .is_some_and(|width| matches!(width, "8" | "16" | "32" | "64"))
    })
}

fn add_flags_width(name: &str) -> Option<u32> {
    match name.strip_prefix("vm_flags_add_w")? {
        "8" => Some(8),
        "16" => Some(16),
        "32" => Some(32),
        "64" => Some(64),
        _ => None,
    }
}

/// Status-flag bits of a `vm_flags_sub` word that depend only on the operands'
/// sign bits and the order of their remaining bits: CF, ZF, SF, OF.
const ORDER_FLAG_BITS: u64 = 1 | (1 << 6) | (1 << 7) | (1 << 11);

/// A register, or a zero extension of one: `(name, register width)`.
fn zero_extended_register(expr: &Expr) -> Option<(&str, BitWidth)> {
    match &expr.kind {
        ExprKind::Register { name } => Some((name, expr.width)),
        ExprKind::Resize { value, .. }
        | ExprKind::Concat {
            high: _,
            low: value,
        } if zero_high(expr) => zero_extended_register(value),
        _ => None,
    }
}

/// Register values that realize every (sign bit, order of the other bits) class
/// of an operand pair at `bits`: `{0, 1, S, S + 1}` with `S` the sign bit, or
/// `{0, 1}` for an operand zero-extended from a narrower register (sign bit 0).
///
/// For `A = sa·S + a'` and `B = sb·S + b'`, `A - B = (sa - sb)·S + (a' - b')`,
/// so ZF is `sa = sb and a' = b'`, CF is `sa < sb or (sa = sb and a' < b')`, SF is
/// `sa ^ sb ^ (a' < b')` and OF is `sa != sb and SF != sa`. The product of these
/// sets over the two operands attains every class, hence every CF/ZF/SF/OF tuple
/// a free pair can produce.
fn sign_order_cover(register_bits: u32, bits: u32) -> Vec<u64> {
    if register_bits < bits {
        return vec![0, 1];
    }
    let sign = 1_u64 << (bits - 1);
    vec![0, 1, sign, sign | 1]
}

/// The register bits that the `demand`ed bits of `expr` are a function of, when
/// `expr` is built from one register by bitwise complement, constant shifts,
/// masks, or-masks, xor, extracts and zero extension only. Every other bit of
/// the register is irrelevant to those result bits.
fn demanded_bits(expr: &Expr, demand: u64) -> Option<(&str, BitWidth, u64)> {
    let demand = demand & mask(expr.width.bits());
    match &expr.kind {
        ExprKind::Register { name } => Some((name, expr.width, demand)),
        ExprKind::Unary {
            op: UnaryOp::Not,
            value,
        }
        | ExprKind::Resize {
            op: ResizeOp::Truncate | ResizeOp::ZeroExtend,
            value,
        } => demanded_bits(value, demand),
        ExprKind::Extract { value, lsb } if *lsb < 64 => demanded_bits(value, demand << lsb),
        ExprKind::Binary { op, left, right } => {
            let (inner, c) = match (constant_of(left), constant_of(right)) {
                (None, Some(c)) => (left, c),
                (Some(c), None) if matches!(op, BinaryOp::And | BinaryOp::Or | BinaryOp::Xor) => {
                    (right, c)
                }
                _ => return None,
            };
            match op {
                BinaryOp::Shr if c < 64 => demanded_bits(inner, demand << c),
                BinaryOp::Shl if c < 64 => demanded_bits(inner, demand >> c),
                BinaryOp::And => demanded_bits(inner, demand & c),
                BinaryOp::Or => demanded_bits(inner, demand & !c),
                BinaryOp::Xor => demanded_bits(inner, demand),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Two distinct full-width registers whose `vm_flags_subN` word is observed only
/// through [`ORDER_FLAG_BITS`]; `partner` is the other register.
#[derive(Clone, Debug, Eq, PartialEq)]
struct PairUse {
    partner: String,
    bits: u32,
}

#[derive(Default)]
struct Uses {
    pair: Option<PairUse>,
    /// `(shift, bits)` of the observed value `register >> shift` in `[0, 2^bits)`.
    view: Option<(u32, u32)>,
    /// Occurs only inside a term the target is independent of.
    ignored: bool,
    width: Option<BitWidth>,
    /// Union of the register bits the target observes directly.
    direct_mask: u64,
    /// Add-flags observations with the flags width they were taken at.
    atoms: Vec<(u32, FlagsAtom)>,
}

struct Scan {
    uses: BTreeMap<String, Uses>,
    refusal: Option<InputDomainRefusal>,
}

impl Scan {
    fn refuse(&mut self, refusal: InputDomainRefusal) {
        self.refusal.get_or_insert(refusal);
    }

    fn note_width(&mut self, register: &str, width: BitWidth) {
        let uses = self.uses.entry(register.to_owned()).or_default();
        if uses.width.is_some_and(|known| known != width) {
            self.refuse(unsupported(register, "register is used at two widths"));
        } else {
            uses.width = Some(width);
        }
    }

    fn visit(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Const { .. } => {}
            ExprKind::Load { .. } => self.refuse(InputDomainRefusal::MemoryDependent),
            ExprKind::Register { name } => {
                self.note_width(name, expr.width);
                let uses = self.uses.entry(name.clone()).or_default();
                uses.direct_mask |= mask(expr.width.bits());
            }
            ExprKind::Binary {
                op: BinaryOp::And,
                left,
                right,
            } if self.visit_flag_bits(left, right)
                || self.visit_masked(expr, left, right)
                || self.visit_demand(expr) => {}
            ExprKind::Unary {
                op: UnaryOp::Not, ..
            } if self.visit_not_or_flags(expr) || self.visit_demand(expr) => {}
            ExprKind::Extract { value, lsb }
                if self.visit_flag_field(expr, value, *lsb)
                    || self.visit_bit_field(expr, value, *lsb)
                    || self.visit_demand(expr) => {}
            ExprKind::Resize {
                op: ResizeOp::Truncate,
                value,
            }
            | ExprKind::Extract { value, lsb: 0 } => {
                if let ExprKind::Register { name } = &value.kind {
                    self.note_width(name, value.width);
                    let uses = self.uses.entry(name.clone()).or_default();
                    uses.direct_mask |= mask(expr.width.bits());
                } else {
                    self.visit(value);
                }
            }
            ExprKind::Intrinsic { name, arguments } => {
                if arguments
                    .iter()
                    .all(|argument| argument.dependencies().is_empty())
                {
                    // Input-independent term: evaluation decides whether it folds.
                    return;
                }
                if is_pure_logical_flags(name) {
                    if self.visit_logical_self(name, arguments) {
                        return;
                    }
                    // A pure function of its operands: the target depends on a
                    // register only through the bits the operands observe.
                    for argument in arguments {
                        self.visit(argument);
                    }
                    return;
                }
                let Some((bits, subtract)) = flags_width(name) else {
                    let register = arguments
                        .iter()
                        .flat_map(Expr::dependencies)
                        .next()
                        .unwrap_or_default();
                    self.refuse(unsupported(
                        &register,
                        "observed through an intrinsic other than add flags",
                    ));
                    return;
                };
                self.visit_add_flags(name, bits, subtract, arguments);
            }
            ExprKind::Unary { value, .. }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => self.visit(value),
            ExprKind::Binary { left, right, .. }
            | ExprKind::Concat {
                high: left,
                low: right,
            } => {
                self.visit(left);
                self.visit(right);
            }
        }
    }

    /// `flags & c` or `(flags >> k) & c` of a free-pair compare.
    fn visit_flag_bits(&mut self, left: &Expr, right: &Expr) -> bool {
        let (c, observed) = match (constant_of(left), constant_of(right)) {
            (Some(c), None) => (c, right),
            (None, Some(c)) => (c, left),
            _ => return false,
        };
        self.flag_bits_of(c, observed)
    }

    /// `!(flags | c)`, which observes the bits of `flags` that `c` leaves clear.
    fn visit_not_or_flags(&mut self, expr: &Expr) -> bool {
        let ExprKind::Unary {
            op: UnaryOp::Not,
            value,
        } = &expr.kind
        else {
            return false;
        };
        let ExprKind::Binary {
            op: BinaryOp::Or,
            left,
            right,
        } = &value.kind
        else {
            return false;
        };
        match (constant_of(left), constant_of(right)) {
            (Some(c), None) => self.flag_bits_of(!c, right),
            (None, Some(c)) => self.flag_bits_of(!c, left),
            _ => false,
        }
    }

    fn flag_bits_of(&mut self, c: u64, observed: &Expr) -> bool {
        let c = c & mask(observed.width.bits());
        // Complementing a word does not change which of its bits are observed.
        let mut observed = observed;
        while let ExprKind::Unary {
            op: UnaryOp::Not,
            value,
        } = &observed.kind
        {
            observed = value;
        }
        if let ExprKind::Binary {
            op: BinaryOp::Shr,
            left: flags,
            right: amount,
        } = &observed.kind
        {
            let Some(k) = constant_of(amount).filter(|k| *k < 64) else {
                return false;
            };
            return self.record_flags(flags, c << k);
        }
        self.record_flags(observed, c)
    }

    /// `flags[lsb .. lsb + width]` of a free-pair compare.
    fn visit_flag_field(&mut self, expr: &Expr, flags: &Expr, lsb: u32) -> bool {
        let bits = expr.width.bits();
        lsb + bits <= flags.width.bits() && self.record_flags(flags, mask(bits) << lsb)
    }

    fn record_flags(&mut self, flags: &Expr, observed: u64) -> bool {
        self.record_pair(flags, observed)
    }

    /// `vm_flags_nor/nandN(a, a)` of an affine view `a`: a function of the cell
    /// of `a` and its residue modulo 256, like the add flags.
    fn visit_logical_self(&mut self, name: &str, arguments: &[Expr]) -> bool {
        let [a, b] = arguments else {
            return false;
        };
        let Some(view) = affine(a).filter(|view| a == b && view.bits == a.width.bits()) else {
            return false;
        };
        let bits = view.bits;
        let atom = FlagsAtom::LogicalSelf {
            view: View {
                negate: view.negate,
                offset: view.offset & mask(bits),
            },
            nand: name.starts_with("vm_flags_nand"),
        };
        self.push_atom(&view, bits, atom);
        true
    }

    /// `expr` observes its register through [`demanded_bits`] only.
    fn visit_demand(&mut self, expr: &Expr) -> bool {
        let Some((register, width, bits)) = demanded_bits(expr, mask(expr.width.bits())) else {
            return false;
        };
        self.note_width(register, width);
        self.uses
            .entry(register.to_owned())
            .or_default()
            .direct_mask |= bits;
        true
    }

    /// Records `vm_flags_subN(a, b)` of two distinct `N`-bit registers observed
    /// through `observed` bits; `false` leaves the expression to the generic scan.
    fn record_pair(&mut self, flags: &Expr, observed: u64) -> bool {
        let ExprKind::Intrinsic { name, arguments } = &flags.kind else {
            return false;
        };
        let (Some((bits, true)), [a, b]) = (flags_width(name), arguments.as_slice()) else {
            return false;
        };
        let (Some((a_name, a_width)), Some((b_name, b_width))) =
            (zero_extended_register(a), zero_extended_register(b))
        else {
            return false;
        };
        if a_name == b_name
            || a.width.bits() != bits
            || b.width.bits() != bits
            || observed & !ORDER_FLAG_BITS != 0
        {
            return false;
        }
        for (register, width, partner) in [(a_name, a_width, b_name), (b_name, b_width, a_name)] {
            self.note_width(register, width);
            let uses = self.uses.entry(register.to_owned()).or_default();
            let pair = PairUse {
                partner: partner.to_owned(),
                bits,
            };
            if uses.pair.get_or_insert_with(|| pair.clone()) != &pair {
                self.refuse(unsupported(
                    register,
                    "register is compared against several partners",
                ));
            }
        }
        true
    }

    /// `register & constant` observes only the constant's bits.
    fn visit_masked(&mut self, expr: &Expr, left: &Expr, right: &Expr) -> bool {
        let (register, constant) = match (&left.kind, &right.kind) {
            (ExprKind::Register { name }, ExprKind::Const { value })
            | (ExprKind::Const { value }, ExprKind::Register { name }) => (name, *value),
            _ => return false,
        };
        self.note_width(register, expr.width);
        let uses = self.uses.entry(register.clone()).or_default();
        uses.direct_mask |= constant & mask(expr.width.bits());
        true
    }

    /// `register[lsb .. lsb + width]` observes just those bits.
    fn visit_bit_field(&mut self, expr: &Expr, value: &Expr, lsb: u32) -> bool {
        let ExprKind::Register { name } = &value.kind else {
            return false;
        };
        let bits = expr.width.bits();
        if lsb + bits > value.width.bits() {
            return false;
        }
        self.note_width(name, value.width);
        let uses = self.uses.entry(name.clone()).or_default();
        uses.direct_mask |= mask(bits) << lsb;
        true
    }

    /// The constant a flags operand is: a literal, or `x | all-ones` / `x & 0`,
    /// whose registers are then irrelevant to the target.
    fn constant_operand(&mut self, operand: &Expr) -> Option<u64> {
        if let Some(value) = constant_of(operand) {
            return Some(value);
        }
        let ExprKind::Binary { op, left, right } = &operand.kind else {
            return None;
        };
        let absorbing = match op {
            BinaryOp::Or => mask(operand.width.bits()),
            BinaryOp::And => 0,
            _ => return None,
        };
        if constant_of(left) != Some(absorbing) && constant_of(right) != Some(absorbing) {
            return None;
        }
        let mut registers = Vec::new();
        operand.walk_distinct(&mut |node| {
            if let ExprKind::Register { name } = &node.kind {
                registers.push((name.clone(), node.width));
            }
        });
        for (register, width) in registers {
            self.note_width(&register, width);
            self.uses.entry(register).or_default().ignored = true;
        }
        Some(absorbing)
    }

    fn visit_add_flags(&mut self, name: &str, bits: u32, subtract: bool, arguments: &[Expr]) {
        let register = arguments
            .iter()
            .flat_map(Expr::dependencies)
            .next()
            .unwrap_or_default();
        let [left, right] = arguments else {
            self.refuse(unsupported(
                &register,
                "add flags need exactly two operands",
            ));
            return;
        };
        let (left_constant, right_constant) =
            (self.constant_operand(left), self.constant_operand(right));
        if subtract && left_constant.is_none() && right_constant.is_none() {
            self.visit_sub_pair(name, bits, &register, left, right);
            return;
        }
        if let (true, Some(constant), None) = (subtract, left_constant, right_constant) {
            let Some(view) = affine(right).filter(|view| view.bits >= bits) else {
                self.refuse(unsupported(
                    &register,
                    &format!("{name} operand is not an affine view of one register"),
                ));
                return;
            };
            let atom = FlagsAtom::SubConstLeft {
                view: View {
                    negate: view.negate,
                    offset: view.offset & mask(bits),
                },
                constant: constant & mask(bits),
            };
            self.push_atom(&view, bits, atom);
            return;
        }
        let (symbolic, constant) = match (left_constant, right_constant) {
            (None, Some(c)) => (left, c),
            (Some(c), None) if !subtract => (right, c),
            _ => {
                self.refuse(unsupported(
                    &register,
                    "add flags of two input-dependent or non-constant operands",
                ));
                return;
            }
        };
        // `sub(a, c)` flags equal `add(a, -c)` flags up to a fixed complement of
        // some of CF, AF and OF for each constant `c`, so the partition of inputs
        // by flag tuple is the add partition at the constant `-c`.
        let constant = if subtract {
            (constant & mask(bits)).wrapping_neg() & mask(bits)
        } else {
            constant
        };
        let Some(view) = affine(symbolic).filter(|view| view.bits >= bits) else {
            self.refuse(unsupported(
                &register,
                &format!("{name} operand is not an affine view of one register"),
            ));
            return;
        };
        let atom = FlagsAtom::AddConst(AddFlagsObservation {
            negate: view.negate,
            offset: view.offset & mask(bits),
            constant: constant & mask(bits),
        });
        self.push_atom(&view, bits, atom);
    }

    /// Files a flags observation of `view`; every observation of a register
    /// must see the same shifted, bounded value `register >> shift`.
    fn push_atom(&mut self, view: &Affine, bits: u32, atom: FlagsAtom) {
        self.note_width(&view.register, view.register_width);
        let range = (view.shift, view.span.min(bits));
        let uses = self.uses.entry(view.register.clone()).or_default();
        // Atoms of different widths are refused later with their own reason.
        let same_width = uses.atoms.iter().all(|(known, _)| *known == bits);
        uses.atoms.push((bits, atom));
        if *uses.view.get_or_insert(range) != range && same_width {
            self.refuse(unsupported(
                &view.register,
                "register is observed through differently shifted or bounded views",
            ));
        }
    }

    /// `sub(s*x + a, s*x + b)` with the same sign on `x`: a pointer-difference
    /// compare whose result is the constant `a - b`.
    fn visit_sub_pair(&mut self, name: &str, bits: u32, register: &str, left: &Expr, right: &Expr) {
        let views = affine(left).zip(affine(right)).filter(|(l, r)| {
            l.register == r.register
                && l.negate == r.negate
                && l.bits >= bits
                && r.bits >= bits
                && (l.shift, l.span) == (r.shift, r.span)
        });
        let Some((l, r)) = views else {
            self.refuse(unsupported(
                register,
                &format!("{name} of two input-dependent operands that are not parallel views of one register"),
            ));
            return;
        };
        let view = |a: &Affine| View {
            negate: a.negate,
            offset: a.offset & mask(bits),
        };
        let atom = FlagsAtom::SubPair {
            left: view(&l),
            right: view(&r),
        };
        self.push_atom(&l, bits, atom);
    }
}

/// A paired register must be used by nothing but its own pair: any other
/// observation would need a joint cover with the partner's values.
fn check_pairs(uses: &BTreeMap<String, Uses>) -> Result<(), InputDomainRefusal> {
    for (register, entry) in uses {
        let Some(pair) = &entry.pair else { continue };
        let mutual = uses
            .get(&pair.partner)
            .and_then(|partner| partner.pair.as_ref())
            .is_some_and(|back| back.partner == *register && back.bits == pair.bits);
        if !mutual {
            return Err(unsupported(
                register,
                "compare partner is observed differently",
            ));
        }
        if entry.direct_mask != 0 || !entry.atoms.is_empty() {
            return Err(unsupported(
                register,
                "free compare operand is also observed beyond its compare flags",
            ));
        }
    }
    Ok(())
}

/// Infers a finite, proven-exhaustive domain for every register the
/// `target` depends on, or refuses with the reason it cannot.
///
/// # Errors
///
/// Refuses memory-dependent targets and any register observation outside the
/// decidable forms documented on the module.
pub fn infer_branch_input_domains(
    target: &Expr,
    limits: InputDomainLimits,
) -> Result<InferredInputDomains, InputDomainRefusal> {
    let target = &abstract_epoch_loads(target)
        .map_err(|error| unsupported("<expression>", &error.to_string()))?;
    let mut scan = Scan {
        uses: BTreeMap::new(),
        refusal: None,
    };
    scan.visit(target);
    if let Some(refusal) = scan.refusal {
        return Err(refusal);
    }
    check_pairs(&scan.uses)?;
    let mut domains = BTreeMap::new();
    let mut certificates = BTreeMap::new();
    for (register, uses) in scan.uses {
        let width = uses
            .width
            .ok_or_else(|| unsupported(&register, "register width unknown"))?;
        if let Some(pair) = &uses.pair {
            domains.insert(
                register.clone(),
                sign_order_cover(width.bits(), pair.bits)
                    .into_iter()
                    .map(|value| Expr::constant(width, value))
                    .collect(),
            );
            certificates.insert(
                register,
                DomainCertificate::SignOrderCover { bits: pair.bits },
            );
            continue;
        }
        let atom_bits = uses
            .atoms
            .iter()
            .map(|(bits, _)| *bits)
            .collect::<BTreeSet<_>>();
        if atom_bits.len() > 1 {
            return Err(unsupported(
                &register,
                "add flags observed at several widths",
            ));
        }
        let atom_width = atom_bits.first().copied();
        let shift = uses.view.map_or(0, |(shift, _)| shift);
        let (values, certificate) = match (uses.direct_mask, atom_width) {
            (direct, atom) if direct != 0 => {
                if shift != 0 {
                    return Err(unsupported(
                        &register,
                        "register is used directly beside a shifted view",
                    ));
                }
                let observed = direct | mask(atom.unwrap_or(0));
                if observed.count_ones() > limits.max_exhaustive_bits {
                    let reason = if register.contains('@') {
                        "read is used directly beyond the exhaustive width and memory bounds none of its bits"
                    } else {
                        "register is used directly beyond the exhaustive width"
                    };
                    return Err(unsupported(&register, reason));
                }
                let certificate = if observed & (observed + 1) == 0 {
                    DomainCertificate::Exhaustive {
                        bits: observed.count_ones(),
                    }
                } else {
                    DomainCertificate::MaskedExhaustive { mask: observed }
                };
                (submasks(observed, limits), certificate)
            }
            (_, Some(bits)) => {
                let observations: Vec<_> = uses.atoms.iter().map(|(_, atom)| *atom).collect();
                let certificate = if bits <= limits.max_exhaustive_bits {
                    DomainCertificate::Exhaustive { bits }
                } else {
                    DomainCertificate::AddFlagsCover {
                        bits,
                        observations: observations.len(),
                    }
                };
                let range_bits = uses.view.map_or(bits, |(_, range)| range);
                (
                    flags_cover_in(bits, &observations, range_bits, limits),
                    certificate,
                )
            }
            (_, None) if uses.ignored => (Some(vec![0]), DomainCertificate::Exhaustive { bits: 0 }),
            (_, None) => {
                return Err(unsupported(&register, "register has no recognized use"));
            }
        };
        let values = values.ok_or_else(|| InputDomainRefusal::TooManyValues {
            register: register.clone(),
            limit: limits.max_values_per_variable,
        })?;
        domains.insert(
            register.clone(),
            values
                .into_iter()
                .map(|value| Expr::constant(width, value << shift))
                .collect(),
        );
        certificates.insert(register, certificate);
    }
    Ok(InferredInputDomains {
        domains,
        certificates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags(bits: u32, left: u64, right: u64) -> u64 {
        let value = Expr::intrinsic(
            BitWidth::W64,
            format!("vm_flags_add_w{bits}"),
            vec![
                Expr::constant(BitWidth::from_bits(bits).expect("width"), left),
                Expr::constant(BitWidth::from_bits(bits).expect("width"), right),
            ],
        );
        let ExprKind::Const { value } = value.kind else {
            panic!("constant flags must fold")
        };
        value
    }

    fn view(observation: AddFlagsObservation, bits: u32, x: u64) -> u64 {
        let m = mask(bits);
        let y = if observation.negate {
            observation.offset.wrapping_sub(x)
        } else {
            x.wrapping_add(observation.offset)
        } & m;
        flags(bits, y, observation.constant)
    }

    fn image(bits: u32, observations: &[AddFlagsObservation], xs: &[u64]) -> BTreeSet<Vec<u64>> {
        xs.iter()
            .map(|x| observations.iter().map(|o| view(*o, bits, *x)).collect())
            .collect()
    }

    fn obs(negate: bool, offset: u64, constant: u64) -> AddFlagsObservation {
        AddFlagsObservation {
            negate,
            offset,
            constant,
        }
    }

    #[test]
    fn subtract_flags_partition_inputs_like_add_flags_of_the_negated_constant() {
        let eval = |name: &str, bits: u32, x: u64, c: u64| {
            evaluate_vm_flags(&format!("{name}_w{bits}"), &[x, c]).expect("flags")
        };
        for (bits, constants) in [
            (8, (0..=255).collect::<Vec<u64>>()),
            (
                16,
                vec![0, 1, 2, 0x10, 0xFF, 0x100, 0x7FFF, 0x8000, 0x8001, 0xFFFF],
            ),
        ] {
            for c in constants {
                let negated = c.wrapping_neg() & mask(bits);
                let mut pairs = BTreeSet::new();
                let mut subs = BTreeSet::new();
                let mut adds = BTreeSet::new();
                for x in 0..=mask(bits) {
                    let sub = eval("vm_flags_sub", bits, x, c);
                    let add = eval("vm_flags_add", bits, x, negated);
                    pairs.insert((sub, add));
                    subs.insert(sub);
                    adds.insert(add);
                }
                assert_eq!(subs.len(), adds.len(), "w{bits} c={c:#x}");
                assert_eq!(pairs.len(), subs.len(), "w{bits} c={c:#x}");
            }
        }
    }

    #[test]
    fn narrow_registers_are_covered_exhaustively() {
        let cover = add_flags_cover(8, &[obs(false, 0, 5)], InputDomainLimits::default())
            .expect("8-bit space fits");
        assert_eq!(cover, (0..=255).collect::<Vec<_>>());
    }

    #[test]
    fn unsupported_width_or_oversized_cover_is_none() {
        assert!(add_flags_cover(12, &[], InputDomainLimits::default()).is_none());
        let tiny = InputDomainLimits {
            max_values_per_variable: 2,
            ..InputDomainLimits::default()
        };
        assert!(add_flags_cover(32, &[obs(false, 0, 5)], tiny).is_none());
    }

    #[test]
    fn sixteen_bit_cover_realizes_the_exact_flag_image() {
        let all: Vec<u64> = (0..=0xFFFF).collect();
        let cases = [
            vec![obs(false, 0, 0xFFFB)],
            vec![obs(false, 0, 0x0005)],
            vec![obs(true, 0xFFFF, 0x0001)],
            vec![obs(false, 0x1234, 0x8000), obs(true, 7, 0x00FF)],
            vec![
                obs(false, 0, 0xFFF0),
                obs(false, 0, 0xFFE0),
                obs(false, 0, 3),
            ],
            vec![obs(true, 0x4000, 0xC000)],
        ];
        for observations in cases {
            let cover = add_flags_cover(16, &observations, InputDomainLimits::default())
                .expect("bounded cover");
            assert!(cover.len() <= 4_096);
            assert_eq!(
                image(16, &observations, &cover),
                image(16, &observations, &all),
                "cover misses a flag tuple for {observations:?}"
            );
        }
    }

    #[test]
    fn sixteen_bit_cover_is_exact_for_many_constants_and_offsets() {
        let all: Vec<u64> = (0..=0xFFFF).collect();
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state & 0xFFFF
        };
        for round in 0..24 {
            let observation = obs(round % 2 == 0, next(), next());
            let cover = add_flags_cover(16, &[observation], InputDomainLimits::default())
                .expect("bounded cover");
            assert_eq!(
                image(16, &[observation], &cover),
                image(16, &[observation], &all),
                "{observation:?}"
            );
        }
    }

    #[test]
    fn thirty_two_bit_cover_is_small_and_realizes_sampled_flag_tuples() {
        let observations = [
            obs(false, 0, 0xFFFF_FFFB),
            obs(false, 0, 0xFFFF_FFF0),
            obs(true, 3, 0x8000_0001),
        ];
        let cover = add_flags_cover(32, &observations, InputDomainLimits::default())
            .expect("bounded cover");
        assert!(cover.len() <= 512, "{} values", cover.len());
        let sampled: Vec<u64> = [0, 1, 4, 5, 6, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF]
            .into_iter()
            .chain((0..20_000).map(|i| i * 0x0001_0F01 + 3))
            .collect();
        assert!(
            image(32, &observations, &sampled).is_subset(&image(32, &observations, &cover)),
            "a sampled value realizes a flag tuple the cover misses"
        );
    }

    #[test]
    fn thirty_two_bit_cover_realizes_random_configs_on_random_inputs() {
        let mut state = 0xD1B5_4A32_D192_ED03_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..12 {
            let count = 1 + round % 3;
            let observations: Vec<_> = (0..count)
                .map(|i| {
                    // Arbitrary constants, plus ones on the sign/carry cut points.
                    let near = [0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF][next() as usize % 5];
                    let constant = if i == 0 { next() & 0xFFFF_FFFF } else { near };
                    obs(next() & 1 == 0, next() & 0xFFFF_FFFF, constant)
                })
                .collect();
            let cover = add_flags_cover(32, &observations, InputDomainLimits::default())
                .expect("bounded cover");
            let covered = image(32, &observations, &cover);
            let mut samples: Vec<u64> = (0..3_000).map(|_| next() & 0xFFFF_FFFF).collect();
            // Rare cells: values whose view lands within 2 of a cut point.
            for o in &observations {
                for base in [
                    0,
                    0x7FFF_FFFF,
                    0x8000_0000,
                    0xFFFF_FFFF,
                    o.constant.wrapping_neg(),
                ] {
                    for delta in 0..5_u64 {
                        let y = base.wrapping_add(delta).wrapping_sub(2);
                        let x = if o.negate {
                            o.offset.wrapping_sub(y)
                        } else {
                            y.wrapping_sub(o.offset)
                        };
                        samples.push(x & 0xFFFF_FFFF);
                    }
                }
            }
            assert!(
                image(32, &observations, &samples).is_subset(&covered),
                "round {round}: cover misses a flag tuple for {observations:?}"
            );
        }
    }

    #[test]
    fn several_comparison_constants_stay_within_the_default_limit() {
        let observations: Vec<_> = (0..6)
            .map(|i| obs(false, 0, 0u64.wrapping_sub(i * 7 + 1)))
            .collect();
        assert!(add_flags_cover(32, &observations, InputDomainLimits::default()).is_some());
    }

    fn target_on(register: &Expr, bits: u32, constant: u64) -> Expr {
        let operand = Expr::resize(
            ResizeOp::Truncate,
            BitWidth::from_bits(bits).expect("width"),
            register.clone(),
        )
        .expect("truncate");
        let flags = Expr::intrinsic(
            BitWidth::W64,
            format!("vm_flags_add_w{bits}"),
            vec![
                operand,
                Expr::constant(BitWidth::from_bits(bits).expect("width"), constant),
            ],
        );
        let zero = Expr::binary(
            BinaryOp::And,
            Expr::binary(BinaryOp::Shr, flags, Expr::constant(BitWidth::W64, 6)).expect("shr"),
            Expr::constant(BitWidth::W64, 1),
        )
        .expect("and");
        let delta =
            Expr::binary(BinaryOp::Mul, zero, Expr::constant(BitWidth::W64, 0x100)).expect("mul");
        Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W64, 0x1_4000_1000),
            delta,
        )
        .expect("add")
    }

    #[test]
    fn cmp_imm_je_on_a_live_in_register_yields_a_two_target_finite_domain() {
        let rcx = Expr::register(BitWidth::W64, "native_rcx_initial");
        let target = target_on(&rcx, 32, 0xFFFF_FFFB);
        let inferred = infer_branch_input_domains(&target, InputDomainLimits::default())
            .expect("decidable shape");
        assert_eq!(
            inferred.certificates["native_rcx_initial"],
            DomainCertificate::AddFlagsCover {
                bits: 32,
                observations: 1
            }
        );
        let report = crate::enumerate_symbolic_value(
            &target,
            &inferred.domains,
            crate::SymbolicDomainLimits {
                max_values_per_variable: 4_096,
                ..crate::SymbolicDomainLimits::default()
            },
        )
        .expect("enumerates");
        assert!(report.complete);
        assert_eq!(report.unique_values, vec![0x1_4000_1000, 0x1_4000_1100]);
        // Independent check at points far from the covered values.
        for x in [0_u64, 5, 6, 0xDEAD_BEEF, 0xFFFF_FFFF_0000_0005, u64::MAX] {
            let concrete = target
                .substitute(&BTreeMap::from([(
                    "native_rcx_initial".to_owned(),
                    Expr::constant(BitWidth::W64, x),
                )]))
                .expect("substitute");
            let ExprKind::Const { value } = concrete.kind else {
                panic!("not concrete")
            };
            assert!(report.unique_values.contains(&value));
            assert_eq!(value == 0x1_4000_1100, x & 0xFFFF_FFFF == 5);
        }
    }

    #[test]
    fn byte_slice_use_is_enumerated_exhaustively() {
        let rcx = Expr::register(BitWidth::W64, "native_rcx_initial");
        let low = Expr::resize(ResizeOp::Truncate, BitWidth::W8, rcx).expect("truncate");
        let wide = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W64, low).expect("extend");
        let target =
            Expr::binary(BinaryOp::Add, Expr::constant(BitWidth::W64, 0x1000), wide).expect("add");
        let inferred = infer_branch_input_domains(&target, InputDomainLimits::default())
            .expect("byte slice is exhaustive");
        assert_eq!(inferred.domains["native_rcx_initial"].len(), 256);
        assert_eq!(
            inferred.certificates["native_rcx_initial"],
            DomainCertificate::Exhaustive { bits: 8 }
        );
    }

    fn unsupported_reason(expr: &Expr, limits: InputDomainLimits) -> (String, String) {
        match infer_branch_input_domains(expr, limits) {
            Err(InputDomainRefusal::Unsupported { register, reason }) => (register, reason),
            other => panic!("expected an Unsupported refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_register_used_at_two_widths_is_refused() {
        let narrow = Expr::register(BitWidth::W32, "native_rcx_initial");
        let wide = Expr::register(BitWidth::W64, "native_rcx_initial");
        let widened = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W64, narrow).expect("zext");
        let target = Expr::binary(BinaryOp::Add, wide, widened).expect("add");
        let (register, reason) = unsupported_reason(&target, InputDomainLimits::default());
        assert_eq!(register, "native_rcx_initial");
        assert!(reason.contains("two widths"), "{reason}");
    }

    #[test]
    fn an_oversized_cover_is_refused_as_too_many_values() {
        let rcx = Expr::register(BitWidth::W64, "native_rcx_initial");
        let target = target_on(&rcx, 32, 0xFFFF_FFFB);
        let limits = InputDomainLimits {
            max_values_per_variable: 2,
            ..InputDomainLimits::default()
        };
        assert_eq!(
            infer_branch_input_domains(&target, limits),
            Err(InputDomainRefusal::TooManyValues {
                register: "native_rcx_initial".to_owned(),
                limit: 2,
            })
        );
    }

    #[test]
    fn truncate_of_a_non_register_recurses_instead_of_panicking() {
        let rcx = Expr::register(BitWidth::W64, "native_rcx_initial");
        let loaded = Expr::load(BitWidth::W64, rcx);
        let truncated = Expr::resize(ResizeOp::Truncate, BitWidth::W8, loaded).expect("t");
        assert_eq!(
            infer_branch_input_domains(&truncated, InputDomainLimits::default()),
            Err(InputDomainRefusal::MemoryDependent)
        );
    }

    #[test]
    fn undecidable_shapes_are_refused() {
        let rcx = Expr::register(BitWidth::W64, "native_rcx_initial");
        let rdx = Expr::register(BitWidth::W64, "native_rdx_initial");
        let limits = InputDomainLimits::default();
        // Jump-table index: the register is used as a value.
        let indexed = Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W64, 0x1000),
            Expr::binary(BinaryOp::Mul, rcx.clone(), Expr::constant(BitWidth::W64, 8))
                .expect("mul"),
        )
        .expect("add");
        let (register, reason) = unsupported_reason(&indexed, limits);
        assert_eq!(register, "native_rcx_initial");
        assert!(reason.contains("beyond the exhaustive width"), "{reason}");
        // Memory-dependent branch.
        let loaded = Expr::load(BitWidth::W64, rcx.clone());
        assert_eq!(
            infer_branch_input_domains(&loaded, limits),
            Err(InputDomainRefusal::MemoryDependent)
        );
        // Flags of two input-dependent operands.
        let both = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_add_w32",
            vec![
                Expr::resize(ResizeOp::Truncate, BitWidth::W32, rcx.clone()).expect("t"),
                Expr::resize(ResizeOp::Truncate, BitWidth::W32, rdx.clone()).expect("t"),
            ],
        );
        let (_, reason) = unsupported_reason(&both, limits);
        assert!(reason.contains("two input-dependent"), "{reason}");
        // Logical flags of a wide operand observe all of its bits.
        let logical = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_nor_w32",
            vec![
                Expr::resize(ResizeOp::Truncate, BitWidth::W32, rcx.clone()).expect("t"),
                Expr::constant(BitWidth::W32, 0xFF),
            ],
        );
        let (_, reason) = unsupported_reason(&logical, limits);
        assert!(reason.contains("beyond the exhaustive width"), "{reason}");
        // An intrinsic with hidden state is not a function of its operands.
        let opaque = Expr::intrinsic(
            BitWidth::W64,
            "native_call_return_epoch_1",
            vec![rcx.clone()],
        );
        let (_, reason) = unsupported_reason(&opaque, limits);
        assert!(reason.contains("other than add flags"), "{reason}");
        // Flags observed at two widths on one register.
        let mixed = Expr::binary(
            BinaryOp::Add,
            target_on(&rcx, 32, 5),
            target_on(&rcx, 16, 5),
        )
        .expect("add");
        let (register, reason) = unsupported_reason(&mixed, limits);
        assert_eq!(register, "native_rcx_initial");
        assert!(reason.contains("several widths"), "{reason}");
        // Direct use of a wide register beside a flags observation.
        let direct = Expr::binary(BinaryOp::Add, target_on(&rcx, 32, 5), rcx).expect("add");
        let (_, reason) = unsupported_reason(&direct, limits);
        assert!(reason.contains("beyond the exhaustive width"), "{reason}");
    }

    fn stack_load(epoch: u64, offset: u64) -> Expr {
        let address = Expr::binary(
            BinaryOp::Add,
            Expr::register(BitWidth::W64, "vsp_initial"),
            Expr::constant(BitWidth::W64, offset),
        )
        .expect("add");
        Expr::intrinsic(
            BitWidth::W32,
            format!("memory_load_epoch_{epoch}"),
            vec![address],
        )
    }

    fn branch_on_sub_flags(operand: Expr, constant: u64) -> Expr {
        let flags = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_sub_w32",
            vec![operand, Expr::constant(BitWidth::W32, constant)],
        );
        let zero = Expr::binary(
            BinaryOp::And,
            Expr::binary(BinaryOp::Shr, flags, Expr::constant(BitWidth::W64, 6)).expect("shr"),
            Expr::constant(BitWidth::W64, 1),
        )
        .expect("and");
        let delta =
            Expr::binary(BinaryOp::Mul, zero, Expr::constant(BitWidth::W64, 0x100)).expect("mul");
        Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W64, 0x1_4000_1000),
            delta,
        )
        .expect("add")
    }

    #[test]
    fn compare_of_an_untracked_stack_read_is_covered_without_the_address() {
        let target = branch_on_sub_flags(stack_load(73, 2936), 5);
        let inferred = infer_branch_input_domains(&target, InputDomainLimits::default())
            .expect("a read of untracked memory is a free 32-bit value");
        assert_eq!(inferred.domains.len(), 1);
        let (name, certificate) = inferred.certificates.iter().next().expect("one");
        assert!(name.starts_with("memory_load_epoch_73@"), "{name}");
        assert!(matches!(
            certificate,
            DomainCertificate::AddFlagsCover { bits: 32, .. }
        ));
        let report = crate::enumerate_symbolic_value(
            &target,
            &inferred.domains,
            crate::SymbolicDomainLimits {
                max_values_per_variable: 4_096,
                ..crate::SymbolicDomainLimits::default()
            },
        )
        .expect("enumerates");
        assert!(report.complete);
        assert_eq!(report.unique_values, vec![0x1_4000_1000, 0x1_4000_1100]);
        // The free variable ranges over everything the read could hold.
        let abstracted = abstract_epoch_loads(&target).expect("abstract");
        let variable = name.clone();
        for x in [0_u64, 4, 5, 6, 0x8000_0000, 0xFFFF_FFFF] {
            let concrete = abstracted
                .substitute(&BTreeMap::from([(
                    variable.clone(),
                    Expr::constant(BitWidth::W32, x),
                )]))
                .expect("substitute");
            let ExprKind::Const { value } = concrete.kind else {
                panic!("not concrete")
            };
            assert!(report.unique_values.contains(&value));
            assert_eq!(value == 0x1_4000_1100, x == 5);
        }
    }

    #[test]
    fn repeated_reads_share_one_variable_and_distinct_reads_stay_apart() {
        let same = Expr::binary(
            BinaryOp::Add,
            branch_on_sub_flags(stack_load(73, 2936), 5),
            branch_on_sub_flags(stack_load(73, 2936), 9),
        )
        .expect("add");
        let inferred = infer_branch_input_domains(&same, InputDomainLimits::default())
            .expect("same read twice");
        assert_eq!(inferred.domains.len(), 1);
        let different = Expr::binary(
            BinaryOp::Add,
            branch_on_sub_flags(stack_load(73, 2936), 5),
            branch_on_sub_flags(stack_load(73, 2944), 5),
        )
        .expect("add");
        let inferred = infer_branch_input_domains(&different, InputDomainLimits::default())
            .expect("two reads");
        assert_eq!(inferred.domains.len(), 2);
    }

    #[test]
    fn untracked_reads_that_are_not_provably_covered_are_refused() {
        let limits = InputDomainLimits::default();
        // The read's value is used directly (jump-table index).
        let indexed = Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W64, 0x1000),
            Expr::resize(ResizeOp::ZeroExtend, BitWidth::W64, stack_load(73, 8)).expect("zext"),
        )
        .expect("add");
        let (register, reason) = unsupported_reason(&indexed, limits);
        assert!(register.contains("untracked memory read"), "{register}");
        assert!(reason.contains("beyond the exhaustive width"), "{reason}");
        // Flags of two unconstrained reads.
        let both = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_sub_w32",
            vec![stack_load(73, 8), stack_load(73, 16)],
        );
        let (_, reason) = unsupported_reason(&both, limits);
        assert!(reason.contains("two input-dependent"), "{reason}");
        // Flags of a non-affine function of the read.
        let masked = Expr::binary(
            BinaryOp::And,
            stack_load(73, 8),
            Expr::constant(BitWidth::W32, 0xF0F0),
        )
        .expect("and");
        let (_, reason) = unsupported_reason(&branch_on_sub_flags(masked, 5), limits);
        assert!(reason.contains("not an affine view"), "{reason}");
    }

    #[test]
    fn two_independent_registers_infer_independent_domains() {
        let rcx = Expr::register(BitWidth::W64, "native_rcx_initial");
        let rdx = Expr::register(BitWidth::W64, "native_rdx_initial");
        let target =
            Expr::binary(BinaryOp::Add, target_on(&rcx, 8, 5), target_on(&rdx, 8, 7)).expect("add");
        let inferred =
            infer_branch_input_domains(&target, InputDomainLimits::default()).expect("both narrow");
        assert_eq!(inferred.domains.len(), 2);
    }

    #[test]
    fn affine_views_compose_through_not_and_constant_offsets() {
        let rcx = Expr::register(BitWidth::W64, "native_rcx_initial");
        let low = Expr::resize(ResizeOp::Truncate, BitWidth::W32, rcx).expect("t");
        let reflected = Expr::unary(UnaryOp::Not, low);
        let shifted =
            Expr::binary(BinaryOp::Add, reflected, Expr::constant(BitWidth::W32, 10)).expect("add");
        let view = affine(&shifted).expect("affine");
        assert!(view.negate);
        assert_eq!(view.bits, 32);
        // ~x + 10 == -x + 9
        assert_eq!(view.offset, 9);
    }

    fn enumerate(target: &Expr, inferred: &InferredInputDomains) -> Vec<u64> {
        let report = crate::enumerate_symbolic_value(
            target,
            &inferred.domains,
            crate::SymbolicDomainLimits {
                max_values_per_variable: 4_096,
                ..crate::SymbolicDomainLimits::default()
            },
        )
        .expect("enumerates");
        assert!(report.complete);
        report.unique_values
    }

    fn nor_flags_of_byte(register: &Expr) -> Expr {
        let low = Expr::resize(ResizeOp::Truncate, BitWidth::W8, register.clone()).expect("t");
        let flags = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_nor_w8",
            vec![
                Expr::unary(UnaryOp::Not, low),
                Expr::constant(BitWidth::W8, 0),
            ],
        );
        let zero = Expr::binary(
            BinaryOp::And,
            Expr::binary(BinaryOp::Shr, flags, Expr::constant(BitWidth::W64, 6)).expect("shr"),
            Expr::constant(BitWidth::W64, 1),
        )
        .expect("and");
        Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W64, 0x1_4000_1000),
            Expr::binary(BinaryOp::Mul, zero, Expr::constant(BitWidth::W64, 0x100)).expect("mul"),
        )
        .expect("add")
    }

    #[test]
    fn logical_flags_of_a_byte_slice_of_a_wide_register_are_proven_exhaustive() {
        let rbx = Expr::register(BitWidth::W64, "native_rbx_initial");
        let target = nor_flags_of_byte(&rbx);
        let inferred = infer_branch_input_domains(&target, InputDomainLimits::default())
            .expect("pure logical flags of a byte");
        assert_eq!(
            inferred.certificates["native_rbx_initial"],
            DomainCertificate::Exhaustive { bits: 8 }
        );
        let values = enumerate(&target, &inferred);
        assert_eq!(values, vec![0x1_4000_1000, 0x1_4000_1100]);
        // Independent check on values outside the enumerated domain.
        for x in [
            0_u64,
            0xFF,
            0x100,
            0xDEAD_BE00,
            u64::MAX,
            0x1234_5678_9ABC_DEF0,
        ] {
            let concrete = target
                .substitute(&BTreeMap::from([(
                    "native_rbx_initial".to_owned(),
                    Expr::constant(BitWidth::W64, x),
                )]))
                .expect("substitute");
            let ExprKind::Const { value } = concrete.kind else {
                panic!("not concrete")
            };
            assert!(values.contains(&value));
        }
    }

    #[test]
    fn logical_flags_of_a_wide_untracked_read_stay_refused() {
        let wide = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_nor_w32",
            vec![stack_load(73, 8), Expr::constant(BitWidth::W32, 0)],
        );
        let (register, reason) = unsupported_reason(&wide, InputDomainLimits::default());
        assert!(register.contains("untracked memory read"), "{register}");
        assert!(reason.contains("beyond the exhaustive width"), "{reason}");
    }

    #[test]
    fn a_raw_untracked_read_as_the_target_is_refused_as_unbounded() {
        let raw = Expr::intrinsic(
            BitWidth::W64,
            "memory_load_epoch_166",
            vec![Expr::register(BitWidth::W64, "vsp_initial")],
        );
        let (register, reason) = unsupported_reason(&raw, InputDomainLimits::default());
        assert!(register.contains("untracked memory read"), "{register}");
        assert!(
            reason.contains("memory bounds none of its bits"),
            "{reason}"
        );
    }

    #[test]
    fn constant_masks_and_bit_fields_are_enumerated_over_their_observed_bits() {
        let rcx = Expr::register(BitWidth::W64, "native_rcx_initial");
        let limits = InputDomainLimits::default();
        // x & 0xF0F0 observes 8 non-contiguous bits.
        let masked = Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W64, 0x1000),
            Expr::binary(
                BinaryOp::And,
                rcx.clone(),
                Expr::constant(BitWidth::W64, 0xF0F0),
            )
            .expect("and"),
        )
        .expect("add");
        let inferred = infer_branch_input_domains(&masked, limits).expect("8 observed bits");
        assert_eq!(
            inferred.certificates["native_rcx_initial"],
            DomainCertificate::MaskedExhaustive { mask: 0xF0F0 }
        );
        let values = enumerate(&masked, &inferred);
        let expected: BTreeSet<u64> = (0..=0xFFFF_u64).map(|x| 0x1000 + (x & 0xF0F0)).collect();
        assert_eq!(values.into_iter().collect::<BTreeSet<_>>(), expected);
        // Bits 16..24 as a field.
        let field = Expr::extract(BitWidth::W8, 16, rcx.clone()).expect("extract");
        let inferred = infer_branch_input_domains(&field, limits).expect("byte field");
        assert_eq!(
            inferred.certificates["native_rcx_initial"],
            DomainCertificate::MaskedExhaustive { mask: 0xFF_0000 }
        );
        assert_eq!(inferred.domains["native_rcx_initial"].len(), 256);
        // Nine observed bits exceed the exhaustive width.
        let nine =
            Expr::binary(BinaryOp::And, rcx, Expr::constant(BitWidth::W64, 0x1FF0)).expect("and");
        let (_, reason) = unsupported_reason(&nine, limits);
        assert!(reason.contains("beyond the exhaustive width"), "{reason}");
    }

    fn pair_image(bits: u32, atoms: &[FlagsAtom], xs: &[u64]) -> BTreeSet<Vec<u64>> {
        xs.iter()
            .map(|x| {
                atoms
                    .iter()
                    .map(|atom| atom.tuple_word(bits, *x).expect("flags"))
                    .collect()
            })
            .collect()
    }

    fn pair(negate: bool, a: u64, b: u64) -> FlagsAtom {
        FlagsAtom::SubPair {
            left: View { negate, offset: a },
            right: View { negate, offset: b },
        }
    }

    #[test]
    fn parallel_view_compare_cover_realizes_the_exact_flag_image() {
        let all: Vec<u64> = (0..=0xFFFF).collect();
        let mut state = 0xA076_1D64_78BD_642F_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state & 0xFFFF
        };
        let mut cases = vec![
            vec![pair(false, 0x2A2, 0xA9E)],
            vec![pair(true, 0, 0x7FFF)],
            vec![pair(false, 0x8000, 0)],
            vec![pair(false, 5, 5)],
            vec![pair(false, 3, 0xFFFD), pair(true, 9, 0x8001)],
        ];
        for round in 0..24 {
            cases.push(vec![pair(round % 2 == 0, next(), next())]);
        }
        for atoms in cases {
            let cover =
                flags_cover(16, &atoms, InputDomainLimits::default()).expect("bounded cover");
            assert_eq!(
                pair_image(16, &atoms, &cover),
                pair_image(16, &atoms, &all),
                "{atoms:?}"
            );
        }
    }

    #[test]
    fn parallel_view_compare_cover_realizes_sampled_flags_at_64_bits() {
        let atoms = [pair(false, 0x2A2, 0xA9E)];
        let cover = flags_cover(64, &atoms, InputDomainLimits::default()).expect("cover");
        assert!(cover.len() <= 1_024, "{}", cover.len());
        let mut samples = vec![
            0_u64,
            1,
            0x7FFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000,
            u64::MAX,
        ];
        for base in [0_u64, 0x8000_0000_0000_0000, 0x7FFF_FFFF_FFFF_FFFF] {
            for delta in 0..4096_u64 {
                samples.push(base.wrapping_add(delta).wrapping_sub(0xB00));
            }
        }
        samples.extend((0..20_000_u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        assert!(pair_image(64, &atoms, &samples).is_subset(&pair_image(64, &atoms, &cover)));
    }

    #[test]
    fn pointer_difference_compare_on_a_live_in_register_is_covered() {
        let vsp = Expr::register(BitWidth::W64, "vsp_initial");
        let at = |c: u64| {
            Expr::binary(BinaryOp::Add, vsp.clone(), Expr::constant(BitWidth::W64, c)).expect("add")
        };
        let flags = Expr::intrinsic(BitWidth::W64, "vm_flags_sub_w64", vec![at(674), at(2718)]);
        let carry =
            Expr::binary(BinaryOp::And, flags, Expr::constant(BitWidth::W64, 1)).expect("and");
        let target = Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W64, 0x1_4000_1000),
            Expr::binary(BinaryOp::Mul, carry, Expr::constant(BitWidth::W64, 0x100)).expect("mul"),
        )
        .expect("add");
        let inferred = infer_branch_input_domains(&target, InputDomainLimits::default())
            .expect("parallel views");
        assert!(matches!(
            inferred.certificates["vsp_initial"],
            DomainCertificate::AddFlagsCover { bits: 64, .. }
        ));
        let values = enumerate(&target, &inferred);
        assert_eq!(values, vec![0x1_4000_1000, 0x1_4000_1100]);
        // Opposite directions have a non-constant difference: refused.
        let neg = Expr::binary(BinaryOp::Sub, Expr::constant(BitWidth::W64, 9), vsp.clone())
            .expect("sub");
        let skew = Expr::intrinsic(BitWidth::W64, "vm_flags_sub_w64", vec![at(1), neg]);
        let (_, reason) = unsupported_reason(&skew, InputDomainLimits::default());
        assert!(reason.contains("not parallel views"), "{reason}");
    }

    fn reg(bits: u32, name: &str) -> Expr {
        Expr::register(BitWidth::from_bits(bits).expect("width"), name)
    }

    fn zext(bits: u32, value: Expr) -> Expr {
        Expr::resize(
            ResizeOp::ZeroExtend,
            BitWidth::from_bits(bits).expect("width"),
            value,
        )
        .expect("zext")
    }

    fn flags_of(name: &str, bits: u32, arguments: Vec<Expr>) -> Expr {
        Expr::intrinsic(BitWidth::W64, format!("{name}_w{bits}"), arguments)
    }

    fn flag_bit(flags: Expr, bit: u64) -> Expr {
        Expr::binary(
            BinaryOp::And,
            Expr::binary(BinaryOp::Shr, flags, Expr::constant(BitWidth::W64, bit)).expect("shr"),
            Expr::constant(BitWidth::W64, 1),
        )
        .expect("and")
    }

    fn two_way(bit: Expr) -> Expr {
        let delta =
            Expr::binary(BinaryOp::Mul, bit, Expr::constant(BitWidth::W64, 0x100)).expect("mul");
        Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W64, 0x1_4000_1000),
            delta,
        )
        .expect("add")
    }

    const TWO_TARGETS: [u64; 2] = [0x1_4000_1000, 0x1_4000_1100];

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn order_flags(bits: u32, a: u64, b: u64) -> u64 {
        evaluate_vm_flags(&format!("vm_flags_sub_w{bits}"), &[a, b]).expect("flags")
            & ORDER_FLAG_BITS
    }

    #[test]
    fn sign_order_cover_attains_every_order_flag_tuple_of_a_free_pair() {
        let samples = |bits: u32| -> Vec<u64> {
            if bits == 8 {
                return (0..=255).collect();
            }
            let sign = 1_u64 << (bits - 1);
            let mut values = vec![
                0,
                1,
                2,
                3,
                sign - 2,
                sign - 1,
                sign,
                sign + 1,
                mask(bits) - 1,
            ];
            values.push(mask(bits));
            let mut state = 0x9E37_79B9_7F4A_7C15_u64;
            values.extend((0..300).map(|_| xorshift(&mut state) & mask(bits)));
            values
        };
        let image = |bits: u32, a_set: &[u64], b_set: &[u64]| {
            a_set
                .iter()
                .flat_map(|a| b_set.iter().map(|b| order_flags(bits, *a, *b)))
                .collect::<BTreeSet<_>>()
        };
        for bits in [8_u32, 16, 32, 64] {
            let wide = sign_order_cover(bits, bits);
            assert_eq!(
                image(bits, &wide, &wide),
                image(bits, &samples(bits), &samples(bits)),
                "w{bits}"
            );
            if bits > 8 {
                // Operand A zero-extended from 8 bits.
                assert_eq!(
                    image(bits, &sign_order_cover(8, bits), &wide),
                    image(bits, &(0..=255).collect::<Vec<u64>>(), &samples(bits)),
                    "w{bits} narrow A"
                );
            }
        }
    }

    #[test]
    fn free_pair_order_flag_branches_infer_the_two_target_domain() {
        let (a, b) = (reg(64, "free_a"), reg(64, "free_b"));
        let cmp = flags_of("vm_flags_sub", 64, vec![a.clone(), b.clone()]);
        let not_of = |e: Expr| Expr::unary(UnaryOp::Not, e);
        let complement_or = Expr::unary(
            UnaryOp::Not,
            Expr::binary(BinaryOp::Or, Expr::constant(BitWidth::W64, !1), cmp.clone()).expect("or"),
        );
        for (name, target) in [
            ("zf", two_way(flag_bit(cmp.clone(), 6))),
            ("sf", two_way(flag_bit(cmp.clone(), 7))),
            ("of", two_way(flag_bit(cmp.clone(), 11))),
            (
                "not zf",
                two_way(
                    Expr::binary(
                        BinaryOp::And,
                        Expr::constant(BitWidth::W64, 1),
                        not_of(
                            Expr::binary(
                                BinaryOp::Shr,
                                cmp.clone(),
                                Expr::constant(BitWidth::W64, 6),
                            )
                            .expect("shr"),
                        ),
                    )
                    .expect("and"),
                ),
            ),
            ("not or cf", two_way(complement_or)),
        ] {
            let inferred = infer_branch_input_domains(&target, InputDomainLimits::default())
                .unwrap_or_else(|refusal| panic!("{name}: {refusal}"));
            for register in ["free_a", "free_b"] {
                assert_eq!(
                    inferred.certificates[register],
                    DomainCertificate::SignOrderCover { bits: 64 }
                );
                assert_eq!(inferred.domains[register].len(), 4);
            }
            assert_eq!(enumerate(&target, &inferred), TWO_TARGETS, "{name}");
        }
        // A narrow operand is only a zero-extended register: no sign bit to vary.
        let narrow = flags_of(
            "vm_flags_sub",
            32,
            vec![zext(32, reg(16, "free_n")), reg(32, "free_w")],
        );
        let target = two_way(flag_bit(narrow, 0));
        let inferred = infer_branch_input_domains(&target, InputDomainLimits::default())
            .expect("narrow free pair");
        assert_eq!(inferred.domains["free_n"].len(), 2);
        assert_eq!(inferred.domains["free_w"].len(), 4);
        assert_eq!(enumerate(&target, &inferred), TWO_TARGETS);
    }

    #[test]
    fn free_pair_observed_beyond_order_flags_stays_refused() {
        let limits = InputDomainLimits::default();
        let (a, b, c) = (reg(64, "free_a"), reg(64, "free_b"), reg(64, "free_c"));
        let cmp = flags_of("vm_flags_sub", 64, vec![a.clone(), b.clone()]);
        // Parity depends on the low bytes of both operands.
        let (_, reason) = unsupported_reason(&two_way(flag_bit(cmp.clone(), 2)), limits);
        assert!(reason.contains("not parallel views"), "{reason}");
        // The whole word is observed.
        let whole = Expr::binary(BinaryOp::Add, cmp.clone(), Expr::constant(BitWidth::W64, 1))
            .expect("add");
        let (_, reason) = unsupported_reason(&whole, limits);
        assert!(reason.contains("not parallel views"), "{reason}");
        // One operand is also used as a value.
        let both =
            Expr::binary(BinaryOp::Add, two_way(flag_bit(cmp.clone(), 6)), a.clone()).expect("add");
        let (_, reason) = unsupported_reason(&both, limits);
        assert!(reason.contains("beyond its compare flags"), "{reason}");
        // One operand is compared against two different partners.
        let other = flags_of("vm_flags_sub", 64, vec![a, c]);
        let chained = Expr::binary(
            BinaryOp::Add,
            two_way(flag_bit(cmp, 6)),
            two_way(flag_bit(other, 6)),
        )
        .expect("add");
        let (_, reason) = unsupported_reason(&chained, limits);
        assert!(reason.contains("several partners"), "{reason}");
    }

    #[test]
    fn shifted_narrow_register_compare_is_covered_over_its_reachable_range() {
        // `(zext16 L >> 1) < 0x104` and `0 - (zext16 L >> 1) != 0`.
        let shifted = Expr::binary(
            BinaryOp::Shr,
            zext(32, reg(16, "narrow")),
            Expr::constant(BitWidth::W32, 1),
        )
        .expect("shr");
        let against = flags_of(
            "vm_flags_sub",
            32,
            vec![shifted.clone(), Expr::constant(BitWidth::W32, 0x104)],
        );
        let from_zero = flags_of(
            "vm_flags_sub",
            32,
            vec![Expr::constant(BitWidth::W32, 0), shifted],
        );
        for flags in [against, from_zero] {
            for bit in [0, 6, 7, 11] {
                let target = two_way(flag_bit(flags.clone(), bit));
                let inferred = infer_branch_input_domains(&target, InputDomainLimits::default())
                    .expect("shifted view");
                let all: BTreeMap<String, Vec<Expr>> = BTreeMap::from([(
                    "narrow".to_owned(),
                    (0..=0xFFFF)
                        .map(|v| Expr::constant(BitWidth::W16, v))
                        .collect(),
                )]);
                let exhaustive = crate::enumerate_symbolic_value(
                    &target,
                    &all,
                    crate::SymbolicDomainLimits {
                        max_values_per_variable: 70_000,
                        max_combinations: 70_000,
                        ..crate::SymbolicDomainLimits::default()
                    },
                )
                .expect("enumerates");
                assert_eq!(
                    enumerate(&target, &inferred),
                    exhaustive.unique_values,
                    "bit {bit}"
                );
            }
        }
    }

    #[test]
    fn range_restricted_covers_realize_the_exact_flag_image() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        for round in 0..48_u64 {
            let (a, b) = (xorshift(&mut state) & 0xFFFF, xorshift(&mut state) & 0xFFFF);
            let view = View {
                negate: round % 2 == 0,
                offset: a,
            };
            let atoms = [
                FlagsAtom::SubConstLeft { view, constant: b },
                FlagsAtom::LogicalSelf {
                    view,
                    nand: round % 3 == 0,
                },
            ];
            for range_bits in [16_u32, 15, 9] {
                let cover = flags_cover_in(16, &atoms, range_bits, InputDomainLimits::default())
                    .expect("bounded cover");
                assert!(cover.iter().all(|x| *x < 1 << range_bits));
                let all: Vec<u64> = (0..1_u64 << range_bits).collect();
                assert_eq!(
                    pair_image(16, &atoms, &cover),
                    pair_image(16, &atoms, &all),
                    "round {round} range {range_bits}"
                );
            }
        }
    }

    #[test]
    fn bit_demand_through_complements_and_shifts_enumerates_only_that_bit() {
        // `1 & !(!L >> 6)` observes bit 6 of the 64-bit read alone.
        let load = reg(64, "wide");
        let shifted = Expr::binary(
            BinaryOp::Shr,
            Expr::unary(UnaryOp::Not, load),
            Expr::constant(BitWidth::W64, 6),
        )
        .expect("shr");
        let bit = Expr::binary(
            BinaryOp::And,
            Expr::constant(BitWidth::W64, 1),
            Expr::unary(UnaryOp::Not, shifted),
        )
        .expect("and");
        let target = two_way(bit);
        let inferred = infer_branch_input_domains(&target, InputDomainLimits::default())
            .expect("one observed bit");
        assert_eq!(
            inferred.certificates["wide"],
            DomainCertificate::MaskedExhaustive { mask: 1 << 6 }
        );
        assert_eq!(enumerate(&target, &inferred), TWO_TARGETS);
        // Arithmetic on the register is not a bit selection.
        let sum = Expr::binary(
            BinaryOp::Add,
            reg(64, "wide"),
            Expr::constant(BitWidth::W64, 1),
        )
        .expect("add");
        let masked =
            Expr::binary(BinaryOp::And, sum, Expr::constant(BitWidth::W64, 1)).expect("and");
        let (_, reason) = unsupported_reason(&masked, InputDomainLimits::default());
        assert!(reason.contains("beyond the exhaustive width"), "{reason}");
    }

    #[test]
    fn logical_self_flags_and_absorbed_operands_are_covered() {
        let load = reg(32, "word");
        let negated = Expr::unary(UnaryOp::Not, load.clone());
        let logical = flags_of("vm_flags_nor", 32, vec![negated.clone(), negated]);
        let target = two_way(flag_bit(logical, 6));
        let inferred = infer_branch_input_domains(&target, InputDomainLimits::default())
            .expect("logical flags of one view");
        assert!(matches!(
            inferred.certificates["word"],
            DomainCertificate::AddFlagsCover { bits: 32, .. }
        ));
        assert_eq!(enumerate(&target, &inferred), TWO_TARGETS);
        // `x | all-ones` is the constant all-ones: `x` is irrelevant.
        let absorbed = Expr::binary(
            BinaryOp::Or,
            Expr::resize(ResizeOp::Truncate, BitWidth::W32, reg(64, "ignored")).expect("t"),
            Expr::constant(BitWidth::W32, 0xFFFF_FFFF),
        )
        .expect("or");
        let target = two_way(flag_bit(
            flags_of("vm_flags_sub", 32, vec![absorbed, load]),
            6,
        ));
        let inferred = infer_branch_input_domains(&target, InputDomainLimits::default())
            .expect("constant left operand");
        assert_eq!(inferred.domains["ignored"].len(), 1);
        assert_eq!(enumerate(&target, &inferred), TWO_TARGETS);
    }
}
