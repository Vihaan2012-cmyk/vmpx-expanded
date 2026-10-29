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
//! * `vm_flags_add_wN(±reg + c1, c2)` status-flag words of one common width,
//!   which depend on the register only through the cell it falls in between a
//!   fixed set of cut points and its residue modulo 256 (see [`add_flags_cover`]).
//!
//! Any other use (direct arithmetic, loads, opaque or logical-flag intrinsics,
//! flags of two input-dependent operands, mixed use kinds) is refused: the
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
    /// Cell/residue cover of add-flag observations of `observations` atoms.
    AddFlagsCover { bits: u32, observations: usize },
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
    InputDomainRefusal::Unsupported {
        register: register.to_owned(),
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
    if !matches!(bits, 8 | 16 | 32 | 64) {
        return None;
    }
    let space = 1_u128 << bits;
    let max_values = limits.max_values_per_variable;
    if bits <= limits.max_exhaustive_bits {
        return exhaustive(bits, limits);
    }
    let half = space >> 1;
    let mut cuts: BTreeSet<u128> = BTreeSet::from([0]);
    for observation in observations {
        let offset = u128::from(observation.offset & mask(bits));
        let constant = u128::from(observation.constant & mask(bits));
        // Thresholds of `y`: its own sign/zero boundary and where `y + k`
        // reaches zero, one (ZF cell), and the sign boundary.
        let thresholds = [
            0,
            half,
            (space - constant) % space,
            (space + 1 - constant) % space,
            (half + space - constant) % space,
        ];
        for threshold in thresholds {
            let x = if observation.negate {
                (offset + space - threshold) % space
            } else {
                (threshold + space - offset) % space
            };
            cuts.insert(x);
            cuts.insert((x + 1) % space);
        }
    }
    let starts: Vec<u128> = cuts.into_iter().collect();
    // The target depends on `x` only through these flag words, so one value per
    // distinct tuple realizes the same image as the sampled cells.
    let name = format!("vm_flags_add_w{bits}");
    let mut seen: BTreeSet<Vec<u64>> = BTreeSet::new();
    let mut values = Vec::new();
    for (index, start) in starts.iter().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(space);
        let take = (end - start).min(256);
        for value in *start..start + take {
            let value = u64::try_from(value).ok()?;
            let tuple = observations
                .iter()
                .map(|observation| {
                    let offset = observation.offset & mask(bits);
                    let operand = if observation.negate {
                        offset.wrapping_sub(value)
                    } else {
                        value.wrapping_add(offset)
                    } & mask(bits);
                    evaluate_vm_flags(&name, &[operand, observation.constant & mask(bits)])
                })
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

/// `expr == (±register + offset) mod 2^bits`.
struct Affine {
    register: String,
    register_width: BitWidth,
    bits: u32,
    negate: bool,
    offset: u64,
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

fn add_flags_width(name: &str) -> Option<u32> {
    match name.strip_prefix("vm_flags_add_w")? {
        "8" => Some(8),
        "16" => Some(16),
        "32" => Some(32),
        "64" => Some(64),
        _ => None,
    }
}

#[derive(Default)]
struct Uses {
    width: Option<BitWidth>,
    /// Widest directly observed low slice.
    direct_bits: Option<u32>,
    /// Add-flags observations with the flags width they were taken at.
    atoms: Vec<(u32, AddFlagsObservation)>,
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
                uses.direct_bits = uses.direct_bits.max(Some(expr.width.bits()));
            }
            ExprKind::Resize {
                op: ResizeOp::Truncate,
                value,
            }
            | ExprKind::Extract { value, lsb: 0 }
                if matches!(value.kind, ExprKind::Register { .. }) =>
            {
                let ExprKind::Register { name } = &value.kind else {
                    unreachable!("guard checked the operand is a register")
                };
                self.note_width(name, value.width);
                let uses = self.uses.entry(name.clone()).or_default();
                uses.direct_bits = uses.direct_bits.max(Some(expr.width.bits()));
            }
            ExprKind::Intrinsic { name, arguments } => {
                if arguments
                    .iter()
                    .all(|argument| argument.dependencies().is_empty())
                {
                    // Input-independent term: evaluation decides whether it folds.
                    return;
                }
                let Some(bits) = add_flags_width(name) else {
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
                self.visit_add_flags(name, bits, arguments);
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

    fn visit_add_flags(&mut self, name: &str, bits: u32, arguments: &[Expr]) {
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
        let (symbolic, constant) = match (constant_of(left), constant_of(right)) {
            (None, Some(c)) => (left, c),
            (Some(c), None) => (right, c),
            _ => {
                self.refuse(unsupported(
                    &register,
                    "add flags of two input-dependent or non-constant operands",
                ));
                return;
            }
        };
        let Some(view) = affine(symbolic).filter(|view| view.bits >= bits) else {
            self.refuse(unsupported(
                &register,
                &format!("{name} operand is not an affine view of one register"),
            ));
            return;
        };
        self.note_width(&view.register, view.register_width);
        self.uses.entry(view.register).or_default().atoms.push((
            bits,
            AddFlagsObservation {
                negate: view.negate,
                offset: view.offset & mask(bits),
                constant: constant & mask(bits),
            },
        ));
    }
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
    let mut scan = Scan {
        uses: BTreeMap::new(),
        refusal: None,
    };
    scan.visit(target);
    if let Some(refusal) = scan.refusal {
        return Err(refusal);
    }
    let mut domains = BTreeMap::new();
    let mut certificates = BTreeMap::new();
    for (register, uses) in scan.uses {
        let width = uses
            .width
            .ok_or_else(|| unsupported(&register, "register width unknown"))?;
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
        let (values, certificate) = match (uses.direct_bits, atom_width) {
            (Some(direct), atom) => {
                let bits = direct.max(atom.unwrap_or(0));
                if bits > limits.max_exhaustive_bits {
                    return Err(unsupported(
                        &register,
                        "register is used directly beyond the exhaustive width",
                    ));
                }
                (
                    exhaustive(bits, limits),
                    DomainCertificate::Exhaustive { bits },
                )
            }
            (None, Some(bits)) => {
                let observations: Vec<_> = uses.atoms.iter().map(|(_, o)| *o).collect();
                let certificate = if bits <= limits.max_exhaustive_bits {
                    DomainCertificate::Exhaustive { bits }
                } else {
                    DomainCertificate::AddFlagsCover {
                        bits,
                        observations: observations.len(),
                    }
                };
                (add_flags_cover(bits, &observations, limits), certificate)
            }
            (None, None) => {
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
                .map(|value| Expr::constant(width, value))
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
        assert!(infer_branch_input_domains(&indexed, limits).is_err());
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
        assert!(infer_branch_input_domains(&both, limits).is_err());
        // Logical (nor) flags need a mask cover, which is not implemented.
        let logical = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_nor_w32",
            vec![
                Expr::resize(ResizeOp::Truncate, BitWidth::W32, rcx.clone()).expect("t"),
                Expr::constant(BitWidth::W32, 0xFF),
            ],
        );
        assert!(infer_branch_input_domains(&logical, limits).is_err());
        // Flags observed at two widths on one register.
        let mixed = Expr::binary(
            BinaryOp::Add,
            target_on(&rcx, 32, 5),
            target_on(&rcx, 16, 5),
        )
        .expect("add");
        assert!(infer_branch_input_domains(&mixed, limits).is_err());
        // Direct use of a wide register beside a flags observation.
        let direct = Expr::binary(BinaryOp::Add, target_on(&rcx, 32, 5), rcx).expect("add");
        assert!(infer_branch_input_domains(&direct, limits).is_err());
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
}
