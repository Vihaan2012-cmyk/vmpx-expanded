use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;
use vmpx_pe::{PeImage, Rva, Va};

use crate::{BitWidth, Expr, ExprError, ExprKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageConcretizationLimits {
    pub max_expression_nodes: usize,
    pub max_checksum_bytes: usize,
}

impl Default for ImageConcretizationLimits {
    fn default() -> Self {
        Self {
            max_expression_nodes: 200_000,
            max_checksum_bytes: 1_048_576,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ImageConcretizationReport {
    pub expression: Expr,
    pub resolved_intrinsics: usize,
    pub unresolved_intrinsics: BTreeSet<String>,
    pub complete: bool,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ImageConcretizationError {
    #[error("expression contains more than {limit} nodes")]
    ExpressionLimit { limit: usize },
    #[error("intrinsic assignment {name} has width {actual:?}, expected {expected:?}")]
    AssignmentWidth {
        name: String,
        expected: BitWidth,
        actual: BitWidth,
    },
    #[error("expression reconstruction failed: {0}")]
    Expression(#[from] ExprError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModuloImageTargetLimits {
    pub concretization: ImageConcretizationLimits,
    pub max_residues: usize,
}

impl Default for ModuloImageTargetLimits {
    fn default() -> Self {
        Self {
            concretization: ImageConcretizationLimits::default(),
            max_residues: 65_536,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RecoveredImageTarget {
    pub va: Va,
    pub rva: Option<Rva>,
    pub executable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModuloImageTargetReport {
    pub schema_version: u32,
    pub driver_intrinsic: String,
    pub divisor: u64,
    pub residues_evaluated: usize,
    pub targets: Vec<RecoveredImageTarget>,
    pub unresolved_intrinsics: BTreeSet<String>,
    pub complete: bool,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ModuloImageTargetError {
    #[error(
        "continuation expression contains {count} unsigned-division drivers; exactly one is required"
    )]
    DriverCount { count: usize },
    #[error("unsigned-division driver {name} has no unique expression node")]
    AmbiguousDriverNode { name: String },
    #[error("unsigned-division driver shape is unsupported: {message}")]
    UnsupportedDriver { message: String },
    #[error("driver divisor {divisor} exceeds residue limit {limit}")]
    ResidueLimit { divisor: u64, limit: usize },
    #[error("symbolic driver rewrite failed: {0}")]
    Expression(#[from] ExprError),
    #[error("image concretization failed: {0}")]
    Concretization(#[from] ImageConcretizationError),
}

fn constant(expression: &Expr) -> Option<u64> {
    match expression.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

fn read_image_value(pe: &PeImage<'_>, address: u64, width: BitWidth) -> Option<u64> {
    let rva = pe.optional.image_base.checked_rva_of(Va(address))?;
    let mut value = 0_u64;
    for byte in 0..width.bytes() {
        let rva = rva.checked_add(byte)?;
        let offset = usize::try_from(pe.rva_to_file_offset(rva).ok()?.get()).ok()?;
        let input = *pe.data().get(offset)?;
        value |= u64::from(input) << (byte * 8);
    }
    Some(width.truncate(value))
}

fn checked_pointer_add(address: u64, stride: i8, index: usize) -> Option<u64> {
    let index = i64::try_from(index).ok()?;
    let delta = index.checked_mul(i64::from(stride))?;
    if delta >= 0 {
        address.checked_add(delta.cast_unsigned())
    } else {
        address.checked_sub(delta.unsigned_abs())
    }
}

fn resolve_checksum(pe: &PeImage<'_>, arguments: &[Expr], byte_limit: usize) -> Option<u64> {
    let [
        accumulator,
        pointer,
        length,
        table,
        shift_bits,
        shift_right,
        pointer_stride,
        iteration_xor,
        _snapshot,
    ] = arguments
    else {
        return None;
    };
    let mut accumulator = u32::try_from(constant(accumulator)?).ok()?;
    let pointer = constant(pointer)?;
    let length = usize::try_from(constant(length)?).ok()?;
    let table = constant(table)?;
    let shift_bits = u32::try_from(constant(shift_bits)?).ok()?;
    let shift_right = constant(shift_right)? != 0;
    let pointer_stride = u8::try_from(constant(pointer_stride)?).ok()?.cast_signed();
    let iteration_xor = u32::try_from(constant(iteration_xor)?).ok()?;
    if length > byte_limit || shift_bits >= 32 {
        return None;
    }
    for index in 0..length {
        let input_address = checked_pointer_add(pointer, pointer_stride, index)?;
        let input = u8::try_from(read_image_value(pe, input_address, BitWidth::W8)?).ok()?;
        let table_index = usize::from(accumulator.to_le_bytes()[0] ^ input);
        let table_offset = u64::try_from(table_index.checked_mul(4)?).ok()?;
        let entry_address = table.checked_add(table_offset)?;
        let entry = u32::try_from(read_image_value(pe, entry_address, BitWidth::W32)?).ok()?;
        let shifted = if shift_right {
            accumulator.wrapping_shr(shift_bits)
        } else {
            accumulator.wrapping_shl(shift_bits)
        };
        accumulator = shifted ^ entry ^ iteration_xor;
    }
    Some(u64::from(accumulator))
}

fn collect_named_intrinsic_arguments(expression: &Expr, name: &str, output: &mut Vec<Vec<Expr>>) {
    match &expression.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => {}
        ExprKind::Unary { value, .. }
        | ExprKind::Load { address: value }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => {
            collect_named_intrinsic_arguments(value, name, output);
        }
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat {
            high: left,
            low: right,
        } => {
            collect_named_intrinsic_arguments(left, name, output);
            collect_named_intrinsic_arguments(right, name, output);
        }
        ExprKind::Intrinsic {
            name: intrinsic_name,
            arguments,
        } => {
            if intrinsic_name == name {
                output.push(arguments.clone());
            }
            for argument in arguments {
                collect_named_intrinsic_arguments(argument, name, output);
            }
        }
    }
}

/// Rewrites assigned opaque inputs and image-backed loads in an expression.
///
/// This routine deliberately does not infer memory immutability. Callers may
/// use the result as a proof only after separately establishing that all
/// represented memory epochs preserve the referenced image bytes.
///
/// # Errors
///
/// Rejects over-budget expressions, cross-width assignments, and malformed
/// expression reconstruction.
pub fn concretize_with_immutable_image(
    pe: &PeImage<'_>,
    expression: &Expr,
    intrinsic_assignments: &BTreeMap<String, Expr>,
    limits: ImageConcretizationLimits,
) -> Result<ImageConcretizationReport, ImageConcretizationError> {
    if expression.node_count_bounded(limits.max_expression_nodes) > limits.max_expression_nodes {
        return Err(ImageConcretizationError::ExpressionLimit {
            limit: limits.max_expression_nodes,
        });
    }
    let mut resolved_intrinsics = 0_usize;
    let mut assignment_width_error = None;
    let expression = expression.rewrite_intrinsics(&mut |width, name, arguments| {
        if let Some(assignment) = intrinsic_assignments.get(name) {
            if assignment.width != width {
                assignment_width_error = Some(ImageConcretizationError::AssignmentWidth {
                    name: name.to_owned(),
                    expected: width,
                    actual: assignment.width,
                });
                return None;
            }
            resolved_intrinsics = resolved_intrinsics.saturating_add(1);
            return Some(assignment.clone());
        }
        if name.starts_with("memory_load_epoch_")
            && let [address] = arguments
            && let Some(address) = constant(address)
            && let Some(value) = read_image_value(pe, address, width)
        {
            resolved_intrinsics = resolved_intrinsics.saturating_add(1);
            return Some(Expr::constant(width, value));
        }
        if name == "table_checksum_low_byte_xor_recurrence_w32"
            && width == BitWidth::W32
            && let Some(value) = resolve_checksum(pe, arguments, limits.max_checksum_bytes)
        {
            resolved_intrinsics = resolved_intrinsics.saturating_add(1);
            return Some(Expr::constant(width, value));
        }
        None
    })?;
    if let Some(error) = assignment_width_error {
        return Err(error);
    }
    let unresolved_intrinsics = expression.intrinsic_dependencies();
    Ok(ImageConcretizationReport {
        complete: unresolved_intrinsics.is_empty(),
        expression,
        resolved_intrinsics,
        unresolved_intrinsics,
    })
}

/// Recovers every image-backed target selected by a W32 unsigned-division
/// remainder. The proof substitutes the complete architectural remainder
/// domain, rejects expressions that retain the quotient or pre-division
/// counter, and requires every residue to concretize without opaque inputs.
///
/// As with [`concretize_with_immutable_image`], the caller must independently
/// prove that the referenced image bytes are immutable for the represented
/// execution path.
///
/// # Errors
///
/// Rejects ambiguous/non-W32 drivers, quotient-dependent expressions,
/// over-budget residue domains, and failed image concretization.
#[allow(clippy::too_many_lines)]
pub fn recover_modulo_image_targets(
    pe: &PeImage<'_>,
    expression: &Expr,
    limits: ModuloImageTargetLimits,
) -> Result<ModuloImageTargetReport, ModuloImageTargetError> {
    let drivers = expression
        .intrinsic_dependencies()
        .into_iter()
        .filter(|name| name.starts_with("x86_unsigned_divide_w32_"))
        .collect::<Vec<_>>();
    let [driver] = drivers.as_slice() else {
        return Err(ModuloImageTargetError::DriverCount {
            count: drivers.len(),
        });
    };
    let mut occurrences = Vec::new();
    collect_named_intrinsic_arguments(expression, driver, &mut occurrences);
    let Some(arguments) = occurrences.first() else {
        return Err(ModuloImageTargetError::AmbiguousDriverNode {
            name: driver.clone(),
        });
    };
    if occurrences.iter().any(|candidate| candidate != arguments) {
        return Err(ModuloImageTargetError::AmbiguousDriverNode {
            name: driver.clone(),
        });
    }
    let [high, _low, divisor] = arguments.as_slice() else {
        return Err(ModuloImageTargetError::UnsupportedDriver {
            message: "driver does not have high, low, and divisor operands".to_owned(),
        });
    };
    if arguments
        .iter()
        .any(|argument| argument.width != BitWidth::W32)
        || constant(high) != Some(0)
    {
        return Err(ModuloImageTargetError::UnsupportedDriver {
            message: "only W32 DIV with a zero high dividend is currently proven".to_owned(),
        });
    }
    let divisor = constant(divisor)
        .filter(|value| *value != 0)
        .ok_or_else(|| ModuloImageTargetError::UnsupportedDriver {
            message: "divisor is zero or symbolic".to_owned(),
        })?;
    let residue_count =
        usize::try_from(divisor).map_err(|_| ModuloImageTargetError::ResidueLimit {
            divisor,
            limit: limits.max_residues,
        })?;
    if residue_count > limits.max_residues {
        return Err(ModuloImageTargetError::ResidueLimit {
            divisor,
            limit: limits.max_residues,
        });
    }

    let symbolic_packed = Expr::concat(
        Expr::register(BitWidth::W32, "__vmpx_div_remainder"),
        Expr::register(BitWidth::W32, "__vmpx_div_quotient"),
    )?;
    let sensitivity = expression
        .rewrite_intrinsics(&mut |_, name, _| (name == driver).then(|| symbolic_packed.clone()))?;
    let dependencies = sensitivity.dependencies();
    if dependencies.contains("__vmpx_div_quotient") {
        return Err(ModuloImageTargetError::UnsupportedDriver {
            message: "continuation target depends on the DIV quotient".to_owned(),
        });
    }
    if sensitivity
        .intrinsic_dependencies()
        .iter()
        .any(|name| name.starts_with("read_timestamp_counter_"))
    {
        return Err(ModuloImageTargetError::UnsupportedDriver {
            message: "timestamp-counter data escapes the proven remainder driver".to_owned(),
        });
    }

    let mut target_values = BTreeSet::new();
    let mut unresolved_intrinsics = BTreeSet::new();
    let mut all_constant = true;
    for remainder in 0..divisor {
        let packed = remainder << 32;
        let concretized = concretize_with_immutable_image(
            pe,
            expression,
            &BTreeMap::from([(driver.clone(), Expr::constant(BitWidth::W64, packed))]),
            limits.concretization,
        )?;
        if let Some(value) = constant(&concretized.expression) {
            target_values.insert(value);
        } else {
            all_constant = false;
            unresolved_intrinsics.extend(concretized.unresolved_intrinsics);
        }
    }
    let targets = target_values
        .into_iter()
        .map(|value| {
            let va = Va(value);
            let rva = pe.optional.image_base.checked_rva_of(va);
            let executable = rva.is_some_and(|rva| {
                pe.sections.iter().any(|section| {
                    section.contains_mapped_rva(rva) && section.characteristics & 0x2000_0000 != 0
                })
            });
            RecoveredImageTarget {
                va,
                rva,
                executable,
            }
        })
        .collect::<Vec<_>>();
    let complete = all_constant && unresolved_intrinsics.is_empty() && !targets.is_empty();
    let mut diagnostics = Vec::new();
    if targets.iter().any(|target| !target.executable) {
        diagnostics
            .push("one or more recovered targets are not mapped executable image bytes".to_owned());
    }
    Ok(ModuloImageTargetReport {
        schema_version: 1,
        driver_intrinsic: driver.clone(),
        divisor,
        residues_evaluated: residue_count,
        targets,
        unresolved_intrinsics,
        complete,
        diagnostics,
    })
}
