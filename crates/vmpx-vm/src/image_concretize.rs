use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;
use vmpx_pe::{PeImage, Rva, Va};

use crate::{
    BinaryOp, BitWidth, Expr, ExprError, ExprKind, InputDomainLimits, ResizeOp,
    SymbolicDomainLimits, infer_branch_input_domains,
};

const SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
const SCN_MEM_READ: u32 = 0x4000_0000;
const SCN_MEM_WRITE: u32 = 0x8000_0000;

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

/// True when every one of `len` bytes starting at `address` lies in read-only
/// image memory: the PE headers, or a section whose characteristics are
/// readable, NOT writable, and file-backed (not uninitialized/`.bss`). Under the
/// preferred-base load this tool assumes, a non-writable, file-backed byte
/// cannot change at runtime, so a load from it has one definitive value equal to
/// the image bytes and may be concretized. Writable, uninitialized, or unmapped
/// bytes are rejected (fail closed). The read itself ([`read_image_value`])
/// further fails when a byte falls in a section's virtual zero-fill tail beyond
/// its raw data.
fn readonly_image_range(pe: &PeImage<'_>, address: u64, len: u32) -> bool {
    if len == 0 {
        return false;
    }
    for byte in 0..len {
        let Some(va) = address.checked_add(u64::from(byte)) else {
            return false;
        };
        let Some(rva) = pe.optional.image_base.checked_rva_of(Va(va)) else {
            return false;
        };
        if rva.0 < pe.optional.size_of_headers {
            continue;
        }
        let Some(section) = pe.sections.iter().find(|s| s.contains_mapped_rva(rva)) else {
            return false;
        };
        let characteristics = section.characteristics;
        if characteristics & SCN_MEM_READ == 0
            || characteristics & SCN_MEM_WRITE != 0
            || characteristics & SCN_CNT_UNINITIALIZED_DATA != 0
        {
            return false;
        }
    }
    true
}

fn read_readonly_image_value(pe: &PeImage<'_>, address: u64, width: BitWidth) -> Option<u64> {
    if readonly_image_range(pe, address, width.bytes()) {
        read_image_value(pe, address, width)
    } else {
        None
    }
}

fn register_widths(expr: &Expr) -> BTreeMap<String, BitWidth> {
    let mut widths = BTreeMap::new();
    expr.walk_distinct(&mut |current| {
        if let ExprKind::Register { name } = &current.kind {
            widths.entry(name.clone()).or_insert(current.width);
        }
    });
    widths
}

fn coerce_width(expr: Expr, width: BitWidth) -> Result<Expr, ExprError> {
    use std::cmp::Ordering;
    match expr.width.bits().cmp(&width.bits()) {
        Ordering::Equal => Ok(expr),
        Ordering::Greater => Expr::extract(width, 0, expr),
        Ordering::Less => Expr::resize(ResizeOp::ZeroExtend, width, expr),
    }
}

/// Classifies a (bottom-up-rewritten) load address as a read-only image table
/// load. Setting its index/base registers to zero yields the table anchor; the
/// load is kept for concretization only when that anchor is a constant that
/// lands in read-only image memory. Misclassifying a genuine input load as a
/// table (or the reverse) only forfeits completeness: the per-assignment fold
/// re-checks read-only membership at each concrete address and fails closed.
fn is_readonly_table_address(pe: &PeImage<'_>, address: &Expr, width: BitWidth) -> bool {
    let zeros: BTreeMap<String, Expr> = register_widths(address)
        .into_iter()
        .map(|(name, register_width)| (name, Expr::constant(register_width, 0)))
        .collect();
    let Ok(anchor) = address.substitute(&zeros) else {
        return false;
    };
    match anchor.simplify().kind {
        ExprKind::Const { value } => readonly_image_range(pe, value, width.bytes()),
        _ => false,
    }
}

fn is_epoch_load(name: &str, arguments: &[Expr]) -> bool {
    name.starts_with("memory_load_epoch_") && arguments.len() == 1
}

/// Flattens the maximal additive chain rooted at `expr` into its non-`Add` terms.
fn add_chain_terms<'a>(expr: &'a Expr, terms: &mut Vec<&'a Expr>) {
    if let ExprKind::Binary {
        op: BinaryOp::Add,
        left,
        right,
    } = &expr.kind
    {
        add_chain_terms(left, terms);
        add_chain_terms(right, terms);
    } else {
        terms.push(expr);
    }
}

/// True when every occurrence of register `name` is a bare-register term of a
/// maximal additive chain whose constant part is an in-image VA — the
/// grouping-insensitive signature of an image-base / relocation-delta addend
/// (added to in-image preferred-base addresses, including the image base itself,
/// to relocate them). A jump table's `image_base + table[base + index]` mixes
/// this delta into the same sum as the index, which the adjacent `reg + const`
/// form cannot see. Any non-additive use, or an additive chain whose constant is
/// not in-image, makes this false, so the delta is bound to zero only when it is
/// provably just that.
fn relocation_delta_only(expr: &Expr, name: &str, in_image: &impl Fn(u64) -> bool) -> bool {
    match &expr.kind {
        ExprKind::Const { .. } => true,
        ExprKind::Register { name: other } => other != name,
        ExprKind::Binary {
            op: BinaryOp::Add, ..
        } => {
            let mut terms = Vec::new();
            add_chain_terms(expr, &mut terms);
            let is_name = |term: &&Expr| matches!(&term.kind, ExprKind::Register { name: other } if other == name);
            let contains_name = terms.iter().any(is_name);
            // `name` must not reach the chain through any non-additive term.
            if !terms
                .iter()
                .all(|term| is_name(term) || relocation_delta_only(term, name, in_image))
            {
                return false;
            }
            if contains_name {
                let constant_sum = terms
                    .iter()
                    .filter_map(|term| match term.kind {
                        ExprKind::Const { value } => Some(value),
                        _ => None,
                    })
                    .fold(0_u64, u64::wrapping_add);
                if !in_image(constant_sum) {
                    return false;
                }
            }
            true
        }
        ExprKind::Unary { value, .. }
        | ExprKind::Load { address: value }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => relocation_delta_only(value, name, in_image),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat {
            high: left,
            low: right,
        } => {
            relocation_delta_only(left, name, in_image)
                && relocation_delta_only(right, name, in_image)
        }
        ExprKind::Intrinsic { arguments, .. } => arguments
            .iter()
            .all(|argument| relocation_delta_only(argument, name, in_image)),
    }
}

/// Binds every provable image-base / relocation-delta register in `expr` to zero
/// under the preferred-base load (see [`relocation_delta_only`]). Returns the
/// rewritten expression and whether any register was bound.
fn bind_relocation_deltas(pe: &PeImage<'_>, expr: &Expr) -> Result<(Expr, bool), ExprError> {
    let in_image = |value: u64| pe.optional.image_base.checked_rva_of(Va(value)).is_some();
    let bindings: BTreeMap<String, Expr> = register_widths(expr)
        .into_iter()
        .filter(|(name, width)| {
            *width == BitWidth::W64 && relocation_delta_only(expr, name, &in_image)
        })
        .map(|(name, width)| (name, Expr::constant(width, 0)))
        .collect();
    if bindings.is_empty() {
        return Ok((expr.clone(), false));
    }
    Ok((expr.substitute(&bindings)?, true))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImageTableBranchLimits {
    pub input_domain: InputDomainLimits,
    pub domain: SymbolicDomainLimits,
    pub concretization: ImageConcretizationLimits,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ImageTableBranchReport {
    pub schema_version: u32,
    pub index_registers: BTreeSet<String>,
    pub table_loads: usize,
    pub combinations_evaluated: usize,
    pub targets: Vec<u64>,
    pub complete: bool,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ImageTableBranchError {
    #[error("branch target reads no concretizable read-only image table")]
    NoReadOnlyTableLoad,
    #[error("expression contains more than {limit} nodes")]
    ExpressionLimit { limit: usize },
    #[error("index-domain inference refused: {0}")]
    Domain(String),
    #[error("index domain cross-product {count} exceeds limit {limit}")]
    CombinationLimit { count: usize, limit: usize },
    #[error("branch target remained symbolic after a complete index assignment")]
    NonConcrete,
    #[error("expression reconstruction failed: {0}")]
    Expression(#[from] ExprError),
}

struct TableEnumeration<'a> {
    pe: &'a PeImage<'a>,
    expression: &'a Expr,
    variables: Vec<(String, Vec<Expr>)>,
    replacements: BTreeMap<String, Expr>,
    targets: BTreeSet<u64>,
    evaluated: usize,
}

impl TableEnumeration<'_> {
    /// Substitutes one complete index assignment, folds the now-constant
    /// read-only image loads to their definitive bytes, and records the
    /// resulting target. A load that does not land in read-only image (or an
    /// expression that stays symbolic) fails the whole recovery closed.
    fn evaluate(&mut self) -> Result<(), ImageTableBranchError> {
        let pe = self.pe;
        let substituted = self.expression.substitute(&self.replacements)?;
        let folded = substituted.rewrite_intrinsics(&mut |width, name, arguments| {
            if is_epoch_load(name, arguments)
                && let ExprKind::Const { value } = arguments[0].kind
                && let Some(byte) = read_readonly_image_value(pe, value, width)
            {
                return Some(Expr::constant(width, byte));
            }
            None
        })?;
        match folded.simplify().kind {
            ExprKind::Const { value } => {
                self.targets.insert(value);
                self.evaluated += 1;
                Ok(())
            }
            _ => Err(ImageTableBranchError::NonConcrete),
        }
    }

    fn walk(&mut self, index: usize) -> Result<(), ImageTableBranchError> {
        if index == self.variables.len() {
            return self.evaluate();
        }
        let (name, values) = self.variables[index].clone();
        for value in values {
            self.replacements.insert(name.clone(), value);
            self.walk(index + 1)?;
        }
        self.replacements.remove(&name);
        Ok(())
    }
}

/// Recovers the finite set of concrete branch targets of a computed branch whose
/// next-VIP reads a read-only image jump/const table at a proven-finite index
/// (`target = f(table[index], index)`).
///
/// The address of each `memory_load_epoch_*` whose anchor lies in read-only
/// image is kept symbolic; every other load is abstracted to a free input
/// register. The input registers' finite domains are then proven by the shared
/// [`infer_branch_input_domains`] machinery (never invented here). The domain
/// cross-product is enumerated, and at each assignment the table addresses
/// become constant and are folded to their definitive read-only image bytes.
///
/// # Soundness
///
/// A load is concretized only when its concrete address provably lies in a
/// read-only, initialized, non-writable image section (so its value cannot
/// change at runtime under the preferred-base load), and only over an index
/// domain the domain machinery proved complete. Any address outside read-only
/// image, any unproven index, or any residual symbolic result fails the whole
/// recovery closed, leaving the existing behavior untouched.
///
/// # Errors
///
/// Returns an error when no read-only table load is present, when a domain
/// cannot be proven, when the cross-product is over budget, or when a complete
/// assignment leaves a symbolic target.
pub fn recover_image_table_targets(
    pe: &PeImage<'_>,
    target: &Expr,
    limits: ImageTableBranchLimits,
) -> Result<ImageTableBranchReport, ImageTableBranchError> {
    let max_nodes = limits.concretization.max_expression_nodes;
    if target.node_count_bounded(max_nodes) > max_nodes {
        return Err(ImageTableBranchError::ExpressionLimit { limit: max_nodes });
    }
    // Keep read-only image-table loads symbolic; abstract every other load to a
    // free input register (the same naming convention as `abstract_epoch_loads`).
    let mut table_loads = 0_usize;
    let abstracted = target.rewrite_intrinsics(&mut |width, name, arguments| {
        if is_epoch_load(name, arguments) {
            if is_readonly_table_address(pe, &arguments[0], width) {
                table_loads += 1;
                return None;
            }
            return Some(Expr::register(width, format!("{name}@{arguments:?}")));
        }
        None
    })?;
    if table_loads == 0 {
        return Err(ImageTableBranchError::NoReadOnlyTableLoad);
    }
    // Bind image-base / relocation-delta registers to zero under the
    // preferred-base load, leaving the table indices as the only free inputs.
    let (abstracted, _) = bind_relocation_deltas(pe, &abstracted)?;
    // Recount after shared-subexpression deduplication during the rewrite.
    let table_loads = abstracted
        .intrinsic_dependencies()
        .iter()
        .filter(|name| name.starts_with("memory_load_epoch_"))
        .count();
    if table_loads == 0 {
        return Err(ImageTableBranchError::NoReadOnlyTableLoad);
    }
    // The index registers live inside the kept table addresses; expose them to
    // the domain machinery by replacing each table load with its own address,
    // then prove each input register's finite domain.
    let probe = abstracted.rewrite_intrinsics(&mut |width, name, arguments| {
        if is_epoch_load(name, arguments) {
            return coerce_width(arguments[0].clone(), width).ok();
        }
        None
    })?;
    let inferred = infer_branch_input_domains(&probe, limits.input_domain)
        .map_err(|refusal| ImageTableBranchError::Domain(refusal.to_string()))?;
    let index_registers: BTreeSet<String> = inferred.domains.keys().cloned().collect();

    let mut combinations = 1_usize;
    for values in inferred.domains.values() {
        combinations = combinations.checked_mul(values.len()).ok_or(
            ImageTableBranchError::CombinationLimit {
                count: usize::MAX,
                limit: limits.domain.max_combinations,
            },
        )?;
        if combinations > limits.domain.max_combinations {
            return Err(ImageTableBranchError::CombinationLimit {
                count: combinations,
                limit: limits.domain.max_combinations,
            });
        }
    }

    let mut enumeration = TableEnumeration {
        pe,
        expression: &abstracted,
        variables: inferred.domains.into_iter().collect(),
        replacements: BTreeMap::new(),
        targets: BTreeSet::new(),
        evaluated: 0,
    };
    enumeration.walk(0)?;
    let targets: Vec<u64> = enumeration.targets.into_iter().collect();
    let complete = !targets.is_empty();
    Ok(ImageTableBranchReport {
        schema_version: 1,
        index_registers,
        table_loads,
        combinations_evaluated: enumeration.evaluated,
        targets,
        complete,
    })
}

#[cfg(test)]
mod table_tests {
    use super::*;
    use crate::BinaryOp;
    use crate::immutable_image::tests::{DATA_VA, RDATA_VA, two_section_pe64};
    use vmpx_pe::PeImage;

    // `.rdata` of the synthetic image holds 0xDEAD_BEEF at its start followed by
    // zeroes, so a W32 table over [0, 4) reads {0xDEAD_BEEF, 0, 0, 0}.
    fn indexed_load(base: u64) -> Expr {
        let index = Expr::register(BitWidth::W64, "switch");
        let masked =
            Expr::binary(BinaryOp::And, index, Expr::constant(BitWidth::W64, 3)).expect("mask");
        let scaled =
            Expr::binary(BinaryOp::Mul, masked, Expr::constant(BitWidth::W64, 4)).expect("scale");
        let address =
            Expr::binary(BinaryOp::Add, Expr::constant(BitWidth::W64, base), scaled).expect("addr");
        Expr::intrinsic(BitWidth::W32, "memory_load_epoch_7", vec![address])
    }

    #[test]
    fn readonly_indexed_load_enumerates_its_targets() {
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let report = recover_image_table_targets(
            &pe,
            &indexed_load(RDATA_VA),
            ImageTableBranchLimits::default(),
        )
        .expect("read-only indexed load over a proven domain resolves");
        assert!(report.complete);
        assert_eq!(report.table_loads, 1);
        assert_eq!(
            report.index_registers,
            BTreeSet::from(["switch".to_owned()])
        );
        assert_eq!(report.targets, vec![0, 0xDEAD_BEEF]);
    }

    #[test]
    fn writable_memory_load_stays_fail_closed() {
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        // `.data` is writable, so its bytes could change at runtime: no load is
        // kept and the recovery fails closed rather than concretizing.
        let error = recover_image_table_targets(
            &pe,
            &indexed_load(DATA_VA),
            ImageTableBranchLimits::default(),
        )
        .expect_err("a writable-memory load must not be concretized");
        assert_eq!(error, ImageTableBranchError::NoReadOnlyTableLoad);
    }

    #[test]
    fn unproven_index_stays_fail_closed() {
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        // A read-only load whose index is used at full width has no proven finite
        // domain, so the branch fails closed instead of guessing a range.
        let index = Expr::register(BitWidth::W64, "unbounded");
        let scaled =
            Expr::binary(BinaryOp::Mul, index, Expr::constant(BitWidth::W64, 4)).expect("scale");
        let address = Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W64, RDATA_VA),
            scaled,
        )
        .expect("addr");
        let load = Expr::intrinsic(BitWidth::W32, "memory_load_epoch_7", vec![address]);
        let error = recover_image_table_targets(&pe, &load, ImageTableBranchLimits::default())
            .expect_err("an unproven index must fail closed");
        assert!(matches!(error, ImageTableBranchError::Domain(_)));
    }
}
