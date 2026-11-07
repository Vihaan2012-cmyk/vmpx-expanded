//! Evidence-based annotation of recovered code.
//!
//! Devirtualized output is correct but anonymous: every statement looks alike,
//! so the reader cannot tell the protector's self-defence apart from the work
//! the function actually does. This pass labels statements with what the IR
//! *shows*, so the interesting lines announce themselves.
//!
//! Every tag is derived from a term that is actually present -- an intrinsic
//! name, a comparison shape -- and says which. Nothing here guesses at intent:
//! a label names the MECHANISM it found, not a motive it cannot see. A reader
//! can recognise a license check from "compares a checksum against a constant
//! and returns a constant"; the pass would be lying if it printed "license
//! check" on its own authority.

use vmpx_vm::{Expr, ExprKind};

use crate::ir::{Cond, Store};
use crate::opt::Stmt;

/// A kind of protector machinery, with the evidence that identified it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub enum Tag {
    /// Depends on a table-checksum recurrence over image bytes.
    Integrity,
    /// Reads the timestamp counter: timing / anti-debug.
    Timing,
    /// Reads the CPU identification leaves: environment fingerprint.
    Fingerprint,
    /// Depends on a read the recovery could not resolve.
    UntrackedRead,
    /// Depends on the result of an imported call.
    ImportCall,
    /// Depends on a freshly allocated address.
    FreshAllocation,
    /// A branch comparing a computed value against a constant, whose body
    /// returns a constant: the shape of a guard.
    ConstantGate,
    /// Reaches the network, by an import this function names.
    Network,
    /// Licensing: VMProtect's serial API, or a gate on an integrity or
    /// environment check -- the shape a licence/anti-tamper decision has.
    License,
    /// Inspects the process for a debugger.
    AntiDebug,
    /// Resolves or loads code at run time.
    DynamicLoad,
}

impl Tag {
    /// The comment text, as `LABEL -> what was found`.
    #[must_use]
    pub const fn text(self) -> &'static str {
        match self {
            Self::Integrity => "INTEGRITY -> table checksum over image bytes",
            Self::Timing => "TIMING -> reads the timestamp counter (rdtsc)",
            Self::Fingerprint => "FINGERPRINT -> reads cpu identification leaves",
            Self::UntrackedRead => "MEMORY -> reads memory the recovery could not resolve",
            Self::ImportCall => "PAYLOAD -> uses the result of an imported call",
            Self::FreshAllocation => "PAYLOAD -> uses a freshly allocated address",
            Self::ConstantGate => "GATE -> compares against a constant and returns a constant",
            Self::Network => "NETWORK -> calls an imported network function",
            Self::License => "LICENSE -> licensing api, or a gate on an integrity/environment check",
            Self::AntiDebug => "ANTI-DEBUG -> inspects the process for a debugger",
            Self::DynamicLoad => "DYNAMIC LOAD -> resolves or loads code at run time",
        }
    }
}

impl From<crate::symbols::Facility> for Tag {
    fn from(facility: crate::symbols::Facility) -> Self {
        use crate::symbols::Facility;
        match facility {
            Facility::Network => Self::Network,
            Facility::License => Self::License,
            Facility::AntiDebug => Self::AntiDebug,
            Facility::DynamicLoad => Self::DynamicLoad,
        }
    }
}

/// The tag an intrinsic name implies, if any.
fn tag_of_intrinsic(name: &str) -> Option<Tag> {
    if name.starts_with("table_checksum") {
        Some(Tag::Integrity)
    } else if name.starts_with("read_timestamp_counter") {
        Some(Tag::Timing)
    } else if name.starts_with("x86_cpuid") || name.contains("cpuid") {
        Some(Tag::Fingerprint)
    } else if name.starts_with("memory_load_epoch") || name.starts_with("memory_snapshot_epoch") {
        Some(Tag::UntrackedRead)
    } else if name.starts_with("fresh_allocation_return") {
        Some(Tag::FreshAllocation)
    } else if name.contains("call_result") || name.starts_with("native_call") {
        Some(Tag::ImportCall)
    } else {
        None
    }
}

/// Every tag the expression's terms imply, including the facilities of any
/// import whose IAT slot address appears as a constant.
fn tags_of_with(expr: &Expr, symbols: &crate::symbols::Symbols, out: &mut Vec<Tag>) {
    let mut pending = vec![expr.clone()];
    let mut seen = std::collections::HashSet::new();
    while let Some(current) = pending.pop() {
        if !seen.insert(current.clone()) {
            continue;
        }
        match &current.kind {
            ExprKind::Const { value } => {
                if let Some(facility) = symbols.facility_at(*value) {
                    out.push(facility.into());
                }
            }
            ExprKind::Register { .. } => {}
            ExprKind::Intrinsic { name, arguments } => {
                if let Some(tag) = tag_of_intrinsic(name) {
                    out.push(tag);
                }
                pending.extend(arguments.iter().cloned());
            }
            ExprKind::Load { address: value }
            | ExprKind::Unary { value, .. }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => pending.push((**value).clone()),
            ExprKind::Binary { left, right, .. }
            | ExprKind::Concat { high: left, low: right } => {
                pending.push((**left).clone());
                pending.push((**right).clone());
            }
        }
    }
}

/// Every tag the expression's terms imply.
fn tags_of(expr: &Expr, out: &mut Vec<Tag>) {
    let mut pending = vec![expr.clone()];
    let mut seen = std::collections::HashSet::new();
    while let Some(current) = pending.pop() {
        if !seen.insert(current.clone()) {
            continue;
        }
        match &current.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => {}
            ExprKind::Intrinsic { name, arguments } => {
                if let Some(tag) = tag_of_intrinsic(name) {
                    out.push(tag);
                }
                pending.extend(arguments.iter().cloned());
            }
            ExprKind::Load { address: value }
            | ExprKind::Unary { value, .. }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => pending.push((**value).clone()),
            ExprKind::Binary { left, right, .. }
            | ExprKind::Concat { high: left, low: right } => {
                pending.push((**left).clone());
                pending.push((**right).clone());
            }
        }
    }
}

fn cond_tags(cond: &Cond, out: &mut Vec<Tag>) {
    match cond {
        Cond::Cmp { a, b, .. } => {
            tags_of(a, out);
            tags_of(b, out);
        }
        Cond::Opaque { args, .. } => {
            for arg in args {
                tags_of(arg, out);
            }
        }
    }
}

fn is_constant(expr: &Expr) -> bool {
    matches!(expr.kind, ExprKind::Const { .. })
}

/// True when a branch compares something computed against a constant and the
/// body it guards returns a constant -- the shape a license or anti-tamper
/// check has, stated as the shape rather than as the motive.
fn constant_gate(cond: &Cond, body: &[Stmt]) -> bool {
    let Cond::Cmp { a, b, .. } = cond else {
        return false;
    };
    if is_constant(a) == is_constant(b) {
        return false;
    }
    body.iter().any(|stmt| match stmt {
        Stmt::Return(values) => values.iter().all(|(_, e)| is_constant(e)),
        _ => false,
    })
}

fn dedup(mut tags: Vec<Tag>) -> Vec<Tag> {
    tags.sort_unstable();
    tags.dedup();
    tags
}

/// Tags an expression carries, for annotating a definition that is emitted
/// outside the statement list (a hoisted global or shared subtree).
#[must_use]
pub fn tags_for(expr: &Expr) -> Vec<Tag> {
    tags_for_with(expr, &crate::symbols::Symbols::default())
}

/// As [`tags_for`], also resolving import slots through `symbols`.
#[must_use]
pub fn tags_for_with(expr: &Expr, symbols: &crate::symbols::Symbols) -> Vec<Tag> {
    let mut tags = Vec::new();
    tags_of_with(expr, symbols, &mut tags);
    dedup(promote_license(tags))
}

/// A gate whose decision rests on an integrity or environment check IS the
/// licence/anti-tamper shape, so it is reported as one. Stated as a promotion
/// of two tags that are both present, not as a guess about a lone comparison.
fn promote_license(mut tags: Vec<Tag>) -> Vec<Tag> {
    let gated = tags.contains(&Tag::ConstantGate);
    let checked = tags.contains(&Tag::Integrity) || tags.contains(&Tag::Fingerprint);
    if gated && checked {
        tags.push(Tag::License);
    }
    tags
}

/// Inserts a comment above every statement whose terms carry evidence.
///
/// Only the tags NEW to a statement are printed: a value derived from a
/// checksum stays derived from it for the rest of the function, and repeating
/// the label on every line would be the noise this pass exists to remove.
pub fn annotate(stmts: &mut Vec<Stmt>) {
    annotate_inner(stmts, &mut Vec::new(), &crate::symbols::Symbols::default());
}

/// As [`annotate`], but skipping tags `already` said elsewhere -- used so a
/// label printed on a hoisted definition is not repeated in the body.
pub fn annotate_with(stmts: &mut Vec<Stmt>, already: &mut Vec<Tag>) {
    annotate_inner(stmts, already, &crate::symbols::Symbols::default());
}

/// As [`annotate_with`], also resolving import slots through `symbols`.
pub fn annotate_with_symbols(
    stmts: &mut Vec<Stmt>,
    already: &mut Vec<Tag>,
    symbols: &crate::symbols::Symbols,
) {
    annotate_inner(stmts, already, symbols);
}

fn annotate_inner(
    stmts: &mut Vec<Stmt>,
    already: &mut Vec<Tag>,
    symbols: &crate::symbols::Symbols,
) {
    let mut out: Vec<Stmt> = Vec::with_capacity(stmts.len());
    for stmt in stmts.drain(..) {
        let mut tags = Vec::new();
        match &stmt {
            Stmt::Assign(_, e) => tags_of_with(e, symbols, &mut tags),
            Stmt::Store(Store { address, value }) => {
                tags_of_with(address, symbols, &mut tags);
                tags_of_with(value, symbols, &mut tags);
            }
            Stmt::Return(values) => {
                for (_, e) in values {
                    tags_of_with(e, symbols, &mut tags);
                }
            }
            Stmt::If { cond, then, els } => {
                cond_tags(cond, &mut tags);
                if constant_gate(cond, then) || constant_gate(cond, els) {
                    tags.push(Tag::ConstantGate);
                }
            }
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                cond_tags(cond, &mut tags);
                if constant_gate(cond, body) {
                    tags.push(Tag::ConstantGate);
                }
            }
            Stmt::IfGoto { cond, .. } => cond_tags(cond, &mut tags),
            _ => {}
        }
        let fresh: Vec<Tag> = dedup(promote_license(tags))
            .into_iter()
            .filter(|t| !already.contains(t))
            .collect();
        for tag in &fresh {
            out.push(Stmt::Comment(tag.text().to_owned()));
            already.push(*tag);
        }
        // Recurse into nested bodies with the tags seen so far, so a label is
        // not repeated inside a branch that already carried it outside.
        let mut stmt = stmt;
        match &mut stmt {
            Stmt::If { then, els, .. } => {
                annotate_inner(then, already, symbols);
                annotate_inner(els, already, symbols);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Loop(body) => {
                annotate_inner(body, already, symbols);
            }
            _ => {}
        }
        out.push(stmt);
    }
    *stmts = out;
}

#[cfg(test)]
mod tests {
    use vmpx_vm::{BinaryOp, BitWidth};

    use super::*;
    use crate::ir::Cmp;

    fn reg(name: &str) -> Expr {
        Expr::register(BitWidth::W32, name)
    }

    fn k(v: u64) -> Expr {
        Expr::constant(BitWidth::W32, v)
    }

    fn intrinsic(name: &str) -> Expr {
        Expr::intrinsic(BitWidth::W32, name, vec![reg("a")])
    }

    fn comments(stmts: &[Stmt]) -> Vec<String> {
        stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::Comment(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_checksum_term_is_labelled_integrity() {
        let mut stmts = vec![Stmt::Assign(
            "x".to_owned(),
            intrinsic("table_checksum_low_byte_xor_recurrence_w32"),
        )];
        annotate(&mut stmts);
        assert!(comments(&stmts).iter().any(|c| c.starts_with("INTEGRITY ->")));
    }

    #[test]
    fn a_timestamp_read_is_labelled_timing() {
        let mut stmts =
            vec![Stmt::Assign("x".to_owned(), intrinsic("read_timestamp_counter_7"))];
        annotate(&mut stmts);
        assert!(comments(&stmts).iter().any(|c| c.starts_with("TIMING ->")));
    }

    #[test]
    fn a_label_is_not_repeated_once_it_has_been_said() {
        let mut stmts = vec![
            Stmt::Assign("x".to_owned(), intrinsic("read_timestamp_counter_7")),
            Stmt::Assign("y".to_owned(), intrinsic("read_timestamp_counter_7")),
        ];
        annotate(&mut stmts);
        let timing = comments(&stmts).iter().filter(|c| c.starts_with("TIMING ->")).count();
        assert_eq!(timing, 1, "the label belongs on the first occurrence only");
    }

    #[test]
    fn a_comparison_against_a_constant_guarding_a_constant_return_is_a_gate() {
        let mut stmts = vec![Stmt::If {
            cond: Cond::Cmp { op: Cmp::Eq, a: reg("checksum"), b: k(0xdead) },
            then: vec![Stmt::Return(vec![("r".to_owned(), k(0))])],
            els: Vec::new(),
        }];
        annotate(&mut stmts);
        assert!(comments(&stmts).iter().any(|c| c.starts_with("GATE ->")));
    }

    #[test]
    fn ordinary_arithmetic_earns_no_label() {
        let mut stmts = vec![Stmt::Assign(
            "x".to_owned(),
            Expr::binary(BinaryOp::Add, reg("a"), reg("b")).expect("add"),
        )];
        annotate(&mut stmts);
        assert!(comments(&stmts).is_empty(), "a plain add is not protector machinery");
    }

    #[test]
    fn a_branch_that_does_not_return_a_constant_is_not_a_gate() {
        let mut stmts = vec![Stmt::If {
            cond: Cond::Cmp { op: Cmp::Eq, a: reg("x"), b: k(1) },
            then: vec![Stmt::Return(vec![("r".to_owned(), reg("a"))])],
            els: Vec::new(),
        }];
        annotate(&mut stmts);
        assert!(comments(&stmts).is_empty());
    }
}
