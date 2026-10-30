//! Core `--neutralize` transform: identifies recovered anti-debug /
//! anti-tamper guards and, when explicitly enabled, rewrites them to their
//! non-detecting path. This is a deliberately non-equivalent transform (it
//! changes observable behavior), so it only ever acts when asked, reports
//! every change it makes, and leaves anything it cannot confidently classify
//! untouched.
//!
//! Recognition is grounded in two intrinsic shapes the recovered-semantics
//! pipeline already produces:
//! - the table-checksum recurrence intrinsic (`semantic_exec::checksum_recurrence_intrinsic`),
//!   the recovered value of an integrity checksum;
//! - the RDTSC intrinsic family (`VmHandlerSemantic::ReadTimestampCounter`
//!   lowering), whose repeated use in one condition is the recovered shape of
//!   a timing/single-step check.
//! A third shape — a native call's opaque result correlated with its resolved
//! import name (e.g. `IsDebuggerPresent`) — is supported through a caller-
//! supplied provenance map, since no in-tree pass yet stitches an import name
//! onto the call-result intrinsic itself (see the SP5 report's
//! deferred-to-integration notes).

use std::collections::BTreeMap;

use serde::Serialize;
use vmpx_pe::Rva;

use crate::{BitWidth, Expr};

/// Canonical name of the in-tree table-checksum recurrence intrinsic.
pub const INTEGRITY_CHECKSUM_INTRINSIC: &str = "table_checksum_low_byte_xor_recurrence_w32";

/// Prefix of the in-tree RDTSC intrinsic family.
pub const TIMESTAMP_COUNTER_INTRINSIC_PREFIX: &str = "read_timestamp_counter_";

/// Minimum number of distinct RDTSC reads a single condition must depend on
/// before it is treated as a timing/single-step check rather than an
/// unrelated use of the cycle counter.
const MIN_TIMESTAMP_READS_FOR_TIMING_CHECK: usize = 2;

/// Imported APIs whose direct, unmodified call result is itself a
/// debugger-presence boolean. Kept short and exact: a fuzzy match here would
/// risk silently neutralizing an unrelated check.
const DEBUGGER_PRESENCE_IMPORTS: &[&str] = &["IsDebuggerPresent", "CheckRemoteDebuggerPresent"];

/// Which outcome of a guard's condition is the "detected" disposition
/// (debugger present / integrity mismatch) rather than the clean
/// continuation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectedWhen {
    /// "Detected" is taken when `condition` evaluates to nonzero.
    NonZero,
    /// "Detected" is taken when `condition` evaluates to zero.
    Zero,
}

/// One recovered conditional guard a routine uses to pick between a
/// "detected" disposition and a clean continuation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RecoveredGuard {
    pub site: Rva,
    pub condition: Expr,
    pub detected_when: DetectedWhen,
}

/// The anti-debug/anti-tamper construct a guard was recognized as.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NeutralizeConstruct {
    /// The condition depends on the recovered table-checksum recurrence.
    IntegrityChecksum,
    /// The condition depends on two or more distinct RDTSC reads.
    DebuggerPresenceTiming,
    /// The condition depends on a recognized debugger-detection API's call
    /// result, via the caller-supplied provenance map.
    DebuggerPresenceImport { import: String },
}

/// One change `neutralize` made, so the caller can report exactly what was
/// altered.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NeutralizeAction {
    pub site: Rva,
    pub construct: NeutralizeConstruct,
    pub evidence: Vec<String>,
}

/// Opt-in switch for the transform. `enabled: false` is the faithful,
/// equivalence-preserving mode and must return the input unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NeutralizeOptions {
    pub enabled: bool,
}

fn classify(
    condition: &Expr,
    imports: &BTreeMap<String, String>,
) -> Option<(NeutralizeConstruct, Vec<String>)> {
    let intrinsics = condition.intrinsic_dependencies();
    if intrinsics.contains(INTEGRITY_CHECKSUM_INTRINSIC) {
        return Some((
            NeutralizeConstruct::IntegrityChecksum,
            vec![format!(
                "condition depends on intrinsic `{INTEGRITY_CHECKSUM_INTRINSIC}`"
            )],
        ));
    }
    let timestamp_reads = intrinsics
        .iter()
        .filter(|name| name.starts_with(TIMESTAMP_COUNTER_INTRINSIC_PREFIX))
        .count();
    if timestamp_reads >= MIN_TIMESTAMP_READS_FOR_TIMING_CHECK {
        return Some((
            NeutralizeConstruct::DebuggerPresenceTiming,
            vec![format!(
                "condition depends on {timestamp_reads} distinct `{TIMESTAMP_COUNTER_INTRINSIC_PREFIX}*` reads"
            )],
        ));
    }
    for name in &intrinsics {
        if let Some(import) = imports.get(name)
            && DEBUGGER_PRESENCE_IMPORTS
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(import))
        {
            return Some((
                NeutralizeConstruct::DebuggerPresenceImport {
                    import: import.clone(),
                },
                vec![format!(
                    "condition depends on intrinsic `{name}`, resolved to import `{import}`"
                )],
            ));
        }
    }
    None
}

/// The constant that forces a guard's non-detecting path, given which
/// outcome was "detected".
fn pass_path_condition(width: BitWidth, detected_when: DetectedWhen) -> Expr {
    match detected_when {
        DetectedWhen::NonZero => Expr::constant(width, 0),
        DetectedWhen::Zero => Expr::constant(width, width.mask()),
    }
}

/// Core anti-debug/anti-tamper neutralize transform.
///
/// With `opts.enabled == false` (faithful mode) this returns the input
/// unchanged and reports no actions — it never alters behavior silently.
/// With it enabled, each guard whose condition is recognized as a
/// debugger-presence or integrity-checksum construct is rewritten to force
/// its non-detecting path and reported in the returned action list. A guard
/// this function cannot confidently classify is left exactly as given.
#[must_use]
pub fn neutralize(
    guards: &[RecoveredGuard],
    imports: &BTreeMap<String, String>,
    opts: NeutralizeOptions,
) -> (Vec<RecoveredGuard>, Vec<NeutralizeAction>) {
    if !opts.enabled {
        return (guards.to_vec(), Vec::new());
    }
    let mut rewritten = Vec::with_capacity(guards.len());
    let mut actions = Vec::new();
    for guard in guards {
        match classify(&guard.condition, imports) {
            Some((construct, evidence)) => {
                rewritten.push(RecoveredGuard {
                    site: guard.site,
                    condition: pass_path_condition(guard.condition.width, guard.detected_when),
                    detected_when: guard.detected_when,
                });
                actions.push(NeutralizeAction {
                    site: guard.site,
                    construct,
                    evidence,
                });
            }
            None => rewritten.push(guard.clone()),
        }
    }
    (rewritten, actions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BinaryOp;

    #[test]
    fn rewrites_integrity_checksum_guard_to_pass_path_and_reports_it() {
        let checksum = Expr::intrinsic(
            BitWidth::W32,
            INTEGRITY_CHECKSUM_INTRINSIC,
            vec![Expr::register(BitWidth::W32, "accumulator")],
        );
        let condition = Expr::binary(BinaryOp::Xor, checksum, Expr::constant(BitWidth::W32, 0xDEAD_BEEF))
            .expect("same width");
        let guard = RecoveredGuard {
            site: Rva(0x1000),
            condition,
            detected_when: DetectedWhen::NonZero,
        };
        let (rewritten, actions) =
            neutralize(&[guard], &BTreeMap::new(), NeutralizeOptions { enabled: true });
        assert_eq!(rewritten.len(), 1);
        assert_eq!(rewritten[0].condition, Expr::constant(BitWidth::W32, 0));
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].site, Rva(0x1000));
        assert_eq!(actions[0].construct, NeutralizeConstruct::IntegrityChecksum);
    }

    #[test]
    fn rewrites_debugger_presence_import_guard_to_pass_path_and_reports_it() {
        let condition = Expr::intrinsic(BitWidth::W32, "native_call_return_epoch_7", Vec::new());
        let mut imports = BTreeMap::new();
        imports.insert(
            "native_call_return_epoch_7".to_owned(),
            "IsDebuggerPresent".to_owned(),
        );
        let guard = RecoveredGuard {
            site: Rva(0x2000),
            condition,
            detected_when: DetectedWhen::NonZero,
        };
        let (rewritten, actions) =
            neutralize(&[guard], &imports, NeutralizeOptions { enabled: true });
        assert_eq!(rewritten[0].condition, Expr::constant(BitWidth::W32, 0));
        assert_eq!(
            actions[0].construct,
            NeutralizeConstruct::DebuggerPresenceImport {
                import: "IsDebuggerPresent".to_owned()
            }
        );
    }

    #[test]
    fn rewrites_timing_check_guard_with_two_distinct_rdtsc_reads() {
        let first = Expr::intrinsic(BitWidth::W64, "read_timestamp_counter_3", Vec::new());
        let second = Expr::intrinsic(BitWidth::W64, "read_timestamp_counter_9", Vec::new());
        let condition = Expr::binary(BinaryOp::Sub, second, first).expect("same width");
        let guard = RecoveredGuard {
            site: Rva(0x5000),
            condition,
            detected_when: DetectedWhen::Zero,
        };
        let (rewritten, actions) =
            neutralize(&[guard], &BTreeMap::new(), NeutralizeOptions { enabled: true });
        assert_eq!(
            rewritten[0].condition,
            Expr::constant(BitWidth::W64, u64::MAX)
        );
        assert_eq!(
            actions[0].construct,
            NeutralizeConstruct::DebuggerPresenceTiming
        );
    }

    #[test]
    fn leaves_unrelated_guard_untouched() {
        let condition = Expr::binary(
            BinaryOp::Sub,
            Expr::register(BitWidth::W32, "counter"),
            Expr::constant(BitWidth::W32, 1),
        )
        .expect("same width");
        let guard = RecoveredGuard {
            site: Rva(0x3000),
            condition,
            detected_when: DetectedWhen::Zero,
        };
        let (rewritten, actions) = neutralize(
            &[guard.clone()],
            &BTreeMap::new(),
            NeutralizeOptions { enabled: true },
        );
        assert_eq!(rewritten, vec![guard]);
        assert!(actions.is_empty());
    }

    #[test]
    fn faithful_mode_returns_input_unchanged_even_for_a_recognizable_guard() {
        let checksum = Expr::intrinsic(BitWidth::W32, INTEGRITY_CHECKSUM_INTRINSIC, Vec::new());
        let guard = RecoveredGuard {
            site: Rva(0x4000),
            condition: checksum,
            detected_when: DetectedWhen::NonZero,
        };
        let (rewritten, actions) = neutralize(
            &[guard.clone()],
            &BTreeMap::new(),
            NeutralizeOptions::default(),
        );
        assert_eq!(rewritten, vec![guard]);
        assert!(actions.is_empty());
    }

    #[test]
    fn action_list_matches_exactly_the_guards_that_were_rewritten() {
        let checksum_guard = RecoveredGuard {
            site: Rva(0x100),
            condition: Expr::intrinsic(BitWidth::W32, INTEGRITY_CHECKSUM_INTRINSIC, Vec::new()),
            detected_when: DetectedWhen::NonZero,
        };
        let unrelated_guard = RecoveredGuard {
            site: Rva(0x200),
            condition: Expr::constant(BitWidth::W32, 1),
            detected_when: DetectedWhen::Zero,
        };
        let import_guard = RecoveredGuard {
            site: Rva(0x300),
            condition: Expr::intrinsic(BitWidth::W32, "native_call_return_epoch_1", Vec::new()),
            detected_when: DetectedWhen::Zero,
        };
        let mut imports = BTreeMap::new();
        imports.insert(
            "native_call_return_epoch_1".to_owned(),
            "CheckRemoteDebuggerPresent".to_owned(),
        );
        let (rewritten, actions) = neutralize(
            &[
                checksum_guard.clone(),
                unrelated_guard.clone(),
                import_guard.clone(),
            ],
            &imports,
            NeutralizeOptions { enabled: true },
        );
        assert_eq!(rewritten[0].condition, Expr::constant(BitWidth::W32, 0));
        assert_eq!(rewritten[1], unrelated_guard);
        assert_eq!(
            rewritten[2].condition,
            Expr::constant(BitWidth::W32, u64::from(u32::MAX))
        );
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].site, Rva(0x100));
        assert_eq!(actions[1].site, Rva(0x300));
    }
}
