use std::collections::BTreeSet;

use vmpx_pe::Rva;

/// Outcome of deciding the original entry point from continuation candidates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OepDecision {
    pub recovered_entry: Option<Rva>,
    pub assessment: &'static str,
    pub evidence: Vec<String>,
}

/// Decides the original entry point from the header entry's recovered paths.
/// `paths` holds one entry per recovered path: its single-resolved continuation
/// RVA, or `None` when the path has no native exit or no unique concrete target.
/// Recovery is strict: the path family must be proven complete, every path must
/// resolve to a native continuation, and all of them must agree on one RVA.
#[must_use]
pub fn decide_oep(
    paths: &[Option<Rva>],
    family_complete: bool,
    is_native: impl Fn(Rva) -> bool,
) -> OepDecision {
    decide_oep_with_returns(paths, &[], family_complete, is_native)
}

/// As [`decide_oep`], with the paths that RETURN to their caller marked.
///
/// A path whose VM exit is an exception unwind back through the original native
/// frame never transfers into original code, so it contributes no entry-point
/// candidate. Counting it as a path that failed to resolve one is a category
/// error, and it is the reason this returned "unresolved" once every header
/// path finally decoded: two of the four unwind, by design.
///
/// `returns_to_caller` is positional against `paths`; a shorter slice leaves the
/// remaining paths treated as transfers. Everything else is unchanged -- the
/// surviving paths must all agree on one RVA and it must be native -- so the
/// failure mode is declining to recover an entry point, not inventing one.
#[must_use]
pub fn decide_oep_with_returns(
    paths: &[Option<Rva>],
    returns_to_caller: &[bool],
    family_complete: bool,
    is_native: impl Fn(Rva) -> bool,
) -> OepDecision {
    let returning = |index: usize| returns_to_caller.get(index).copied().unwrap_or(false);
    let transfers: Vec<Option<Rva>> = paths
        .iter()
        .enumerate()
        .filter(|(index, _)| !returning(*index))
        .map(|(_, target)| *target)
        .collect();
    let returned = paths.len() - transfers.len();
    if returned != 0 && !transfers.is_empty() {
        let mut decision = decide_oep_inner(&transfers, family_complete, is_native);
        decision.evidence.push(format!(
            "{returned} path(s) unwind to the caller and supply no entry-point candidate"
        ));
        return decision;
    }
    decide_oep_inner(paths, family_complete, is_native)
}

fn decide_oep_inner(
    paths: &[Option<Rva>],
    family_complete: bool,
    is_native: impl Fn(Rva) -> bool,
) -> OepDecision {
    let unresolved = |mut evidence: Vec<String>, reason: String| {
        evidence.push(reason);
        OepDecision {
            recovered_entry: None,
            assessment: "unresolved",
            evidence,
        }
    };
    let mut evidence = vec![format!("{} recovered header path(s)", paths.len())];
    if !family_complete {
        return unresolved(
            evidence,
            "the header path family is not proven complete".to_owned(),
        );
    }
    if paths.is_empty() {
        return unresolved(evidence, "no header path was recovered".to_owned());
    }
    let missing = paths.iter().filter(|path| path.is_none()).count();
    if missing != 0 {
        return unresolved(
            evidence,
            format!("{missing} path(s) have no unique concrete continuation target"),
        );
    }
    let distinct: BTreeSet<Rva> = paths.iter().flatten().copied().collect();
    if let Some(rva) = distinct.iter().find(|rva| !is_native(**rva)) {
        return unresolved(
            evidence,
            format!(
                "continuation {rva} is not native (non-executable, protector section, VM entry or header entry)"
            ),
        );
    }
    let mut distinct = distinct.into_iter();
    match (distinct.next(), distinct.next()) {
        (Some(only), None) => {
            evidence.push(format!(
                "all paths continue at native {only}; candidate original entry (native continuation)"
            ));
            OepDecision {
                recovered_entry: Some(only),
                assessment: "recovered_native_continuation",
                evidence,
            }
        }
        _ => {
            evidence.push("paths continue at more than one native RVA".to_owned());
            OepDecision {
                recovered_entry: None,
                assessment: "ambiguous",
                evidence,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Option<Rva> = Some(Rva(0x4000));
    const B: Option<Rva> = Some(Rva(0x5000));
    const VMP: Option<Rva> = Some(Rva(0x9000));

    fn native_below_9000(rva: Rva) -> bool {
        rva.0 < 0x9000
    }

    #[test]
    fn all_paths_agreeing_on_one_native_rva_is_the_oep() {
        let d = decide_oep(&[A, A, A], true, native_below_9000);
        assert_eq!(d.recovered_entry, A);
        assert_eq!(d.assessment, "recovered_native_continuation");
        assert!(!d.evidence.is_empty());
    }

    #[test]
    fn a_failed_path_among_resolved_ones_vetoes() {
        let d = decide_oep(&[A, None, A], true, native_below_9000);
        assert_eq!(d.recovered_entry, None);
        assert_eq!(d.assessment, "unresolved");
    }

    #[test]
    fn a_native_and_non_native_mix_vetoes() {
        let d = decide_oep(&[A, VMP], true, native_below_9000);
        assert_eq!(d.recovered_entry, None);
        assert_eq!(d.assessment, "unresolved");
        assert!(d.evidence.iter().any(|line| line.contains("9000")));
    }

    #[test]
    fn a_lone_non_native_target_is_not_oep() {
        assert_eq!(
            decide_oep(&[VMP], true, native_below_9000).recovered_entry,
            None
        );
    }

    #[test]
    fn distinct_native_targets_are_ambiguous() {
        let d = decide_oep(&[A, B], true, |_| true);
        assert_eq!(d.recovered_entry, None);
        assert_eq!(d.assessment, "ambiguous");
    }

    #[test]
    fn no_paths_is_unresolved() {
        let d = decide_oep(&[], true, |_| true);
        assert_eq!(d.recovered_entry, None);
        assert_eq!(d.assessment, "unresolved");
    }

    #[test]
    fn an_incomplete_family_never_recovers() {
        let d = decide_oep(&[A, A], false, |_| true);
        assert_eq!(d.recovered_entry, None);
        assert_eq!(d.assessment, "unresolved");
    }
}
