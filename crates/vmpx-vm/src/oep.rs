use std::collections::BTreeSet;

use vmpx_pe::Rva;

/// Outcome of deciding the original entry point from continuation candidates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OepDecision {
    pub recovered_entry: Option<Rva>,
    pub assessment: &'static str,
    pub evidence: Vec<String>,
}

/// Decides the original entry point from the single-resolved continuation RVAs
/// of the header entry's recovered paths. Only a unique native continuation is
/// accepted; anything else leaves the entry unrecovered.
#[must_use]
pub fn decide_oep(candidates: &[Rva], is_native: impl Fn(Rva) -> bool) -> OepDecision {
    let distinct: BTreeSet<Rva> = candidates.iter().copied().collect();
    let mut evidence = vec![format!(
        "{} continuation candidate(s), {} distinct",
        candidates.len(),
        distinct.len()
    )];
    let mut native = Vec::new();
    for rva in distinct {
        if is_native(rva) {
            native.push(rva);
        } else {
            evidence.push(format!(
                "candidate {rva} is not native (protector section or VM entry) and is excluded"
            ));
        }
    }
    match native.as_slice() {
        [only] => {
            evidence.push(format!(
                "the single native continuation {only} is the original entry point"
            ));
            OepDecision {
                recovered_entry: Some(*only),
                assessment: "recovered_native_continuation",
                evidence,
            }
        }
        [] => {
            evidence.push("no native continuation was resolved".to_owned());
            OepDecision {
                recovered_entry: None,
                assessment: "unresolved",
                evidence,
            }
        }
        many => {
            evidence.push(format!(
                "{} distinct native continuations ({}); the entry point is ambiguous",
                many.len(),
                many.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
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

    #[test]
    fn single_native_continuation_is_the_oep() {
        let d = decide_oep(&[Rva(0x4000)], |r| r == Rva(0x4000));
        assert_eq!(d.recovered_entry, Some(Rva(0x4000)));
        assert_eq!(d.assessment, "recovered_native_continuation");
        assert!(!d.evidence.is_empty());
    }

    #[test]
    fn a_vmp_or_ambiguous_target_is_not_oep() {
        assert_eq!(decide_oep(&[Rva(0x9000)], |_| false).recovered_entry, None);
        assert_eq!(
            decide_oep(&[Rva(0x4000), Rva(0x5000)], |_| true).recovered_entry,
            None
        );
        assert_eq!(decide_oep(&[], |_| true).recovered_entry, None);
    }

    #[test]
    fn assessments_distinguish_unresolved_from_ambiguous() {
        assert_eq!(decide_oep(&[], |_| true).assessment, "unresolved");
        assert_eq!(
            decide_oep(&[Rva(0x9000)], |_| false).assessment,
            "unresolved"
        );
        assert_eq!(
            decide_oep(&[Rva(0x4000), Rva(0x5000)], |_| true).assessment,
            "ambiguous"
        );
    }

    #[test]
    fn duplicate_paths_to_one_native_target_still_recover() {
        let d = decide_oep(&[Rva(0x4000), Rva(0x4000), Rva(0x4000)], |_| true);
        assert_eq!(d.recovered_entry, Some(Rva(0x4000)));
    }

    #[test]
    fn a_vmp_candidate_does_not_make_a_native_one_ambiguous_but_is_recorded() {
        let d = decide_oep(&[Rva(0x4000), Rva(0x9000)], |r| r == Rva(0x4000));
        assert_eq!(d.recovered_entry, Some(Rva(0x4000)));
        assert!(d.evidence.iter().any(|line| line.contains("9000")));
    }
}
