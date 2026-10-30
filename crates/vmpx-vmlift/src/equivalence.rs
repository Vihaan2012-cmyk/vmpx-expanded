//! Differential equivalence core: compares a VM routine's recovered exit
//! against a replacement's modeled exit as data, over a finite input domain.
//!
//! Anything that cannot be proven equal is a divergence; nothing is passed
//! silently.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use vmpx_vm::{Expr, NativeExitState, X64NativeRegister};

use crate::{CollectedCall, CollectedEffects, CollectedStore};

pub type Assignment = BTreeMap<String, u64>;

/// One side's complete observable outcome for a single input assignment.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExitSemantics {
    pub exit: NativeExitState,
    pub effects: CollectedEffects,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Vm,
    Replacement,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Divergence {
    Register {
        register: X64NativeRegister,
        vm: Option<Expr>,
        replacement: Option<Expr>,
    },
    Rflags {
        vm: Expr,
        replacement: Expr,
    },
    ReturnAddress {
        vm: Expr,
        replacement: Expr,
    },
    FrameSize {
        vm: u32,
        replacement: u32,
    },
    StoreMismatch {
        index: usize,
        vm: CollectedStore,
        replacement: CollectedStore,
    },
    StoreCount {
        vm: usize,
        replacement: usize,
    },
    CallMismatch {
        index: usize,
        vm: CollectedCall,
        replacement: CollectedCall,
    },
    CallCount {
        vm: usize,
        replacement: usize,
    },
    /// A side carries effects the collector could not represent, so
    /// equivalence cannot be established.
    Unproven {
        side: Side,
        reasons: Vec<String>,
    },
    /// A side's exit could not be modeled for this input at all.
    Unmodeled {
        reason: String,
    },
}

fn same_expr(a: &Expr, b: &Expr) -> bool {
    a == b || a.simplify() == b.simplify()
}

fn same_exprs(a: &[Expr], b: &[Expr]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_expr(x, y))
}

fn same_store(a: &CollectedStore, b: &CollectedStore) -> bool {
    a.width == b.width && same_expr(&a.address, &b.address) && same_expr(&a.value, &b.value)
}

fn same_call(a: &CollectedCall, b: &CollectedCall) -> bool {
    a.iat_rva == b.iat_rva
        && a.result_intrinsic == b.result_intrinsic
        && same_exprs(&a.arguments, &b.arguments)
}

/// Compares two exits field by field and returns every divergence found.
/// An empty result means the two are structurally equivalent after
/// simplification.
#[must_use]
pub fn compare_exits(vm: &ExitSemantics, replacement: &ExitSemantics) -> Vec<Divergence> {
    let mut out = Vec::new();
    for (side, semantics) in [(Side::Vm, vm), (Side::Replacement, replacement)] {
        if !semantics.effects.unsupported.is_empty() {
            out.push(Divergence::Unproven {
                side,
                reasons: semantics.effects.unsupported.clone(),
            });
        }
    }

    let (a, b) = (&vm.exit, &replacement.exit);
    let registers = a
        .registers
        .keys()
        .chain(b.registers.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    for register in registers {
        let (left, right) = (a.registers.get(&register), b.registers.get(&register));
        let equal = matches!((left, right), (Some(l), Some(r)) if same_expr(l, r));
        if !equal {
            out.push(Divergence::Register {
                register,
                vm: left.cloned(),
                replacement: right.cloned(),
            });
        }
    }
    if !same_expr(&a.rflags, &b.rflags) {
        out.push(Divergence::Rflags {
            vm: a.rflags.clone(),
            replacement: b.rflags.clone(),
        });
    }
    if !same_expr(&a.return_address, &b.return_address) {
        out.push(Divergence::ReturnAddress {
            vm: a.return_address.clone(),
            replacement: b.return_address.clone(),
        });
    }
    if a.frame_size != b.frame_size {
        out.push(Divergence::FrameSize {
            vm: a.frame_size,
            replacement: b.frame_size,
        });
    }

    let (vs, rs) = (&vm.effects.stores, &replacement.effects.stores);
    if vs.len() != rs.len() {
        out.push(Divergence::StoreCount {
            vm: vs.len(),
            replacement: rs.len(),
        });
    }
    for (index, (l, r)) in vs.iter().zip(rs).enumerate() {
        if !same_store(l, r) {
            out.push(Divergence::StoreMismatch {
                index,
                vm: l.clone(),
                replacement: r.clone(),
            });
        }
    }

    let (vc, rc) = (&vm.effects.calls, &replacement.effects.calls);
    if vc.len() != rc.len() {
        out.push(Divergence::CallCount {
            vm: vc.len(),
            replacement: rc.len(),
        });
    }
    for (index, (l, r)) in vc.iter().zip(rc).enumerate() {
        if !same_call(l, r) {
            out.push(Divergence::CallMismatch {
                index,
                vm: l.clone(),
                replacement: r.clone(),
            });
        }
    }
    out
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AssignmentDivergence {
    pub assignment: Assignment,
    pub divergence: Divergence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Equivalent,
    Divergent,
    /// No inputs were checked, so nothing was verified.
    EmptyDomain,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VerificationReport {
    pub schema_version: u32,
    pub verdict: Verdict,
    pub assignments_checked: usize,
    pub divergent_assignments: usize,
    pub divergences: Vec<AssignmentDivergence>,
}

impl VerificationReport {
    #[must_use]
    pub fn is_equivalent(&self) -> bool {
        self.verdict == Verdict::Equivalent
    }
}

/// Runs `model` for every assignment and aggregates the comparisons. The
/// model returns the VM and replacement exits, or a reason it could not
/// produce them (recorded as a divergence). An empty domain is never
/// reported as equivalent.
pub fn verify_over_domain<F>(domain: &[Assignment], mut model: F) -> VerificationReport
where
    F: FnMut(&Assignment) -> Result<(ExitSemantics, ExitSemantics), String>,
{
    let mut divergences = Vec::new();
    let mut divergent_assignments = 0;
    for assignment in domain {
        let found = match model(assignment) {
            Ok((vm, replacement)) => compare_exits(&vm, &replacement),
            Err(reason) => vec![Divergence::Unmodeled { reason }],
        };
        if !found.is_empty() {
            divergent_assignments += 1;
        }
        divergences.extend(found.into_iter().map(|divergence| AssignmentDivergence {
            assignment: assignment.clone(),
            divergence,
        }));
    }
    let verdict = if domain.is_empty() {
        Verdict::EmptyDomain
    } else if divergences.is_empty() {
        Verdict::Equivalent
    } else {
        Verdict::Divergent
    };
    VerificationReport {
        schema_version: 1,
        verdict,
        assignments_checked: domain.len(),
        divergent_assignments,
        divergences,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::{BinaryOp, BitWidth};

    fn c(v: u64) -> Expr {
        Expr::constant(BitWidth::W64, v)
    }

    fn reg(name: &str) -> Expr {
        Expr::register(BitWidth::W64, name)
    }

    fn semantics() -> ExitSemantics {
        ExitSemantics {
            exit: NativeExitState {
                schema_version: 1,
                registers: X64NativeRegister::ALL
                    .into_iter()
                    .map(|r| (r, c(0)))
                    .collect(),
                rflags: c(0x246),
                return_address: c(0x1000),
                frame_size: 136,
            },
            effects: CollectedEffects {
                stores: vec![
                    CollectedStore {
                        address: c(0x10),
                        value: c(1),
                        width: BitWidth::W64,
                    },
                    CollectedStore {
                        address: c(0x18),
                        value: c(2),
                        width: BitWidth::W64,
                    },
                ],
                calls: vec![CollectedCall {
                    result_intrinsic: "call0".to_owned(),
                    iat_rva: 0x2000,
                    arguments: vec![c(5)],
                }],
                unsupported: Vec::new(),
            },
        }
    }

    #[test]
    fn identical_exits_are_equivalent() {
        assert!(compare_exits(&semantics(), &semantics()).is_empty());
    }

    #[test]
    fn differing_register_is_named() {
        let mut r = semantics();
        r.exit.registers.insert(X64NativeRegister::Rcx, c(9));
        let d = compare_exits(&semantics(), &r);
        assert_eq!(d.len(), 1);
        assert!(matches!(
            &d[0],
            Divergence::Register {
                register: X64NativeRegister::Rcx,
                ..
            }
        ));
    }

    #[test]
    fn missing_register_is_divergence() {
        let mut r = semantics();
        r.exit.registers.remove(&X64NativeRegister::R9);
        let d = compare_exits(&semantics(), &r);
        assert!(matches!(
            &d[0],
            Divergence::Register {
                register: X64NativeRegister::R9,
                replacement: None,
                ..
            }
        ));
    }

    #[test]
    fn equal_after_simplification_is_equivalent() {
        let mut v = semantics();
        let mut r = semantics();
        v.exit.registers.insert(X64NativeRegister::Rax, reg("x"));
        let plus_zero = Expr::binary(BinaryOp::Add, reg("x"), c(0)).expect("add");
        r.exit.registers.insert(X64NativeRegister::Rax, plus_zero);
        assert!(compare_exits(&v, &r).is_empty());
    }

    #[test]
    fn differing_rflags_is_divergence() {
        let mut r = semantics();
        r.exit.rflags = c(0x247);
        let d = compare_exits(&semantics(), &r);
        assert!(matches!(&d[..], [Divergence::Rflags { .. }]));
    }

    #[test]
    fn store_value_and_order_diverge() {
        let mut r = semantics();
        r.effects.stores[1].value = c(3);
        let d = compare_exits(&semantics(), &r);
        assert!(matches!(
            &d[..],
            [Divergence::StoreMismatch { index: 1, .. }]
        ));

        let mut swapped = semantics();
        swapped.effects.stores.swap(0, 1);
        let d = compare_exits(&semantics(), &swapped);
        assert_eq!(d.len(), 2);
        assert!(
            d.iter()
                .all(|x| matches!(x, Divergence::StoreMismatch { .. }))
        );
    }

    #[test]
    fn store_count_and_call_diffs_diverge() {
        let mut r = semantics();
        r.effects.stores.pop();
        r.effects.calls[0].arguments = vec![c(6)];
        let d = compare_exits(&semantics(), &r);
        assert!(d.iter().any(|x| matches!(
            x,
            Divergence::StoreCount {
                vm: 2,
                replacement: 1
            }
        )));
        assert!(
            d.iter()
                .any(|x| matches!(x, Divergence::CallMismatch { index: 0, .. }))
        );
        r.effects.calls.clear();
        let d = compare_exits(&semantics(), &r);
        assert!(d.iter().any(|x| matches!(x, Divergence::CallCount { .. })));
    }

    #[test]
    fn unsupported_effects_are_never_passed() {
        let mut r = semantics();
        r.effects.unsupported.push("opaque call".to_owned());
        let d = compare_exits(&semantics(), &r);
        assert!(matches!(
            &d[..],
            [Divergence::Unproven {
                side: Side::Replacement,
                ..
            }]
        ));
    }

    #[test]
    fn return_address_and_frame_size_diverge() {
        let mut r = semantics();
        r.exit.return_address = c(0x2000);
        r.exit.frame_size = 144;
        let d = compare_exits(&semantics(), &r);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn domain_aggregation() {
        let domain = (0..3_u64)
            .map(|i| Assignment::from([("in".to_owned(), i)]))
            .collect::<Vec<_>>();
        let ok = verify_over_domain(&domain, |_| Ok((semantics(), semantics())));
        assert!(ok.is_equivalent());
        assert_eq!(ok.assignments_checked, 3);

        let bad = verify_over_domain(&domain, |a| {
            let mut r = semantics();
            if a["in"] == 1 {
                r.exit.registers.insert(X64NativeRegister::Rax, c(7));
            }
            Ok((semantics(), r))
        });
        assert_eq!(bad.verdict, Verdict::Divergent);
        assert_eq!(bad.divergent_assignments, 1);
        assert_eq!(bad.divergences[0].assignment["in"], 1);

        let unmodeled = verify_over_domain(&domain[..1], |_| Err("no model".to_owned()));
        assert_eq!(unmodeled.verdict, Verdict::Divergent);
    }

    #[test]
    fn empty_domain_is_not_equivalent() {
        let report = verify_over_domain(&[], |_| Ok((semantics(), semantics())));
        assert_eq!(report.verdict, Verdict::EmptyDomain);
        assert!(!report.is_equivalent());
    }
}
