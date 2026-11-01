//! Input-dependent branches: the original forks into a path family, the recovered
//! function carries its own `selector == successor` conditions, and every pair
//! is compared under the joint condition. Bounds fail closed.

use std::collections::BTreeMap;

use vmpx_mutation::cfg_recovery::{CfgLimits, CfgOptions, NativeCfg, recover_native_cfg_opts};
use vmpx_verify::branch_paths::diff_against_original_paths;
use vmpx_verify::differential::{
    AbiPolicy, ForkLimits, NativeEnv, NativePath, execute_native_function, execute_native_paths,
};
use vmpx_verify::equiv::{EquivVerdict, Verifier};
use vmpx_vm::{BinaryOp, BitWidth, Expr, NativeExitState, X64NativeRegister as R};
use vmpx_vmlift::{
    CollectedEffects, ExitSemantics, PathCondition, UnrolledFunction, UnrolledPath,
};

const W: BitWidth = BitWidth::W64;
const BASE: u64 = 0x1000;

/// `cmp rcx,5 ; jb L ; mov eax,7 ; ret ; L: mov eax,9 ; ret`
const CLAMP: [u8; 18] = [
    0x48, 0x83, 0xF9, 0x05, 0x72, 0x06, 0xB8, 0x07, 0x00, 0x00, 0x00, 0xC3, 0xB8, 0x09, 0x00, 0x00,
    0x00, 0xC3,
];

/// `L: dec rcx ; jnz L ; ret`: loops until the input reaches zero.
const COUNTDOWN: [u8; 6] = [0x48, 0xFF, 0xC9, 0x75, 0xFB, 0xC3];

fn cfg(code: &[u8]) -> NativeCfg {
    recover_native_cfg_opts(code, BASE, BASE, CfgLimits::default(), CfgOptions::PROVEN).expect("cfg")
}

fn reg(r: R) -> Expr {
    Expr::register(W, r.symbolic_name())
}

fn k(v: u64) -> Expr {
    Expr::constant(W, v)
}

fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr::binary(op, l, r).unwrap()
}

fn paths(code: &[u8]) -> Result<Vec<NativePath>, String> {
    execute_native_paths(&cfg(code), 1000, &NativeEnv::default(), ForkLimits::DEFAULT)
}

/// The VM's view of `rcx < 5`: CF of `vm_flags_sub_w64(rcx, 5)`, as a selector.
fn selector() -> Expr {
    let flags = Expr::intrinsic(W, "vm_flags_sub_w64", vec![reg(R::Rcx), k(5)]);
    bin(BinaryOp::And, flags, k(1))
}

fn recovered_path(rax: u64, selector_equals_one: bool) -> UnrolledPath {
    let mut registers: BTreeMap<R, Expr> = R::ALL.into_iter().map(|r| (r, reg(r))).collect();
    registers.insert(R::Rax, k(rax));
    UnrolledPath {
        conditions: vec![PathCondition { selector: selector(), value: 1, equal: selector_equals_one }],
        semantics: ExitSemantics {
            exit: NativeExitState {
                xmm: BTreeMap::new(),
                schema_version: 1,
                registers,
                rflags: k(0),
                return_address: k(0),
                frame_size: 0,
            },
            effects: CollectedEffects { stores: Vec::new(), calls: Vec::new(), unsupported: Vec::new() },
        },
        store_epochs: Vec::new(),
        rflags_is_entry_flags: true,
        return_pop: 0,
        blocks_executed: 3,
        region_exit: false,
        region: None,
        env_forks: 0,
    }
}

fn recovered(paths: Vec<UnrolledPath>) -> UnrolledFunction {
    UnrolledFunction {
        paths,
        branches_decided: 0,
        forks: 1,
        spin_arms_pruned: 0,
        entry_sp_offset: 136,
        notes: Vec::new(),
    }
}

#[test]
fn an_input_dependent_branch_forks_into_a_partition() {
    let family = paths(&CLAMP).expect("forks");
    assert_eq!(family.len(), 2);
    assert!(family.iter().all(|p| p.literals.len() == 1));
    // The two leaves carry opposite polarities of the same predicate.
    assert_eq!(family[0].literals[0].0, family[1].literals[0].0);
    assert_ne!(family[0].literals[0].1, family[1].literals[0].1);
}

#[test]
fn the_legacy_entry_point_still_refuses_an_input_dependent_branch() {
    let err = execute_native_function(&cfg(&CLAMP), 100).unwrap_err();
    assert!(err.ends_with("depends on the function's inputs"), "{err}");
}

#[test]
fn an_input_dependent_loop_exceeds_the_fork_depth_and_fails_closed() {
    let err = paths(&COUNTDOWN).unwrap_err();
    assert!(err.contains("fork depth") || err.contains("more than"), "{err}");
}

#[test]
fn the_path_budget_fails_closed() {
    let limits = ForkLimits { max_forks_per_path: 10, max_paths: 1 };
    let err = execute_native_paths(&cfg(&CLAMP), 1000, &NativeEnv::default(), limits).unwrap_err();
    assert!(err.contains("more than 1 paths"), "{err}");
}

#[cfg(feature = "smt-z3")]
mod with_z3 {
    use super::*;

    fn diff(rec: &UnrolledFunction) -> vmpx_verify::differential::DiffOutcome {
        let verifier = Verifier::default().with_backend(Box::new(vmpx_verify::smt::Z3Backend));
        diff_against_original_paths(&paths(&CLAMP).unwrap(), rec, &verifier, &AbiPolicy::win64(), false)
    }

    // The leaf order is the fork order: the taken arm (`rcx < 5`, rax = 9) runs first.
    #[test]
    fn matching_paths_under_their_conditions_are_proven() {
        let out = diff(&recovered(vec![recovered_path(9, true), recovered_path(7, false)]));
        assert_eq!(out.verdict, EquivVerdict::Proven, "{}", out.reason);
        assert_eq!(out.paths_compared, 2);
    }

    #[test]
    fn a_swapped_arm_is_refuted() {
        let out = diff(&recovered(vec![recovered_path(7, true), recovered_path(9, false)]));
        assert_eq!(out.verdict, EquivVerdict::Refuted, "{}", out.reason);
    }

    #[test]
    fn a_wrong_value_on_one_arm_is_refuted() {
        let out = diff(&recovered(vec![recovered_path(9, true), recovered_path(8, false)]));
        assert_eq!(out.verdict, EquivVerdict::Refuted, "{}", out.reason);
    }

    #[test]
    fn a_recovered_family_missing_an_arm_is_never_proven() {
        let out = diff(&recovered(vec![recovered_path(9, true)]));
        assert_eq!(out.verdict, EquivVerdict::Timeout, "{}", out.reason);
        assert!(out.reason.contains("incomplete"), "{}", out.reason);
    }
}
