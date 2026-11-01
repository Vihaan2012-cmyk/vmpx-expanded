//! The ground-truth differential on a tiny synthetic pair: a native "original"
//! (a real x86 loop, lifted and executed by the native path) against a
//! protected-style recovered function (an unrolled exit state over the same
//! entry symbols). Equivalent pairs prove; a deliberately divergent one refutes
//! with a counterexample that reproduces; anything unmodeled is not proven.

use std::collections::BTreeMap;

use vmpx_mutation::cfg_recovery::{CfgOptions, recover_native_cfg_opts, CfgLimits};
use vmpx_verify::differential::{AbiPolicy, diff_against_original, execute_native_function};
use vmpx_verify::equiv::{EquivVerdict, Verifier};
use vmpx_vm::{BinaryOp, BitWidth, Expr, NativeExitState, X64NativeRegister as R};
use vmpx_vmlift::{CollectedEffects, CollectedStore, ExitSemantics, UnrolledFunction, UnrolledPath};

const W: BitWidth = BitWidth::W64;
const BASE: u64 = 0x1000;

/// ```text
/// mov rax, rcx ; mov ecx, 3
/// L: add rax, 5 ; rol rax, 3 ; dec ecx ; jnz L
/// ret
/// ```
const LOOP: [u8; 24] = [
    0x48, 0x89, 0xC8, 0xB9, 0x03, 0x00, 0x00, 0x00, 0x48, 0x83, 0xC0, 0x05, 0x48, 0xC1, 0xC0,
    0x03, 0xFF, 0xC9, 0x75, 0xF4, 0xC3, 0x90, 0x90, 0x90,
];

fn reg(r: R) -> Expr {
    Expr::register(W, r.symbolic_name())
}

fn k(v: u64) -> Expr {
    Expr::constant(W, v)
}

fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr::binary(op, l, r).unwrap()
}

fn original() -> ExitSemantics {
    let cfg = recover_native_cfg_opts(&LOOP, BASE, BASE, CfgLimits::default(), CfgOptions::PROVEN)
        .expect("cfg");
    execute_native_function(&cfg, 100).expect("original executes").semantics
}

/// The recovered function: `tail` is added each round (the original adds 5).
fn recovered(tail: u64, clobber_rbx: bool) -> UnrolledFunction {
    let mut x = reg(R::Rcx);
    for _ in 0..3 {
        x = bin(BinaryOp::Rol, bin(BinaryOp::Add, x, k(tail)), k(3));
    }
    let mut registers: BTreeMap<R, Expr> = R::ALL.into_iter().map(|r| (r, reg(r))).collect();
    registers.insert(R::Rax, x);
    registers.insert(R::Rcx, k(0)); // volatile: the original leaves ecx = 0 too
    registers.insert(R::Rdx, k(0xdead)); // volatile and different: must be ignored
    if clobber_rbx {
        registers.insert(R::Rbx, k(7)); // callee-saved: must be caught
    }
    UnrolledFunction {
        paths: vec![UnrolledPath {
            conditions: Vec::new(),
            semantics: ExitSemantics {
                exit: NativeExitState {
                    schema_version: 1,
                    registers,
                    rflags: k(0),
                    return_address: k(0),
                    frame_size: 0,
                },
                effects: CollectedEffects {
                    stores: Vec::<CollectedStore>::new(),
                    calls: Vec::new(),
                    unsupported: Vec::new(),
                },
            },
            store_epochs: Vec::new(),
            rflags_is_entry_flags: true,
            return_pop: 0,
            blocks_executed: 5,
            region_exit: false,
            env_forks: 0,
        }],
        branches_decided: 2,
        forks: 0,
        spin_arms_pruned: 0,
        entry_sp_offset: 136,
    }
}

fn diff(rec: &UnrolledFunction) -> vmpx_verify::differential::DiffOutcome {
    diff_against_original(&original(), rec, &Verifier::default(), &AbiPolicy::win64(), false)
}

#[test]
fn the_loop_is_unrolled_on_the_native_side() {
    let cfg = recover_native_cfg_opts(&LOOP, BASE, BASE, CfgLimits::default(), CfgOptions::PROVEN)
        .unwrap();
    let f = execute_native_function(&cfg, 100).unwrap();
    assert_eq!(f.branches_decided, 3);
    assert_eq!(f.blocks_executed, 5);
}

#[test]
fn equivalent_recovery_is_proven_against_the_original() {
    let out = diff(&recovered(5, false));
    assert_eq!(out.verdict, EquivVerdict::Proven, "{}", out.reason);
    assert_eq!(out.paths_compared, 1);
    assert!(!out.modulo_env_gates());
}

#[test]
fn divergent_recovery_is_refuted_with_a_reproducing_counterexample() {
    let out = diff(&recovered(6, false));
    assert_eq!(out.verdict, EquivVerdict::Refuted, "{}", out.reason);
    let report = out.report.unwrap();
    let (cell, cex) = report.paths[0].first_counterexample().expect("counterexample");
    assert_eq!(format!("{cell:?}"), "Register(Rax)");
    assert_ne!(cex.vm, cex.recovered);
    // Replay: the recovered side computes cex.vm, the original cex.recovered.
    let x = cex.assignment["native_rcx_initial"];
    let step = |tail: u64| {
        let mut v = x;
        for _ in 0..3 {
            v = v.wrapping_add(tail).rotate_left(3);
        }
        v
    };
    assert_eq!(cex.vm, step(6));
    assert_eq!(cex.recovered, step(5));
}

#[test]
fn a_clobbered_callee_saved_register_is_caught() {
    let out = diff(&recovered(5, true));
    assert_eq!(out.verdict, EquivVerdict::Refuted);
}

#[test]
fn gate_failure_arms_are_reported_and_strict_mode_refuses_them() {
    let mut rec = recovered(5, false);
    let mut region = rec.paths[0].clone();
    region.region_exit = true;
    rec.paths.push(region);
    let out = diff(&rec);
    assert_eq!(out.verdict, EquivVerdict::Proven);
    assert!(out.modulo_env_gates());
    let strict = diff_against_original(
        &original(),
        &rec,
        &Verifier::default(),
        &AbiPolicy::win64(),
        true,
    );
    assert_eq!(strict.verdict, EquivVerdict::Timeout);
}

#[test]
fn an_unmodeled_recovered_store_is_never_proven() {
    let mut rec = recovered(5, false);
    rec.paths[0].semantics.effects.stores.push(CollectedStore {
        address: reg(R::Rdx),
        value: k(1),
        width: W,
    });
    rec.paths[0].store_epochs.push(1);
    let out = diff(&rec);
    assert_eq!(out.verdict, EquivVerdict::Timeout);
}

#[test]
fn an_input_dependent_original_branch_is_not_modeled() {
    // cmp rcx, 0 ; jne +1 ; ret ; ret
    let code = [0x48, 0x83, 0xF9, 0x00, 0x75, 0x01, 0xC3, 0xC3];
    let cfg = recover_native_cfg_opts(&code, BASE, BASE, CfgLimits::default(), CfgOptions::PROVEN)
        .unwrap();
    assert!(execute_native_function(&cfg, 100).is_err());
}

#[cfg(feature = "smt-z3")]
#[test]
fn z3_proves_a_differently_shaped_recovery() {
    // The recovered rotate written as shifts: not structurally equal, needs the solver.
    let mut x = reg(R::Rcx);
    for _ in 0..3 {
        let sum = bin(BinaryOp::Add, x, k(5));
        x = bin(BinaryOp::Or, bin(BinaryOp::Shl, sum.clone(), k(3)), bin(BinaryOp::Shr, sum, k(61)));
    }
    let mut rec = recovered(5, false);
    rec.paths[0].semantics.exit.registers.insert(R::Rax, x);
    let verifier = Verifier::default().with_backend(Box::new(vmpx_verify::smt::Z3Backend));
    let out = diff_against_original(&original(), &rec, &verifier, &AbiPolicy::win64(), false);
    assert_eq!(out.verdict, EquivVerdict::Proven, "{}", out.reason);
}
