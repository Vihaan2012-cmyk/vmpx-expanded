//! A recovered exit is lowered, optimized and read back as semantics; the gate
//! then proves it equal to the recovered exit, or refutes a corrupted one with
//! a counterexample that reproduces.

use std::collections::BTreeMap;

use vmpx_recompile::PhysicalRegister as P;
use vmpx_verify::equiv::{CellId, CellVerdict, EquivVerdict, Obligation, Verifier};
use vmpx_vm::{BinaryOp, BitWidth, Expr, NativeExitState, X64NativeRegister as R};
use vmpx_vmlift::{
    CollectedEffects, CollectedStore, ExitSemantics, RecompiledPair, ReplacementConfig,
    recompiled_pair,
};

const W: BitWidth = BitWidth::W64;
const CONTINUATION: u64 = 0x1_4000_1000;

fn reg(r: R) -> Expr {
    Expr::register(W, r.symbolic_name())
}

fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr::binary(op, l, r).unwrap()
}

fn k(v: u64) -> Expr {
    Expr::constant(W, v)
}

fn physical(r: R) -> P {
    match r {
        R::Rax => P::Rax,
        R::Rbx => P::Rbx,
        R::Rcx => P::Rcx,
        R::Rdx => P::Rdx,
        R::Rbp => P::Rbp,
        R::Rsi => P::Rsi,
        R::Rdi => P::Rdi,
        R::R8 => P::R8,
        R::R9 => P::R9,
        R::R10 => P::R10,
        R::R11 => P::R11,
        R::R12 => P::R12,
        R::R13 => P::R13,
        R::R14 => P::R14,
        R::R15 => P::R15,
    }
}

/// rax = (rax + rcx) ^ `mask`, rdx = rdx * 3, everything else kept.
fn recovered(mask: u64) -> ExitSemantics {
    let mut registers: BTreeMap<R, Expr> = R::ALL.into_iter().map(|r| (r, reg(r))).collect();
    registers.insert(
        R::Rax,
        bin(BinaryOp::Xor, bin(BinaryOp::Add, reg(R::Rax), reg(R::Rcx)), k(mask)),
    );
    registers.insert(R::Rdx, bin(BinaryOp::Mul, reg(R::Rdx), k(3)));
    ExitSemantics {
        exit: NativeExitState {
            schema_version: 1,
            registers,
            rflags: k(0x246),
            return_address: k(CONTINUATION),
            frame_size: 136,
        },
        effects: CollectedEffects {
            stores: Vec::new(),
            calls: Vec::new(),
            unsupported: Vec::new(),
        },
    }
}

fn with_store(mut vm: ExitSemantics) -> ExitSemantics {
    vm.effects.stores.push(CollectedStore {
        address: bin(BinaryOp::Add, reg(R::Rbx), k(8)),
        value: reg(R::Rsi),
        width: W,
    });
    vm
}

fn config(vm: &ExitSemantics, restore_flags: bool) -> ReplacementConfig {
    ReplacementConfig {
        inputs: R::ALL
            .into_iter()
            .map(|r| (r.symbolic_name().to_owned(), physical(r)))
            .collect(),
        max_expression_nodes: 10_000,
        max_spill_slots: 64,
        import_calls: BTreeMap::new(),
        continuation: Some(CONTINUATION),
        effects: vm
            .effects
            .stores
            .iter()
            .map(|s| vmpx_vmlift::EffectStore {
                address: s.address.clone(),
                value: s.value.clone(),
                width: s.width,
            })
            .collect(),
        effect_calls: Vec::new(),
        rflags: restore_flags.then(|| vm.exit.rflags.clone()),
        symbol_rewrites: BTreeMap::new(),
    }
}

fn pair(vm: &ExitSemantics, optimize: bool) -> RecompiledPair {
    pair_with(vm, optimize, false)
}

fn pair_with(vm: &ExitSemantics, optimize: bool, restore_flags: bool) -> RecompiledPair {
    let mut readback =
        recompiled_pair(vm, &config(vm, restore_flags), optimize).expect("readback");
    assert_eq!(readback.pairs.len(), 1);
    readback.pairs.remove(0)
}

fn obligation(vm: &ExitSemantics, recompiled: &ExitSemantics) -> Obligation {
    let mut o = Obligation::from_exits(vm, recompiled);
    o.cells.retain(|c| c.id != CellId::Rflags);
    o
}

#[test]
fn correctly_recovered_function_is_proven() {
    for optimize in [false, true] {
        let good = recovered(0x1234);
        let p = pair(&good, optimize);
        assert!(p.return_address_attested && !p.rflags_attested);
        let report = Verifier::default().check_function(&[obligation(&p.vm, &p.recompiled)], true);
        let bad: Vec<_> = report.paths[0]
            .cells
            .iter()
            .filter(|(_, v)| !matches!(v, CellVerdict::Proven(_)))
            .collect();
        assert_eq!(report.verdict, EquivVerdict::Proven, "optimize={optimize}: {bad:#?}");
        assert!(
            report.paths[0]
                .cells
                .iter()
                .all(|(_, v)| matches!(v, CellVerdict::Proven(_)))
        );
    }
}

#[test]
fn corrupted_recovery_is_refuted_with_a_reproducing_counterexample() {
    let truth = pair(&recovered(0x1234), true);
    let corrupted = pair(&recovered(0x1235), true);
    let report = Verifier::default().check_function(
        &[obligation(&truth.vm, &corrupted.recompiled)],
        true,
    );
    assert_eq!(report.verdict, EquivVerdict::Refuted);

    let (cell, cex) = report.paths[0].first_counterexample().expect("counterexample");
    assert_eq!(cell, &CellId::Register(R::Rax));
    assert_ne!(cex.vm, cex.recovered);
    let env: BTreeMap<String, Expr> = cex
        .assignment
        .iter()
        .map(|(name, value)| (name.clone(), k(*value)))
        .collect();
    let eval = |e: &Expr| e.substitute(&env).unwrap().simplify();
    let vm = &truth.vm.exit.registers[&R::Rax];
    let rec = &corrupted.recompiled.exit.registers[&R::Rax];
    assert_eq!(eval(vm), k(cex.vm));
    assert_eq!(eval(rec), k(cex.recovered));
}

#[test]
fn a_dropped_store_is_never_proven() {
    let truth = pair(&with_store(recovered(0x1234)), true);
    let mut dropped = truth.recompiled.clone();
    dropped.effects.stores.clear();
    let report = Verifier::default().check_function(&[obligation(&truth.vm, &dropped)], true);
    assert_ne!(report.verdict, EquivVerdict::Proven);
}

/// Regression guard for two real miscompiles the gate once found in the
/// closed-path `build_replacement`: the effect store lowered its address and
/// value into the same scratch register (storing `[value] <- value`), and the
/// flag value was materialised in a scratch register after the exit registers
/// were loaded (clobbering r10). Both are now fixed (distinct scratch/frame
/// slots), so the gate proves this lowering; it must never refute these cells again.
#[test]
fn gate_proves_fixed_closed_path_lowering() {
    let vm = with_store(recovered(0x1234));
    let p = pair_with(&vm, false, true);
    let mut o = Obligation::from_exits(&p.vm, &p.recompiled);
    o.cells.retain(|c| c.id != CellId::ReturnAddress);
    let report = Verifier::default().check_function(&[o], true);
    assert_eq!(report.verdict, EquivVerdict::Proven);
    let refuted: Vec<&CellId> = report.paths[0]
        .cells
        .iter()
        .filter(|(_, v)| matches!(v, CellVerdict::Refuted(_)))
        .map(|(id, _)| id)
        .collect();
    assert!(refuted.is_empty(), "no cell should refute now: {refuted:?}");
}
