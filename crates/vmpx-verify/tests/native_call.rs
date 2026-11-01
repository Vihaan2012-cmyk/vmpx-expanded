//! Win64 call modeling in the native executor, and composition of a native SSE
//! segment between two VM regions.

use std::collections::BTreeMap;

use vmpx_disasm::{Gpr, lift_bytes_with};
use vmpx_mutation::cfg_recovery::{CfgLimits, CfgOptions, VmAlgebra, recover_native_cfg_opts};
use vmpx_verify::differential::{
    NativeEnv, apply_native_segment, compose_into_next, execute_native_function_in, xmm_entry_symbol,
};
use vmpx_vm::{BitWidth, Expr, ExprKind, NativeExitState, X64NativeRegister};
use vmpx_vmlift::{CollectedEffects, ExitSemantics};

const IMAGE: u64 = 0x1_8000_0000;
const BASE: u64 = IMAGE + 0x1000;

fn cfg_of(code: &[u8]) -> vmpx_mutation::NativeCfg {
    recover_native_cfg_opts(code, BASE, BASE, CfgLimits::default(), CfgOptions::PROVEN_FP)
        .expect("cfg")
}

fn env() -> NativeEnv {
    NativeEnv { image_base: IMAGE, readonly: vec![(IMAGE + 0x2000, IMAGE + 0x3000)] }
}

fn name_of(e: &Expr) -> String {
    match &e.simplify().kind {
        ExprKind::Intrinsic { name, .. } => name.clone(),
        other => panic!("not an intrinsic: {other:?}"),
    }
}

// addsd xmm0,xmm1 ; call qword ptr [rip+0x100] ; mulsd xmm0,xmm0 ; ret
const IMPORT_CALL: [u8; 15] = [
    0xF2, 0x0F, 0x58, 0xC1, 0xFF, 0x15, 0x00, 0x01, 0x00, 0x00, 0xF2, 0x0F, 0x59, 0xC0, 0xC3,
];

#[test]
fn import_call_follows_the_win64_abi() {
    let cfg = cfg_of(&IMPORT_CALL);
    let f = execute_native_function_in(&cfg, 10, &env()).expect("executes");
    let calls = &f.semantics.effects.calls;
    assert_eq!(calls.len(), 1);
    // slot = (BASE + 10) + 0x100, as an RVA
    assert_eq!(u64::from(calls[0].iat_rva), BASE + 10 + 0x100 - IMAGE);
    assert_eq!(calls[0].arguments.len(), 8, "rcx rdx r8 r9 + xmm0-3");
    // The first fp argument is xmm0 after the addsd.
    let ExprKind::Intrinsic { name, .. } = &calls[0].arguments[4].kind else {
        panic!("xmm0 argument is not the addsd result: {:?}", calls[0].arguments[4]);
    };
    assert_eq!(name, "fp_add_f64");
    let xmm0 = &f.semantics.exit.xmm[&0].0;
    assert_eq!(name_of(xmm0), "fp_mul_f64");
    let rax = &f.semantics.exit.registers[&X64NativeRegister::Rax];
    assert!(name_of(rax).ends_with("_rax_f64"));
    // Callee-saved state survives.
    assert_eq!(
        f.semantics.exit.registers[&X64NativeRegister::Rbx],
        Expr::register(BitWidth::W64, "native_rbx_initial")
    );
    assert_eq!(f.semantics.exit.xmm[&6].0, xmm_entry_symbol(6, false));
    assert!(!f.assumptions.is_empty());
}

#[test]
fn call_needs_the_image_base() {
    let cfg = cfg_of(&IMPORT_CALL);
    let err = execute_native_function_in(&cfg, 10, &NativeEnv::default()).unwrap_err();
    assert!(err.contains("image base"), "{err}");
}

#[test]
fn direct_call_is_keyed_by_target() {
    // call rel32 +0 (to the next instruction's end) ; ret
    let code = [0xE8, 0x10, 0x00, 0x00, 0x00, 0xC3];
    let cfg = cfg_of(&code);
    let f = execute_native_function_in(&cfg, 10, &env()).expect("executes");
    let call = &f.semantics.effects.calls[0];
    assert_eq!(u64::from(call.iat_rva & 0x7FFF_FFFF), BASE + 5 + 0x10 - IMAGE);
    assert_ne!(call.iat_rva & 0x8000_0000, 0, "direct calls carry the high bit");
}

#[test]
fn volatile_registers_are_clobbered_by_the_call() {
    // mov rcx, rbx ; call [rip+0x100] ; mov rax, rcx ; ret
    let code = [0x48, 0x89, 0xD9, 0xFF, 0x15, 0x00, 0x01, 0x00, 0x00, 0x48, 0x89, 0xC8, 0xC3];
    let cfg = cfg_of(&code);
    let f = execute_native_function_in(&cfg, 10, &env()).expect("executes");
    let rax = &f.semantics.exit.registers[&X64NativeRegister::Rax];
    assert!(name_of(rax).contains("_rcx_f64"), "{rax:?}");
    // The call saw rcx = rbx.
    assert_eq!(
        f.semantics.effects.calls[0].arguments[0].simplify(),
        Expr::register(BitWidth::W64, "native_rbx_initial")
    );
}

/// `[call [rip+0x100]] ; mov eax, [rip+disp] ; ret` loading from `target`.
fn call_then_load(target: u64) -> Vec<u8> {
    let next_ip = BASE + 12;
    let disp = i32::try_from(i128::from(target) - i128::from(next_ip)).unwrap();
    let mut c = vec![0xFF, 0x15, 0x00, 0x01, 0x00, 0x00, 0x8B, 0x05];
    c.extend_from_slice(&disp.to_le_bytes());
    c.push(0xC3);
    c
}

#[test]
fn loads_after_a_call_fail_closed_unless_still_the_entry_image() {
    let writable = cfg_of(&call_then_load(IMAGE + 0x5000));
    let err = execute_native_function_in(&writable, 10, &env()).unwrap_err();
    assert!(err.contains("after a call"), "{err}");
    let readonly = cfg_of(&call_then_load(IMAGE + 0x2800));
    assert!(execute_native_function_in(&readonly, 10, &env()).is_ok());
}

#[test]
fn without_a_call_loads_are_unchanged() {
    // mov eax, [rip+disp] ; ret  (writable target, no call before it)
    let next_ip = BASE + 6;
    let disp = i32::try_from(i128::from(IMAGE + 0x5000) - i128::from(next_ip)).unwrap();
    let mut code = vec![0x8B, 0x05];
    code.extend_from_slice(&disp.to_le_bytes());
    code.push(0xC3);
    let cfg = cfg_of(&code);
    assert!(execute_native_function_in(&cfg, 10, &env()).is_ok());
}

fn identity_state() -> ExitSemantics {
    ExitSemantics {
        exit: NativeExitState {
            schema_version: 1,
            registers: X64NativeRegister::ALL
                .iter()
                .map(|r| {
                    let g = vmpx_verify::differential::gpr_of(*r);
                    (*r, Expr::register(BitWidth::W64, format!("native_{}_initial", g.name())))
                })
                .collect(),
            xmm: BTreeMap::new(),
            rflags: Expr::register(BitWidth::W64, "native_rflags_initial"),
            return_address: Expr::constant(BitWidth::W64, 0),
            frame_size: 0,
        },
        effects: CollectedEffects { stores: vec![], calls: vec![], unsupported: vec![] },
    }
}

fn segment(code: &[u8]) -> vmpx_disasm::BlockSemantics<Expr> {
    lift_bytes_with(&VmAlgebra, code, BASE, CfgOptions::PROVEN_FP.into()).expect("lift")
}

#[test]
fn a_native_segment_composes_between_vm_regions() {
    // mulsd xmm1, xmm0
    let after = apply_native_segment(&identity_state(), &segment(&[0xF2, 0x0F, 0x59, 0xC8])).unwrap();
    assert_eq!(name_of(&after.exit.xmm[&1].0), "fp_mul_f64");
    assert_eq!(after.exit.xmm[&0].0, xmm_entry_symbol(0, false));
    assert_eq!(after.exit.xmm[&2].0, xmm_entry_symbol(2, false));
    // The next VM region (identity over its own entry symbols) sees that state.
    let next = identity_state();
    let composed = compose_into_next(&after, &next).unwrap();
    assert_eq!(composed.exit.xmm[&1], after.exit.xmm[&1]);
}

#[test]
fn ucomisd_flags_reach_the_next_region_through_rflags() {
    // ucomisd xmm0, xmm1
    let after = apply_native_segment(&identity_state(), &segment(&[0x66, 0x0F, 0x2E, 0xC1])).unwrap();
    let deps = after.exit.rflags.dependencies();
    assert!(deps.contains("native_rflags_initial"), "untouched RFLAGS bits are kept");
    assert!(after.exit.rflags.intrinsic_dependencies().iter().any(|n| n == "fp_cmp_unord_f64"));
}

#[test]
fn unsafe_segments_are_refused() {
    // add rsp, 8  (stack not balanced)
    let err = apply_native_segment(&identity_state(), &segment(&[0x48, 0x83, 0xC4, 0x08])).unwrap_err();
    assert!(err.contains("stack pointer"), "{err}");
    // mov [rbx], rax  (memory write)
    let err = apply_native_segment(&identity_state(), &segment(&[0x48, 0x89, 0x03])).unwrap_err();
    assert!(err.contains("writes memory"), "{err}");
    // mul rcx (flags clobbered)
    let err = apply_native_segment(&identity_state(), &segment(&[0x48, 0xF7, 0xE1])).unwrap_err();
    assert!(err.contains("clobbers flag"), "{err}");
}

#[test]
fn gpr_inputs_are_taken_from_the_previous_state() {
    // mov rax, rcx
    let mut prev = identity_state();
    prev.exit.registers.insert(X64NativeRegister::Rcx, Expr::constant(BitWidth::W64, 7));
    let after = apply_native_segment(&prev, &segment(&[0x48, 0x89, 0xC8])).unwrap();
    assert_eq!(after.exit.registers[&X64NativeRegister::Rax], Expr::constant(BitWidth::W64, 7));
    let _ = Gpr::Rax;
}
