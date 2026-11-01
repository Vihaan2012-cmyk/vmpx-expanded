//! Computed `jmp`/`call` targets in the native executor: a data-derived pointer is
//! an uninterpreted call, a jump-table fetch or intra-function dispatch is refused.

use vmpx_mutation::cfg_recovery::{CfgLimits, CfgOptions, Terminator, recover_native_cfg_opts};
use vmpx_verify::differential::{INDIRECT_CALL_RVA, NativeEnv, execute_native_function_in};
use vmpx_vm::{BinaryOp, BitWidth, Expr};

const IMAGE: u64 = 0x1_8000_0000;
const BASE: u64 = IMAGE + 0x1000;

fn cfg_with(code: &[u8], options: CfgOptions) -> vmpx_mutation::NativeCfg {
    recover_native_cfg_opts(code, BASE, BASE, CfgLimits::default(), options).expect("cfg")
}

fn env() -> NativeEnv {
    NativeEnv { image_base: IMAGE, readonly: vec![(IMAGE + 0x2000, IMAGE + 0x3000)] }
}

fn run(code: &[u8]) -> Result<vmpx_verify::differential::NativeFunction, String> {
    execute_native_function_in(&cfg_with(code, CfgOptions::PROVEN_FP), 10, &env())
}

fn entry(name: &str) -> Expr {
    Expr::register(BitWidth::W64, name)
}

fn c64(v: u64) -> Expr {
    Expr::constant(BitWidth::W64, v)
}

// mov rax,[rcx] ; mov rcx,[rcx+8] ; mov r9,[rcx+0x18] ; mov rcx,rax ; jmp r9
const VTABLE_TAIL: [u8; 16] = [
    0x48, 0x8B, 0x01, 0x48, 0x8B, 0x49, 0x08, 0x4C, 0x8B, 0x49, 0x18, 0x48, 0x89, 0xC1, 0x41, 0xFF,
];

fn vtable_tail() -> Vec<u8> {
    let mut c = VTABLE_TAIL.to_vec();
    c.push(0xE1);
    c
}

#[test]
fn vtable_tail_call_is_an_indirect_call_on_the_loaded_target() {
    let f = run(&vtable_tail()).expect("executes");
    let calls = &f.semantics.effects.calls;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].iat_rva, INDIRECT_CALL_RVA);
    assert_eq!(calls[0].arguments.len(), 9, "rcx rdx r8 r9 + xmm0-3 + target");
    // rcx was overwritten with the first loaded word.
    assert_eq!(calls[0].arguments[0].simplify(), Expr::load(BitWidth::W64, entry("native_rcx_initial")));
    let inner = Expr::load(
        BitWidth::W64,
        Expr::binary(BinaryOp::Add, entry("native_rcx_initial"), c64(8)).unwrap(),
    );
    let target = Expr::load(BitWidth::W64, Expr::binary(BinaryOp::Add, inner, c64(0x18)).unwrap());
    assert_eq!(calls[0].arguments[8].simplify(), target.simplify());
    assert!(f.assumptions.iter().any(|a| a.contains("indirect call target")));
}

#[test]
fn the_computed_target_distinguishes_otherwise_identical_calls() {
    // Same shape, different vtable slot (+0x20): the target argument differs.
    let mut other = vtable_tail();
    other[10] = 0x20;
    let a = run(&vtable_tail()).expect("a");
    let b = run(&other).expect("b");
    assert_ne!(
        a.semantics.effects.calls[0].arguments[8].simplify(),
        b.semantics.effects.calls[0].arguments[8].simplify()
    );
}

#[test]
fn import_tail_call_is_keyed_by_its_slot() {
    // jmp qword ptr [rip+0x100]
    let f = run(&[0xFF, 0x25, 0x00, 0x01, 0x00, 0x00]).expect("executes");
    let call = &f.semantics.effects.calls[0];
    assert_eq!(u64::from(call.iat_rva), BASE + 6 + 0x100 - IMAGE);
    assert_eq!(call.arguments.len(), 8);
    assert!(matches!(
        cfg_with(&[0xFF, 0x25, 0x00, 0x01, 0x00, 0x00], CfgOptions::PROVEN_FP).blocks[&BASE].terminator,
        Terminator::TailCall { .. }
    ));
}

#[test]
fn register_call_mid_function() {
    // sub rsp,0x28 ; call rax ; add rsp,0x28 ; ret
    let code = [0x48, 0x83, 0xEC, 0x28, 0xFF, 0xD0, 0x48, 0x83, 0xC4, 0x28, 0xC3];
    let f = run(&code).expect("executes");
    let call = &f.semantics.effects.calls[0];
    assert_eq!(call.iat_rva, INDIRECT_CALL_RVA);
    assert_eq!(call.arguments[8].simplify(), entry("native_rax_initial"));
}

#[test]
fn memory_operand_call_followed_by_code_fails_closed() {
    // sub rsp,0x28 ; call qword ptr [r8+0x28] ; add rsp,0x28 ; ret
    // The loaded target is an argument of the opaque result the next block uses, and
    // loads after a call are only accepted from read-only image bytes.
    let code = [0x48, 0x83, 0xEC, 0x28, 0x41, 0xFF, 0x50, 0x28, 0x48, 0x83, 0xC4, 0x28, 0xC3];
    let err = run(&code).unwrap_err();
    assert!(err.contains("after a call"), "{err}");
}

#[test]
fn jump_table_fetch_fails_closed() {
    // lea rdx,[rip+0x1000] ; movsxd r8,[rdx+r8*4] ; add r8,rdx ; jmp r8
    let code = [
        0x48, 0x8D, 0x15, 0x00, 0x10, 0x00, 0x00, 0x4E, 0x63, 0x04, 0x82, 0x49, 0x01, 0xD0, 0x41,
        0xFF, 0xE0,
    ];
    let err = run(&code).unwrap_err();
    assert!(err.contains("jump table"), "{err}");
}

#[test]
fn tail_jump_with_the_frame_allocated_fails_closed() {
    // sub rsp,8 ; jmp rax
    let err = run(&[0x48, 0x83, 0xEC, 0x08, 0xFF, 0xE0]).unwrap_err();
    assert!(err.contains("frame still allocated"), "{err}");
}

#[test]
fn a_load_that_may_alias_an_earlier_store_fails_closed() {
    // mov [rdx],rbx ; call qword ptr [rcx] ; ret
    let err = run(&[0x48, 0x89, 0x1A, 0xFF, 0x11, 0xC3]).unwrap_err();
    assert!(err.contains("alias"), "{err}");
}

#[test]
fn segment_relative_target_is_unsupported() {
    // jmp qword ptr gs:[rax]
    let cfg = cfg_with(&[0x65, 0xFF, 0x20], CfgOptions::PROVEN_FP);
    assert!(matches!(cfg.blocks[&BASE].terminator, Terminator::Unsupported(_)));
}

#[test]
fn consumers_that_do_not_model_calls_still_get_unsupported() {
    for code in [vtable_tail(), vec![0xFF, 0x25, 0, 0, 0, 0], vec![0xFF, 0x11, 0xC3]] {
        let cfg = cfg_with(&code, CfgOptions::PROVEN);
        assert!(matches!(cfg.blocks[&BASE].terminator, Terminator::Unsupported(_)), "{code:x?}");
    }
}

#[test]
fn far_jump_is_unsupported() {
    // jmp far qword ptr [rax]  (REX.W FF /5)
    let cfg = cfg_with(&[0x48, 0xFF, 0x28], CfgOptions::PROVEN_FP);
    assert!(matches!(cfg.blocks[&BASE].terminator, Terminator::Unsupported(_)));
}
