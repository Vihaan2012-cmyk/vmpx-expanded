//! Cut-point handling of input-dependent native loops and the inductive prover.

use vmpx_mutation::cfg_recovery::{CfgLimits, CfgOptions, NativeCfg, recover_native_cfg_opts};
use vmpx_verify::differential::native_loops::{components, execute_native_loops, loop_cuts};
use vmpx_verify::differential::{
    ForkLimits, NativeEnv, SegmentEnd, execute_native_paths,
};

fn cfg_at(code: &[u8], base: u64, options: CfgOptions) -> NativeCfg {
    recover_native_cfg_opts(code, base, base, CfgLimits::default(), options).expect("cfg")
}

fn cfg(code: &[u8], base: u64) -> NativeCfg {
    cfg_at(code, base, CfgOptions::PROVEN)
}

/// `xor eax,eax ; L: add rax,rcx ; dec rcx ; jnz L ; ret`
const SUM_DEC: [u8; 11] = [0x31, 0xC0, 0x48, 0x01, 0xC8, 0x48, 0xFF, 0xC9, 0x75, 0xF8, 0xC3];
/// Same function: `sub rcx,1 ; jne L`.
const SUM_SUB: [u8; 12] = [0x31, 0xC0, 0x48, 0x01, 0xC8, 0x48, 0x83, 0xE9, 0x01, 0x75, 0xF7, 0xC3];
/// Same function: `lea rcx,[rcx-1] ; test rcx,rcx ; jnz L`.
const SUM_LEA: [u8; 15] =
    [0x31, 0xC0, 0x48, 0x01, 0xC8, 0x48, 0x8D, 0x49, 0xFF, 0x48, 0x85, 0xC9, 0x75, 0xF4, 0xC3];
/// Different function: adds `rdx` instead of `rcx`.
const SUM_OTHER_ADDEND: [u8; 11] = [0x31, 0xC0, 0x48, 0x01, 0xD0, 0x48, 0xFF, 0xC9, 0x75, 0xF8, 0xC3];
/// Different function: `jg` leaves the loop when the count wraps to -1.
const SUM_SIGNED_EXIT: [u8; 11] = [0x31, 0xC0, 0x48, 0x01, 0xC8, 0x48, 0xFF, 0xC9, 0x7F, 0xF8, 0xC3];
/// `L: dec rcx ; jnz L ; ret`
const COUNTDOWN: [u8; 6] = [0x48, 0xFF, 0xC9, 0x75, 0xFB, 0xC3];
/// Nested: `rax = rcx * r8` by counting.
const NESTED_DEC: [u8; 19] = [
    0x31, 0xC0, 0x48, 0x89, 0xCA, 0x48, 0xFF, 0xC0, 0x48, 0xFF, 0xCA, 0x75, 0xF8, 0x49, 0xFF, 0xC8, 0x75, 0xF0,
    0xC3,
];
const NESTED_SUB: [u8; 21] = [
    0x31, 0xC0, 0x48, 0x89, 0xCA, 0x48, 0xFF, 0xC0, 0x48, 0x83, 0xEA, 0x01, 0x75, 0xF7, 0x49, 0x83, 0xE8, 0x01,
    0x75, 0xEE, 0xC3,
];
/// A loop that stores each iteration: `L: mov [rdx],rcx ; dec rcx ; jnz L ; ret`.
const STORING: [u8; 9] = [0x48, 0x89, 0x0A, 0x48, 0xFF, 0xC9, 0x75, 0xF8, 0xC3];
/// `push rbx ; xor eax,eax ; L: add rax,[rdx] ; add rdx,8 ; dec rcx ; jnz L ; pop rbx ; ret`
const SUM_MEM: [u8; 17] = [
    0x53, 0x31, 0xC0, 0x48, 0x03, 0x02, 0x48, 0x83, 0xC2, 0x08, 0x48, 0xFF, 0xC9, 0x75, 0xF4, 0x5B, 0xC3,
];
/// Same with `lea rdx,[rdx+8]`.
const SUM_MEM_LEA: [u8; 17] = [
    0x53, 0x31, 0xC0, 0x48, 0x03, 0x02, 0x48, 0x8D, 0x52, 0x08, 0x48, 0xFF, 0xC9, 0x75, 0xF4, 0x5B, 0xC3,
];

fn loops(code: &[u8], base: u64) -> Result<vmpx_verify::differential::native_loops::NativeLoops, String> {
    execute_native_loops(&cfg_at(code, base, CfgOptions::DIFFERENTIAL), 10_000, &NativeEnv::default(), ForkLimits::DEFAULT)
}

#[test]
fn a_revisited_fork_is_named_a_loop() {
    let err = execute_native_paths(&cfg(&COUNTDOWN, 0x1000), 1000, &NativeEnv::default(), ForkLimits::DEFAULT)
        .unwrap_err();
    assert!(err.starts_with("input-dependent loop"), "{err}");
}

#[test]
fn one_cut_breaks_the_single_loop() {
    let cuts = loop_cuts(&cfg(&SUM_DEC, 0x1000)).unwrap();
    assert_eq!(cuts.into_iter().collect::<Vec<_>>(), vec![0x1002]);
}

#[test]
fn a_loop_nest_is_cut_where_every_cycle_is_broken() {
    let c = cfg(&NESTED_DEC, 0x1000);
    let cuts = loop_cuts(&c).unwrap();
    assert_eq!(cuts.len(), 1);
    assert!(cuts.contains(&0x1005));
}

#[test]
fn the_loop_is_a_finite_family_of_segments() {
    let n = loops(&SUM_DEC, 0x1000).unwrap();
    assert_eq!(n.entry.len(), 1);
    assert!(matches!(n.entry[0].end, SegmentEnd::Cut { header: 0x1002, .. }));
    let seg = &n.loops[&0x1002];
    assert_eq!(seg.phis.len(), components().len());
    assert_eq!(seg.paths.len(), 2);
    let exits = seg.paths.iter().filter(|p| matches!(p.end, SegmentEnd::Exit(_))).count();
    assert_eq!(exits, 1);
}

#[test]
fn a_loop_that_stores_fails_closed() {
    let err = loops(&STORING, 0x1000).unwrap_err();
    assert!(err.contains("stores or calls"), "{err}");
}

#[test]
fn the_nest_is_a_finite_family_too() {
    let n = loops(&NESTED_DEC, 0x1000).unwrap();
    assert_eq!(n.loops.len(), 1);
    assert_eq!(n.loops[&0x1005].paths.len(), 3);
}

#[test]
fn a_phi_pointer_loads_past_the_frame_with_the_alias_rule() {
    let n = loops(&SUM_MEM, 0x1000).unwrap();
    assert_eq!(n.loops.len(), 1);
    assert!(n.assumptions.iter().any(|a| a.contains("does not alias")), "{:?}", n.assumptions);
}

#[cfg(feature = "smt-z3")]
mod with_z3 {
    use super::*;
    use vmpx_verify::differential::AbiPolicy;
    use vmpx_verify::equiv::{EquivVerdict, Verifier};
    use vmpx_verify::loops::{PairFamily, prove_loop_equivalence};

    fn family(code: &[u8], base: u64) -> PairFamily {
        PairFamily::from_native(&loops(code, base).expect("loops")).expect("family")
    }

    fn verdict(a: &[u8], b: &[u8]) -> (EquivVerdict, String) {
        let verifier = Verifier::default().with_backend(Box::new(vmpx_verify::smt::Z3Backend));
        let out = prove_loop_equivalence(&family(a, 0x1000), &family(b, 0x2000), &verifier, &AbiPolicy::win64());
        (out.verdict, out.reason)
    }

    #[test]
    fn differently_compiled_loops_are_proven_equal() {
        assert_eq!(verdict(&SUM_DEC, &SUM_SUB), (EquivVerdict::Proven, String::new()));
        assert_eq!(verdict(&SUM_DEC, &SUM_LEA), (EquivVerdict::Proven, String::new()));
        assert_eq!(verdict(&SUM_DEC, &SUM_DEC), (EquivVerdict::Proven, String::new()));
    }

    #[test]
    fn differently_compiled_nests_are_proven_equal() {
        assert_eq!(verdict(&NESTED_DEC, &NESTED_SUB), (EquivVerdict::Proven, String::new()));
    }

    #[test]
    fn a_pointer_walk_through_memory_is_proven_equal() {
        assert_eq!(verdict(&SUM_MEM, &SUM_MEM_LEA), (EquivVerdict::Proven, String::new()));
    }

    #[test]
    fn different_loops_are_never_proven() {
        for other in [&SUM_OTHER_ADDEND[..], &SUM_SIGNED_EXIT[..], &NESTED_DEC[..]] {
            let (v, why) = verdict(&SUM_DEC, other);
            assert_eq!(v, EquivVerdict::Timeout, "{why}");
        }
        let (v, why) = verdict(&SUM_MEM, &SUM_DEC);
        assert_eq!(v, EquivVerdict::Timeout, "{why}");
    }
}
