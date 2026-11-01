//! Floating-point soundness gate for the differential, on synthetic f64 pairs.
//!
//! There is no native FP lifter yet, so the "original" side is the exit state a
//! lifter would produce, built by hand: the result bits in `rax` (as
//! `movq rax, xmm0` leaves them) over `native_xmm*_initial` entry symbols. The
//! comparator under test is the real one: `diff_against_original` -> `Verifier`.
//!
//! PROVEN needs a bit-exact structural/canonical match or a Z3 floating-point
//! UNSAT (`--features smt-z3`); REFUTED needs a counterexample replayed through
//! the exact evaluator; everything else, closeness included, stays TIMEOUT.

use std::collections::BTreeMap;

use vmpx_verify::differential::{AbiPolicy, DiffOutcome, diff_against_original};
use vmpx_verify::equiv::{CellVerdict, EquivVerdict, Method, Verifier};
use vmpx_vm::fp::{self, FpBinary as B, FpFormat::F64, FpUnary as U};
use vmpx_vm::{BitWidth, Expr, NativeExitState, X64NativeRegister as R};
use vmpx_vmlift::{CollectedEffects, ExitSemantics, UnrolledFunction, UnrolledPath};

fn xmm(n: u8) -> Expr {
    Expr::register(BitWidth::W64, format!("native_xmm{n}_initial"))
}
fn k(v: f64) -> Expr {
    fp::constant(F64, v)
}
fn bin(op: B, a: Expr, b: Expr) -> Expr {
    fp::binary_expr(op, F64, a, b).unwrap()
}
fn add(a: Expr, b: Expr) -> Expr {
    bin(B::Add, a, b)
}
fn sub(a: Expr, b: Expr) -> Expr {
    bin(B::Sub, a, b)
}
fn mul(a: Expr, b: Expr) -> Expr {
    bin(B::Mul, a, b)
}
fn div(a: Expr, b: Expr) -> Expr {
    bin(B::Div, a, b)
}
fn neg(a: Expr) -> Expr {
    fp::unary_expr(U::Neg, F64, a).unwrap()
}
fn sqrt(a: Expr) -> Expr {
    fp::unary_expr(U::Sqrt, F64, a).unwrap()
}
fn call(target: &str, args: Vec<Expr>) -> Expr {
    fp::call_expr(target, F64, args).unwrap()
}

fn exit(rax: Expr) -> ExitSemantics {
    let registers: BTreeMap<R, Expr> = R::ALL
        .into_iter()
        .map(|r| (r, if r == R::Rax { rax.clone() } else { Expr::register(BitWidth::W64, r.symbolic_name()) }))
        .collect();
    ExitSemantics {
        exit: NativeExitState {
            schema_version: 1,
            registers,
            rflags: Expr::constant(BitWidth::W64, 0),
            return_address: Expr::constant(BitWidth::W64, 0),
            frame_size: 0,
        },
        effects: CollectedEffects { stores: Vec::new(), calls: Vec::new(), unsupported: Vec::new() },
    }
}

fn recovered(rax: Expr) -> UnrolledFunction {
    UnrolledFunction {
        paths: vec![UnrolledPath {
            conditions: Vec::new(),
            semantics: exit(rax),
            store_epochs: Vec::new(),
            rflags_is_entry_flags: true,
            return_pop: 0,
            blocks_executed: 1,
            region_exit: false,
            env_forks: 0,
        }],
        branches_decided: 0,
        forks: 0,
        spin_arms_pruned: 0,
        entry_sp_offset: 0,
    }
}

fn run(original_rax: Expr, recovered_rax: Expr, verifier: &Verifier) -> DiffOutcome {
    diff_against_original(
        &exit(original_rax),
        &recovered(recovered_rax),
        verifier,
        &AbiPolicy::win64(),
        false,
    )
}

fn plain() -> Verifier {
    Verifier::default()
}

#[cfg(feature = "smt-z3")]
fn with_z3() -> Verifier {
    Verifier::default().with_backend(Box::new(vmpx_verify::smt::Z3Backend))
}

fn rax_method(out: &DiffOutcome) -> Method {
    let report = out.report.as_ref().expect("report");
    report.paths[0]
        .cells
        .iter()
        .find_map(|(id, v)| match (format!("{id:?}").as_str(), v) {
            ("Register(Rax)", CellVerdict::Proven(m)) => Some(m.clone()),
            _ => None,
        })
        .expect("rax proven")
}

fn rax_cex(out: &DiffOutcome) -> (u64, u64, BTreeMap<String, u64>) {
    let report = out.report.as_ref().expect("report");
    let (id, cex) = report.paths[0].first_counterexample().expect("counterexample");
    assert_eq!(format!("{id:?}"), "Register(Rax)");
    (cex.vm, cex.recovered, cex.assignment.clone())
}

// (a) structurally identical -> PROVEN with no solver at all

#[test]
fn identical_fp_trees_are_proven_structurally() {
    let f = || add(mul(xmm(0), xmm(1)), sqrt(xmm(2)));
    let out = run(f(), f(), &plain());
    assert_eq!(out.verdict, EquivVerdict::Proven, "{}", out.reason);
    assert_eq!(rax_method(&out), Method::Structural);
    assert_eq!(plain().backend_names(), vec!["native-bounded".to_owned()]);
}

#[test]
fn commuted_operands_are_proven_canonically() {
    let original = add(mul(xmm(0), xmm(1)), mul(xmm(2), k(0.5)));
    let recovered = add(mul(k(0.5), xmm(2)), mul(xmm(1), xmm(0)));
    let out = run(original, recovered, &plain());
    assert_eq!(out.verdict, EquivVerdict::Proven, "{}", out.reason);
    assert_eq!(rax_method(&out), Method::FpCanonical);
}

#[test]
fn double_negation_cancels_canonically() {
    let out = run(sub(xmm(0), xmm(1)), sub(neg(neg(xmm(0))), xmm(1)), &plain());
    assert_eq!(out.verdict, EquivVerdict::Proven, "{}", out.reason);
    assert_eq!(rax_method(&out), Method::FpCanonical);
}

#[test]
fn canonicalization_does_not_reassociate() {
    // (a+b)+c and a+(b+c) are NOT equal in IEEE; canonical equality must not claim them.
    let l = add(add(xmm(0), xmm(1)), xmm(2));
    let r = add(xmm(0), add(xmm(1), xmm(2)));
    assert_ne!(vmpx_verify::fp::canonicalize(&l), vmpx_verify::fp::canonicalize(&r));
    assert_ne!(run(l, r, &plain()).verdict, EquivVerdict::Proven);
}

// (b) equivalent but differently shaped: only the FP solver may prove these

#[test]
fn reassociated_but_equivalent_needs_the_fp_solver() {
    let pairs = [
        (mul(xmm(0), k(2.0)), add(xmm(0), xmm(0))),
        (div(xmm(0), k(4.0)), mul(xmm(0), k(0.25))),
        (sub(xmm(0), xmm(1)), add(xmm(0), neg(xmm(1)))),
    ];
    for (original, recovered) in pairs {
        // no solver: not provable structurally, and no counterexample exists -> TIMEOUT
        let out = run(original.clone(), recovered.clone(), &plain());
        assert_eq!(out.verdict, EquivVerdict::Timeout, "{}", out.reason);
        #[cfg(feature = "smt-z3")]
        {
            let out = run(original, recovered, &with_z3());
            assert_eq!(out.verdict, EquivVerdict::Proven, "{}", out.reason);
            assert_eq!(rax_method(&out), Method::FpSolver("z3".to_owned()));
        }
    }
}

// (c) libm calls: equal target + equal arguments -> equal result

#[test]
fn libm_call_with_identical_arguments_is_proven() {
    let original = mul(call("powf", vec![div(xmm(1), xmm(0)), k(0.2857)]), sqrt(xmm(2)));
    let recovered = mul(sqrt(xmm(2)), call("powf", vec![div(xmm(1), xmm(0)), k(0.2857)]));
    let out = run(original, recovered, &plain());
    assert_eq!(out.verdict, EquivVerdict::Proven, "{}", out.reason);
    assert_eq!(rax_method(&out), Method::FpCanonical);
}

#[test]
fn libm_call_with_arguments_equal_only_by_the_solver() {
    let original = call("sqrt", vec![mul(xmm(0), k(2.0))]);
    let recovered = call("sqrt", vec![add(xmm(0), xmm(0))]);
    assert_eq!(run(original.clone(), recovered.clone(), &plain()).verdict, EquivVerdict::Timeout);
    #[cfg(feature = "smt-z3")]
    {
        let out = run(original, recovered, &with_z3());
        assert_eq!(out.verdict, EquivVerdict::Proven, "{}", out.reason);
    }
    #[cfg(not(feature = "smt-z3"))]
    let _ = (original, recovered);
}

#[test]
fn libm_calls_with_different_target_or_arguments_are_not_proven() {
    let a = call("powf", vec![xmm(0), xmm(1)]);
    for other in [call("pow", vec![xmm(0), xmm(1)]), call("powf", vec![xmm(1), xmm(0)])] {
        let v = run(a.clone(), other.clone(), &plain()).verdict;
        assert_eq!(v, EquivVerdict::Timeout, "calls cannot be evaluated, so never refuted either");
        #[cfg(feature = "smt-z3")]
        assert_ne!(run(a.clone(), other, &with_z3()).verdict, EquivVerdict::Proven);
    }
}

// (d) a genuinely different FP function is refuted with a replayed counterexample

#[test]
fn different_fp_function_is_refuted_with_a_replayed_counterexample() {
    let original = mul(xmm(0), k(2.0));
    let recovered = mul(xmm(0), k(3.0));
    let out = run(original.clone(), recovered.clone(), &plain());
    assert_eq!(out.verdict, EquivVerdict::Refuted);
    let (vm, rec, assignment) = rax_cex(&out);
    let x = f64::from_bits(assignment["native_xmm0_initial"]);
    // the gate's `vm` slot holds the recovered side, `recovered` the original
    assert_eq!(f64::from_bits(vm), x * 3.0);
    assert_eq!(f64::from_bits(rec), x * 2.0);
    assert_ne!(vm, rec);
}

#[test]
fn reassociation_is_refuted_not_proven() {
    let l = add(add(xmm(0), xmm(1)), xmm(2));
    let r = add(xmm(0), add(xmm(1), xmm(2)));
    let out = run(l, r, &plain());
    assert_eq!(out.verdict, EquivVerdict::Refuted);
    let (vm, rec, a) = rax_cex(&out);
    let (x, y, z) = (
        f64::from_bits(a["native_xmm0_initial"]),
        f64::from_bits(a["native_xmm1_initial"]),
        f64::from_bits(a["native_xmm2_initial"]),
    );
    assert_eq!(vm, (x + (y + z)).to_bits());
    assert_eq!(rec, ((x + y) + z).to_bits());
}

// (e) honesty: close is not equal

#[test]
fn one_ulp_apart_is_never_proven() {
    // x * 0.1 and x / 10.0 agree to within an ulp but not bit-for-bit.
    let out = run(mul(xmm(0), k(0.1)), div(xmm(0), k(10.0)), &plain());
    assert_ne!(out.verdict, EquivVerdict::Proven);
    assert_eq!(out.verdict, EquivVerdict::Refuted);
    let (vm, rec, a) = rax_cex(&out);
    let x = f64::from_bits(a["native_xmm0_initial"]);
    assert_eq!(f64::from_bits(vm), x / 10.0);
    assert_eq!(f64::from_bits(rec), x * 0.1);
    let ulps = vm.abs_diff(rec);
    assert!(ulps > 0 && ulps <= 2, "{ulps} ulps apart: close, still not equal");
    #[cfg(feature = "smt-z3")]
    assert_ne!(run(mul(xmm(0), k(0.1)), div(xmm(0), k(10.0)), &with_z3()).verdict, EquivVerdict::Proven);
}

#[test]
fn numerically_close_through_a_libm_call_is_never_proven() {
    // pow(x, y) vs pow(x, y) * (1 + 2^-52): no libm model, so nothing can refute or prove it.
    let p = || call("pow", vec![xmm(0), xmm(1)]);
    let out = run(p(), mul(p(), k(1.0 + f64::EPSILON)), &plain());
    assert_eq!(out.verdict, EquivVerdict::Timeout);
    #[cfg(feature = "smt-z3")]
    assert_ne!(
        run(p(), mul(p(), k(1.0 + f64::EPSILON)), &with_z3()).verdict,
        EquivVerdict::Proven
    );
}

#[test]
fn a_nan_input_probe_does_not_manufacture_a_refutation() {
    // x + 0.0 vs x differ only at -0.0 (-0.0 + 0.0 = +0.0): a real difference, found and replayed.
    let out = run(add(xmm(0), k(0.0)), xmm(0), &plain());
    assert_eq!(out.verdict, EquivVerdict::Refuted);
    let (_, _, a) = rax_cex(&out);
    assert_eq!(a["native_xmm0_initial"], (-0.0_f64).to_bits());
}

// the real corpus shape: compressible subsonic orifice flow (powf / sqrt chain)

/// mdot = Cd*A*P0 * sqrt( 2g/((g-1) R T0) * (r^(2/g) - r^((g+1)/g)) ),  r = P/P0
/// xmm0 = P0, xmm1 = P, xmm2 = T0.
fn orifice(commute: bool, sub_as_add_neg: bool, gamma: f64) -> Expr {
    let (p0, p, t0) = (xmm(0), xmm(1), xmm(2));
    let r = div(p, p0.clone());
    let t1 = call("powf", vec![r.clone(), k(2.0 / gamma)]);
    let t2 = call("powf", vec![r, k((gamma + 1.0) / gamma)]);
    let spread = if sub_as_add_neg { add(t1, neg(t2)) } else { sub(t1, t2) };
    let rt = if commute { mul(t0.clone(), k(287.05)) } else { mul(k(287.05), t0) };
    let scale = div(k(2.0 * gamma / (gamma - 1.0)), rt);
    let root = sqrt(mul(scale, spread));
    let cda = k(0.62 * 0.0123);
    if commute { mul(root, mul(p0, cda)) } else { mul(mul(cda, p0), root) }
}

#[test]
fn orifice_flow_commuted_is_proven_without_a_solver() {
    let out = run(orifice(false, false, 1.4), orifice(true, false, 1.4), &plain());
    assert_eq!(out.verdict, EquivVerdict::Proven, "{}", out.reason);
    assert_eq!(rax_method(&out), Method::FpCanonical);
}

#[test]
fn orifice_flow_with_different_gamma_is_not_proven() {
    let out = run(orifice(false, false, 1.4), orifice(true, false, 1.3), &plain());
    assert_ne!(out.verdict, EquivVerdict::Proven);
}

#[test]
fn orifice_flow_sub_vs_add_neg_needs_the_solver() {
    let out = run(orifice(false, false, 1.4), orifice(false, true, 1.4), &plain());
    assert_eq!(out.verdict, EquivVerdict::Timeout);
    #[cfg(feature = "smt-z3")]
    {
        let out = run(orifice(false, false, 1.4), orifice(false, true, 1.4), &with_z3());
        assert_eq!(out.verdict, EquivVerdict::Proven, "{}", out.reason);
        assert_eq!(rax_method(&out), Method::FpSolver("z3".to_owned()));
    }
}

#[test]
fn nan_only_divergence_is_neither_proven_nor_refuted() {
    // sub vs add-of-neg differ only in the sign of a NaN result: not a value
    // difference, so no counterexample; and not bit-identical, so no proof.
    let out = run(sub(xmm(0), xmm(1)), add(xmm(0), neg(xmm(1))), &plain());
    assert_eq!(out.verdict, EquivVerdict::Timeout, "{}", out.reason);
}
