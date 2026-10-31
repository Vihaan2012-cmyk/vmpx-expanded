//! SP5 differential-execution harness checks against the real unitops.dll:
//! same-function agreement, cross-function divergence, and that an incomplete
//! run is surfaced as incomplete (never silently counted as agreement).

use vmpx_emulate::{Limits, Outcome, Target, differential, input_set, run_function};
use vmpx_pe::PeImage;

const UNITOPS: &[u8] = include_bytes!("E:/VMPX/testcorpus/corpus/unitops.dll");

fn pe() -> PeImage<'static> {
    PeImage::parse(UNITOPS).expect("parse unitops")
}

#[test]
fn same_function_full_agreement() {
    let pe = pe();
    let inputs = input_set(2, 64, 0x5EED);
    let add: Target = "op_add32".into();
    let report = differential(&pe, &add, &pe, &add, &inputs, &Limits::default()).expect("diff");

    assert!(report.all_agreed(), "op_add32 vs itself: {}", report.summary());
    assert_eq!(report.agreed, report.total);
    assert_eq!(report.diverged, 0);
    assert_eq!(report.incomplete, 0);
    assert!(report.total > 100, "edge+random set should be sizeable");
    eprintln!("same-function: {}", report.summary());
}

#[test]
fn different_functions_diverge_at_specific_input() {
    const M: u64 = 0xffff_ffff;
    let pe = pe();
    let inputs = input_set(2, 64, 0x5EED);
    let add: Target = "op_add32".into();
    let sub: Target = "op_sub32".into();
    let report = differential(&pe, &add, &pe, &sub, &inputs, &Limits::default()).expect("diff");

    assert!(!report.all_agreed(), "add vs sub must diverge");
    assert!(report.diverged > 0);
    assert_eq!(report.incomplete, 0, "both are fully modeled, nothing should stop early");

    // The first divergence names a concrete input with a + b != a - b (32-bit).
    let d = report.first_divergence.as_ref().expect("a divergence");
    let (a, b) = (d.input[0] & M, d.input[1] & M);
    assert_ne!((a + b) & M, a.wrapping_sub(b) & M, "divergence input must actually differ");
    assert_eq!(d.rax_a & M, (a + b) & M);
    assert_eq!(d.rax_b & M, a.wrapping_sub(b) & M);
    eprintln!(
        "divergence at a={a:#x} b={b:#x}: add={:#x} sub={:#x} ({})",
        d.rax_a, d.rax_b, report.summary()
    );
}

#[test]
fn incomplete_run_is_not_agreement() {
    let pe = pe();
    // A run that stops before the sentinel (here: a loop starved by a tight step
    // budget, which stands in for any early stop) must never count as agreement.
    let loop_mul: Target = "op_loop_mul".into();
    let tight = Limits { max_steps: 1, ..Limits::default() };
    let r = run_function(&pe, &loop_mul, &[6, 7], &tight).expect("setup");
    assert!(!r.completed(), "a 1-step budget cannot complete: {:?}", r.stop);

    let inputs = input_set(2, 8, 1);
    let report = differential(&pe, &loop_mul, &pe, &loop_mul, &inputs, &tight).expect("diff");
    assert_eq!(report.agreed, 0, "an incomplete run is never agreement");
    assert_eq!(report.incomplete, report.total);
    assert!(!report.all_agreed());
    let inc = report.first_incomplete.as_ref().expect("an incomplete");
    assert!(!inc.completed_a && !inc.completed_b);
    eprintln!("incomplete surfaced: {} first_stop={:?}", report.summary(), inc.stop_a);
}

#[test]
fn records_track_per_input_outcome() {
    let pe = pe();
    let inputs = input_set(2, 8, 2);
    let add: Target = "op_add32".into();
    let sub: Target = "op_sub32".into();
    let report = differential(&pe, &add, &pe, &sub, &inputs, &Limits::default()).expect("diff");

    assert_eq!(report.records.len(), report.total);
    let agree = report.records.iter().filter(|r| r.outcome == Outcome::Agree).count();
    let diverge = report.records.iter().filter(|r| r.outcome == Outcome::Diverge).count();
    assert_eq!(agree, report.agreed);
    assert_eq!(diverge, report.diverged);
}
