//! SP5 foundation: a differential-execution oracle.
//!
//! Execute a function two ways and compare: run the SAME inputs through two
//! entry points (the original VM function and its devirtualized native
//! replacement, once that wiring lands) and assert identical observable
//! results. Built now on the Layer-1 loader ([`crate::load`]) against real PE
//! exports.
//!
//! # Soundness
//!
//! This is an *empirical* (inductive) oracle, not a proof. A pass means "agreed
//! on K/N inputs", never "proven equal". Results are only comparable when BOTH
//! runs reach the return sentinel: a divergence, or an early stop on either
//! side (unsupported instruction, unmodeled API stub, fault, step limit), means
//! "not proven equal". An incomplete run is NEVER counted as agreement.

use vmpx_pe::PeImage;

use crate::emu::Stop;
use crate::loader::{LoadError, LoadOptions, load};

/// Where to position execution within a loaded image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// Resolve an export by name.
    Export(String),
    /// `image_base + rva`.
    Rva(u64),
}

impl From<&str> for Target {
    fn from(s: &str) -> Self {
        Target::Export(s.to_string())
    }
}
impl From<String> for Target {
    fn from(s: String) -> Self {
        Target::Export(s)
    }
}

/// A caller-provided guest memory window to capture and compare after a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputRegion {
    pub addr: u64,
    pub len: u64,
}

/// Execution bounds and the optional output window.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_steps: u64,
    pub stack_size: u64,
    pub output_region: Option<OutputRegion>,
}

impl Default for Limits {
    fn default() -> Self {
        Self { max_steps: 100_000, stack_size: LoadOptions::default().stack_size, output_region: None }
    }
}

/// A setup failure that is the same for every input (bad image or target), as
/// opposed to a per-input runtime [`Stop`], which is recorded in [`RunResult`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    Load(LoadError),
    ExportNotFound(String),
}

impl From<LoadError> for RunError {
    fn from(e: LoadError) -> Self {
        RunError::Load(e)
    }
}

/// The outcome of a single function execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunResult {
    /// Why the run stopped. Only [`Stop::Reached`] is a clean return.
    pub stop: Stop,
    /// RAX at the stop (meaningful only when [`RunResult::completed`]).
    pub rax: u64,
    /// Captured output-region bytes, when a region was requested and the run
    /// completed and the window was readable; `None` otherwise.
    pub output: Option<Vec<u8>>,
    /// Instructions retired.
    pub steps: usize,
}

impl RunResult {
    /// Did the run return cleanly to the sentinel? Only then are `rax`/`output`
    /// comparable.
    #[must_use]
    pub fn completed(&self) -> bool {
        matches!(self.stop, Stop::Reached)
    }
}

/// Load `pe`, position at `target` with Win64 integer `args`, run to the return
/// sentinel, and capture RAX, the output region (if requested) and the stop
/// reason. Runtime problems (unsupported instruction, fault, step limit) are
/// recorded in the returned [`RunResult`], never panics.
///
/// # Errors
/// [`RunError`] for a setup failure: the image fails to load, the stack write
/// for the return sentinel faults, or a named export is absent.
pub fn run_function(
    pe: &PeImage,
    target: &Target,
    args: &[u64],
    limits: &Limits,
) -> Result<RunResult, RunError> {
    let opts = LoadOptions { stack_size: limits.stack_size };
    let mut img = load(pe, &opts)?;
    let rip = match target {
        Target::Export(name) => {
            img.export(name).ok_or_else(|| RunError::ExportNotFound(name.clone()))?
        }
        Target::Rva(rva) => img.image_base.wrapping_add(*rva),
    };
    let sentinel = img.return_sentinel();
    let mut emu = img.prepare(rip, args)?;
    let stop = emu.run_until(sentinel, limits.max_steps);
    let completed = matches!(stop, Stop::Reached);
    let output = match limits.output_region {
        Some(r) if completed => {
            let mut buf = vec![0u8; r.len as usize];
            emu.mem.read(r.addr, &mut buf).ok().map(|()| buf)
        }
        _ => None,
    };
    Ok(RunResult { stop, rax: emu.cpu.regs[0], output, steps: emu.trace.len() })
}

/// Both runs completed but produced different observable results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Divergence {
    pub input: Vec<u64>,
    pub rax_a: u64,
    pub rax_b: u64,
    pub output_a: Option<Vec<u8>>,
    pub output_b: Option<Vec<u8>>,
}

/// At least one side did not return cleanly, so the pair is not comparable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Incomplete {
    pub input: Vec<u64>,
    pub stop_a: Stop,
    pub stop_b: Stop,
    pub completed_a: bool,
    pub completed_b: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Both completed and matched.
    Agree,
    /// Both completed and differed.
    Diverge,
    /// One or both did not complete: not proven equal, not a divergence.
    Incomplete,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffRecord {
    pub input: Vec<u64>,
    pub outcome: Outcome,
}

/// Aggregate result of comparing two functions over an input set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffReport {
    pub total: usize,
    pub agreed: usize,
    pub diverged: usize,
    pub incomplete: usize,
    pub first_divergence: Option<Divergence>,
    pub first_incomplete: Option<Incomplete>,
    pub records: Vec<DiffRecord>,
}

impl DiffReport {
    /// True only when every input agreed and nothing diverged or stopped early.
    /// This is empirical agreement over the set, not a proof of equivalence.
    #[must_use]
    pub fn all_agreed(&self) -> bool {
        self.total > 0 && self.agreed == self.total
    }

    /// One-line human summary, e.g. `agreed 128/130 (diverged 1, incomplete 1)`.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "agreed {}/{} (diverged {}, incomplete {})",
            self.agreed, self.total, self.diverged, self.incomplete
        )
    }
}

/// Run every input in `input_set` through both targets and compare RAX (and the
/// output region, when requested). Reports per-input agreement, the first
/// divergence and the first early stop.
///
/// A fresh image is loaded per run (the loader hands its memory to the
/// emulator), so both sides start from identical state every time.
///
/// # Errors
/// [`RunError`] from the first run that fails to set up (bad image or target);
/// per-input runtime stops are recorded as [`Outcome::Incomplete`], not errors.
pub fn differential(
    pe_a: &PeImage,
    target_a: &Target,
    pe_b: &PeImage,
    target_b: &Target,
    input_set: &[Vec<u64>],
    limits: &Limits,
) -> Result<DiffReport, RunError> {
    let mut report = DiffReport {
        total: input_set.len(),
        agreed: 0,
        diverged: 0,
        incomplete: 0,
        first_divergence: None,
        first_incomplete: None,
        records: Vec::with_capacity(input_set.len()),
    };

    for input in input_set {
        let a = run_function(pe_a, target_a, input, limits)?;
        let b = run_function(pe_b, target_b, input, limits)?;

        let outcome = if !a.completed() || !b.completed() {
            report.incomplete += 1;
            if report.first_incomplete.is_none() {
                report.first_incomplete = Some(Incomplete {
                    input: input.clone(),
                    stop_a: a.stop.clone(),
                    stop_b: b.stop.clone(),
                    completed_a: a.completed(),
                    completed_b: b.completed(),
                });
            }
            Outcome::Incomplete
        } else if a.rax == b.rax && a.output == b.output {
            report.agreed += 1;
            Outcome::Agree
        } else {
            report.diverged += 1;
            if report.first_divergence.is_none() {
                report.first_divergence = Some(Divergence {
                    input: input.clone(),
                    rax_a: a.rax,
                    rax_b: b.rax,
                    output_a: a.output.clone(),
                    output_b: b.output.clone(),
                });
            }
            Outcome::Diverge
        };
        report.records.push(DiffRecord { input: input.clone(), outcome });
    }
    Ok(report)
}

// SplitMix64: a tiny self-contained PRNG so the generator needs no dep.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Structured edge values for one integer argument: zeros/ones, 64- and 32-bit
/// signed/unsigned boundaries, and `2^k`, `2^k-1`, `2^k+1` and single bits for
/// every `k`. This is the edge-case philosophy the mutation scorer reuses.
#[must_use]
pub fn edge_values() -> Vec<u64> {
    let mut v = vec![
        0,
        1,
        2,
        u64::MAX,                // -1
        0x7FFF_FFFF_FFFF_FFFF,   // i64::MAX
        0x8000_0000_0000_0000,   // i64::MIN
        0xFFFF_FFFF,             // u32::MAX
        0x7FFF_FFFF,             // i32::MAX
        0x8000_0000,             // i32::MIN (zero-extended)
        0x1_0000_0000,           // 2^32
    ];
    for k in 0..64 {
        let p = 1u64 << k;
        v.push(p); // 2^k / single bit
        v.push(p.wrapping_sub(1)); // 2^k - 1
        v.push(p.wrapping_add(1)); // 2^k + 1
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Build an input set of `num_args`-tuples: structured edge cases (every arg set
/// to an edge value; then each arg swept over edges with the rest held at 1)
/// plus `random_count` random tuples from `seed`. Deduplicated.
#[must_use]
pub fn input_set(num_args: usize, random_count: usize, seed: u64) -> Vec<Vec<u64>> {
    if num_args == 0 {
        return vec![vec![]];
    }
    let edges = edge_values();
    let mut out: Vec<Vec<u64>> = Vec::new();

    // Diagonal: all args at the same edge value.
    for &e in &edges {
        out.push(vec![e; num_args]);
    }
    // Per-position sweep, remaining args held at a neutral 1.
    for i in 0..num_args {
        for &e in &edges {
            let mut t = vec![1u64; num_args];
            t[i] = e;
            out.push(t);
        }
    }
    // Random tuples.
    let mut st = seed;
    for _ in 0..random_count {
        out.push((0..num_args).map(|_| splitmix64(&mut st)).collect());
    }

    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_structured_and_deduped() {
        let s = input_set(2, 16, 0xABCD);
        assert!(s.iter().all(|t| t.len() == 2));
        assert!(s.contains(&vec![0, 0]));
        assert!(s.contains(&vec![u64::MAX, u64::MAX]));
        assert!(s.contains(&vec![0x8000_0000_0000_0000, 1])); // sweep arg0
        let mut sorted = s.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), s.len(), "no duplicate tuples");
    }

    #[test]
    fn zero_args_yields_single_empty_tuple() {
        assert_eq!(input_set(0, 5, 1), vec![Vec::<u64>::new()]);
    }
}
