# SP6 — Mutation Deobfuscation Engine: Design Spec

**Status:** design / greenfield. Separate subsystem from the SP1–SP5 VM-devirt pipeline.
**Predecessors:** reuses `vmpx-vm` (`expr`, `value_range`), `vmpx-disasm`, `vmpx-recompile`, `vmpx-detect`. Does NOT depend on SP2–SP5 completion (the VM path and the native-mutation path are disjoint).

## 1. Goal

Soundly deobfuscate VMProtect **Mutation** (and the mutation layer of **Ultra**): native-code obfuscation — junk/dead code, opaque predicates, instruction substitution, and mixed boolean-arithmetic (MBA). Output: cleaned, **behavior-equivalent** native code (sound tier), with an **experimental** tier for inductively-validated results. Mutation keeps the code native (no VM), so this is a native→native cleanup, not a devirtualization.

## 2. Architecture — a solver tournament (the agreed model)

- **Players**: individual solvers AND composed chains of solvers, each a competitor.
- **Scorer = differential input-testing** (the workhorse): thousands of concrete inputs per candidate — random PLUS structured edge cases (0, 1, MAX, sign boundaries, 2^k±1, single-bit) — original vs candidate output. Per-instance verdict; this is what makes ELO honest (it catches a solver great on one instance, garbage on the next).
- **ELO ladder**, PER problem-class (linear-MBA / nonlinear-MBA / opaque-predicate / dead-code / substitution) and **cost-aware** (don't crown slow winners). Matchups update ratings; selection routes a sub-problem to the top-rated player(s) for its class.
- **SMT gate (z3) = promotion to the SOUND tier**, not the scorer. Input-testing ranks + selects + feeds the experimental tier; an SMT equivalence proof upgrades a result to proven/behavior-preserving. Both, never either; never block on the proof.
- **Applicator = the vmpx engine**: rewrites the IR and re-emits. Applies ONLY verified transforms (sound tier) or clearly-labeled experimental ones.
- **Internalization is earned**: the thousands of matchups on a REPRESENTATIVE corpus (must include real VMProtect mutation-mode output, not just synthetic) ARE the characterization; the converged per-class ELO says which techniques to reimplement natively. Keep the portfolio until the data earns it.

## 3. Crate layout (new)

- `vmpx-mutation` — core: mutation IR (or a thin wrapper over `vmpx_vm::Expr`), the pass pipeline, re-emit, orchestration.
- `vmpx-solvers` — the `Solver` trait + individual solvers + chain combinators.
- `vmpx-tournament` — ELO registry, per-class ratings, matchmaking, selection policy.
- `vmpx-verify` — differential input-testing harness + the SMT (z3) gate.

Reuse: `vmpx-vm` (`Expr`, hash-consing, `value_range`), `vmpx-disasm` (decode/CFG), `vmpx-recompile` (x64 emit), `vmpx-detect` (classification).

## 4. The parallelization contract (interfaces — WP1)

These signatures are the contract every other package builds against, so work packages proceed in parallel against the spec before the code lands. (Illustrative; WP1 finalizes exact types.)

```rust
// A unit of deobfuscation work: an expression (or code region) to simplify.
pub struct Problem { pub expr: Expr, pub class: ProblemClass, pub ctx: Ctx }
pub enum ProblemClass { LinearMba, NonlinearMba, OpaquePredicate, DeadCode, Substitution, Unknown }

pub struct Candidate { pub expr: Expr, pub by: SolverId, pub cost: Cost }

pub trait Solver {
    fn id(&self) -> SolverId;
    fn applies_to(&self, class: ProblemClass) -> bool;     // routing hint
    fn solve(&self, p: &Problem) -> Option<Candidate>;      // None = pass/giveup
}

// Chains are solvers built from solvers.
pub struct Chain(Vec<Box<dyn Solver>>); // impl Solver

pub trait Scorer { fn score(&self, original: &Expr, cand: &Candidate) -> Score; }   // input-testing
pub enum Verdict { Proven, Refuted, Unknown }
pub trait Gate   { fn verify(&self, original: &Expr, cand: &Expr) -> Verdict; }       // SMT

pub struct Elo(/* per-(player,class) ratings */);
impl Elo {
    pub fn record(&mut self, class: ProblemClass, winner: SolverId, loser: SolverId);
    pub fn rank(&self, class: ProblemClass) -> Vec<SolverId>; // best first
}
```

`Score` carries: correctness (input-test agreement ratio, must be 1.0 to be eligible), simplicity (AST node count / op count), and time. Eligibility = agreement 1.0 over the test set; among eligible, rank by (simplicity, cost). The Gate upgrades an eligible candidate to the sound tier.

## 5. Work packages (≈12 — the fan-out)

**Wave 1 — foundation + independent (parallel):**
- **WP1 — contract** (`vmpx-mutation` types + `Solver`/`Scorer`/`Gate`/`Elo` traits). Small, keystone; everything else builds to it. *Blocks the solver army; land first.*
- **WP2 — differential input-testing harness** (`vmpx-verify`): generate input tuples (random + structured edge cases), evaluate `Expr` (reuse `vmpx_vm` const-folding / an evaluator) on each, compare. The Scorer.
- **WP3 — SMT gate** (`vmpx-verify`): z3 (or a pure-Rust bitvector decision procedure if we avoid the dep) proving `original == candidate` over bitvectors. The Gate.
- **WP5 — native CFG recovery** (`vmpx-mutation`): decode a mutation-protected function to a native CFG (reuse `vmpx-disasm`); the substrate the passes operate on.
- **WP6 — classification** (`vmpx-detect`): Virtualization/Mutation/Ultra. *(already in progress)*
- **WP12 — mutation benchmark corpus** (`testcorpus/corpus`): protect known functions in mutation mode, extract the obfuscated native expressions as ground-truth `(clean expr, obfuscated expr)` pairs → the data that computes the ELOs. Extends existing gen-corpus/build-gen tooling.

**Wave 2 — solvers + passes (parallel, build to WP1 traits):**
- **WP4 — ELO registry + tournament engine** (`vmpx-tournament`): ratings, matchmaking over the corpus, selection policy (route by class → top ELO; cost-aware).
- **WP7 — dead-code / junk elimination** (`vmpx-mutation` pass): liveness + sound removal of effect-free instructions.
- **WP8 — opaque-predicate prover** (solver): prove a branch condition constant via `value_range` + bounded symbolic reasoning (BINSEC backward-bounded DSE style); fold the dead arm.
- **WP9 — linear-MBA simplifier** (solver): SiMBA-style canonicalization of linear MBA to a normal form.
- **WP10 — general-MBA solver** (solver): GAMBA-style algebraic rewrite rules + an optional synthesis wrapper (Xyntia-style), all behind the gate.
- **WP11 — substitution canonicalizer** (solver): peephole reversal of VMProtect's instruction-substitution patterns.

**Wave 3 — integration:**
- Wire pass pipeline (DCE → substitution → MBA → opaque-fold, iterate to fixpoint) with the tournament selecting per sub-problem; re-emit via `vmpx-recompile`; honest sound/experimental tiering; a `vmpx demutate` CLI command.

## 6. Dependency graph (what blocks what)

WP1 blocks WP4, WP7–WP11 (they use its traits). WP2, WP3, WP5, WP6, WP12 are independent of WP1 (define/adapt their own small interfaces, integrate at the trait boundary). So Wave 1 is ~6 agents in parallel; once WP1 lands, Wave 2 fans to ~7 more; Wave 3 integrates. The only hard serialization is WP1 → solver army.

## 7. Soundness (inherited from the vmpx framework)

- A transform ships in the SOUND tier only if the Gate proves equivalence; otherwise it is EXPERIMENTAL, labeled, never silently applied as proven.
- Input-testing is inductive (passing N inputs ≠ proof) → it ranks and gates the experimental tier, never certifies soundness alone.
- Fail closed; never drop an effect; preserve observable behavior.

## 8. Definition-of-done ladder

- **M1:** contract + input-testing + SMT gate + corpus → the tournament can *score* a solver on real data.
- **M2:** ≥2 solvers per class registered and ELO-ranked on the mutation corpus.
- **M3:** end-to-end `demutate` on `coredll.mutation.dll`: DCE + opaque-fold + MBA cleanup applied, output behavior-verified (input-tested; SMT where it closes) against the original.
- **M4:** earned internalization — the per-class winners reimplemented natively; portfolio kept for the tail.
