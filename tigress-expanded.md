<h1 align="center">tigress-expanded</h1>

<p align="center">
  <b>Static deobfuscation of <a href="https://tigress.wtf/">Tigress</a>-obfuscated binaries.</b><br>
  Virtualization, control-flow flattening, MBA and opaque predicates — recovered,
  and <i>validated against ground truth</i>.
</p>

<p align="center">
  <code>status: planned / design</code>
</p>

---

Tigress (Collberg et al.) is a **source-to-source** C obfuscator/diversifier. It
is a different beast from VMProtect in a way that is mostly *in our favour*: the
output is a normal compiled binary — no packing, no encrypted IAT, no `.vmp`
sections, no commercial anti-tamper by default. The hard parts of the VMProtect
pipeline (unpacking, strip, import recovery, mutation relocation) simply do not
apply.

## The thing that changes the economics: ground truth

With VMProtect we have no oracle — every result is "proven under an assumption,"
and we can never be fully certain a recovery is correct. Tigress is source-level,
so we can generate unlimited `(clean source → obfuscated binary → known output)`
triples and **measure correctness as a number, then hill-climb** — something we
cannot do on VMProtect at all. That single fact makes most of Tigress a
*lower-risk, higher-confidence* target than the work already shipped in
[vmpx-expanded](README.md).

## Difficulty, calibrated against the VMProtect work

Easiest → hardest, relative to what vmpx-expanded already does:

| transform | difficulty | why |
|---|---|---|
| `EncodeArithmetic` (MBA) | **easy** | exactly what the solver tournament (`vmpx-solvers` / `vmpx-tournament`) already does |
| `Flatten` (control-flow flattening) | **easy–medium** | a dispatcher + state var; de-flattening is well-trodden and our CFG + state-model recovery apply |
| `EncodeLiterals` / `EncodeData` | **easy–medium** | decode by symbolic execution |
| `Virtualize` (basic) | **medium** | *easier* than VMProtect's VM — clean compiler output, normal calling conventions, `.pdata` present; new piece is recognizing a randomized dispatch ISA, which is our structural-handler + semantic-executor approach |
| `AddOpaque` (opaque predicates) | **medium** | `vmpx-solvers::opaque_predicate` + the z3 gate prove them away |
| `Split` / `Merge` | **medium** | inlining / outlining recovery |
| `Jit` (static runtime codegen) | **hard** | the real code isn't in the binary — lift and fold the *generator*; deterministic, so foldable (reuses `concretize`) |
| `JitDynamic`, input-dependent | **frontier** | the wall — when the emitted code genuinely depends on runtime input there is no single static program; recover the generator's intent and label the rest *observed* |

## What transfers directly (the expensive parts, already built)

- The lifting engine: `vmpx-vm` (symbolic execution, `Expr` IR, concretization),
  the roller/unroller, and `build_replacement` emission — largely VM-agnostic.
- Structural handler discovery — recognizing handlers by shape, not signature,
  which is exactly right for Tigress's *randomized* instruction sets.
- The native-mutation **solver tournament** (`vmpx-solvers`, `vmpx-tournament`,
  `vmpx-verify`) — MBA, opaque predicates, synthesis, differential input-testing
  scorer, z3 gate.
- The CFG recovery and the equivalence machinery.
- The readable emitter (`vmpx-decompile --readable --dialect …`).

## Scope

A **scoped, demonstrable win** is very achievable and lower-risk than the
VMProtect devirt: *recover + deobfuscate standard Tigress (`Virtualize` +
`Flatten` + MBA + opaque), validated end-to-end against a ground-truth corpus
with real equivalence proofs.* The engine is already most of the way there.
"Handle arbitrary `JitDynamic` + nested virtualization" is the research tail, not
the main body.

## First step

The corpus harness — a script that runs Tigress over a seed of C programs across
a config matrix, compiles, and records the `(clean, obfuscated, expected-output)`
triples. It is the piece that makes every step after it *measurable*, and it is a
day of work because the corpus is free to generate.

---

Same discipline as the rest of the project: static-first, fail-closed, labeled
assumptions, and the intended-use terms in [`SECURITY.md`](SECURITY.md).
