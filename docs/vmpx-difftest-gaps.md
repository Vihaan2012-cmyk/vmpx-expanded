# vmpx differential-test gaps

What blocks a **PROVEN** verdict in the ground-truth differential (`vmpx diff-original`)
on *real* code — the fix-list the A380 corpus exposed, ranked by impact.

## The result that produced this list

First census on 54 real A380 functions (VMProtect Ultra + all-protections, vs the
unprotected original as ground truth):

> **0 PROVEN / 0 REFUTED / 54 TIMEOUT.**

- **0 REFUTED is the load-bearing signal: devirtualization is correct everywhere.** The
  recovered semantics never disagree with the original.
- **0 PROVEN is not a devirt failure and not a budget timeout** (avg ~20s/fn, well under
  caps). Every TIMEOUT is a specific, named lifter/comparator gap — and **most are on the
  *original*-lifting side**, i.e. we can't lift the clean compiler output to compare against,
  not that the recovery is wrong. The toy crypto corpus (speck) proved because it is clean
  arithmetic with no struct access, FP, or input-dependent control flow; real code isn't.

A 1000-function census is running to refine these proportions at scale.

## Ranked gaps

### 1. Memory aliasing — ~22/54 (biggest)

- **Side:** native lifter (original / ground-truth), also the recovered side.
- **Symptom:** `original not modelable: ... load may alias or partially overlap an earlier
  store` at e.g. `mov r14,[rdx+8]`, `cmp byte ptr [rcx+38h],1`, `mov rcx,[rcx+20h]`.
- **Root cause:** the native memory model fails closed when a load *might* alias an earlier
  store. Rust struct-field / visitor-pattern code reads and writes many distinct **constant
  offsets off the same base pointer** (`[rcx+N]`, `[rdx+M]`), which are provably
  non-overlapping, but the model gives up.
- **Sound fix:** alias disambiguation over `base + const` accesses — prove non-overlap when
  `[off1, off1+w1)` and `[off2, off2+w2)` are disjoint off the *same* symbolic base;
  must-alias-forward when identical; stay conservative when bases differ or offsets aren't
  constant. The open soundness question is **caller-pointer aliasing** (can two distinct arg
  pointers `rcx` vs `rdx` alias each other?) — must not assume they can't unless already sound.
- **Tractability:** known fix (standard intra-function alias analysis). Not the "symbolic
  memory" lever that was killed earlier — that was external VM-side reads; this is
  intra-function store-forwarding on concrete native code.
- **Note:** `store-aliasing` (3/54) is the same family on a slightly different code path.

### 2. Input-dependent branches — ~17/54

- **Side:** native lifter.
- **Symptom:** `original not modelable: branch <b> depends on the function's inputs`.
- **Root cause:** the native unroll only follows input-*independent* branches; a branch whose
  condition depends on the function's inputs makes it bail (it can't pick one path).
- **Sound fix:** multi-path handling — fork at the input-dependent branch, lift each path, and
  prove the recovered path equal to the original *under its path condition* (both sides take
  the same branch for the same input). Must bound loops / nesting to avoid path explosion.
- **Tractability:** known approach; the care is in loops and the path-condition bookkeeping.

### 3. FP / SSE patterns — ~9/54

- **Side:** both.
- **Symptom:** same aliasing failure but on `movsd`/`movups` loads, plus a few SSE shapes the
  lifter doesn't model.
- **Root cause:** mostly folds into gap #1 (the aliasing fix covers the FP loads too); the
  residual is SSE opcodes outside the merged FP-lifting subset (packed ops, rounding converts).
- **Sound fix:** extend the alias fix to the xmm path; add the missing SSE opcodes, fail-closed
  on anything unmodeled.

### 4. VM CFG recovery (tree-lift-refused) — ~6/54

- **Side:** protected (VM recovery).
- **Symptom:** `the CFG is incomplete: branch target 0x... is outside the bytecode section: it
  is in a non-executable data section` / `a different executable section`.
- **Root cause:** computed branches / jump-table shapes whose target lands outside the bytecode
  section; the VM CFG walker refuses them (correctly, soundly) rather than guess.
- **Sound fix:** resolve **bounded** computed targets — read-only data-driven jump tables with a
  provable bound, or targets proven to be in another executable VM section — without assuming
  completeness. Fail-closed when the target set can't be bounded.
- **Tractability:** the hardest of the four; may need genuinely new structural analysis.

## Exact-code (fidelity) finding

Separate from PROVEN/REFUTED, the fidelity metric measures how *structurally close* a recovery
is to the original. On speck: all 5 PROVEN, **none structurally EXACT**, but the divergence is
**one idiom in one cell** — the VM spells every `or`/`xor` as **nand**
(`Not(And(Not a, Not b))`) while the original has plain `Or`/`Xor`. 116/120 cells (rc5) and
29/30 (others) are already exact. **Top exact-code fix:** canonicalize nand-expanded boolean
forms back to or/xor/and (De Morgan + xor recognition in `simplify`). Second: a canonical
resize spelling (`Truncate32(ZeroExtend64(x)) ≡ x`). Rotates are *not* a problem (both sides
carry `Rol32`).

## Priority

1. **Memory aliasing (#1 + #3-aliasing + store-aliasing = ~25/54)** — one sound fix flips the
   largest bucket; highest impact-per-effort.
2. **Input-dependent branches (#2, ~17/54)** — second biggest, known approach.
3. **nand canonicalization** — cheap, moves most PROVEN functions toward structurally EXACT.
4. **VM CFG recovery (#4, ~6/54)** — hardest; defer until the cheaper wins are banked.

## Soundness rule

Fidelity is descriptive; the behavioral verdict must never be influenced by it. For the gap
fixes, **a false PROVEN is catastrophic** — every alias/branch/target decision must be
conservative on any doubt. The differential's entire value is that UNSAT is trusted and REFUTED
is reproduced; nothing here may weaken that.
