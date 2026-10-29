# SP2 — OEP Recovery + Generalized Virtualized-Function Recovery: Design Spec

- **Date:** 2026-10-29
- **Status:** Draft (autonomous build)
- **Branch:** `sp2-oep-generalized` (off `main` @ 2883c34, which contains SP1)
- **Predecessor:** SP1 (native reintegration) — complete. See `2026-10-29-vmpx-sp1-native-reintegration-design.md` for the overall roadmap and the A/C soundness strategy, which this spec inherits.

## 1. Context

SP1 made the pipeline splice recovered native replacements into a new section and redirect the module entry + TLS callbacks, behind the `--experimental` gate. Sound `output` still refuses on protected binaries because of the remaining blockers. SP2 attacks the two that gate real output:

- **"original module entry point is not recovered"** — the `oep` command and the `clean` gate never learn the original entry; `analyze_oep` hardcodes `recovered_entry = None`.
- **"VMProtect sections remain referenced by unresolved paths"** — only the module entry and TLS callbacks are recovered today; arbitrary virtualized functions scattered through the binary are not, so references into `.vmp*` remain.

SP2 also absorbs the old "executable (non-DLL) entry finite input-domain recovery is not implemented" gap (`clean.rs:624`).

## 2. Goal and honest scope

**Goal:** prove the original entry point, and discover + recover *every* reachable virtualized routine (not just the module entry and TLS callbacks), lifting and reintegrating each with SP1's machinery, so the "OEP" and "`.vmp` referenced" blockers can clear and sound output becomes possible for samples we fully cover.

**Honest scope.** This is the research-heavy milestone. It splits into a tractable spine and a hard tail:

- **Tractable (expected to land, unit/synthetic-verifiable):** OEP recovery from the already-proven continuation target; generalized VM-entry discovery across the binary (the scanning machinery exists); calling-convention / live-in inference per entry; and lift+reintegrate for functions whose symbolic execution reaches a native exit **without** depending on an unbounded input (input-independent or symbolically-closed).
- **Hard tail (grind, don't park):** functions whose VM control flow branches on a runtime input need the function's calling convention and a **finite input domain** inferred before the entry-path recovery can enumerate them. For the general case this is at the frontier; the plan attacks it with inferred live-in sets + bounded domain enumeration and keeps widening coverage. Where a function cannot be recovered **soundly**, it remains a blocker (sound mode refuses; `--experimental` still emits what it has) — but per the run's standing instruction this tail is ground on, not parked, and "recovered" means the differential/round-trip checks pass, never an assertion.

**Verification reality.** There is no protected test corpus in-repo yet (the VMProtect demo could be downloaded but not executed under the sandbox; `testcorpus/PROTECT-RECIPE.md` lets the user produce protected+original pairs locally). So SP2 is verified at the **unit/synthetic level** plus the repo's existing `p06_*` fixtures; true behavioral verification on real VMProtect awaits the user running the recipe. Every milestone states what it proved and what it did not.

## 3. Decomposition

- **P1 — OEP recovery.** From the header entry's recovered path(s), when `path.native_exit` is `Some` and `recover_modulo_image_targets(&pe, &exit.return_address, ..)` is `complete` with a single resolved `rva` that lies in a native (non-`.vmp`, non-VM-entry) region, that RVA is the recovered original entry. Populate `OepAnalysis.recovered_entry`, surface it in the `oep` command, and let the `clean` gate drop the OEP blocker when it is proven. Build on `recover_modulo_image_targets` (`image_concretize.rs`), `NativeExitState` (`native_exit_state.rs`), `recover_entry_paths` (`entry_paths.rs`). DLL case first (the pipeline already recovers DLL entry paths); EXE case depends on P4.
- **P2 — Generalized VM-entry discovery.** Enumerate every VM entry in the image, not just the header/TLS roots: run `discover_vm_entries` with executable-stub scanning and fold in call-site/xref discovery (the export-dependency walk in `clean.rs` already finds VM entries reachable from exports via `analyze_vm_entry_at` with confidence ≥ 60). Produce the full set of candidate virtualized routines with their confidence.
- **P3 — Calling-convention / live-in inference.** For a discovered entry that is not the DLL entry, infer which architectural registers (and stack slots) are live-in, using `analyze_x64_native_frame` (already identifies the stack-cursor register) plus a liveness pass over the recovered handler effects. This produces the `inputs` set a replacement needs — today SP1 reuses the header entry's inputs for TLS, which SP2 replaces with per-function inference.
- **P4 — Input-domain handling.** First, recover functions that need **no** domain enumeration: those whose symbolic execution reaches a native exit expressed purely over the inferred live-in set (input-independent or symbolically-closed) — lift+reintegrate directly. Then the hard tail: when the VM branches on an input, infer a finite domain (bounded enumeration over the inferred live-in values, with the existing `VmProfileExplorerLimits`) and recover each branch; prove completeness or leave a blocker. This also implements the EXE-entry domain the old `clean.rs:624` refused.
- **P5 — Lift + reintegrate each recovered function.** Feed each recovered function's exit state(s) + per-function `inputs` into SP1's grouping/`reintegrate` machinery (which already lays out multiple groups, builds fallback dispatchers, and redirects). Nested-continuation grouping (deferred from SP1) is also handled here, since it is the same "re-entrant VM entry" problem.
- **P6 — Clear blockers.** When the OEP is proven (P1) and every reachable virtualized routine is recovered + reintegrated (P2–P5), drop the "OEP not recovered" and "`.vmp` referenced" blockers so sound `output` can emit for fully-covered samples. (`.vmp` stripping itself is SP4; SP2 clears the *reference* blocker by covering the routines.)

## 4. Cross-cutting rules (inherited from SP1)

- Soundness default; `--experimental` stays the only gate-relaxed path; behavior-changing modes opt-in and labeled.
- Fully static; fixture-first; equivalence-preserving.
- "Recovered" requires passing verification (synthetic round-trip / differential), never assertion.

## 5. Verification strategy

- **Unit/synthetic (CI):** new fixtures in the `p06_*` style — a synthetic VM entry that transitions to a known native RVA (P1); a synthetic image with several VM entries (P2); functions with known live-in sets (P3); input-independent and small-finite-domain functions with known exit states (P4); multi-function reintegration (P5).
- **The repo's existing `p06_*` + discovery/symbolic tests** must stay green throughout.
- **Out-of-band (user):** once the user protects the `testcorpus` originals (which have known-good oracle outputs) via `PROTECT-RECIPE.md`, run `vmpx clean --experimental` and compare behavior — the strongest check, but not available in CI.

## 6. Risks

- **P4 is the wall.** General per-function input-domain inference is unsolved in the literature for arbitrary VMProtect functions. Expectation: cover input-independent and small-finite-domain functions; the long tail stays a blocker and is ground on, with honest per-function status (recovered / partial / blocked).
- **Soundness of generalized discovery.** A mis-identified VM entry (false positive) must not produce a wrong replacement; confidence gating + the existing proof gates guard this. A missed entry (false negative) leaves a `.vmp` reference → sound mode correctly refuses.
- **No real corpus in CI.** Synthetic verification can be unrepresentative; flagged, and the user's recipe is the real check.

## 7. Plan order

Execute via Subagent-Driven Development, tractable spine first so each step is a verifiable win:
1. **P1 OEP recovery** (concrete, verifiable now).
2. **P2 generalized VM-entry discovery** (wire existing machinery into a full inventory).
3. **P3 live-in / calling-convention inference** (replaces SP1's header-inputs reuse).
4. **P4a input-independent / symbolically-closed function recovery** (+ reintegrate via P5 path).
5. **P4b finite-domain enumeration + EXE-entry domain** — the hard tail; spike feasibility first, then grind, verifying each increment.
6. **P5/P6 multi-function reintegration + blocker clearing.**

Detailed per-task steps live in the SP2 implementation plan (`docs/superpowers/plans/`). SP3–SP5 remain sketched in the SP1 roadmap and are not planned until SP2's lessons land.
