# SP2 — OEP Recovery + Generalized Virtualized-Function Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Prove the original entry point and recover/reintegrate arbitrary virtualized routines (not just the module entry + TLS), so the OEP and `.vmp`-referenced blockers can clear.

**Architecture:** Build on SP1's reintegration machinery. Keep each recovery decision a pure, unit-testable function (the pipeline wiring around it is verified out-of-band against the user's protected corpus, which does not exist in CI). Tractable spine first (OEP, discovery, live-in inference, input-independent recovery), then the hard input-domain tail (spike, then grind).

**Tech Stack:** Rust (edition 2024, rustc 1.88+), existing crates `vmpx-vm` (entry_paths, image_concretize, native_exit_state, native_context, native_continuation, native_vm_chain), `vmpx-cli`, `vmpx-rebuild` (SP1 reintegration), `vmpx-recompile`, `vmpx-vmlift`.

**Spec:** `docs/superpowers/specs/2026-10-29-vmpx-sp2-oep-generalized-recovery-design.md`

## Global Constraints

- Rust edition 2024; `unsafe_code = "forbid"`. No new external deps in the core/library crates.
- Lean comments. Commits: GitHub noreply author, NO Co-Authored-By line.
- Soundness default; `--experimental` is the only gate-relaxed path. "Recovered" requires passing verification, never assertion.
- **Verification reality:** no protected corpus in CI → verify the decidable/pure parts with constructed unit tests + keep the repo's existing tests (239) green; end-to-end recovery is proven out-of-band via `testcorpus/PROTECT-RECIPE.md`. Every task states what it proved vs deferred.

---

## Task 1 — OEP recovery from the proven continuation

Recover the original entry point: when the header entry's recovered path has a native exit whose return-address resolves to a single concrete RVA in a native (non-`.vmp`) region, that RVA is the OEP. Surface it in the `oep` command and let the `clean` gate drop the "OEP not recovered" blocker when proven.

**Files:**
- Create: `crates/vmpx-vm/src/oep.rs` (a pure decision function + its exports) — or add to an existing module if cleaner.
- Modify: `crates/vmpx-vm/src/lib.rs` (export), `crates/vmpx-cli/src/main.rs` (`analyze_oep`), `crates/vmpx-cli/src/clean.rs` (`CleanOepReport` + drop the OEP blocker when recovered).
- Test: unit tests in `oep.rs`.

**Interfaces:**
- Consumes: `NativeExitState` (`native_exit_state.rs`: has `return_address`, `registers`), `recover_modulo_image_targets(pe, expr, ModuloImageTargetLimits) -> Result<ModuloImageTargetReport,_>` (report has `complete: bool`, `targets: Vec<{rva: Option<Rva>, va: Va}>`), `RecoveredEntryPathFamily`/`RecoveredEntryPath` (`entry_paths.rs`), `PeImage`, `Rva`.
- Produces:
  ```rust
  pub struct OepDecision { pub recovered_entry: Option<Rva>, pub assessment: &'static str, pub evidence: Vec<String> }
  // Pure: given the resolved single-target continuation RVA(s) of the header entry's paths and a predicate for "native (non-vmp, non-vm-entry) RVA", decide the OEP.
  pub fn decide_oep(candidates: &[Rva], is_native: impl Fn(Rva) -> bool) -> OepDecision;
  ```
  The pipeline computes `candidates` from each header path: `path.native_exit` → `recover_modulo_image_targets(&exit.return_address)` where `report.complete && report.targets.len()==1 && targets[0].rva.is_some()`.

- [ ] **Step 1: Write the failing test** for `decide_oep`:

```rust
#[test]
fn single_native_continuation_is_the_oep() {
    let d = decide_oep(&[Rva(0x4000)], |r| r == Rva(0x4000)); // 0x4000 is native
    assert_eq!(d.recovered_entry, Some(Rva(0x4000)));
    assert_eq!(d.assessment, "recovered_native_continuation");
}
#[test]
fn a_vmp_or_ambiguous_target_is_not_oep() {
    assert_eq!(decide_oep(&[Rva(0x9000)], |_| false).recovered_entry, None);       // target in .vmp
    assert_eq!(decide_oep(&[Rva(0x4000), Rva(0x5000)], |_| true).recovered_entry, None); // ambiguous
    assert_eq!(decide_oep(&[], |_| true).recovered_entry, None);                    // none resolved
}
```

- [ ] **Step 2: Run to verify it fails** — `cargo test -p vmpx-vm oep` → FAIL (undefined).
- [ ] **Step 3: Implement** `decide_oep`: unique native candidate → `Some`, `"recovered_native_continuation"`; empty/ambiguous/non-native → `None`, `"unresolved"`; record evidence strings. Export from lib.rs.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Wire into the pipeline** (no new test required; existing tests must stay green): in `analyze_oep` (main.rs), after recovering the header entry's paths (reuse the DLL `native_rdx_initial` 0..=3 domain + limits pattern from `clean.rs`), gather the single-resolved continuation RVAs and call `decide_oep` with `is_native = rva not inside a .vmp section and not an `analyze_vm_entry_at` ≥60 stub`; set `OepAnalysis.recovered_entry`/assessment/evidence. In `clean.rs`, compute the same and set `CleanOepReport.recovered_entry`/`complete`; and when it is `complete`, do NOT push the `"original module entry point is not recovered"` blocker. Keep all other blockers.
- [ ] **Step 6: Verify** `cargo test --workspace` green; `cargo build --workspace` clean. Note in the report that end-to-end OEP recovery on a real sample is pending the protected corpus; the pure `decide_oep` logic + the wiring compile and the existing suite is green.
- [ ] **Step 7: Commit** `feat(vm): recover original entry point from the proven native continuation`.

---

## Task 2 — Generalized VM-entry inventory

Produce the full set of virtualized routines in the image, not just header/TLS. Wire `discover_vm_entries` (with executable-stub scanning) together with the export-reachable VM entries (the `export_dependency_report` walk in clean.rs already collects `analyze_vm_entry_at` ≥60 stubs). Deduplicate by RVA; attach confidence and discovery source. Output a `Vec<VmEntryCandidate>` covering the binary. Add a CLI surfacing (extend `vm-list`, or a report field). Unit-test the dedup/merge logic with constructed candidate lists. Keep existing tests green.

## Task 3 — Per-function live-in / calling-convention inference

Replace SP1's "reuse the header entry's `inputs`" for non-header entries with per-function inference: use `analyze_x64_native_frame` (identifies the stack-cursor register) + a liveness pass over the recovered handler effects to determine which architectural registers/stack slots are live-in. Produce the `inputs: Vec<(String, PhysicalRegister)>` a replacement needs for that specific entry. Unit-test the liveness/live-in decision on constructed effect sequences.

## Task 4a — Input-independent / symbolically-closed function recovery

Recover functions whose symbolic execution reaches a native exit expressed purely over the inferred live-in set (no branch on an unbounded input): run `recover_entry_paths` with the inferred inputs and an empty/trivial domain, confirm a native exit, and feed the exit state + per-function inputs into SP1's reintegration grouping. Unit-test the "is this exit input-independent / closed over live-ins?" predicate.

## Task 4b — Finite-domain enumeration + EXE entry domain (the hard tail)

**Spike first** (feasibility, labeled throwaway): can a finite input domain be inferred for a branching virtualized function from the inferred live-in set + the VM's observed comparisons, within `VmProfileExplorerLimits`? Report findings. Then implement bounded enumeration for the cases the spike shows are decidable, including the executable (non-DLL) entry domain the old `clean.rs:624` refused. Prove completeness per function or leave a blocker. Grind on widening coverage; never assert recovery that the verification (synthetic round-trip) doesn't support.

## Task 5 — Multi-function reintegration + nested continuations

Feed every recovered function (Tasks 1,4a,4b) into SP1's multi-group `reintegrate` path with per-function inputs (Task 3). Handle the nested-continuation grouping deferred from SP1 (re-entrant VM entries) here. Unit-test group assembly for several functions.

## Task 6 — Clear the blockers

When the OEP is proven (Task 1) and every reachable virtualized routine (Task 2 inventory) is recovered + reintegrated (Tasks 4a/4b/5), drop the `"OEP not recovered"` and `"VMProtect sections remain referenced by unresolved paths"` blockers so sound `output` can emit for fully-covered samples. Gate strictly: any uncovered entry keeps the blocker. Unit-test the gate logic (covered-set vs discovered-set).

---

## Notes for the executor

- Tasks 1–3 and 4a are the tractable spine — each is a real, unit-verifiable increment.
- Task 4b is the research wall; spike before committing detail, grind, and report honest per-function status (recovered / partial / blocked).
- Real-sample end-to-end verification is out-of-band (user's protected corpus via PROTECT-RECIPE.md). CI proves the pure logic + keeps the 239 existing tests green.
