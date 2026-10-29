# vmpx — Devirtualized Output: Design Spec

- **Date:** 2026-10-29
- **Status:** Draft, pending author review
- **Scope of this document:** Full design of **SP1 (native reintegration + image emission)**, plus the roadmap (SP2–SP6) that SP1 is the backbone of. Each later sub-project gets its own spec → plan → implementation cycle.

---

## 1. Context & problem

`vmpx` is a soundness-first static recovery toolkit for VMProtect-protected x64 Windows PEs. The `clean` pipeline (`crates/vmpx-cli/src/clean.rs`) already:

- unpacks packed virtual sections into a mapped analysis image,
- detects VMProtect and discovers VM entry stubs,
- symbolically executes handler chains for the DLL entry and TLS callbacks,
- recovers exact native exit states and follows continuations / nested VM entries, and
- **compiles correct native replacement machine code** for each recovered path (`vmpx_vmlift::build_replacement`, `clean.rs:1038`), recording where each body would be placed.

It then **discards** those bytes: `emitted_section` is built and immediately dropped (`clean.rs:1099`, `let _ = emitted_section;`). The pipeline refuses to emit a devirtualized binary, and hard-codes four blockers for any protected input (`clean.rs:1112`):

1. original module entry point (OEP) not recovered,
2. application import set not independently proven,
3. no devirtualized replacement image produced,
4. `.vmp` sections remain referenced by unresolved paths.

Plus conditional gaps — notably finite input-domain recovery for executable (non-DLL) entries is flatly *not implemented* (`clean.rs:624`).

**Goal of the overall effort:** take `vmpx` to the point of producing clean, behavior-preserving ("reversible") devirtualized output, as a direction with provable milestones rather than a single finish line.

**Completion strategy (decided):** a mix of **A (soundness-first)** and **C (hybrid/decomposed)**. The sound, proof-gated path is the real deliverable and the default. An explicit, loudly-labeled *experimental* emission mode exists only as scaffolding to exercise the machinery against real samples early; it is never the default and is quarantined to its own output path.

---

## 2. Goals & non-goals

### Goals
- Turn the already-compiled replacement bytes into a loadable, structurally-valid PE.
- Preserve the existing proof gate: sound emission stays refused until the relevant proofs pass.
- Provide an experimental emission mode so the machinery can be tested end-to-end immediately.
- Decompose the remaining work into independently valuable, independently testable sub-projects.

### Non-goals (and a critical caveat)
- **"Reversible" means behavior-reversible, not bit-reversible.** VMProtect virtualization is lossy: original instruction encodings, function boundaries, padding, and the pristine `.text` are destroyed. The output reproduces *semantics*, not the original bytes. It will not match a hash of the unprotected original, and a byte/disassembly diff against the original will look nothing alike even when behavior is identical. This caveat must not be silently dropped in later work.
- **"Any binary" is a direction, not a guarantee.** Correctness is *proven* on synthetic fixtures we construct and *empirically confirmed* on local samples. Coverage is bounded by the VMProtect constructs we have lifted and proved; outside that, sound mode refuses.
- SP1 does not attempt OEP recovery, import reconstruction, `.vmp` stripping, or optimization — those are SP2/SP3/SP5/SP6.

---

## 3. Validation strategy

- **Primary oracle (CI):** synthetic fixtures, extending the repo's existing `p06_*` fixture style. Reproducible, requires no proprietary samples, runs in CI.
- **Secondary oracle (out-of-band):** real VMProtect-protected samples held locally on the author's Windows machine (payware addons). Used for behavioral spot-checks only; **never committed, never shared, never uploaded.**
- Consequence: universal correctness is not claimed. Each milestone's "done" is defined against fixtures plus a local behavioral spot-check.

---

## 4. Roadmap (SP1–SP6) & definition-of-done ladder

| # | Sub-project | Clears / purpose | Depends on |
|---|-------------|------------------|------------|
| **SP1** | **Native reintegration + image emission** (backbone; experimental mode lives here) | blocker 3 ("no replacement image produced") | — |
| SP2 | OEP recovery (prove the real entry so the redirect is sound) | blocker 1 | SP1 |
| SP3 | Import-set reconstruction (reduce protector-era imports to the proven app set) | blocker 2 | SP1 |
| SP4 | Executable (non-DLL) entry finite input-domain recovery (`clean.rs:624`) | widens coverage beyond DLL/TLS | SP1 |
| SP5 | Dangling `.vmp` reference resolution + section stripping | blocker 4 | SP1–SP3 |
| SP6 | Optimizing recompilation via VTIL + differential verification engine | cleanest output; quality, not coverage | SP1; independently sequenceable |

### DoD ladder (each rung independently useful and testable)
- **L1 — SP1 experimental:** produces a loadable PE with replacements spliced into a new section and the entry redirected; reparses, passes `validate_for_loader`, and runs against a local VMP sample. Default/sound mode still refuses. *First rung that yields a testable binary.*
- **L2 — +SP2:** entry redirect targets a *proven* OEP → sound mode can emit for simple samples.
- **L3 — +SP3:** imports reduced to the proven application set.
- **L4 — +SP4/SP5:** executable entries covered; `.vmp` sections stripped with no dangling references → "clean" in the strong sense (the `clean <in> <out>` command finally writes `<out>`).
- **L5 — +SP6:** recovered code optimized/normalized to compiler-quality, each pass behavior-verified.

### Cross-cutting rules for every sub-project
1. **Soundness default.** The existing proof gate stays; sound emission is unlocked only when a sub-project's proof genuinely passes. Experimental emission is a separate, explicit, labeled flag.
2. **Fixture-first.** Every component gets synthetic fixtures as the CI oracle; local samples are the out-of-band behavioral check.
3. **Equivalence-preserving.** No transformation (especially SP6 optimization) may change observable behavior; where a transformation could, it ships with a verification check, not faith.

---

## 5. SP1 — Native reintegration + image emission (full design)

### 5.1 Current emission state (what SP1 builds on)
In the recompilation stage (`clean.rs:962`–`1100`):
- `next_section_rva(&mapped)` gives the RVA where a new section would begin; `va = image_base + next_rva`.
- `replacement_inputs(...)` binds symbolic inputs to physical registers (including the `vsp_initial` stack-cursor binding).
- For each exit state (header + nested + TLS), `build_replacement(&exit.registers, &config, CodeAddress(va + total_bytes))` compiles a body; `total_bytes` accumulates body sizes, so **body _i_ is compiled assuming it starts at `next_rva + Σ(previous body sizes)`**.
- `placed: Vec<(assignments, entry_rva)>` records each body's RVA; `functions: Vec<(assignments, Vec<u8>)>` holds the bytes; `emitted_section = Some(functions)` — then dropped.

SP1 stops the drop and consumes `functions` + `placed`.

### 5.2 Components (library-level; CLI stays thin)

**(1) `append_code_section` — new `crates/vmpx-rebuild/src/reintegrate.rs`.**
Appends one executable section (default name `.vmpx0`, characteristics R+X: `IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_CNT_CODE`) to the mapped image at `next_section_rva`. Writes a new section header, places the payload, bumps `SizeOfImage`, and respects `SectionAlignment` / `FileAlignment`.
- **Hard constraint:** there must be room in the section-header table (the region between the end of the optional header + existing section headers and the first section's data). If there is no room for another 40-byte `IMAGE_SECTION_HEADER`, return a typed error — this surfaces as a reintegration diagnostic, never a panic. (Header-rebuild to make room is explicitly out of scope for SP1; recorded as a future enhancement.)
- Signature sketch:
  ```rust
  pub struct AppendedSection { pub rva: Rva, pub size: u32 }
  pub fn append_code_section(
      image: &[u8], name: &str, payload: &[u8],
  ) -> Result<(Vec<u8>, AppendedSection), ReintegrateError>;
  ```

**(2) `emit_domain_dispatcher` — new `crates/vmpx-recompile/src/dispatch.rs`.**
An entry can recover several paths — one per value of its finite input domain (e.g. DllMain reason in `rdx` = 0..=3). Each path is a separately compiled body. At runtime the replacement must select the right body, so this emits a prologue: a `cmp <domain reg>, k; je body_k` chain over the recovered values, with a **fall-through `jmp` to the preserved original entry RVA** as the default. If the domain is a single value, no dispatcher is emitted and the entry points straight at the one body.
- Encoded with the iced-x86 assembler already used in `vmpx-recompile`.
- Unit-tested by re-decoding the emitted bytes and asserting each `je`/`jmp` rel32 resolves to the intended target.
- Signature sketch:
  ```rust
  pub struct DispatchArm { pub value: u64, pub target: Rva }
  pub fn emit_domain_dispatcher(
      domain_reg: PhysicalRegister,
      arms: &[DispatchArm],
      fallback: Rva,         // preserved original entry
      dispatcher_rva: Rva,   // where this dispatcher is placed
  ) -> Result<Vec<u8>, EncodeError>;
  ```

**(3) `redirect_entry` / `redirect_tls_callback` — in `reintegrate.rs`.**
Sound and non-destructive: **does not patch `.vmp` bytes.**
- `redirect_entry`: rewrites `AddressOfEntryPoint` to the dispatcher (or sole body) RVA. The *original* `AddressOfEntryPoint` value is captured first and passed as the dispatcher fallback, so an input outside the recovered domain still reaches the original stub.
- `redirect_tls_callback`: rewrites the relevant pointer in the TLS callback array (an absolute VA) to the replacement's VA.

**(4) `reintegrate_image` — orchestrator in `reintegrate.rs`.**
Takes the mapped image plus per-entry groups and produces the final image.
- Input group shape:
  ```rust
  pub enum RedirectKind { HeaderEntry, TlsCallback { array_index: usize } }
  pub struct ReintegrationBody { pub assignments: BTreeMap<String,u64>, pub bytes: Vec<u8>, pub rva: Rva }
  pub struct ReintegrationEntry {
      pub redirect: RedirectKind,
      pub domain_register: Option<PhysicalRegister>, // None => single body, no dispatcher
      pub domain_variable: Option<String>,           // which assignment key the register carries
      pub bodies: Vec<ReintegrationBody>,            // already compiled at their rva
      pub original_entry_rva: Rva,                   // dispatcher fallback
  }
  pub struct ReintegrationReport {
      pub section: AppendedSection,
      pub entries_redirected: usize,
      pub dispatchers: usize,
      pub loader_valid: bool,
      pub diagnostics: Vec<String>,
  }
  pub fn reintegrate_image(
      mapped: &[u8], entries: &[ReintegrationEntry],
  ) -> Result<(Vec<u8>, ReintegrationReport), ReintegrateError>;
  ```

### 5.3 Layout contract (avoids recompiling bodies)
Bodies are already compiled assuming they start at `next_section_rva` and pack sequentially. Therefore the new section is laid out **bodies first, dispatchers after**:

```
[ body0 ][ body1 ] ... [ bodyN ][ dispatcher_a ][ dispatcher_b ] ...
^ next_section_rva
```

- Body addresses are left untouched → their RIP-relative references and continuation tail-jumps stay valid.
- Each dispatcher is compiled at its own known post-bodies RVA (base = `next_section_rva + Σ(body sizes) + Σ(preceding dispatcher sizes)`), because its `je`/`jmp` targets are rel32 from its own location.
- Entries are redirected to their dispatcher RVA (or, for a single-body domain, directly to the body RVA).

### 5.4 Sound vs experimental mode (the A/C mix)
- **`output` (existing positional arg) is unchanged:** sound emission, written only when `blockers` is empty (`ready_to_write`). SP1 alone does not clear the OEP/import blockers, so sound `output` still correctly refuses until SP2/SP3. Reintegration is *built and run* in SP1, but its result is gated out of `output`.
- **New `--experimental <path>`** (distinct from `output`): writes the reintegrated image regardless of blockers. Always accompanied by:
  - a stderr warning line, and
  - `experimental: true` plus the full `blockers` list in the JSON report.
  Quarantined to its own path; never changes sound-`output` semantics or aliases input/output/report (reuse the existing alias guards in `main.rs`).
- The `reintegration` stage report flips to `complete: true` when the *machinery* succeeds (section + redirect + relayout + disk reparse), independent of `ready_to_write`, and carries metrics (section RVA/size, entries redirected, dispatcher count).

### 5.5 Error handling
- No header-table room, 32-bit size overflow, dispatcher rel32 out of range → typed `ReintegrateError`, surfaced as diagnostics; never panics.
- The final image must reparse (`PeImage::parse`) before any write.
- If `validate_for_loader` fails, the *experimental* file is still written for inspection but flagged `loader_valid: false` with a warning. Sound `output` is never written on any reparse/validation failure.

### 5.6 Scope boundaries (YAGNI — explicitly out of SP1)
- No OEP recovery (experimental mode assumes the header entry's recovered paths are the entry; SP2 makes it sound).
- No import reconstruction (protector imports left as-is; SP3).
- No `.vmp` stripping (replacements live in the new section *alongside* the untouched protector sections; SP5).
- No header-table rebuild to create section-header room (future enhancement).
- No optimization of emitted code (SP6).

### 5.7 Testing plan (fixture-first)
- **Unit:**
  - `append_code_section`: header-room math (success and the no-room error), `SizeOfImage` bump, alignment of RVA and raw pointers.
  - `redirect_entry` / `redirect_tls_callback`: field and array-pointer rewrites; original entry captured for fallback.
  - `emit_domain_dispatcher`: multi-arm `cmp/je` chain + fallback `jmp`, verified by re-decoding and checking resolved targets; single-arm → no dispatcher.
- **Integration (synthetic fixture, `p06_*` style):** a minimal PE with a trivial recovered VM entry → `reintegrate_image` → assert: new section present with R+X characteristics, `AddressOfEntryPoint == dispatcher RVA`, dispatcher branches to the body, output reparses and passes `validate_for_loader`.
- **End-to-end:** `clean --experimental <out>` on the existing fixture asserts the produced file reparses and `reintegration.complete == true` and `experimental == true` in the report.
- **Out-of-band (manual, local):** `clean --experimental` on a local VMP addon; run on Windows and observe behavior.

### 5.8 Files touched
- **New:** `crates/vmpx-rebuild/src/reintegrate.rs`, `crates/vmpx-recompile/src/dispatch.rs`, fixtures + tests under the respective `tests/` dirs.
- **Edited:** `crates/vmpx-rebuild/src/lib.rs` and `crates/vmpx-recompile/src/lib.rs` (module + re-exports); `crates/vmpx-cli/src/clean.rs` (gather per-entry groups from `functions`/`placed` instead of dropping them; call `reintegrate_image`; populate the `reintegration` stage report); `crates/vmpx-cli/src/main.rs` (`--experimental <path>` arg on `Command::Clean`, alias guards, write path, stderr warning).

### 5.9 SP1 definition of done
- All unit + integration + e2e tests above pass in CI against synthetic fixtures.
- `clean --experimental` produces a reparsing, loader-valid PE on the fixture, with correct entry redirect and dispatcher behavior.
- Sound `output` behavior is unchanged (still refuses on protected inputs; still a verified no-op on clean inputs).
- A local VMP sample has been run through `clean --experimental` and behaviorally spot-checked by the author.

---

## 6. SP2–SP6 sketches (each to be fully specified in its own cycle)

- **SP2 — OEP recovery.** Prove the original entry (e.g. via the first native-transition / VM-exit target reached from the protected header entry) so `redirect_entry`'s target is a proven original-entry semantics rather than an assumption. Clears blocker 1; promotes experimental entry emission to sound.
- **SP3 — Import-set reconstruction.** Reduce the parsed protector-era import set to the application's real imports (resolve indirect/VM-mediated IAT usage to concrete symbols), and rebuild a conventional import directory. Clears blocker 2. Builds on `vmpx-rebuild/src/import_table.rs` and the existing `Imports` analysis.
- **SP4 — Executable (non-DLL) entry domain recovery.** Implement finite input-domain recovery for executable entry points (currently `clean.rs:624` refuses). Widens coverage beyond the DLL-entry/TLS `rdx` reason domain.
- **SP5 — Dangling `.vmp` reference resolution + section stripping.** Once entry, TLS, and reachable routines are devirtualized and imports reconstructed, prove no remaining references into `.vmp*` sections, then strip them. Clears blocker 4; produces "clean" output in the strong sense.
- **SP6 — Optimizing recompilation (VTIL) + differential verification engine.** Wire the already-present VTIL bridge (`native/vtil-bridge/`, `vmpx_vtil_optimize` → `vtil::optimizer::apply_all`) into the lift → recompile path (the stubbed `vtil` stage at `clean.rs:1255`), plus the repo's existing expression simplification. Lower the *optimized* IR to minimal native code so output reads like compiler output.
  - **Equivalence-preserving requirement:** every pass must preserve observable behavior. SP6 ships with a **differential verification engine** — the "debug engine" — that executes the original VM and the optimized native replacement over the recovered input domain and asserts identical register/memory/effect results. This is the pass validator and the local-sample equivalence oracle.
  - **Independently sequenceable:** VTIL optimization sits between lift and recompile and can land as a quality improvement before SP5, turned on by default only once its verifier is trusted.
  - **Open question (carried to SP6 planning):** whether the "debug engine" should *also* emit readable debug artifacts (PDB symbols or a commented annotated disassembly of the output) in addition to being the differential verifier. Default assumption: differential verifier first; readable-symbols as a separate optional deliverable.

---

## 7. Risks & open questions
- **Header-table room (SP1).** Some VMProtect layouts may leave no room for an extra section header. SP1 fails cleanly here; a header-rebuild enhancement may be needed for real coverage.
- **Dispatcher fallback correctness (SP1).** Falling back to the original (still-protected) entry is only a safety net for inputs outside the recovered domain; if the domain is proven complete, the fallback is unreachable. Experimental mode with an incomplete domain relies on it and must label that.
- **Soundness of experimental output.** Experimental emission can be confidently wrong; mitigated by quarantine (separate path), loud labeling, and the blockers list in the report. It is a test artifact, not a product.
- **Behavior-vs-bit reversibility.** Must remain explicit in user-facing docs so "clean/reversible" is not misread as original-file reconstruction.
- **SP6 verifier scope.** Differential verification over a *finite recovered domain* proves equivalence on that domain, not universally; this bounds the strength of the "cleanest output" claim.

---

## 8. Immediate next step
On author approval of this spec, proceed to the **writing-plans** skill to produce the SP1 implementation plan (TDD, task-by-task). SP2–SP6 are not planned until SP1 lands and its lessons are folded back in.
