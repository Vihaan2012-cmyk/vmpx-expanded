# vmpx-expanded build journal

A detailed, step-by-step log of the autonomous build of the vmpx devirtualization roadmap.
Newest entries appended at the bottom. The terse recovery ledger lives separately at
`.superpowers/sdd/2026-10-29-vmpx-sp1-native-reintegration/progress.md`; this file is the
human-readable narrative with reasoning behind every decision.

Repo: private, github.com/Vihaan2012-cmyk/vmpx-expanded · Branch: `sp1-native-reintegration` (off `main` @ 4bfc2fa)
Commit author: GitHub noreply, no Co-Authored-By line. Code comments kept lean (user preference).

---

## 0. Standing instructions from the user (apply for the whole run)

- Run autonomously overnight, agent-driven, until all SPs are done. Do NOT stop to check in.
- "Take the best choice at every crossroad" — I make rulings and keep going; log each one.
- Do NOT park SP2's hard problem (per-function input-domain / calling-convention inference) — keep chewing on it, escalating models and switching approaches, until it genuinely verifies. Never fabricate a win; the differential verifier + tests are the gate.
- Do SP3, SP4, SP5 too. Do not stop at all.
- Avoid `rm -rf` — skip the SDD workspace-delete step; leave scratch dirs in place.
- Keep really detailed notes at every step (this file).
- Lean code comments; noreply author, no co-author line.

## 1. Origin & verdict (malware review)

User asked to scan github.com/skyboundfs/vmpx for malware. Cloned, inspected statically. Verdict: **no malware** — a VMProtect static-analysis/devirtualization toolkit in Rust (~42k LOC). `unsafe` forbidden workspace-wide, no `build.rs`, no network/process/dynamic-loading/crypto-exfil surface. The one scare — `serde_json` depending on `zmij` instead of `ryu` — turned out legitimate: serde_json genuinely migrated its float formatting from Ryū to Żmij. The native C++ VTIL bridge is a clean FFI wrapper.

## 2. Goal & strategy (brainstorming outcomes)

- Goal: carry the pipeline to **clean, behavior-preserving ("reversible") devirtualized output**, as provable milestones not a single finish line.
- "Reversible" = behavior-reversible, NOT bit-reversible (virtualization is lossy; we rebuild semantics, not original bytes).
- Strategy: **mix of A (soundness-first) and C (hybrid/decomposed)**. Sound, proof-gated path is the default deliverable; an explicit, labeled `--experimental` emission mode is scaffolding to test the machinery early. Soundness is the spine.
- Validation oracle: **synthetic fixtures (CI) + the user's local VMProtect payware samples (out-of-band, never committed)**. Universal "any binary" correctness stays aspirational; each milestone's "done" is fixtures + local spot-check.
- Anti-debug: our analysis is **static**, so runtime anti-debug never fires against us. Output policy is **user-selectable**: faithful reproduction (default, equivalent) vs `--neutralize` (debuggable, deliberately non-equivalent). Integrity/checksum checks interact with reintegration (SP1 risk; resolved at `.vmp` stripping).
- **Fully static** — no dynamic dumper (that would reintroduce all anti-debug problems).
- UI: **egui/eframe**, folded into current work.

## 3. Roadmap (current, after user reprioritization)

- **SP1** — Native reintegration + image emission (backbone; `--experimental` mode). *IN PROGRESS.*
- **SP2** — OEP recovery **+ generalized virtualized-function recovery** (prove original entry; discover every VM entry via call-site/xref; infer per-function calling convention + finite input domain incl. non-DLL executable entries; lift + reintegrate each). The hard research weight; user said keep chewing until it verifies.
- **SP3** — Import-set reconstruction.
- **SP4** — Dangling `.vmp` reference resolution + section stripping.
- **SP5** — Optimizing recompilation (VTIL) + differential verification engine + anti-debug `--neutralize`.
- **UI** — egui `vmpx-ui` crate, folded into current work (built this session, build-verified only; author runs it).

Full design + roadmap: `docs/superpowers/specs/2026-10-29-vmpx-sp1-native-reintegration-design.md`.
SP1 implementation plan: `docs/superpowers/plans/2026-10-29-vmpx-sp1-native-reintegration.md`.

## 4. SP1 execution (Subagent-Driven Development, Sonnet 5.5 agents)

Key discovery before planning: `vmpx-rebuild` ALREADY implements + tests `append_section` (file-backed section append with full header-slack/security/alignment/reparse validation) and `set_entry_point` (entry redirect). And `compile()` emits a Win64 frame, so a dispatcher must be a NAKED trampoline (hand-encoded via iced `code_asm`). That trimmed SP1 to: dispatcher encoder, TLS redirect, orchestration seam, clean.rs/main.rs wiring, `--experimental`.

Pre-flight conflict scan rulings:
- **R1** — Tasks 2 & 3 share a minimal-PE test helper: T2 creates module-level `minimal_conventional_pe()` (+ `pe_with_one_tls_callback()` layered on it); T3 reuses it. (Satisfied.)
- **R2** — T5 also edits clean.rs to add `CleanReport.experimental` (struct lives there).

### Task 1 — naked domain-dispatcher encoder (vmpx-recompile/dispatch.rs)
- Implementer: Sonnet. Commit `ae54768`. 3/3 dispatch tests (incl. a decode test asserting je/jmp targets resolve to the right absolute VAs), clippy clean.
- Review (Sonnet): **spec ✅, quality approved**, 5 Minors deferred (cmp-immediate values untested; zero-arm case untested; width-check message says "32-bit" but i32::try_from rejects 0x8000_0000..=u32::MAX; no flags-clobber comment; first test subsumed by decode test).
- iced `je(u64)`/`jmp(u64)`/`cmp(reg,i32)` worked directly — no label workaround needed.
- **Complete** (4bfc2fa..ae54768, review clean).

### Task 2 — TLS callback redirect (vmpx-rebuild/reintegrate.rs)
- Implementer: Sonnet, in an **isolated git worktree** (run in parallel with T1 — disjoint crates, to honor the user's "use multiple agents at once" while avoiding a shared-index race). Branch `worktree-agent-a513...`, commit `11ae143`. 23/23 crate tests, TDD RED verified by stubbing.
- Design note surfaced for T3: `minimal_conventional_pe()` `.text` spans RVA [0x1000,0x3000).
- Review (Sonnet): **spec ✅, quality approved, R1 satisfied**, Minors deferred (NoTlsDirectory/non-executable paths untested; doc says "file-backed" but check is contains_mapped_rva; PE32+ layout hardcoded without PeKind check; verbose `# Errors` doc; new private read_u64 helper possible dup).
- Merged into the branch via **cherry-pick** (disjoint files → clean, linear history) as `e6aa8af`; worktree removed (no rm -rf — used `git worktree remove`). Merged tree 23/23 green.
- **Complete.**

### Task 3 — reintegration orchestration seam (vmpx-rebuild/reintegrate.rs)
- Implementer: Sonnet. Commit `2cadf54`. 24/24 crate tests. Chains append_section → set_entry_point → set_tls_callback(per redirect) → validate_for_loader (records loader_valid as bool, doesn't fail on invalid). Section chars 0x6000_0020.
- Review (Sonnet): **spec ✅, quality approved, R1 satisfied (reuses the helper, no dup), no codegen-crate dependency, ZERO findings at any severity.**
- **Complete** (no deferred minors).

### Roadmap change (mid-run, user-directed)
User: put generalized function recovery in the earliest SP, combined with OEP; and build the UI now. Edited the spec: **SP2 = OEP + generalized recovery** (absorbs old exec-entry item), imports→SP3, stripping→SP4, optimization→SP5, UI folded into current work. Docs-only commits `bef9964`, `0d8988f` (pushed). Done while T4 implementer ran — committed docs-only to avoid sweeping its in-flight code.

### Task 4 — wire reintegration into the clean pipeline (clean.rs + main.rs)
- Implementer: Sonnet (the hardest task; ~18 min, 220k subagent tokens). Commit `3f6fdd6`, status **DONE_WITH_CONCERNS**. Implemented `layout_reintegration_payload` + its unit test (bodies-first/dispatcher-after), wired the HEADER entry fully, fixed the core addressing invariant (bodies compiled at `image_base + next_section_rva(conventional)`, not the RVA-mapped image), `analyze_clean_with_image` now returns a 3-tuple with both callers updated, reintegration stage reports complete+metrics. Concern: TLS + nested-continuation bodies compiled/counted but not grouped into redirects.
- Review (Sonnet): **spec ❌ (one gap)** + quality approved with Minors. Crucial catch: the "fallback" I put in my dispatch prompt does NOT exist in the brief/plan — plan Step 3 says wire `tls_redirects`, and Task 5's self-review expects TLS metrics, so **TLS wiring was in scope**. The reviewer verified the addressing fix is correct (the hard part). Minors: `entries_redirected` hardcoded to 1; no dedup/validation of domain values; header path missing `native_rdx_initial` silently excluded; redundant second `relayout_conventional`.
- **Ruling:** TLS-redirect wiring is in-scope and enters the fix loop (the T2/T3 plumbing exists). Nested-continuation grouping (second-order rewrite of continuation targets that re-enter the VM — genuinely unspecified in the plan) is DEFERRED to SP2 generalized recovery. Reason: plan text + Task-5 self-review require TLS metrics and the plumbing exists; nested continuation is a harder second-order change. Cost if wrong: TLS index-mapping subtlety → rework; nested-continuation deferral means experimental output won't redirect VM-reentry paths (acceptable for SP1 L1).
- **Fix round 1 dispatched** (resumed the original implementer): wire TLS redirects (tag each tls_exit_state with its TLS entry RVA → map to callback index → lay out its bodies after the header group → populate `tls_redirects`), derive `entries_redirected`, dedup domain values, diagnose missing-key header paths. Nested-continuation explicitly left out. *IN PROGRESS.*

---
