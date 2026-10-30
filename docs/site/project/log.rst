Development Log
===============

A record of how the pipeline was built: each task, its verification, the review verdict
and the commit. The git history is the authoritative record; this page is the readable
narrative.

Origin
------

The project began as a security review of the upstream
`skyboundfs/vmpx <https://github.com/skyboundfs/vmpx>`_ repository, cloned and inspected
statically. Verdict: no malware. It is a VMProtect static-analysis and devirtualization
toolkit in Rust of roughly 42,000 lines, with ``unsafe`` forbidden workspace-wide, no
``build.rs``, and no network, process, dynamic-loading or exfiltration surface. The one
unusual dependency, ``serde_json`` pulling ``zmij`` in place of ``ryu``, is legitimate:
``serde_json`` migrated its float formatting from Ryū to Żmij. The native VTIL bridge is
a clean FFI wrapper.

SP1: Native reintegration
-------------------------

Built with subagent-driven development: an implementer and a reviewer per task, plus a
final whole-branch review. The key discovery was that ``vmpx-rebuild`` already implemented
and tested ``append_section`` and ``set_entry_point``, and that ``compile()`` emits a
Win64 frame, so the dispatcher had to be a naked trampoline hand-encoded with iced. That
trimmed SP1 to five tasks.

Task 1: naked domain-dispatcher encoder
    ``vmpx-recompile/src/dispatch.rs`` emits ``cmp reg, imm; je body`` per recovered
    value, then ``jmp fallback``. Commit ``ae54768``; 3 tests, including a decode test
    asserting that ``je``/``jmp`` targets resolve to the right absolute addresses.

Task 2: TLS-callback redirect
    ``set_tls_callback`` in ``vmpx-rebuild``. Rejects a non-executable target, rewrites
    the TLS array slot, reparse-validates. Commit ``e6aa8af``.

Task 3: reintegration orchestration seam
    ``reintegrate()`` chains ``append_section``, ``set_entry_point``, ``set_tls_callback``
    and ``validate_for_loader``, recording ``loader_valid`` as a bool without failing on an
    invalid result. Commit ``2cadf54``. Reviewed with zero findings.

Roadmap change (mid-run)
    Generalized function recovery moved into the earliest sub-project, combined with OEP,
    making SP2 "OEP plus generalized recovery". Imports became SP3, stripping SP4,
    optimization SP5; the egui UI was folded into the current work.

Task 4: wire reintegration into the clean pipeline
    Stops discarding the compiled replacements; lays out bodies first and dispatchers
    after; fixes the addressing invariant (bodies compiled at
    ``image_base + next_section_rva(conventional)``). Review found a spec gap (TLS
    redirects hardcoded empty), then a cursor-drift bug when a group's dispatcher assembly
    fails. Both fixed; the second by rolling the layout back on failure. Commits
    ``3f6fdd6``, ``6bdee34``, ``c6d6835``.

Task 5: ``--experimental`` output path
    Sound output unchanged. ``--experimental <path>`` writes the reintegrated image
    regardless of blockers, with a stderr warning and ``CleanReport.experimental``.
    Commit ``fb69a40``.

``vmpx-ui``
    An egui/eframe crate driving the CLI. Build-verified (no display in CI). Commit
    ``0258159``.

Final review and merge
    An Opus whole-branch review was clean to merge. One fix wave (commit ``2883c34``): a
    single-body group whose domain is not provably complete now gets a dispatcher with a
    fallback; dead code removed; a checked add for a VA; ``set_tls_callback`` tightened;
    the UI handles a worker panic. SP1 merged to ``main`` by fast-forward with 239 workspace
    tests passing.

SP2: OEP and generalized recovery
---------------------------------

A sequential spine: OEP, then discovery, live-in inference, input-closed and
decidable-branching recovery, emission, clearing blockers.

Task 1: OEP recovery
    ``decide_oep`` in ``vmpx-vm/src/oep.rs``. Round 1 review found a latent unsound case (a
    partial family could declare a wrong OEP) and a native candidate next to a non-native one
    still recovering. Both fixed; 246 tests. Commits ``64cd125``, ``90d9f38``.

Task 2: generalized VM-entry inventory
    A deduplicated scan merged with export cross-references into ``CleanReport.vm_inventory``.
    Ruling: an export entry that cannot be re-resolved must be treated as uncovered by the
    coverage gate, so it can never clear by vanishing. Commit ``1dfe723``; 253 tests.

Task 3: per-function live-in and calling-convention inference
    ``infer_live_in`` and a ``calling_convention`` adapter. Ruling: the adapter under-reports,
    so it is diagnostic only; emission binds all architectural registers. Commit
    ``c1ebf00``; 264 tests.

Task 4a: input-closed function recovery
    ``is_input_closed`` and ``recover_input_closed_functions``. Reviewed sound. Ruling:
    carry the collected effects (stores and calls) as well as the exit. Commit ``e9d31c7``;
    270 tests.

Task 4b: finite-domain branching recovery and executable entry
    A spike concluded that the general branching case is undecidable, so a sound decidable
    subset was implemented and everything else refused: 8-bit slices and add-flags covers. The
    reviewer independently re-derived the 32/64-bit cover. Commit ``81a7a24``; 286 tests.

Task 5: reintegrate recovered functions
    ``redirect_vm_stub`` writes a ``jmp`` over each recovered single-body function's VM
    stub. The step edits existing code bytes of a protected binary, so it is confined to the
    ``--experimental`` image. Commit ``616aad8``.

Task 6: strict coverage gate
    A pure ``decide_coverage`` where every doubt keeps its blocker. Commit ``1452bb8``. The
    UI was updated to surface the SP2 fields (commit ``31d4d53``).

Synthetic end-to-end fixture
    A synthetic VM-entry fixture and pipeline test (commit ``7946fd8``).

SP3, SP4, SP5 cores
-------------------

SP3: application import-set reconstruction
    ``import_reconstruct.rs`` (commit ``d12fc16``).

SP4: dangling ``.vmp`` reference checker
    ``dangling.rs`` (commit ``4bbc81f``).

SP5: differential equivalence verifier and neutralize transform
    ``equivalence.rs`` (commit ``0d05adf``) and ``neutralize.rs`` (commit ``908dc0b``).

VTIL bridge
    The feature-gated Rust FFI binding in ``vmpx-vtil-sys`` (commit ``a4ea913``).

Verification scope
------------------

There is no protected corpus in CI. Correctness is verified by unit and synthetic tests plus
out-of-band runs against locally protected samples. The goal remains behaviour-reversible
output, not bit-reversible.

Design documents
----------------

* ``docs/superpowers/specs/2026-10-29-vmpx-sp1-native-reintegration-design.md``
* ``docs/superpowers/plans/2026-10-29-vmpx-sp1-native-reintegration.md``
* ``docs/superpowers/specs/2026-10-29-vmpx-sp2-oep-generalized-recovery-design.md``
* ``docs/superpowers/plans/2026-10-29-vmpx-sp2-oep-generalized-recovery.md``
