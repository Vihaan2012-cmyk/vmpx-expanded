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

Value dispatcher for branching functions
    ``emit_value_dispatcher`` and ``emit_displaced_stub`` in ``vmpx-recompile``, used by
    ``clean`` to reintegrate recovered branching functions. Commit ``4ef23a2``.

Wiring the soundness cores into ``clean``
    ``clean_wiring.rs`` connects the coverage gate, the dangling-reference checker and
    the import reduction to the pipeline (commit ``5b96caa``, ``CleanReport`` schema 3).
    The OEP, reintegration and ``.vmp`` blockers now drop only when their proof holds.
    The import blocker always stays, because the referenced-IAT-slot set is intentionally
    empty. Anti-debug is reported recognition-only. The full clearing path is
    unexercised end-to-end, since no real VMProtect corpus is available in CI.

Synthetic end-to-end fixture
    A synthetic VM-entry fixture and pipeline test (commit ``7946fd8``).

SP3, SP4, SP5 cores
-------------------

SP3: application import-set reconstruction
    ``import_reconstruct.rs`` (commit ``d12fc16``), later wired into ``clean``.

SP4: dangling ``.vmp`` reference checker
    ``dangling.rs`` (commit ``4bbc81f``).

SP5: differential equivalence verifier and neutralize transform
    ``equivalence.rs`` (commit ``0d05adf``) and ``neutralize.rs`` (commit ``908dc0b``).

VTIL bridge
    The feature-gated Rust FFI binding in ``vmpx-vtil-sys`` (commit ``a4ea913``).

Real-sample decode and lift (VMProtect 3.11 Compact)
----------------------------------------------------

The first real protected sample exposed a VM unlike the synthetic fixtures: keyless,
with table-indexed dispatch. It was decoded in stages, each a commit on the
``sp2-oep-generalized`` line.

Dispatch (``a86c405``, ``194a09c``)
    Table-indexed keyless dispatch modelled; postlude-branch dispatch and the
    branch handlers recognised.
Operands and ALU (``2ed6552``)
    Keyless raw operand recovery, ``Add`` and ``Sub`` classification and value flow,
    16-bit virtual-register slot indices.
Stream decode (``9801512``)
    Table base propagated through the stream; linear VM instruction stream decoded.
Mul, Div, narrow widths (``2fd129c``)
    ``DIV``/``MUL`` and narrow-width handlers classified.
CFG walk (``6b19f26``)
    Control-flow walk over VM branches for complete function decode.
Branches and checksum (``9e50df6``)
    Intra-branch and checksum handlers decoded; this closed the handler set at
    256 of 256 classified.
Native calls (``237919f``)
    Keyless native-call bridge handlers decoded.
Performance (``dcecea0``)
    Structural sharing and simplification bound symbolic expression size.
Lift (``a09a025``, ``1e5d8ad``)
    Decoded-stream lift adapter lifts linear functions to native code; a CFG-level
    lift handles branching functions and bridges native calls.
Branch domains (``921ab03``)
    Finite domains proven for non-affine ``vm_flags_sub`` branch conditions.

Memory and imports (``07f631d``, ``5e2f3af``, ``06913ae``)
    Statically constant memory and known TEB reads folded, from read-only sections
    only; VM native-call targets resolved to concrete imports, recovering
    ``HeapAlloc`` and ``HeapFree``.
Branching lift (``ee20cb4``, ``b98c781``)
    Memory reads hoisted with alias checks so branching functions lift; domains
    proven for untracked-read and non-add-flags branch selectors.
Hybrid image (``5a723b0``, ``eab392f``)
    Decoded-stream lifts spliced into the ``--experimental`` image behind
    ``--decoded-lift``, with the stubs redirected.
Nested and continuation exits (``5d86035``, ``f13772f``)
    Nested VM-to-VM continuations followed in the CFG; region (continuation) exits
    lifted. The first branching function (``0x126b02``) devirtualizes to native code.

Result: linear and branching functions devirtualize; the output is a valid hybrid PE
with the lifted functions spliced in. The sound frontier and what is refused are written up in
:doc:`/vmprotect/devirtualization/static-frontier`.

Generalizing past the first sample
----------------------------------

The first sample was one build of one function family. Testing against freshly
protected builds showed that the decoder was over-fitted to it, and that re-protecting
the same input could swing results by tens of points. The work that followed was
generalization, each step judged against several builds rather than one.

Packer and layout (``b79bc9a``, ``27246b7``, ``1e39f41``)
    Real VMProtect 3.11 packer layouts, including multi-section streams bound by
    end-of-stream length and destinations that are uninitialized or not stream
    backed, unpack and relayout correctly.
Handlers and codecs (``f420459``, ``3e17e2e``, ``01c2214``)
    Division, modulo and bit-permutation handler specifications; the dispatch and
    keyed-operand codec of the bit-permutation family; two-candidate bootstrap role
    assignment disambiguated by decode-coherence evidence.
Branch resolution (``d3f80b2``, ``ec080cf``, ``8201b32``)
    The CPUID processor-signature anti-analysis gate resolved by a sound both-target
    over-approximation; read-only image loads concretized to resolve jump and
    constant tables; stack and local stores proven non-aliasing so post-store
    branches resolve.
Lift coverage (``12597c3``, ``e048d64``, ``5dd7897``)
    Remaining intrinsics and flag operations lowered; natural and reducible
    multi-exit loops emitted; memory reads captured in order across CFG nodes.

Mutation-mode deobfuscation (SP6)
---------------------------------

Mutated code is treated as a program, not a pattern (:doc:`/vmprotect/devirtualization/attacking-mutation`).
The design is a solver tournament.

Contract and solvers (``0844d0e``, ``7e8aaa4``, ``7531419``, ``19dc413``)
    Shared problem and candidate types, then a portfolio of solvers: linear MBA,
    algebraic MBA rewriting, opaque predicates, substitution canonicalization, and
    later dictionary and synthesis solvers.
Scoring and ELO (``8ac6607``, ``2215117``)
    A differential input-testing scorer ranks candidates; a cost-aware ELO ladder
    routes each problem class to the best-rated solvers.
SMT gate (``ad2398c``, ``f2ae51d``)
    An equivalence gate is the only route from the experimental tier to the sound
    tier. A candidate that merely passes input testing is not promoted.
Native CFG and end to end (``cea4ecc``, ``44ade88``, ``b35a8c0``)
    Multi-block native CFG recovery, including decoding at control-flow-reached
    offsets to resolve overlapping-instruction anti-disassembly, and an end-to-end
    binary demutation pass. Measured against a synthetic benchmark with a ground
    truth oracle (``09bc712``); coverage on real mutated binaries is partial.

Protection removal and packers
------------------------------

Section stripping (``618a0f9``, ``14146f9``)
    A stripping primitive that refuses when any reference would dangle, exposed as the
    ``strip`` subcommand and an opt-in ``clean --experimental --strip`` stage.
Packer front-end (``35bf859``)
    Detection of UPX, ASPack, PECompact, MPRESS and FSG, reported by ``detect`` and
    routed by ``generic-unpack`` (:doc:`/tool/cli/generic-unpack`).
Real UPX end to end (``c0f5cfb``)
    The emulator runs a real UPX unpack stub to its original entry point with a
    faithful model of image-page writes. This is the one packer validated on a real
    sample.

SP7: contained dynamic assist
-----------------------------

Several gaps were confirmed to need runtime observation: import protection,
runtime-initialized globals and recursion depth. The design
(:doc:`/vmprotect/devirtualization/dynamic-assist`) was built as ``vmpx-emulate``,
a pure in-process interpreter.

Foundation (``222d972``, ``033022e``, ``9b35eb3``)
    PE loader and minimal environment, mutable in-callback hooks and an
    ``ObservedFacts`` hand-off format, then Win32 API stubs with import observation.
Dump and OEP (``b3f5496``, ``fdbaa06``)
    OEP detection by write-then-execute, image dump, runtime-global capture, VM
    dispatcher observation and recursion-depth capture.
ISA and environment (``6f4e629``, ``369b859``, ``3fb1b59``, ``b680e98``, ``0feb633``)
    The modelled instruction set widened step by step, with exact flag semantics or a
    stop; loader environment, module export tables and ntdll syscall stubs modelled
    so protector startup reaches the OEP.
Bridges into the static model (``d03c40f``, ``f448c7b``, ``28518aa``, ``cfce0e5``)
    Observed imports for reconstruction, observed runtime globals to seed symbolic
    reads, observed recursion depth for a bounded unroll, and observed branch targets
    for computed targets. All are experimental and labelled.
Containment (``d03e35f``)
    Cargo is forced offline for the whole workspace (``net.offline = true``), and the
    emulator's API layer checks a network-API denylist before any API is modelled.

Cross-profile VM transitions and the measured ceiling
------------------------------------------------------

The largest late gain came from the walker. On a large real-world protected module,
many functions stalled because a VM exit popped a target that begins a block in a
*different* register-role and direction profile.

Cross-profile walker (``e4395a1``, ``2d15f41``)
    The CFG walker follows a proven popped-target VM transition and re-resolves
    register roles per block. A per-entry node and instruction budget makes the
    walk fail closed; the first version needed that bound because memory use grew
    on large inputs.
Dispatch residual and operand timing (``f35a48e``)
    Further recovery of dispatch residuals, operand-timing and nested-profile
    transitions.
CPUID leaf-only variant (``3c569ba``)
    Another handler variant modelled.

Measured result: **313 of 814 (38.5%)** virtualized functions on that module are
soundly recovered, under the all-paths-reach-exit oracle. The remaining 501 were
examined and are documented as beyond sound static reach rather than left as
open bugs: branch targets computed from runtime-pointer memory reads, nonlinear
``SHLD``/``SHRD`` double-shift targets with no finite input domain, and handlers
outside the file-backed image or behind import protection
(:doc:`/vmprotect/devirtualization/static-frontier`). The emulator can observe some
of these facts, but only on the experimental tier.

Verification scope
------------------

There is no protected corpus in CI. Correctness is verified by unit and synthetic tests plus
out-of-band runs against locally protected samples, including multi-build corpora, where
variance between builds of the same input is the main hazard. The goal remains
behaviour-reversible output, not bit-reversible.

Design documents
----------------

* ``docs/superpowers/specs/2026-10-29-vmpx-sp1-native-reintegration-design.md``
* ``docs/superpowers/plans/2026-10-29-vmpx-sp1-native-reintegration.md``
* ``docs/superpowers/specs/2026-10-29-vmpx-sp2-oep-generalized-recovery-design.md``
* ``docs/superpowers/plans/2026-10-29-vmpx-sp2-oep-generalized-recovery.md``
