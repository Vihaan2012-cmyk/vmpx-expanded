Roadmap
=======

vmpx-expanded takes the upstream `vmpx <https://github.com/skyboundfs/vmpx>`_ analysis
framework toward clean, behaviour-preserving devirtualized output. The upstream project
recovers a great deal (unpacking, detection, PE modelling, disassembly, VM handler
discovery, symbolic execution, native continuations) but deliberately refuses to emit a
devirtualized binary until the original entry point and application import set are
proven. This fork plans and implements the remaining work in sound, independently
testable stages.

Completion strategy: a mix of *soundness-first* and *hybrid/decomposed*. The sound,
proof-gated path is the real deliverable and the default. An explicit, loudly labelled
experimental mode exists as scaffolding to exercise the machinery and to carry facts
that were observed rather than proven.

Real-sample status
------------------

Static devirtualization now fully recovers real, freshly-protected VMProtect 3.11
"Compact" functions end to end: opaque bytecode, a complete control-flow graph, and
lifted native IR. Completeness is certified by an inflation-proof oracle
(``all_paths_reach_vm_exit``): a function counts as recovered only if every path of
the walked CFG provably reaches a VM exit, so a partial walk cannot be reported as
complete. The Compact handler set is fully classified (256 of 256), table-indexed
keyless dispatch is modelled, branches are resolved by symbolic execution and proven
input domains, VM-side calls resolve to concrete imports, and nested VM-to-VM
continuations are followed. ``clean --experimental --decoded-lift`` splices lifted
functions into a hybrid PE with the stubs redirected. See
:doc:`/tool/pipeline/compact-vm` and :doc:`/tool/pipeline/function-lift`.

The current measured ceiling, on a large real-world protected module with 814
virtualized functions, is **313 of 814 (38.5%) soundly recovered**. The key lever past
the earlier ceiling was a *cross-profile VM-transition walker*: the recovery walker
switches the active register-role and direction profile at a proven popped-target
transition (register roles are re-resolved per block), under a fail-closed per-entry
node and instruction budget.

The other 501 functions are not recovered, and this is a stated limit, not a pending
fix. They fall into three groups that are beyond sound static reach:

* branch targets computed from runtime-pointer memory reads;
* nonlinear ``SHLD``/``SHRD`` double-shift targets with no finite input domain;
* handlers outside the file-backed image, or behind import protection.

The reasons are written up in :doc:`/vmprotect/devirtualization/static-frontier`.
Build-to-build variance is also large (re-protecting the same input moves results a
long way), so a figure measured on one build is not a promise about another. What the
dynamic engine adds, and what it cannot, is in
:doc:`/vmprotect/devirtualization/dynamic-assist`.

Sub-projects
------------

.. list-table::
   :header-rows: 1
   :widths: 8 30 20 42

   * - SP
     - Scope
     - Blocker addressed
     - Status
   * - SP1
     - Native reintegration and image emission
     - No devirtualized image produced
     - Complete and merged. Section append, header and TLS redirects, domain
       dispatchers, ``--experimental``, ``vmpx-ui``
   * - SP2
     - OEP recovery and generalized virtualized-function recovery
     - Original entry point not recovered; ``.vmp`` still referenced
     - Implemented and exercised on real protected binaries: OEP decision, VM-entry
       inventory, live-in inference, input-closed and decidable branching recovery,
       the cross-profile transition walker, value dispatchers and the strict
       coverage gate. Measured sound recovery on a large module is 313 of 814
       functions (38.5%); the rest is documented as beyond static reach. Nested
       continuations are followed and lifted, but the sound gate still does not count
       a nested entry as covered
   * - SP3
     - Application import-set reconstruction
     - Import set not proven
     - Proven-set reduction and a conventional import-directory builder
       (``reconstruct_clean_imports``, proven bindings only) are wired into ``clean``.
       VMProtect 3.11 import protection is resolved at run time, so it is
       dynamic-only: the sound path keeps the import blocker, and bindings observed in
       the contained emulator feed only the labelled experimental tier
   * - SP4
     - Dangling ``.vmp`` reference resolution and stripping
     - ``.vmp`` sections remain referenced
     - A section-stripping primitive that refuses on any dangling reference, a
       ``strip`` subcommand, and an opt-in ``clean --experimental --strip`` stage.
       Real VMProtect 3.11 packer unpacking (LZMA, multi-section layouts) and
       conventional relayout are in place. Stripping is not part of the sound
       ``OUTPUT`` path
   * - SP5
     - Optimizing recompilation (VTIL), differential verification, anti-debug policy
     - Quality, not coverage
     - Differential equivalence core as a library, plus an emulator-based
       differential harness for checking lifts against the original. The
       ``neutralize`` transform is recognition-only in ``clean`` (``applied: false``).
       VTIL bridge binding exists behind a feature, not yet in the pipeline
   * - SP6
     - Mutation-mode deobfuscation
     - Mutated native code and handlers
     - Implemented: a solver tournament (linear and general MBA, arithmetic
       identities, opaque predicates, substitution canonicalization, dictionary and
       synthesis solvers) ranked per problem class by ELO on differential
       input-testing, native CFG recovery, dead-term elimination, an end-to-end
       binary pass, and an SMT equivalence gate that is the only route to the sound
       tier. Validated against a synthetic benchmark with ground truth; coverage on
       real mutated binaries is partial
   * - SP7
     - Contained, offline dynamic assist for user-mode samples
     - Static limits: runtime globals, import protection, recursion depth, packers
     - Implemented as ``vmpx-emulate`` plus bridges into the static model, on a
       labelled experimental tier only; packer detection and a bounded
       ``generic-unpack`` path. Observed facts never reach the sound output. See
       :doc:`/vmprotect/devirtualization/dynamic-assist`

What remains open
-----------------

* **The static ceiling.** About 61% of the functions on the large measured module are
  dynamic or runtime-dependent. Raising 313 of 814 further, soundly, needs new proof
  techniques rather than tuning; some of the remainder is irreducible without the
  runtime environment.
* **Generalization across builds.** Per-build variance in the VM instance means a fix
  must be judged against several protections of the same input, not one.
* **Host-called virtualized functions.** Observing them in the emulator needs a
  host-API driver that supplies the caller's inputs; some branches are irreducibly
  environment-dependent.
* **Sound import set.** Blocked by design while import protection is resolved at run
  time.
* **Non-VMProtect packers.** UPX has been validated end to end in the emulator.
  ASPack, PECompact and MPRESS are routed to the same bounded path but have not been
  validated on real samples; FSG is detected by a low-confidence heuristic and is not
  routed.

Definition-of-done ladder
-------------------------

Each rung is independently useful and testable.

L1: SP1, experimental
    A loadable PE with the entry and TLS replacements spliced into a new section and
    the entry redirected. Reparses, passes loader validation. Sound mode still refuses.
    The first rung that yields a testable binary. Reached.

L2: SP2
    A proven OEP and every reachable virtualized routine devirtualized and
    reintegrated, not just the entry and TLS callbacks. Sound mode can emit for whole
    samples. Reached per function (313 of 814 on the measured module), not per module.

L3: SP3
    Imports reduced to the proven application set. Blocked while imports are
    resolved at run time.

L4: SP4
    ``.vmp`` sections stripped with no dangling references: "clean" in the strong sense,
    where ``clean <in> <out>`` finally writes ``<out>``. The stripping machinery exists;
    the sound gate is not cleared on a real module.

L5: SP5
    Recovered code optimized and normalized, each pass behaviour-verified.

Cross-cutting rules
-------------------

#. **Soundness default.** The proof gate stays; sound emission is unlocked only when a
   sub-project's proof genuinely passes.
#. **Fixture-first.** Every component has synthetic fixtures as the CI oracle.
#. **Equivalence-preserving.** No transformation may change observable behaviour
   without a verification check.
#. **Static by default; dynamic only when contained.** The sound path never runs the
   target. The one dynamic component is a software interpreter with no native
   execution, network, host syscall, filesystem or FFI path, and what it observes is
   kept on a labelled, experimental tier.
#. **Behaviour-changing modes are opt-in and labelled.**

Risks
-----

Per-function input-domain inference is the wall
    General inference is unsolved for arbitrary VMProtect functions. The measured
    outcome is 38.5% sound recovery on a large module; the long tail stays a blocker
    with honest per-function status.

Header room
    Some layouts leave no room for an extra section header; ``append_section`` fails
    cleanly and a header rebuild may be needed for real coverage.

Integrity interaction
    An edited image can trip a live checksum. The append-only design mitigates the common
    case; stripping the protector resolves it.

Soundness of experimental output
    It can be confidently wrong, and an observed fact describes one execution. Mitigated
    by quarantine, labelling and the blockers list.

Behaviour versus bit reversibility
    "Clean" and "reversible" must not be read as reconstruction of the original file.
