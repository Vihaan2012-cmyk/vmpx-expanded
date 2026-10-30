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
experimental mode exists only as scaffolding to exercise the machinery early.

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
     - OEP decision, VM-entry inventory, live-in inference, input-closed and
       decidable branching recovery, executable-entry recovery, VM-stub redirect,
       strict coverage-gate decision are implemented. The coverage gate is not yet
       consumed by ``clean``; nested-continuation redirects and branching
       value-dispatch emission remain
   * - SP3
     - Application import-set reconstruction
     - Import set not proven
     - Core implemented in ``vmpx-rebuild``: proven-set reduction and a conventional
       import directory builder. Not yet wired into ``clean``; thunk and bridge
       resolution remains
   * - SP4
     - Dangling ``.vmp`` reference resolution and stripping
     - ``.vmp`` sections remain referenced
     - Reference checker implemented in ``vmpx-disasm``. Stripping not yet performed
   * - SP5
     - Optimizing recompilation (VTIL), differential verification, anti-debug policy
     - Quality, not coverage
     - Differential equivalence core and ``--neutralize`` transform implemented as
       library functions; VTIL bridge binding exists behind a feature, not yet in the
       pipeline

Definition-of-done ladder
-------------------------

Each rung is independently useful and testable.

L1: SP1, experimental
    A loadable PE with the entry and TLS replacements spliced into a new section and
    the entry redirected. Reparses, passes loader validation. Sound mode still refuses.
    The first rung that yields a testable binary.

L2: SP2
    A proven OEP and every reachable virtualized routine devirtualized and
    reintegrated, not just the entry and TLS callbacks. Sound mode can emit for whole
    samples.

L3: SP3
    Imports reduced to the proven application set.

L4: SP4
    ``.vmp`` sections stripped with no dangling references: "clean" in the strong sense,
    where ``clean <in> <out>`` finally writes ``<out>``.

L5: SP5
    Recovered code optimized and normalized, each pass behaviour-verified.

Cross-cutting rules
-------------------

#. **Soundness default.** The proof gate stays; sound emission is unlocked only when a
   sub-project's proof genuinely passes.
#. **Fixture-first.** Every component has synthetic fixtures as the CI oracle.
#. **Equivalence-preserving.** No transformation may change observable behaviour
   without a verification check.
#. **Static only.** No sub-project runs the target.
#. **Behaviour-changing modes are opt-in and labelled.**

Risks
-----

Per-function input-domain inference is the wall
    General inference is unsolved for arbitrary VMProtect functions. Expectation: cover
    input-independent and small-finite-domain functions; the long tail stays a blocker
    with honest per-function status.

Header room
    Some layouts leave no room for an extra section header; ``append_section`` fails
    cleanly and a header rebuild may be needed for real coverage.

Integrity interaction
    An edited image can trip a live checksum. The append-only design mitigates the common
    case; stripping the protector resolves it.

Soundness of experimental output
    It can be confidently wrong. Mitigated by quarantine, labelling and the blockers list.

Behaviour versus bit reversibility
    "Clean" and "reversible" must not be read as reconstruction of the original file.
