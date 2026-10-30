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

Real-sample status
------------------

The first real VMProtect 3.11 sample uses the "Compact" VM, and it is now fully
decoded: 256 of 256 handlers classified with none unknown or failed, table-indexed
keyless dispatch modelled, and the stream decoded through a CFG walk. Linear
functions devirtualize fully to native x86-64. Branching functions decode with
branches resolved by symbolic execution and proven input domains and lift
through the CFG-level lift, including continuation exits; the first branching
function (``0x126b02``) devirtualizes to native code. VM-side calls resolve to
concrete imports (``HeapAlloc`` and ``HeapFree`` are recovered), and nested
VM-to-VM continuations are followed in the CFG. ``clean --experimental
--decoded-lift`` splices the lifted functions into a valid hybrid PE with the
stubs redirected. The honest limits of the static approach (unbounded
input branches, re-entrant continuations, protector-continuation exits and
environment-dependent branches) are in
:doc:`/vmprotect/devirtualization/static-frontier`; dynamic assist is the
intended way past them, and its design (SP7, emulation first) is in
:doc:`/vmprotect/devirtualization/dynamic-assist`. See
:doc:`/tool/pipeline/compact-vm` and :doc:`/tool/pipeline/function-lift`. In
``clean`` these stages run only under the opt-in ``--decoded-lift``.

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
       value dispatchers for branching functions and the strict coverage gate are
       implemented, and the gate now drives blocker clearing in ``clean``. With no
       real VMProtect corpus in CI, the full clearing path is unexercised
       end-to-end. Nested-continuation redirects remain deferred
   * - SP3
     - Application import-set reconstruction
     - Import set not proven
     - Proven-set reduction and a conventional import directory builder in
       ``vmpx-rebuild``, wired into ``clean``. The referenced-IAT-slot set is
       intentionally empty (no sound static way to compute it yet), so the import
       blocker always stays: proven-incomplete by design, not unimplemented
   * - SP4
     - Dangling ``.vmp`` reference resolution and stripping
     - ``.vmp`` sections remain referenced
     - Reference checker in ``vmpx-disasm``, wired into ``clean`` as a backstop for
       the ``.vmp`` blocker (any unresolved reference keeps it). Stripping not yet
       performed
   * - SP5
     - Optimizing recompilation (VTIL), differential verification, anti-debug policy
     - Quality, not coverage
     - Differential equivalence core is a library function (not called by ``clean``).
       The ``neutralize`` transform is surfaced in ``clean`` as a recognition-only
       ``anti_debug`` report field (``applied: false``); nothing is neutralized in the
       sound path. VTIL bridge binding exists behind a feature, not yet in the
       pipeline
   * - SP7
     - Emulation-first dynamic assist for user-mode samples
     - Static limits: unbounded input branches, environment reads, re-entrant
       continuations
     - Design only; see :doc:`/vmprotect/devirtualization/dynamic-assist`

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
