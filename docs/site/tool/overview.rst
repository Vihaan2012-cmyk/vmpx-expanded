Overview
========

vmpx is an evidence-driven static-analysis toolkit for VMProtect-protected
Windows PE files. The Rust workspace is edition 2024, ``rust-version = "1.88"``,
MIT licensed, and forbids ``unsafe`` code workspace-wide (the one exception is
the feature-gated FFI binding in :doc:`crates/vtil-sys`). The command-line
binary is named ``vmpx`` (package ``vmpx-cli``); the analysis logic lives in
library crates that can be used independently.

The sound path is entirely static. An input file is parsed, unpacked by
re-implementing the protector's own decompression, and symbolically executed.
No target instruction is ever run natively. Two consequences follow, and they
shape the rest of the design:

* Runtime defenses never fire. There is no live process, no PEB, no timing
  source and no breakpoint for an anti-debug check to observe. The checks are
  still present in the protected code, and the tool recovers the handlers that
  implement them as static semantics (see :doc:`/vmprotect/anti-analysis`).
* In the sound path nothing is observed; everything is proven. A path through a
  virtualized routine is either decoded and symbolically executed to a native
  exit state, or it is reported as a blocker with the reason it could not be.

A separate, opt-in dynamic-assist engine can *observe* a sample inside a
contained software emulator, for the facts static analysis cannot prove. It is
described under :ref:`overview-tiers` below and never feeds the sound output.

Current capabilities
--------------------

* **Devirtualization.** Real, freshly-protected VMProtect 3.11 "Compact"
  functions are recovered end to end: opaque bytecode, a complete control-flow
  graph, and lifted native IR. Completeness is certified by an inflation-proof
  oracle (``all_paths_reach_vm_exit``), so a partial walk cannot be reported as
  complete. On a large real-world protected module with 814 virtualized
  functions, **313 of 814 (38.5%) are soundly recovered**. That is the current
  measured sound ceiling, not a target: the remainder depends on runtime
  facts (see :doc:`/vmprotect/devirtualization/static-frontier`).
* **Cross-profile VM-transition walker.** The recovery walker switches the
  active register-role and direction profile at a proven popped-target
  transition, re-resolving register roles per block, under a fail-closed
  per-entry budget. It was the main lever past the earlier ceiling.
* **Mutation-mode deobfuscation.** A tournament of simplifiers (MBA and
  arithmetic-identity solvers, opaque-predicate and substitution solvers) is
  ranked by differential input-testing; an SMT equivalence gate decides whether a
  result is proven. Coverage on real mutated binaries is partial.
* **Protection removal.** Real VMProtect 3.11 LZMA unpacking, section stripping
  that refuses on any dangling reference, and import reconstruction from proven
  bindings.
* **Packer front-end.** ``detect`` classifies UPX, ASPack, PECompact, MPRESS and
  FSG; ``generic-unpack`` runs a bounded, contained unpack to the original entry
  point (:doc:`cli/generic-unpack`). Real UPX is validated end to end in the
  emulator; the other packers are routed but not validated on real samples.
* **Contained dynamic assist.** ``vmpx-emulate`` runs a module in an in-process
  interpreter to observe an unpacked image, resolved imports, runtime globals,
  recursion depth and branch targets, and feeds them back on the experimental
  tier only (:doc:`/vmprotect/devirtualization/dynamic-assist`).

.. _overview-tiers:

Sound and experimental tiers
----------------------------

Output and facts come in two tiers that never mix.

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * -
     - Sound (proven)
     - Experimental (observed or unverified)
   * - Source
     - Static analysis that passed a verification gate: the completeness oracle,
       structurally proven input domains, the SMT gate for mutation results
     - Gate-relaxed emission (``--experimental``), and facts observed in one
       emulated run (``--dynamic``, ``generic-unpack``)
   * - Written to
     - ``clean``'s ``OUTPUT``, only when no blocker remains
     - Separate, labelled paths; the report names every observed fact used
   * - Can be wrong
     - Only through a bug in the gate; every doubt keeps a blocker
     - Yes. An observed fact is evidence about one execution, not a proof

Where a run contradicts a static result, the static result is reported as
blocked rather than silently patched. See :doc:`soundness`.

Goals
-----

The long-term goal is clean, behavior-preserving ("reversible") devirtualized
output: a PE that does what the protected file did, with the VMProtect
sections removed. The project reaches that goal as a ladder of independently
testable milestones rather than a single finish line:

.. list-table::
   :header-rows: 1
   :widths: 10 55 32

   * - Rung
     - Milestone
     - Status
   * - L1
     - Native replacements spliced into a new section, entry and TLS callbacks redirected
     - Implemented and exercised on real protected binaries; behind ``--experimental``
   * - L2
     - Proven original entry point and every reachable virtualized routine recovered
     - Per-function recovery works on real binaries (313 of 814 soundly on one large module); whole-module sound clearing is not reached
   * - L3
     - Imports reduced to the proven application set
     - Wired into ``clean`` as an always-incomplete proof: import protection is resolved at run time, so the import blocker stays in the sound path
   * - L4
     - Dangling ``.vmp`` references resolved and sections stripped
     - Reference checker wired in; a ``strip`` subcommand and an opt-in experimental ``--strip`` stage exist; stripping is not part of the sound ``OUTPUT``
   * - L5
     - Optimized recompilation and differential verification
     - Equivalence verifier and emulator differential harness are library-level; anti-debug recognition is reported (nothing is applied in the sound path)

See :doc:`/project/roadmap` for the detailed status of each sub-project.

Separately from the rungs, the VMProtect 3.11 Compact VM is fully decoded
(256 of 256 handlers classified); linear and branching functions lift to native
x86-64 and ``clean --experimental --decoded-lift`` splices them into a hybrid PE;
see :doc:`pipeline/compact-vm` and :doc:`pipeline/function-lift`.

What "recovered" means
----------------------

Without a protected corpus in continuous integration, "recovered" does not mean
that a behaviour was observed. It means the result passed verification: the
completeness oracle and proven domains on real samples measured out of band, and
synthetic verification and review in CI. Recovery is deliberately strict: where soundness is
not provable the tool refuses and keeps a *blocker* rather than guessing.
Ambiguity is preserved in a report and produces no result. Many doc comments in
the code say this explicitly, for example "no unknown value is invented" and
"it never guesses a missing operation".

The full model, including the proof gate, the two kinds of output and the list
of blockers, is described in :doc:`soundness`.

Anatomy of a run
----------------

The ``clean`` subcommand chains the libraries in a fixed order:

#. :doc:`crates/detect` classifies the input and its layout.
#. :doc:`crates/unpack` restores packed sections when the layout is
   ``PackedVirtualSections``; otherwise the input is analysed as-is.
#. :doc:`crates/vm` discovers VM entries, recovers the bootstrap and handler
   codecs, and symbolically executes VM paths to native exit states.
#. :doc:`crates/vmlift` collects the path effects and builds replacements;
   :doc:`crates/recompile` lowers them to x86-64.
#. :doc:`crates/rebuild` relayouts the image to a conventional PE, validates
   it for the Windows loader, and, when replacements exist, appends a section
   and redirects entry points.

Each stage is described in :doc:`pipeline/index`.

Scope and non-goals
-------------------

* vmpx targets PE32+ (x86-64) images. PE32 files parse, but the reintegration
  machinery is PE32+ only.
* "Any binary" is a direction, not a guarantee. Correctness is proven on the
  synthetic fixtures the repository constructs and confirmed out-of-band on
  local samples. Coverage is bounded by the VMProtect constructs that have been
  lifted and proved; outside that, the sound path refuses.
* Behavior-reversible, not bit-reversible. See the note on the front page.
* vmpx is a research and malware-analysis tool. It never executes the input
  natively. The one dynamic component, ``vmpx-emulate``, is a software
  interpreter with no host syscall, filesystem, FFI or network path; a network-API
  denylist stops a run that reaches one, and the workspace builds offline. The
  CLI and library crates have no network, process-spawning or dynamic-loading
  surface (only the optional :doc:`ui` launches a process: the ``vmpx`` CLI
  itself).
* A figure such as 313 of 814 is a measurement on one module and one build.
  Re-protecting the same input moves results considerably, so read it as a
  ceiling observed, not a guaranteed rate.
