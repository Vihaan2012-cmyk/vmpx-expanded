The Soundness Model
===================

vmpx is a soundness-first tool. Producing *some* output is easy; producing output
that is **correct** is the whole point, and the two diverge badly for a protected
binary. This page defines what the tool promises, what it withholds, and what the
words "recovered", "blocker" and "experimental" mean.

Principles
----------

Fail closed
    Ambiguity is kept in a report and yields no result, rather than a guess. Many
    functions have a branch of the form "if I cannot prove this, return an error or
    a blocker": "no unknown value is invented", "it never guesses a missing
    operation", "shape agreement alone is not sufficient".

Static by default
    The core devirtualizer never executes the input. This has a loud
    consequence: runtime anti-debug and anti-tamper never fire against the
    analysis, because there is no process, no PEB, no timing source and no
    breakpoint. The flip side is that anything that only exists at runtime — a
    self-unpacked OEP image, an on-demand import table, a PEB-dependent branch —
    is invisible to a purely static pass.

    Dynamic assist (SP7) is a *separate, opt-in* architecture that closes that
    gap: the sample is detonated inside a contained pure-Rust emulator, allowed
    to self-unpack, and the resulting image is dumped and handed back to the
    static pipeline. It deliberately reintroduces the anti-debug and containment
    problem — which is exactly why it runs under emulation with a fail-closed
    device model, not on the host. Its results carry an ``observed-not-proven``
    trust tier, kept distinct from the proven static output; it never relaxes
    the fail-closed rules above.

Behaviour, not bytes
    "Reversible" means *behaviour-reversible*. Virtualization is lossy:
    original instruction encodings, function boundaries, padding and the pristine
    ``.text`` are destroyed. The output reproduces semantics. It will not match a
    hash of the unprotected original, and a disassembly diff will look nothing
    alike even when behaviour is identical.

"Any binary" is a direction
    Correctness is *proven* on synthetic fixtures and *empirically confirmed* on
    local samples. Coverage is bounded by the VMProtect constructs that have been
    lifted and proved; outside them, the sound path refuses.

What "recovered" means
----------------------

There is no protected corpus in continuous integration. "Recovered" therefore means
the result passed synthetic verification and review. It does not mean a behaviour
was observed on a real sample. Recovery is deliberately strict: where soundness is
not provable the tool refuses and keeps a blocker.

Each recovered thing carries its own proof obligation:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Recovered item
     - What must hold
   * - A VM entry
     - Evidence-scored at confidence 60 or more; the bootstrap yields a
       behaviourally valid dispatch slice under some role assignment
   * - A handler's semantics
     - The normalized effect shape is unambiguous *and* a def-use proof of value
       flow exists; otherwise it stays ``Unknown``
   * - A path
     - The decoder reached ``VmExit`` with the exact native exit state read from
       the terminal memory snapshot
   * - A path family
     - ``domain_complete`` and ``all_paths_reach_vm_exit``
   * - The OEP
     - The path family is complete and every path resolves to one native
       continuation (:ref:`oep-recovery`)
   * - A function's input domain
     - Structurally proven exhaustive (never sampled)
   * - A replacement
     - Passes the readiness audit and compiles under ABI validation; equivalence
       over the recovered domain is checked by the differential verifier
   * - An output image
     - Reparses, and passes ``validate_for_loader``

Blockers and the proof gate
---------------------------

A *blocker* is a statement of a proof that does not yet exist. ``clean`` collects
them in ``CleanReport.blockers``. ``ready_to_write`` is true only if the list is
empty, and the sound ``OUTPUT`` is written only then.

Blockers on a protected input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Blocker
     - Cleared by
   * - ``original module entry point is not recovered``
     - A successful OEP decision (:doc:`pipeline/oep-recovery`)
   * - ``application import set is not proven independently reconstructed``
     - Never, today. The import reduction is wired in, but the set of referenced
       IAT slots it consumes is intentionally empty (there is no sound static way to
       compute it yet), so this blocker always stays. The reconstruction is proven
       incomplete by design, not unimplemented
   * - ``native code reintegration has not produced a devirtualized replacement image``
     - A reintegrated image was produced
   * - ``VMProtect sections remain referenced by unresolved paths``
     - The coverage gate passes (every inventory entry covered, none dropped), a
       reintegrated image exists, and the dangling-reference checker reports no
       unresolved reference (SP4)

Conditional blockers
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Blocker
     - Raised when
   * - ``executable entry has VM-entry confidence N, below 60``
     - The executable's entry is not a confident VM entry
   * - ``protected header entry has no recovered VM profile``
     - The input has VMProtect evidence but the header entry is not a VM entry
   * - ``header VM entry path recovery is incomplete``
     - Path recovery for the header entry failed with an error
   * - ``entry path family is not proven safe for native replacement``
     - The effect-liveness audit does not clear the family
   * - ``N loader-reachable TLS VM entry/entries remain semantically unresolved``
     - A TLS callback VM entry was not fully recovered
   * - ``executable entry is not recoverable as an input-closed single path: ...``
     - A non-DLL header entry could not be recovered as a single input-closed path
   * - ``input has no strong VMProtect evidence but its layout is not a conventional PE file``
     - ``NoStrongEvidence`` with a non-conventional layout

The verified no-op
~~~~~~~~~~~~~~~~~~

An input classified ``NoStrongEvidence`` with a ``ConventionalFile`` layout is the
only case in which ``clean`` clears its gate today. ``OUTPUT`` is a byte-for-byte
copy of ``INPUT`` that has been reparsed from disk. This matters mostly as a
safety property: vmpx will not "clean" a file it cannot show to be protected.

Sound output and experimental output
-------------------------------------

There are two output paths, and they never mix.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - Sound ``OUTPUT``
     - ``--experimental PATH``
   * - Written when
     - ``blockers`` is empty
     - A reintegrated image exists, regardless of blockers
   * - Meaning
     - Every relevant proof passed
     - Machinery exercised; unverified
   * - Labelling
     - None needed
     - Warning on stderr, ``experimental: true`` and the full ``blockers`` list in
       the report
   * - On loader-validation failure
     - Never written
     - Written for inspection, flagged
   * - Aliasing guards
     - Yes
     - Yes

``--experimental`` does **not** relax the gate. It is scaffolding so the pipeline
can be exercised against real samples early, quarantined to its own path.

Known risks of experimental output:

* it can be confidently wrong;
* a dispatcher falls back to the original protected entry for inputs outside the
  recovered domain, which is only a safety net when the domain is complete;
* an edited image interacts with any live integrity check (see
  :doc:`pipeline/verification`).

Behaviour-changing modes
------------------------

Two modes each weaken a guarantee. Both are off by default, labelled, and do not
change the semantics of the sound ``OUTPUT``:

``--experimental``
    Gate-relaxed emission. Implemented on ``clean``.

``--neutralize``
    Anti-debug stripping. The transform is a library function. ``clean`` surfaces
    only a *recognition-only* ``anti_debug`` report field (``applied: false``);
    neutralization is never applied in the sound path.

Where soundness still has gaps
------------------------------

The tool is explicit about its limits:

* no protected corpus in CI, so synthetic fixtures may be unrepresentative, and the
  full blocker-clearing path has not been exercised end-to-end on real protected
  input;
* a missed VM entry (false negative) leaves a ``.vmp`` reference, so the sound
  mode correctly refuses;
* a mis-identified VM entry (false positive) must not produce a wrong
  replacement; confidence gating and the proof gates guard against it;
* general per-function input-domain inference is unsolved. Only a decidable subset
  is implemented: 8-bit slices and add-flags covers, with jump tables, memory
  dependent targets, loops, nested input branches, mixed widths, ``nor``/``nand``
  and opaque intrinsics refused;
* output equivalence is verified over the *finite recovered domain*, not
  universally.
