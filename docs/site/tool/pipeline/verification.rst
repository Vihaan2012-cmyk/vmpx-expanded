Verification and Anti-Debug Policy
==================================

*Crates:* :doc:`/tool/crates/vmlift` (``equivalence.rs``),
:doc:`/tool/crates/vm` (``neutralize.rs``), :doc:`/tool/crates/vtil-sys`

SP5 has two jobs: make recompiled output *checkable*, and give the user an
explicit choice about how anti-debug logic is treated.

Differential equivalence
------------------------

A replacement is correct only if it matches the virtual routine it replaces.
``compare_exits`` compares the two as **data**, over a finite input domain, rather
than by running either one:

.. code-block:: rust

    pub struct ExitSemantics {
        pub exit: NativeExitState,
        pub effects: CollectedEffects,
    }

    pub fn compare_exits(vm: &ExitSemantics, replacement: &ExitSemantics) -> Vec<Divergence>;

    pub fn verify_over_domain<F>(domain: &[Assignment], model: F) -> VerificationReport
    where
        F: FnMut(&Assignment) -> Result<(ExitSemantics, ExitSemantics), String>;

Anything that cannot be proven equal is a divergence. Nothing is passed silently.
The ``Divergence`` variants name exactly what differs:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Divergence
     - Meaning
   * - ``Register``
     - A native register's exit expression differs (or exists on one side only)
   * - ``Rflags``, ``ReturnAddress``
     - The flags or return address differ
   * - ``FrameSize``
     - The stack frame sizes differ
   * - ``StoreMismatch``, ``StoreCount``
     - An observable store differs, or the number of stores differs
   * - ``CallMismatch``, ``CallCount``
     - An imported call differs (IAT slot, result intrinsic, arguments), or the
       number of calls differs
   * - ``Unproven``
     - A side carries effects the collector could not represent, so equivalence
       cannot be established
   * - ``Unmodeled``
     - A side's exit could not be modelled for this input at all

Expressions are compared after simplification (``a == b`` or ``a.simplify() ==
b.simplify()``). An empty result means the two are structurally equivalent.

``verify_over_domain`` runs a model for every assignment and aggregates the
results into a ``VerificationReport``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Contents
   * - ``verdict``
     - ``equivalent``, ``divergent`` or ``empty_domain``
   * - ``assignments_checked``
     - Number of input assignments compared
   * - ``divergent_assignments``
     - Number with at least one divergence
   * - ``divergences``
     - ``{assignment, divergence}`` for every difference

An empty domain is **never** reported as equivalent: nothing was checked, so
nothing was verified.

.. note::

    Differential verification over a *finite recovered domain* proves
    equivalence on that domain, not universally. That bounds the strength of
    any "cleanest output" claim built on it.

Anti-debug policy
-----------------

Anti-debug and anti-tamper logic is part of a virtualized routine's semantics.
The emit stage therefore offers two behaviours:

*Faithful reproduction* (default)
    The checks are reproduced exactly, so the output stays behaviour-equivalent
    and still detects debuggers.

*Neutralization* (explicit)
    A deliberately **non-equivalent** mode that forces the "no debugger" path
    for a debuggable clean binary. Never the default.

Because analysis is static, runtime anti-debug never fires against vmpx either
way. The policy only governs what the *output* does.

The neutralize transform
~~~~~~~~~~~~~~~~~~~~~~~~

``neutralize`` takes the guards a routine uses to choose between a "detected"
disposition and a clean continuation, and (when enabled) rewrites each recognised
guard to force the non-detecting path:

.. code-block:: rust

    pub struct RecoveredGuard {
        pub site: Rva,
        pub condition: Expr,
        pub detected_when: DetectedWhen,   // NonZero or Zero
    }

    pub struct NeutralizeOptions {
        pub enabled: bool,                 // false = faithful, input returned unchanged
    }

With ``enabled: false`` the input is returned unchanged and no actions are
reported. With it enabled, each recognised guard is rewritten and reported as a
``NeutralizeAction { site, construct, evidence }``. A guard that cannot be
confidently classified is left untouched.

Recognition is grounded in intrinsic shapes the pipeline already produces:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Construct
     - Evidence
   * - ``IntegrityChecksum``
     - The condition depends on the table-checksum recurrence intrinsic
       ``table_checksum_low_byte_xor_recurrence_w32``
   * - ``DebuggerPresenceTiming``
     - The condition depends on two or more *distinct* ``read_timestamp_counter_*``
       reads. One read alone is treated as an unrelated use of the cycle counter.
   * - ``DebuggerPresenceImport``
     - The condition depends on the result of a native call that a caller-supplied
       provenance map resolves to ``IsDebuggerPresent`` or
       ``CheckRemoteDebuggerPresent``. The list is short and exact: a fuzzy match
       would risk neutralizing an unrelated check.

No in-tree pass yet stitches an import name onto a call-result intrinsic, which is
why the third shape takes a provenance map from the caller.

Why neutralization is safe to offer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* it is off by default and labelled in output;
* it reports every change it makes;
* the differential verifier confirms that faithful mode is unchanged and documents
  exactly what neutralization alters;
* it never alters the semantics of the sound ``OUTPUT``.

There is no ``--neutralize`` flag on the ``clean`` command. The transform is a
tested library function, and ``clean`` wires it in **recognition-only**: the
``anti_debug`` field of the report (``CleanAntiDebugReport``) lists the guards
``neutralize`` recognised and carries ``applied: false``. The rewritten guards are
discarded, so neutralization is never applied in the sound path. The pipeline does not
yet extract conditional guards from recovered paths, so today the field examines no
guards. The differential equivalence verifier is likewise not yet called by ``clean``.

Integrity interaction
---------------------

An experimental image that has been *edited* (a stub overwritten by
``redirect_vm_stub``, or a header changed) interacts with any live integrity check
in the original: if a self-checksum covers the edited bytes, the check will fail at
runtime while the VM and its checksum handler are still present. The append-only
section plus header-only edits avoid touching checksummed ``.vmp`` bytes, which
mitigates the common case. Full resolution comes from stripping the protector
sections, which removes the checksum handlers. See
:doc:`/vmprotect/integrity`.

VTIL optimization
-----------------

The VTIL optimizer is the planned quality pass: route lifted IR through
``vtil::optimizer::apply_all`` so the output reads like compiler output. It sits
between lifting and recompilation, is independently sequenceable, and is turned on
by default only once its verifier is trusted. The Rust binding exists
(:doc:`/tool/crates/vtil-sys`), gated behind the ``vtil`` feature.
