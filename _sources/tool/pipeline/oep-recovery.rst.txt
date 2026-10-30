.. _oep-recovery:

OEP Recovery
============

*Crate:* :doc:`/tool/crates/vm` (``oep.rs``)  *Command:* :doc:`/tool/cli/oep`

In a virtualized binary, the header entry point is not the program's original
entry point (OEP). It is a protector stub that sets up the runtime and, much
later, transfers control to the application's real startup code. A devirtualizer
cannot redirect or replace the entry without knowing where that real startup
code is, so the OEP is the first of the :doc:`blockers </tool/soundness>`.

Approach
--------

vmpx does not search for the OEP by pattern. It *derives* it from a proven fact:
the native address the VM returns to after the header entry finishes.

#. Recover every path of the header entry (see :doc:`symbolic-execution`). For a
   DLL there is one path per value of the ``DllMain`` reason domain.
#. For each path, read the native exit state. When ``path.native_exit`` is
   present, the return address is an expression; ``recover_modulo_image_targets``
   resolves it to a concrete set of image-backed targets.
#. A path contributes a *continuation* only if that set resolves to a single
   concrete RVA.
#. ``decide_oep`` accepts the continuation as the OEP only if all of the
   following hold.

``decide_oep``
~~~~~~~~~~~~~~

.. code-block:: rust

    pub fn decide_oep(
        paths: &[Option<Rva>],
        family_complete: bool,
        is_native: impl Fn(Rva) -> bool,
    ) -> OepDecision;

``paths`` holds one entry per recovered path: its single-resolved continuation
RVA, or ``None``. The decision is strict, in this order:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Condition
     - Outcome if violated
   * - The path family is proven complete (``family_complete``)
     - ``unresolved``: "the header path family is not proven complete"
   * - At least one path was recovered
     - ``unresolved``: "no header path was recovered"
   * - Every path has a unique concrete continuation
     - ``unresolved``: "N path(s) have no unique concrete continuation target"
   * - Every distinct continuation is native: executable, file-backed, outside the protector sections, not a VM entry and not the header entry
     - ``unresolved``: "continuation X is not native"
   * - All paths agree on exactly one RVA
     - ``ambiguous`` when there are several distinct native targets

When every condition holds the result is ``recovered_entry = Some(rva)`` with
assessment ``recovered_native_continuation`` and evidence such as "all paths
continue at native 0x...; candidate original entry (native continuation)".

Why so strict
~~~~~~~~~~~~~

An earlier version of the decision had two latent unsound cases, both found in
review and fixed: a *partial* path family could declare a wrong OEP, and a
native candidate sitting next to a non-native one could still recover. The tests
now pin the veto cases: a failed path among resolved ones vetoes the decision, a
native and non-native mix vetoes it, and a lone non-native target is not an OEP.

What it unlocks
~~~~~~~~~~~~~~~

* ``vmpx oep`` populates ``recovered_entry``.
* ``clean`` drops the "original module entry point is not recovered" blocker,
  and ``CleanReport.oep`` reports ``complete: true``.

Limitations
-----------

* Recovery is implemented for DLL entries. For an executable (non-DLL) header
  entry, ``oep`` states that recovery from the entry continuation is not
  implemented; ``clean`` has a separate executable-entry path (see
  :doc:`function-recovery`).
* "Recovered" means the continuation passed the checks above. There is no
  unprotected semantic oracle in the workspace to compare it with; confirming
  the OEP against a real unprotected build happens out-of-band.
