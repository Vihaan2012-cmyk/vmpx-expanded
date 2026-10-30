oep
===

.. program:: vmpx oep

Ranks module-entry evidence without assuming that the protected header entry
is the original entry point.

Synopsis
--------

.. code-block:: console

    vmpx oep <INPUT> [--json]

Options
-------

.. option:: INPUT

   The PE to analyse.

.. option:: --json

   Emit ``OepAnalysis`` as JSON.

Method
------

The command runs VM discovery restricted to loader roots (no executable scan).

* If the header entry is a discovered VM entry, ``header_entry_assessment`` is
  ``protector_vm_entry`` with its confidence and evidence; otherwise
  ``unresolved``.
* TLS roots that were rejected as VM entries are listed as
  ``native_tls_candidates``.
* For a DLL whose header entry is a VM entry, the entry's paths are recovered
  and the strict OEP decision is applied (:ref:`oep-recovery`). When it
  succeeds, ``recovered_entry`` holds the original native module entry and the
  assessment becomes ``recovered_native_continuation``. Other outcomes are
  ``unresolved`` and ``ambiguous``.
* For an executable (non-DLL) header, recovery from the entry continuation is
  not implemented, and a warning says so.

Text output
-----------

.. code-block:: text

    [oep] header entry <rva> assessment=<assessment> confidence=<n|unknown>
    [oep] recovered native module entry: <rva|unresolved>
      <evidence line>
    [oep] warning: <warning>

JSON schema
-----------

``OepAnalysis`` (``schema_version`` 1):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Field
     - Contents
   * - ``header_entry``
     - ``AddressOfEntryPoint`` from the optional header
   * - ``header_entry_vm_confidence``
     - Confidence of the VM entry at the header entry, or null
   * - ``header_entry_assessment``
     - ``protector_vm_entry``, ``recovered_native_continuation``,
       ``ambiguous`` or ``unresolved``
   * - ``recovered_entry``
     - The proven original entry RVA, or null
   * - ``native_tls_candidates``
     - TLS callbacks that are native rather than VM entries
   * - ``evidence``
     - Strings; those from the OEP decision are prefixed ``oep:``
   * - ``warnings``
     - Strings

Limitations
-----------

``recovered_entry`` is null unless a VM-exit/native-transition path was fully
decoded. The warnings always state that the workspace has no unprotected
semantic oracle: the recovered entry is proven to be where the VM hands control
to native code, not compared to an unprotected build of the same program.
