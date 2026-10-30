Attacking Anti-Analysis
=======================

*Mechanism:* :doc:`/vmprotect/anti-analysis`, :doc:`/vmprotect/integrity`,
:doc:`/vmprotect/exceptions`

Static analysis is not a trick that happens to avoid anti-debug. It follows from the
way the problem is posed: if the analysis never executes the target, no checks of the
execution environment can run.

What a static analysis neutralises by construction
--------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Defence
     - Why it cannot fire
   * - ``IsDebuggerPresent``, ``PEB`` flags, ``NtQueryInformationProcess``
     - There is no process, so no PEB and no debug port
   * - Hardware and software breakpoints
     - Nothing is stepped; nothing is patched
   * - Timing (``rdtsc``, ``GetTickCount``)
     - No wall-clock execution is observed
   * - Hypervisor, device and registry checks
     - There is no guest environment to inspect
   * - Anti-dump, header erasure
     - No memory image is taken from a running process
   * - Packer self-decryption
     - The decompressor is reproduced, not run

What remains: the checks as semantics
-------------------------------------

The checks are still *in the code*. A function that begins with an anti-debug test
returns something different when the test fails. Because the analysis recovers the
function's **semantics**, the test becomes part of the recovered semantics:

* a ``rdtsc`` becomes an intrinsic expression ``read_timestamp_counter_*``;
* a native call becomes a ``NativeCall`` effect with an epoch, and its result an opaque
  intrinsic;
* a checksum becomes either an exactly-evaluated constant or a symbolic
  ``table_checksum_low_byte_xor_recurrence_w32``;
* the reaction is a conditional on those values, a *guard*.

The recovered routine therefore faithfully **contains** the check. When it is compiled
back to native code, the output performs the same test at run time. That is the default,
*faithful* policy.

Neutralizing
------------

If a debuggable binary is wanted, the guards can be rewritten. The transform recognises
a guard by its inputs:

* depends on the checksum intrinsic: **integrity check**;
* depends on two or more *distinct* timestamp-counter reads: **timing check** (one read
  is an unrelated use of the cycle counter);
* depends on the result of ``IsDebuggerPresent`` or ``CheckRemoteDebuggerPresent`` (via
  a provenance map): **debugger-presence check**.

and replaces the condition with the constant that selects the non-detecting outcome. The
list of APIs is short and exact on purpose: a fuzzy match could silently remove an
unrelated check.

The transform is **not equivalence preserving** and says so. It is disabled by default,
reports every change, and never touches the sound output. See
:doc:`/tool/pipeline/verification`.

Exceptions and anti-analysis
----------------------------

Exception-driven control flow is also anti-analysis in effect, because a disassembler
that does not model it misses real edges. The attack is the same as for any other
protector feature: model it precisely, and refuse when the model is not exact. See
:doc:`/vmprotect/exceptions`.

Integrity in the rebuilt file
-----------------------------

Modifying the file is itself a form of tampering. The defence against integrity checks
in the *output* is to avoid modifying anything that is checksummed while the checks
are alive (append instead of patch), and then to remove the checks by stripping the
protector. That is why stripping is the last gate.
