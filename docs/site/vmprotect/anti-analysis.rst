Anti-Debug, Anti-VM and Anti-Dump
=================================

A VMProtect-protected program checks, from several directions, whether it is being
observed. The checks fall into three families, each with a different purpose and a
different answer from a static analysis.

.. note::

    Which checks are present depends on the version and on the options chosen
    ("debugger detection", "virtualization-tools detection", memory protection).
    The techniques below are the ones generally reported for the product and for
    this class of protector. vmpx's own source contains **no anti-debug handling
    code**; what it encodes is the recovery of the *handlers* that implement
    timing and integrity checks (``ReadTimestampCounter``, ``TableChecksum``), as
    static semantics.

Why static analysis sidesteps most of this
------------------------------------------

Every check on this page works by *observing the environment while the program
runs*. A static analysis has no environment: no process, no PEB, no debug registers,
no scheduler, no hypervisor, no timing source. The checks are therefore never
evaluated, and cannot fail or alert. The cost is that the analysis cannot rely on
observing the program's behaviour either, so it has to understand the checks
well enough to **model** their outcome. This is the reasoning behind the project's
"static only" principle and behind making anti-debug handling an *output policy*,
not an analysis concern (:doc:`/tool/pipeline/verification`).

Anti-debug
----------

User-mode debugger detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Process environment block flags.*
    ``PEB.BeingDebugged`` is set by the loader when a debugger is attached;
    ``IsDebuggerPresent`` reads it. A protected program may read the PEB directly
    (``gs:[0x60]`` on x64) so there is no import to hook. ``PEB.NtGlobalFlag`` has
    heap-debugging bits (``0x70``) set when a process is created under a debugger.
    The process heap's flags and force-flags fields are altered by a debugged
    process.

*Debug objects through the native API.*
    ``NtQueryInformationProcess`` with ``ProcessDebugPort`` (7),
    ``ProcessDebugObjectHandle`` (30) or ``ProcessDebugFlags`` (31) reveals the
    attachment. ``CheckRemoteDebuggerPresent`` wraps the first.

*Thread hiding.*
    ``NtSetInformationThread`` with ``ThreadHideFromDebugger`` stops delivering
    debug events for a thread, so a later breakpoint hangs the debugger.

*Hardware breakpoints.*
    The debug registers ``DR0`` to ``DR7`` hold hardware breakpoints. They are not
    readable from user mode directly, but the context captured for an exception
    contains them. A program can provoke an exception, read ``Dr0..Dr3`` from the
    ``CONTEXT`` its handler is given, and test for non-zero values.

*Software breakpoints.*
    An ``int 3`` (``0xCC``) written by a debugger changes the code bytes. Detected
    by scanning code for ``0xCC``, or by the integrity check below.

*Exception behaviour.*
    An ``int 3`` or an ``int 2d`` under a debugger is swallowed or handled
    differently from the same instruction run natively; the trap flag (``TF``) set
    through ``pushfq``/``popfq`` produces a single-step exception that a debugger
    may consume. A program that expects to receive the exception and does not has
    detected the debugger.

*Timing.*
    ``rdtsc`` read twice around a short sequence normally yields a small difference;
    under single-stepping it is thousands of times larger. The VM contains a handler
    that executes ``rdtsc`` and pushes the two halves as separate dwords, and
    protected code compares the difference to a threshold. This is the shape vmpx
    recognises as ``ReadTimestampCounter``. ``GetTickCount`` and
    ``QueryPerformanceCounter`` serve the same purpose with coarser resolution.

*Window and process enumeration.*
    Checks for the windows or process names of known debuggers.

Kernel-mode debugger detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Querying for a kernel debugger: ``NtQuerySystemInformation`` with
``SystemKernelDebuggerInformation`` (35) returns whether a kernel debugger is enabled
and not hidden. Opening the device names used by classic kernel debuggers is the
older variant.

Integrity-based detection
~~~~~~~~~~~~~~~~~~~~~~~~~

A debugger that sets software breakpoints or patches code modifies bytes the
protector has checksummed. This is covered in :doc:`integrity`.

Anti-VM and anti-sandbox
------------------------

The "virtualization tools detection" option asks whether the program runs inside a
virtual machine or a sandbox. (*Virtualization* here means hypervisors, not the VMProtect VM.) Techniques in general
use:

*CPUID.*
    ``cpuid`` leaf 1 returns the **hypervisor-present bit** in ``ECX`` bit 31.
    Leaf ``0x40000000`` returns a vendor string in ``EBX:ECX:EDX``:
    ``"VMwareVMware"``, ``"VBoxVBoxVBox"``, ``"KVMKVMKVM"``, ``"Microsoft Hv"``.
    The VM contains a ``cpuid`` handler that exposes this to protected code.

*Hypervisor ports.*
    VMware's "backdoor" is an ``in eax, dx`` with ``EAX = 0x564D5868`` (``"VMXh"``)
    and ``DX = 0x5658`` (``"VX"``); outside VMware the instruction raises an
    exception, inside it returns data.

*Device and registry artefacts.*
    Names of virtual disks, adapters, drivers and registry keys that belong to
    guest tools.

*Hardware fingerprints.*
    MAC address prefixes assigned to hypervisor vendors; SMBIOS and ACPI
    strings; unusually low CPU counts or memory.

*Sandbox artefacts.*
    The presence of a particular module (a hooking DLL) in the process, or a
    particular named object.

Anti-dump
---------

Protectors also defend against taking a memory image of the unpacked program:

* **Header erasure or alteration.** The PE header is overwritten in memory after load
  (zeroed or modified), so a naive dump cannot be parsed.
* **Memory-region protection.** The restored sections have page protection altered,
  so that reading them from another process or rewriting them behaves unexpectedly.
* **Late binding of imports.** If the import address table is only filled in on
  demand, a dump taken early is incomplete (:doc:`imports`).
* **Erasing the unpacker.** After restoring, the packer's own code can overwrite itself.

Static unpacking is immune by construction: it never relies on a process to have
unpacked itself.

Integrating the checks into the protected code
----------------------------------------------

Detection results are not typically followed by an immediate ``exit(1)``. A visible
reaction identifies the check, so a sophisticated protector makes the consequence
**delayed and indirect**: a flag stored away that corrupts a value used later, a
checksum that decrypts the next block incorrectly, a branch taken ten functions
later. Two consequences matter for analysis:

* The check and the reaction are separated, so patching out one visible
  ``IsDebuggerPresent`` is not enough.
* A check implemented *inside the VM* is just another computation: its inputs are
  intrinsics (``read_timestamp_counter_*``, a native call result, a checksum) and its
  outcome is a branch. In recovered form it is a conditional guard on a value derived
  from one of those intrinsics.

How vmpx treats it
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Aspect
     - vmpx behaviour
   * - Running the target
     - Never. No check can fire.
   * - Timing checks
     - ``ReadTimestampCounter`` handler recognised; value is an opaque
       ``read_timestamp_counter_*`` intrinsic
   * - Integrity checks
     - ``TableChecksum`` handler recognised; evaluated against the analysed image or
       left symbolic (:doc:`integrity`)
   * - Environment API calls
     - Native-call bridge resolves the target against imports; unknown targets
       havoc memory
   * - Output policy (default)
     - **Faithful**: the output reproduces the checks, so it is equivalent and still
       detects debuggers
   * - Output policy (opt-in)
     - **Neutralize**: guards recognised as checksum, timing or debugger-presence
       constructs are rewritten to force the non-detecting path. Deliberately not
       equivalent; a library transform, surfaced by ``clean`` only as a
       recognition-only report field (:doc:`/tool/pipeline/verification`)
