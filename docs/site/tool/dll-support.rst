DLL Support
===========

Dynamic-link libraries are a first-class target. Most real-world VMProtect samples
(plugins, addons, game modules) are DLLs, and a DLL's entry is the case the
pipeline can model most completely.

Why DLLs are tractable
----------------------

A DLL's module entry is ``DllMain(hinstDLL, fdwReason, lpvReserved)`` under the
Windows x64 calling convention, so the inputs are the registers ``rcx``, ``rdx``
and ``r8``. Of those, only ``fdwReason`` (in ``rdx``) steers behaviour in a way the
loader defines, and it takes exactly four values:

.. list-table::
   :header-rows: 1
   :widths: 10 40 50

   * - Value
     - Constant
     - Meaning
   * - 0
     - ``DLL_PROCESS_DETACH``
     - The DLL is being unloaded
   * - 1
     - ``DLL_PROCESS_ATTACH``
     - The DLL is being loaded into a process
   * - 2
     - ``DLL_THREAD_ATTACH``
     - A new thread is being created
   * - 3
     - ``DLL_THREAD_DETACH``
     - A thread is exiting

The input domain of the header entry is therefore **finite and known**:
``native_rdx_initial`` over 0 to 3. The pipeline runs every value as an independent
path (``recover_entry_paths``), and a path family is *domain complete* when every
value was explored and each path reaches ``VmExit``.

Recovery limits for the DLL entry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pipeline tightens the profile explorer for the header entry: at most 32
segments, 2,048 instructions per segment and 8,192 instructions in total.
Exceeding a limit is reported as a stop reason (``SegmentLimit``,
``TotalInstructionLimit``), never ignored.

From paths to a replacement
---------------------------

Each recovered path becomes one compiled replacement body. The domain dispatcher
then selects among them at run time:

.. code-block:: nasm

    ; entry redirected here
    cmp  rdx, 0
    je   body_detach
    cmp  rdx, 1
    je   body_attach
    cmp  rdx, 2
    je   body_thread_attach
    cmp  rdx, 3
    je   body_thread_detach
    jmp  original_protected_entry     ; only for out-of-domain values

When the domain is proven complete, the final ``jmp`` is unreachable. A single-body
group whose domain is *not* proven complete still receives a dispatcher with a
fallback, so the body can never run on an out-of-domain input.

The original entry point
------------------------

For a DLL, the OEP is derived from the native continuation all four paths reach
once the VM has finished initialising the runtime: ``decide_oep`` requires them to
agree on exactly one native RVA (:doc:`pipeline/oep-recovery`).

TLS callbacks
-------------

VMProtect commonly installs TLS callbacks that run before the module entry (and
for each thread). They are loader-reachable roots for VM discovery, so each TLS
callback that is a VM entry is analysed on its own and reported in
``CleanReport.tls_entries``. A callback is a blocker only if its VM behaviour is
still unknown.

For reintegration, each recovered callback is redirected by rewriting its slot in
the TLS callback array to the replacement's VA (``set_tls_callback``), and a
callback that has several recovered paths gets a dispatcher exactly as the header
entry does.

Exports
-------

A DLL's exports are the program's API surface, and an export is only eligible for
native replacement once every branch and call reachable from it stays outside the
protector sections. The export-dependency walk reports, per export,
``reachable_blocks``, ``vm_references`` and ``vm_entries`` (see
:doc:`cli/reports`). Those entry RVAs feed the VM-entry inventory
(:doc:`pipeline/function-recovery`), so the work remaining for each export is
addressable rather than a bare count.

Addressing and relocations
--------------------------

Replacement bodies are compiled at ``image_base + next_section_rva`` (the preferred
base), which is the core addressing invariant of reintegration. The conventional
relayout keeps the original relocation directory in force by clearing
``IMAGE_FILE_RELOCS_STRIPPED`` when a relocation directory exists, so the loader
still applies it, and ``validate_for_loader`` checks that every relocation target is
mapped.

Executables
-----------

A non-DLL executable has no ``DllMain`` reason. Its entry point (CRT start) is
invoked once, so its invocation domain is a single path. It is recovered through
the empty-domain input-closed route (:doc:`pipeline/function-recovery`), and
``vmpx oep`` states that OEP recovery from the entry continuation is only
implemented for DLL entries.
