vmpx-cli
========

The command-line front end. The binary target is ``vmpx`` (``src/main.rs``). It
exposes no library API. It depends on all other library crates plus ``clap``,
``serde``, ``serde_json``, ``sha2`` and ``anyhow``.

.. code-block:: toml

    [[bin]]
    name = "vmpx"
    path = "src/main.rs"

Modules
-------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - Responsibility
   * - ``main.rs``
     - The clap ``Command`` enum, the eight analysis subcommands, JSON
       output structs (``InfoOutput``, ``Comparison``, ``FileSummary``,
       ``ImportAnalysis``, ``OepAnalysis``) and the file-writing safety checks
   * - ``clean.rs``
     - The proof-gated pipeline. ``analyze_clean`` and
       ``analyze_clean_with_image`` build the ``CleanReport`` (schema version 2),
       the optional conventional image and the optional reintegrated image
   * - ``vm_inventory.rs``
     - ``VmInventoryEntry`` and the deterministic ``merge_vm_inventory``
   * - ``input_closed_functions.rs``
     - Recovery of input-closed and branching virtualized functions, and the
       executable-entry decision (``decide_exe_entry``)
   * - ``coverage_gate.rs``
     - ``decide_coverage``, the strict gate for the OEP and ``.vmp`` blockers

Key functions in ``clean.rs``
-----------------------------

``analyze_clean_with_image(input_path, input) -> Result<(CleanReport, Option<Vec<u8>>, Option<Vec<u8>>)>``
    Runs the whole pipeline and returns the report, the conventional-layout image
    and the reintegrated image. ``main.rs`` decides what to write.

``export_dependency_report``
    Walks every branch and direct call reachable from each export (CFG limits of
    2,048 blocks and 60,000 instructions) and records distinct virtualized routines.

``build_vm_inventory``
    Merges scan results and export references into the inventory.

``replacement_inputs``
    Binds symbolic inputs to physical registers, including the ``vsp_initial``
    stack-cursor binding and its rewritten name.

``dll_entry_recovery_inputs``
    The ``native_rdx_initial`` domain 0 to 3 and the tightened explorer limits
    (see :doc:`/tool/dll-support`).

``layout_reintegration_payload`` and ``commit_group_layout``
    Lay out bodies first and dispatchers after, and roll the cursor back when a
    group cannot be assembled.

``header_entry_continuations`` and ``is_native_continuation``
    Feed the OEP decision, shared with ``vmpx oep``.

Write safety
------------

``main.rs`` enforces, before and after running a command that writes:

* the output must differ from the input, directly and through a canonicalized
  alias;
* the output must not already exist;
* ``--report``, ``--reconstructed`` and ``--experimental`` must not alias the
  input, the output, or one another;
* every file written is reparsed from disk.

Process behaviour
-----------------

The binary reads its input into memory, never opens a network connection, never
spawns a process and never loads a library. Errors propagate through ``anyhow``
and produce a non-zero exit status with the chain of causes on standard error.
