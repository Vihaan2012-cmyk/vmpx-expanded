The CleanReport Schema
======================

``vmpx clean --report`` writes a ``CleanReport`` as pretty-printed JSON. The
current ``schema_version`` is **2**. The report is the complete, machine-readable
record of a run, and it is written even when the proof gate refuses output.
:doc:`/tool/ui` renders it.

Top level
---------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Contents
   * - ``schema_version``
     - ``2``
   * - ``input``
     - ``{path, sha256, file_size}``
   * - ``pe``
     - Summary of the parsed image (below)
   * - ``protection``
     - The full ``ProtectionReport`` from :doc:`detect`
   * - ``unpacking``
     - ``{performed, mapped_image_size, report}``; ``report`` is the
       ``UnpackReport`` from :doc:`unpack` when unpacking was performed
   * - ``oep``
     - ``{protected_header_entry, recovered_entry, complete}``
   * - ``imports``
     - ``{parsed_modules, parsed_symbols, reconstruction_complete, unresolved}``
   * - ``vm_instances``
     - VM instances (bootstrap, entry RVAs, distinct seeds)
   * - ``vm_entries``
     - ``VmEntryCandidate`` list (see :doc:`vm-list`)
   * - ``vm_inventory``
     - Deduplicated inventory of virtualized routines (below)
   * - ``input_closed_functions``
     - Stage report for input-closed function recovery
   * - ``handlers``
     - ``{unique, classified, unknown}`` handler counts
   * - ``unknown_handlers``
     - RVAs of handlers that could not be classified
   * - ``virtualized_routines``
     - One ``CleanVmPathReport`` per recovered header-entry path
   * - ``tls_entries``
     - One ``CleanTlsEntryReport`` per TLS callback VM entry
   * - ``exports``
     - Per-export dependency on protector code
   * - ``native_continuations``
     - Analysis of native code reached after a VM exit
   * - ``vtil``
     - Stage report; the VTIL optimizer stage
   * - ``recompilation``
     - Stage report for native code generation
   * - ``reintegration``
     - Stage report for section append and redirects
   * - ``reconstruction``
     - Stage report for the overall reconstruction proof
   * - ``layout``
     - Conventional relayout and loader validation
   * - ``validation``
     - ``{input_parsed, mapped_image_parsed, output_written, output_reparsed}``
   * - ``warnings``, ``errors``
     - Strings
   * - ``blockers``
     - Why sound output was refused; sorted and deduplicated
   * - ``ready_to_write``
     - True only when ``blockers`` is empty
   * - ``experimental``
     - True when ``--experimental`` was requested

``pe``
~~~~~~

``kind``, ``machine``, ``image_base``, ``entry_point``, ``size_of_image``,
``sections``, ``import_modules``, ``import_symbols``, ``tls_callbacks``,
``exception_functions`` and ``structural_diagnostics`` (the last six are
counts).

Stage reports
~~~~~~~~~~~~~

Several fields share the ``CleanStageReport`` shape:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Contents
   * - ``complete``
     - True when the stage's machinery succeeded. For ``reintegration`` this
       is independent of ``ready_to_write``: it reports whether the section
       append, redirects, relayout and disk reparse worked, not whether the
       result is proven correct.
   * - ``metrics``
     - A string-to-integer map. ``reintegration`` carries ``section_rva``,
       ``section_size``, ``entries_redirected``, ``dispatchers``,
       ``single_body_candidates``, ``single_body_redirected``,
       ``single_body_bytes`` and ``tls_redirected``. ``recompilation`` carries
       ``emitted`` and ``bytes``.
   * - ``diagnostics``
     - Human-readable lines, including where each replacement body was placed.

``vm_inventory``
~~~~~~~~~~~~~~~~

One entry per distinct virtualized routine, merged from every discovery source
(the executable scan and export cross-references):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Contents
   * - ``rva``
     - Entry stub location
   * - ``confidence``
     - Highest confidence over all sources
   * - ``sources``
     - Union of source labels
   * - ``reachable_from_export``
     - Combined with logical OR across sources

The list is sorted by RVA and deterministic.

``virtualized_routines`` (``CleanVmPathReport``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One entry per recovered path of the header entry. For a DLL there is one per
value of the finite input domain (the ``DllMain`` reason).

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Contents
   * - ``assignments``
     - The input assignment that selects this path, for example
       ``{"native_rdx_initial": 1}``
   * - ``segments``
     - Number of VM segments executed
   * - ``decoded_operations``
     - Number of VM operations decoded
   * - ``stop_reason``
     - Why exploration stopped; ``VmExit`` means the path completed
   * - ``native_exit_recovered``
     - Whether the exact native exit state was read out
   * - ``continuation_targets``
     - Image-backed continuation targets (``ModuloImageTargetReport``)
   * - ``exception_coverage``
     - Whether external reads route through exception handlers
   * - ``issues``
     - Strings describing gaps

``tls_entries`` (``CleanTlsEntryReport``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``entry``, ``confidence``, ``recovered``, ``domain_complete``,
``all_paths_reach_vm_exit``, ``segments``, ``decoded_operations``,
``stop_reason``, ``native_exit_recovered`` and ``issues``. A TLS callback is a
blocker only if its VM behaviour is still unknown; analysing each turns "N
protected TLS entries" into per-entry evidence.

``exports`` (``CleanExportReport``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every non-forwarded export: ``name``, ``rva``, ``reachable_blocks``,
``vm_references`` (the number of *distinct* virtualized routines the export can
reach), ``vm_entries`` (their RVAs) and ``vm_free``. An export is eligible for
native replacement only once every branch and call reachable from it stays
outside the protector sections.

``native_continuations`` (``CleanNestedContinuationReport``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where a VM exit returns into another VM entry, or into ordinary native code,
the pipeline follows it: ``source_assignments``, ``target``,
``vm_entry_confidence``, ``decoded_operations``, ``stop_reason``,
``exception_coverage``, ``native_cfg_blocks``, ``native_analysis`` and
``issues``.

``layout`` (``CleanLayoutReport``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``attempted``, ``file_size``, ``size_of_headers``, ``sections``,
``zero_fill_sections``, ``loader_valid``, ``loader_summary`` and
``diagnostics``. ``loader_valid`` records whether every directory the Windows
loader consumes was walked successfully. The layout is a transform, not
devirtualization: every VM handler and protector section is still present.

Reading a report
----------------

A practical way to triage a run:

#. Look at ``blockers``. If it is empty the run produced sound output.
#. Look at ``oep.complete`` and ``oep.recovered_entry``.
#. Check ``vm_inventory`` against ``input_closed_functions`` and
   ``reintegration.metrics``: the gap is the work remaining.
#. Check ``exports[*].vm_entries`` to see which exported functions still reach
   virtualized code.
#. Use ``virtualized_routines[*].issues`` and ``unknown_handlers`` to find what
   stopped a path.
