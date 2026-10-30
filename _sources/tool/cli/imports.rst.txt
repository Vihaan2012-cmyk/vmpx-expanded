imports
=======

.. program:: vmpx imports

Inventories the import table and the direct native RIP-relative references to
its slots.

Synopsis
--------

.. code-block:: console

    vmpx imports <INPUT> [--json]

Options
-------

.. option:: INPUT

   The PE to analyse.

.. option:: --json

   Emit ``ImportAnalysis`` as JSON.

Method
------

The command counts modules, named versus ordinal symbols, and *duplicate
identity groups* (the same symbol appearing at more than one IAT slot;
duplicates are preserved as evidence, never collapsed). It then runs two
resynchronizing decode scans looking for RIP-relative references to exact IAT
slots:

#. one over exception-table function ranges;
#. one over the remaining executable bytes.

Each scan is given a budget of 30,000,000 candidate decodes. Every reference
found is tagged ``exception_range_resync_candidate`` or
``executable_resync_candidate`` in the ``discovery`` field.

Text output
-----------

.. code-block:: text

    [imports] <m> modules, <s> symbols (<n> named, <o> ordinal)
    [imports] <d> duplicate identity groups; <k> IAT slots have direct native references
    [imports] analyzed <r> function ranges / <bytes> modeled bytes; file-backed executable sections total <bytes> bytes
    [imports] exception-range recovery attempted <c> independent candidate decodes (truncated=<bool>)
    [imports] remaining-executable fallback attempted <c> independent candidate decodes (truncated=<bool>)
    <instruction_rva> -> <slot> (<module>!<symbol>; <discovery>)
    [imports] warning: <warning>

JSON schema
-----------

``ImportAnalysis`` (``schema_version`` 1):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Contents
   * - ``modules``, ``symbols``
     - Import module and symbol counts
   * - ``named_symbols``, ``ordinal_symbols``
     - Symbols imported by name versus by ordinal
   * - ``duplicate_symbol_groups``
     - Groups of IAT slots with the same identity
   * - ``iat_rva``, ``iat_size``
     - Location and size of the import address table
   * - ``analyzed_function_ranges``, ``analyzed_function_bytes``
     - Coverage of the exception-table scan
   * - ``executable_file_backed_bytes``
     - Total file-backed bytes in executable sections
   * - ``function_resync_candidate_decodes``, ``function_resync_resource_limited``
     - Work done and whether the budget truncated the exception-range scan
   * - ``resync_candidate_decodes``, ``resync_resource_limited``
     - The same for the remaining-executable scan
   * - ``direct_native_references``
     - ``{reference, discovery, module, symbol, ordinal}`` per reference;
       ``reference`` is an ``IpRelativeReference`` with the instruction RVA and
       the resolved slot
   * - ``slots_with_direct_native_references``
     - Number of slots referenced at least once
   * - ``slots_without_direct_native_references``
     - Number never referenced
   * - ``warnings``
     - Strings

Warnings
--------

The report always carries these caveats, because each is a limit of the method:

* Absence of a direct reference does not prove an import unused.
* Duplicates are preserved as evidence.
* Exception metadata alone does not prove reachability.
* Independent-start findings are decoded candidates, not proven reachable.

Truncation warnings are added when a scan exhausts its budget.

Relation to import reconstruction
---------------------------------

``imports`` is the evidence input for the application import-set
reconstruction described in :doc:`/tool/pipeline/imports-and-stripping`: a slot
becomes part of the application's proven import set only when it is referenced
*and* carries exactly one unambiguous identity.
