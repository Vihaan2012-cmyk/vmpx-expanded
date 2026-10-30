vm-list
=======

.. program:: vmpx vm-list

Discovers evidence-scored VM entry stubs and the mutated bootstrap profiles
they share.

Synopsis
--------

.. code-block:: console

    vmpx vm-list <INPUT> [--json] [--no-scan]

Options
-------

.. option:: INPUT

   The PE to scan. For a packed file, pass the analysis image produced by
   :doc:`unpack` or let ``clean`` do it.

.. option:: --json

   Emit the full ``VmDiscoveryReport``, including ``rejected_roots``.

.. option:: --no-scan

   Restrict discovery to the PE entry point and TLS callbacks (sets
   ``scan_executable_stubs = false``). Without this flag the executable
   sections are also scanned for entry-stub candidates.

How discovery works
-------------------

A virtualized routine is entered through a small native stub shaped
``push <immediate seed>; call <bootstrap>``. Candidates are *roots*:

* the PE entry point;
* every TLS callback;
* unless ``--no-scan``, an executable scan of sections that own the entry
  point or have entropy at least 7.0, prefiltered on the opcodes ``0x68`` and
  ``0x6A`` (``push imm``) and validated by decoding.

Each candidate is scored by evidence. A candidate needs a confidence of 60 or
more to be reported as an entry.

.. list-table::
   :header-rows: 1
   :widths: 75 25

   * - Evidence
     - Weight
   * - The root is loader-reachable (entry point or TLS callback)
     - +10
   * - Pushes an immediate seed
     - +15
   * - Reaches the bootstrap through a direct call
     - +15
   * - The downstream chain saves at least 8 distinct registers
     - +15
   * - The chain saves flags
     - +5
   * - Reserves between 0x80 and 0x1000 bytes of stack
     - +20
   * - Aligns the stack
     - +5
   * - Builds a pointer from a high aligned base
     - +5
   * - Performs state-dependent memory fetches
     - +5
   * - Ends in an indirect dispatch
     - +10

Entries that reach the same stack-reservation instruction share a bootstrap and
are grouped as one *VM instance*; the set of distinct seeds per instance is
recorded. One protected binary can therefore contain several independently
mutated VM instances (see :doc:`/vmprotect/vm/entry`).

Reachability is tracked: entry-point and TLS roots are ``LoaderReachable``;
scanned stubs are ``UnprovenExecutableCandidate``.

Text output
-----------

.. code-block:: text

    [vm] <n> entries grouped into <m> bootstrap profiles (<k> stub candidates checked)
    [vm] instance <id> bootstrap=<rva> entries=<n> seeds=<n>
      entry=<rva> source=<source> reachability=<reach> confidence=<n> seed=<hex> call=<rva> bootstrap=<rva>
    [vm] warning: <diagnostic>

JSON schema
-----------

``VmDiscoveryReport``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Contents
   * - ``schema_version``
     - Report format version
   * - ``scanned_stub_candidates``
     - Number of candidate stubs checked
   * - ``resource_limited``
     - True if a scan limit (``max_stub_candidates`` and friends) cut the scan
       short
   * - ``entries``
     - ``VmEntryCandidate`` list (below)
   * - ``instances``
     - ``{id, bootstrap_rva, entry_rvas, distinct_seeds}`` per VM instance
   * - ``rejected_roots``
     - Candidates that were examined and fell below the confidence floor
   * - ``diagnostics``
     - Strings

``VmEntryCandidate``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Contents
   * - ``rva``, ``va``
     - Location of the entry stub
   * - ``source``
     - ``pe_entry_point``, ``tls_callback`` or ``executable_scan``
   * - ``reachability``
     - ``loader_reachable`` or ``unproven_executable_candidate``
   * - ``confidence``
     - 0 to 100
   * - ``seed``
     - The pushed immediate, or null
   * - ``call_target``
     - Target of the call, or null
   * - ``bootstrap_rva``
     - Start of the bootstrap this entry shares, or null
   * - ``evidence``
     - ``{weight, statement}`` per scoring finding

Default limits (``VmDiscoveryLimits``)
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Limit
     - Default
     - Meaning
   * - ``scan_executable_stubs``
     - true
     - Scan executable sections in addition to the loader roots
   * - ``max_stub_candidates``
     - 25,000
     - Candidates examined before the scan reports ``resource_limited``
   * - ``max_chain_fragments``
     - 512
     - Direct-jump fragments followed per chain
   * - ``max_chain_instructions``
     - 20,000
     - Instructions decoded per chain
   * - ``minimum_confidence``
     - 60
     - Floor for reporting an entry
