detect
======

.. program:: vmpx detect

Profiles VMProtect evidence, the packed or mapped layout of an image, the
per-function protection mode, and any common non-VMProtect packer. It looks
mainly at structure (section geometry, entropy, directory placement), never at a
single mandatory byte signature, and it ignores the file extension.

Synopsis
--------

.. code-block:: console

    vmpx detect <INPUT> [--json]

Options
-------

.. option:: INPUT

   The PE to profile.

.. option:: --json

   Emit the full ``ProtectionReport``.

Text output
-----------

.. code-block:: text

    [vmp] verdict=<verdict> confidence=<0-100> layout=<layout>
      +20 vmp_section_names: <statement>
      +25 packed_virtual_sections: <statement>
    [mode] <mode> (probed=N clean_vm=N mutated_vm=N no_vm=N)
    [mode] signal: <statement>
    [packer] <kind> (confidence <0-100>)
      <evidence statement>
    [region] .text    rva=0x1000 size=0x... entropy=... entry=false
    [vmp] warning: <warning>

One weighted evidence line per finding (``+weight id: statement``), the protection
mode and its probe counts, the packer classification, one ``[region]`` line per
executable section (RVA, size, Shannon entropy and whether it owns the entry
point), and then any warnings.

The evidence model
------------------

Each finding adds points. The total, capped at 100, is the confidence.

.. list-table::
   :header-rows: 1
   :widths: 38 8 54

   * - Evidence id
     - Weight
     - Condition
   * - ``vmp_section_names``
     - +20
     - Section names starting with ``.vmp`` (case-insensitive). Weak evidence, because names can be changed.
   * - ``packed_virtual_sections``
     - +25
     - Three or more initialized virtual-only sections.
   * - ``rva_mapped_analysis_image``
     - +10
     - The image is already an RVA-layout analysis image.
   * - ``high_entropy_entry_region``
     - +20
     - The section owning the entry point has entropy of at least 7.2 bits per byte.
   * - ``entry_in_shared_protector_candidate``
     - +5
     - The entry-owning section is both readable and executable.
   * - ``multiple_high_entropy_executable_regions``
     - +10
     - Two or more executable sections with entropy at least 7.2.
   * - ``imports_in_entry_region``
     - +5
     - The import directory lives inside the entry-owning executable section.
   * - ``exceptions_in_entry_region``
     - +5
     - The exception directory lives inside the entry-owning executable section.
   * - ``entry_and_tls_share_protector_region``
     - +10
     - The entry point and the first TLS callback fall in the same executable region.

An *initialized virtual-only section* has ``PointerToRawData == 0``,
``SizeOfRawData == 0``, a non-zero ``VirtualSize`` and no uninitialized-data
flag. It has no bytes on disk; the protector's loader code fills it at runtime.

Verdict thresholds:

* ``VmprotectLikely`` at confidence 60 or more;
* ``Suspicious`` at 30 or more;
* ``NoStrongEvidence`` otherwise.

When the verdict is ``VmprotectLikely`` but no ``.vmp`` names exist, a warning
states that the classification is structural because randomized section names
prevent name confirmation.

Image layout
------------

The same pass classifies how the file is laid out, which decides whether
unpacking is required:

``RvaMappedAnalysisImage``
    At least 80% of sections have raw pointer equal to RVA and raw size equal
    to virtual size. This is the shape ``vmpx unpack`` produces.
``PackedVirtualSections``
    Otherwise, three or more initialized virtual-only sections.
``ConventionalFile``
    Otherwise, zero initialized virtual-only sections.
``Ambiguous``
    Anything else (one or two virtual-only sections).

Protection mode
---------------

``[mode]`` classifies how the entry stubs behave, by probing them:

``virtualization``
    A VM bootstrap dispatch is recoverable and its handlers decode into a coherent
    instruction stream.
``mutation``
    Entry stubs lead to native code with no recoverable VM dispatch.
``ultra``
    A VM bootstrap is recoverable but its handlers do not decode, the signature of
    virtualization combined with mutation.
``unknown``
    No entry stubs to probe, or the signals contradict.

The counts say how many stubs were probed and how many fell into each class. It is a
triage signal over a sample of stubs, not a per-function guarantee.

Packer classification
---------------------

The same pass recognizes a common non-VMProtect packer. VMProtect evidence is not
consulted here, so the two can appear side by side. The classification uses section
names and, where useful, a marker in the head or tail of the file.

.. list-table::
   :header-rows: 1
   :widths: 20 14 66

   * - Kind
     - Confidence
     - Evidence
   * - ``UPX``
     - 90, or 97
     - ``UPX0``, ``UPX1`` or ``UPX2`` section name; 97 when the ``UPX!`` magic is also present
   * - ``ASPack``
     - 90
     - ``.aspack`` or ``.adata`` section
   * - ``PECompact``
     - 90, or 70
     - ``pec1`` or ``pec2`` section; 70 for a bare ``PEC2`` marker
   * - ``MPRESS``
     - 90
     - ``.MPRESS1`` or ``.MPRESS2`` section
   * - ``FSG``
     - 35
     - Heuristic only: at most two sections with the entry in a writable and
       executable, non-standard-named first section
   * - ``none``
     - 0
     - Nothing above matched

Section names can be changed, so this is evidence, not identification, and the FSG
heuristic in particular can misfire. A detection is the input to
:doc:`generic-unpack`, which routes only UPX, ASPack, PECompact and MPRESS to a
dynamic unpack and never FSG.

JSON schema
-----------

``ProtectionReport``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Field
     - Contents
   * - ``schema_version``
     - Report format version
   * - ``verdict``
     - ``no_strong_evidence``, ``suspicious`` or ``vmprotect_likely``
   * - ``confidence``
     - Integer 0 to 100
   * - ``image_layout``
     - ``conventional_file``, ``packed_virtual_sections``, ``rva_mapped_analysis_image`` or ``ambiguous``
   * - ``entry_point``
     - Entry RVA
   * - ``entry_section``
     - Index of the section owning the entry, or null
   * - ``initialized_virtual_only_sections``
     - Count
   * - ``rva_mapped_sections``
     - Count of sections with raw pointer equal to RVA
   * - ``executable_regions``
     - ``{section_index, section_name, start, size, entropy, owns_entry_point}`` per executable section
   * - ``evidence``
     - ``{id, weight, statement}`` per finding
   * - ``protection_mode``
     - ``{mode, probed_entries, clean_vm_entries, mutated_vm_entries, no_vm_entries, signal}``
   * - ``packer``
     - ``{kind, confidence, evidence}``: the packer classification above
   * - ``warnings``
     - Strings

.. note::

    Detection is a triage tool, not a classifier of record. An unprotected
    program with a packed, high-entropy ``.text`` can score as
    ``Suspicious``; a protected file whose sections were rearranged can score
    lower than its content deserves. The ``clean`` pipeline therefore treats
    ``NoStrongEvidence`` together with ``ConventionalFile`` as the only
    combination eligible for the verified no-op.
