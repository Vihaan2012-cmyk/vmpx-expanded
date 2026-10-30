unpack
======

.. program:: vmpx unpack

Recovers VMProtect-packed sections into an explicit RVA-layout analysis image.
This is the static equivalent of "dumping the process after the packer stub
has run": the LZMA-compressed section contents are decompressed and placed at
their virtual addresses without executing a single target instruction.

Synopsis
--------

.. code-block:: console

    vmpx unpack <INPUT> <OUTPUT> [--report <PATH>] [--conventional <PATH>]

Options
-------

.. option:: INPUT

   A packed PE. Its layout should be ``PackedVirtualSections`` (see
   :doc:`detect`).

.. option:: OUTPUT

   The RVA-layout analysis image. It is **not** loadable by the Windows
   loader: every section has ``PointerToRawData == VirtualAddress`` and
   ``SizeOfRawData == VirtualSize``. The command refuses to overwrite the
   input, directly or via an alias, and the written file is reparsed from
   disk.

.. option:: --report PATH

   Write the versioned JSON ``UnpackReport`` (see below).

.. option:: --conventional PATH

   Also write the image in conventional file layout, loadable by the Windows
   loader, after running loader validation. The path must not alias the input,
   the output or the report.

Text output
-----------

``[unpack]``
    ``<kind> PACKER_INFO at <file offset> with <n> blocks``, one
    ``block i:`` line per block (``src``, ``dst``, decompressed size over
    capacity, section virtual size), any warnings, and, with
    :option:`--conventional`, the loader-validation counts and a layout
    summary.
``[verify]``
    Confirmation that each written file reparsed from disk.
``[report]``
    The report path.

``UnpackReport``
----------------

=======================  =====================================================
Field                    Contents
=======================  =====================================================
``schema_version``       Report format version
``input_size``           Input size in bytes
``output_size``          Size of the analysis image
``packed_sections``      Number of packed destination sections
``packer_table``         ``{kind, file_offset, destination_key,
                         lzma_properties, entries}``
``blocks``               One entry per decompressed block
``in_memory_reparsed``   The image parsed successfully in memory
``disk_reparsed``        The written file parsed successfully from disk
``warnings``             For example a block that decompressed to fewer bytes
                         than its section's virtual size
=======================  =====================================================

``packer_table.kind`` is ``LegacyPlaintext`` or ``EncodedRotatingKey`` (see
:doc:`/vmprotect/packing`). ``destination_key`` is the recovered rotating key
for the encoded layout and null for the legacy layout. Each entry in ``entries``
is ``{source_rva, destination_rva}``. Each entry in ``blocks`` is:

========================  ====================================================
Field                     Contents
========================  ====================================================
``section_index``         Index of the destination section
``section_name``          Name of the destination section
``source_rva``            RVA of the compressed stream
``destination_rva``       RVA the stream expands to
``section_virtual_size``  The section's virtual size
``destination_capacity``  Bytes available before the next section
``decompressed_size``     Bytes actually produced
========================  ====================================================

Failure modes
-------------

``UnpackError`` has four variants, shown in the error text with a stage tag:

``[pe]``
    The input could not be parsed.
``[locate]``
    ``PACKER_INFO`` was absent or ambiguous (more than one validated table).
``[decompress]``
    A block failed to decompress: ``block <i> (<section>): <message>``.
``[reconstruct]``
    The mapped image could not be assembled, for instance because
    ``SizeOfImage`` or a section range cannot be represented or overflows.

See :doc:`/tool/pipeline/unpacking` for the algorithm.
