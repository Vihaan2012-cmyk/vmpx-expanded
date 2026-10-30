Unpacking
=========

*Crate:* :doc:`/tool/crates/unpack`  *Command:* :doc:`/tool/cli/unpack`

A packed binary keeps its protected content compressed in the file; the section
table still describes where that content lands in memory. vmpx reproduces the
packer's own decompression statically.

Destination sections
--------------------

Destination sections are those with ``PointerToRawData == 0``,
``SizeOfRawData == 0``, a non-zero ``VirtualSize`` and no uninitialized-data
flag: *initialized virtual-only* sections. They exist in the section table but
have no bytes on disk. The protector's loader fills them at runtime, and the
unpacker reproduces that fill.

Locating ``PACKER_INFO``
------------------------

``PACKER_INFO`` is an array of 8-byte entries ``{ source_rva: u32,
destination_rva: u32 }``, one per destination section. It is located by
scanning the whole file; there is no fixed offset.

A candidate is accepted only if every ``source_rva`` maps to a file byte equal
to ``0x00`` (the first byte of a raw LZMA range-coder stream). At most 16
candidates are collected, and the search requires exactly one validated table;
otherwise it reports ambiguity or absence.

Two layouts are tried, legacy first. The encoded layout is searched only when
no legacy table validates.

``LegacyPlaintext``
    The destination RVAs are stored in the clear. Entry *i* validates when its
    destination field equals the virtual address of destination section *i*.
    The dword located 8 bytes before the table is an RVA that points at the
    five LZMA property bytes, so the properties come from the file
    (``destination_key = None``).

``EncodedRotatingKey``
    The destination RVAs are obfuscated with a rotating key. The LZMA
    properties are not read from the file: the fixed constant
    ``5D 00 00 00 04`` (``DEFAULT_LZMA_PROPERTIES``) is used.

The rotating-key scheme
~~~~~~~~~~~~~~~~~~~~~~~

For entry index *i* with stored value ``s[i]``, the decoded destination is::

    destination[i] = s[i] XOR rotl32(key, (i * 7) & 31)

The key is not stored. The code derives it by assuming ``destination[0]`` equals
the virtual address of each candidate destination section in turn:
``key = s[0] XOR seed_section_VA``. A key is accepted only if every decoded
destination is the virtual address of a destination section *and* the decoded
set is a permutation of those sections: no duplicates, none missing. As cheap
pre-filters, the first two entries' source RVAs must already validate before
the full table is checked at a byte offset.

Decompression
-------------

Each block is a headerless raw LZMA stream. The unpacker synthesizes the 13-byte
standard ``.lzma`` header (five property bytes followed by an unpacked size of
``u64::MAX``, meaning unknown) and feeds the stream, relying on the end marker.

The stream is followed by unrelated PE bytes, and a standalone decoder rejects
trailing input. On failure the code therefore retries with exactly the number
of bytes the first pass consumed up to the end marker. That length is
evidence-derived, not a guessed compressed size.

Decompression is bounded: output cannot exceed the destination span, and the
decoder has a 128 MiB memory limit. ``UnpackLimits`` default to a 2 GiB maximum
image and 256 packed sections.

Reconstructing the mapped image
-------------------------------

#. Allocate a zeroed buffer of ``SizeOfImage`` bytes and copy the headers.
#. Copy every file-backed section to its virtual address in the buffer.
#. Rewrite each section header so ``SizeOfRawData = VirtualSize`` and
   ``PointerToRawData = VirtualAddress``. This is the *RVA layout*: file offset
   equals RVA.
#. For each table entry, decompress into the destination section. The writable
   span is the distance to the next section's start (or to ``SizeOfImage`` for
   the last), and it must be at least the section's virtual size. A block that
   decompresses to fewer bytes than the virtual size produces a warning, not an
   error.
#. Reparse the result in memory (``in_memory_reparsed``). The CLI additionally
   reparses the written file from disk (``disk_reparsed``).

.. note::

    The output is an analysis artifact and is not a Windows-loadable file.
    Producing one is a separate step, ``relayout_conventional`` in
    :doc:`/tool/crates/rebuild`.

Conventional relayout
---------------------

Because the analysis image preserves every section's ``VirtualAddress``,
conversion to a conventional file never needs to rewrite a directory, a
relocation or a branch. Only ``PointerToRawData`` and ``SizeOfRawData`` change,
and zero-fill sections keep no raw data. The tested invariant is that the bytes
the loader maps are unchanged apart from the section header table. The
limits are: PE32+ only, at most 96 sections, no security directory, and no
overlapping sections.

Protector sections are kept as evidence at this point, so
``normalize_characteristics`` clears ``IMAGE_FILE_RELOCS_STRIPPED`` when a
relocation directory exists, so that the Windows loader still applies base
relocations.

See :doc:`/vmprotect/packing` for how VMProtect's packer works and why this
approach defeats it.
