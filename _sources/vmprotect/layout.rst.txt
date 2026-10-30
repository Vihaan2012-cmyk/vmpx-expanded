Sections and Image Layout
=========================

Section anatomy
---------------

A protected file adds several sections to the program's own. By default the
protector's sections are named ``.vmp0``, ``.vmp1`` and sometimes ``.vmp2`` (the
numbering follows the order the protector created them). Names are configurable and
randomizable, so they are weak evidence; vmpx counts them as +20 of a possible 100
confidence points and explicitly warns when a verdict is reached without them.

What lives in the protector's sections varies by version and options, but the
categories are stable:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Content
     - Role
   * - Runtime stub
     - The code that runs first: unpacking, initialization, environment checks
   * - Compressed original sections
     - The LZMA streams that the ``PACKER_INFO`` table points at
   * - VM entry stubs' targets: bootstraps
     - The shared code that builds a VM frame (:doc:`vm/entry`)
   * - VM handlers
     - Scattered, mutated native code implementing the VM's operations
   * - Bytecode
     - The encoded virtual instructions of every virtualized function
   * - Tables
     - The checksum table (:doc:`integrity`), import-resolution data
       (:doc:`imports`), and the ``PACKER_INFO`` table itself
   * - Exception metadata
     - ``RUNTIME_FUNCTION`` and ``UNWIND_INFO`` records for the handlers that use
       exception-driven control flow (:doc:`exceptions`)

Because the code, the data and the tables share sections, such a section is
frequently **readable, writable and executable at once**. vmpx treats a
readable-and-executable entry section as extra evidence (+5).

The memory image versus the file
--------------------------------

A PE has two geometries. The *file* geometry (``PointerToRawData``,
``SizeOfRawData``) says where each section's bytes sit in the file. The *memory*
geometry (``VirtualAddress``, ``VirtualSize``) says where the loader puts them. In
an ordinary file the two agree up to alignment padding. In a packed file they
diverge sharply:

.. code-block:: text

    section   VirtualAddress  VirtualSize  PointerToRawData  SizeOfRawData
    .text         0x1000        0x58000             0              0       <- virtual-only
    .rdata       0x59000        0x1C000             0              0       <- virtual-only
    .data        0x75000         0x8000             0              0       <- virtual-only
    .vmp0        0x7D000        0x40000         0x400          0x40000     <- the packed payload
    .vmp1        0xBD000        0x12000       0x40400          0x12000     <- runtime stub

(Illustrative; the numbers are not from a real sample.)

The *RVA-layout analysis image* that ``vmpx unpack`` produces is the opposite
extreme: ``PointerToRawData == VirtualAddress`` and ``SizeOfRawData ==
VirtualSize`` for every section, so a file offset *is* an RVA. It is the most
convenient shape for analysis because there is nothing to translate, and the
detector recognises it (``RvaMappedAnalysisImage``) when at least 80% of sections
have that shape.

Directories that the protector touches
--------------------------------------

Import directory
    Replaced or hollowed out by import protection. The imports that remain are
    what the *protector* needs; the program's own are resolved some other way
    (:doc:`imports`).

TLS directory
    Frequently used. A TLS callback runs before the module entry for every process
    attach and every thread attach, so it is an early foothold for initialization
    and for anti-debug checks. vmpx treats TLS callbacks as loader-reachable roots
    for VM discovery.

Exception directory
    The x64 ``.pdata`` table. Handler fragments are registered with exception
    handlers for the exception-driven control flow of :doc:`exceptions`, which also
    means the table is a map of the handler code that the analysis can use as a
    function-range oracle.

Base relocation directory
    May be stripped (``IMAGE_FILE_RELOCS_STRIPPED``), in which case the image must
    load at its preferred base. When present, the unpacker's restored sections
    contain absolute addresses that the loader fixes up.

Resources
    Optionally encrypted until first use.

Load configuration, debug directory
    The debug directory may be removed. The load configuration may be altered to
    fit the protector's own use of structured exception handling.

Header edits worth knowing
--------------------------

* ``AddressOfEntryPoint`` points into the protector.
* ``SizeOfImage`` covers the restored sections, not the file.
* The section table has little or no spare room. This matters to anything that
  tries to *add* a section: the space between the end of the section table and the
  first section's raw data is usually exhausted, which is why
  ``append_section`` in :doc:`/tool/crates/rebuild` requires an unused 40-byte
  slot and returns a typed error when there is none.
* Protection invalidates any existing Authenticode signature, so a signed
  protected file was signed *after* protection. vmpx's section appending refuses an
  image with a security directory, because relocating the certificate is not
  implemented.

Reading the layout quickly
--------------------------

A short checklist an analyst can apply by eye, which is exactly what
:doc:`/tool/cli/detect` automates:

#. Is there an executable section whose entropy is near 8?
#. Are there sections with a virtual size but no raw size?
#. Does the entry point sit in one of the odd sections?
#. Do the import, exception and TLS directories point into that same section?
#. Do the section names follow ``.vmpN``?

Three or more "yes" answers is strong structural evidence even when the names have
been randomized.
