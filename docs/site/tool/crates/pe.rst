vmpx-pe
=======

Strict PE/COFF parsing and the address model used by every other crate. It has
no vmpx dependencies.

Address types
-------------

Four newtypes keep different address spaces from being mixed up. All are
``#[serde(transparent)]``, so they serialize as plain integers, and all display
as ``0x`` followed by uppercase hexadecimal.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Type
     - Inner
     - Meaning
   * - ``Rva``
     - ``u32``
     - Relative virtual address; supports ``checked_add``
   * - ``Va``
     - ``u64``
     - Absolute virtual address
   * - ``FileOffset``
     - ``u64``
     - Offset within the file
   * - ``ImageBase``
     - ``u64``
     - Preferred load address; ``checked_add_rva`` and ``checked_rva_of``
       convert between ``Rva`` and ``Va``

The parser
----------

.. code-block:: rust

    impl<'a> PeImage<'a> {
        pub fn parse(data: &'a [u8]) -> Result<Self, PeError>;
        pub fn data(&self) -> &[u8];
        pub fn directory(&self, index: usize) -> Option<&DataDirectory>;
        pub fn report(&self) -> PeReport;
        pub fn rva_to_file_offset(&self, rva: Rva) -> Result<FileOffset, PeError>;
        pub fn file_offset_to_rva(&self, offset: FileOffset) -> Result<Rva, PeError>;
        pub fn parse_unwind_info(&self, rva: Rva) -> Result<UnwindInfo, PeError>;
        pub fn runtime_functions_containing(&self, rva: Rva) -> Vec<RuntimeFunction>;
    }

Parsing is strict and bounds-checked. Errors are returned through ``PeError``;
softer inconsistencies are collected as ``Diagnostic`` values in the report
rather than aborting.

Model types
-----------

``PeKind`` (``Pe32`` or ``Pe32Plus``), ``CoffHeader``, ``OptionalHeader``,
``Section`` (with ``mapped_span`` and ``contains_mapped_rva``), ``DataDirectory``,
``DirectoryAddressKind``, ``ImportModule``, ``ImportSymbol``, ``ExportDirectory``,
``ExportSymbol``, ``TlsDirectory``, ``RuntimeFunction``, ``UnwindInfo``,
``UnwindCodeSlot``, ``BaseRelocationBlock``, ``BaseRelocation``, ``DebugEntry``,
``LoadConfig``, ``Diagnostic`` and ``PeReport``.

``PeReport`` is the serializable summary printed by :doc:`/tool/cli/info`. Its
fields are described there.

Directory constants
-------------------

``IMAGE_DIRECTORY_ENTRY_EXPORT``, ``_IMPORT``, ``_RESOURCE``, ``_EXCEPTION``,
``_SECURITY``, ``_BASERELOC``, ``_DEBUG``, ``_TLS``, ``_LOAD_CONFIG``, ``_IAT``
and ``_COM_DESCRIPTOR``.

The mapped span and raw/virtual geometry
----------------------------------------

Much of vmpx's reasoning depends on the difference between a section's *file*
geometry (``PointerToRawData``, ``SizeOfRawData``) and its *memory* geometry
(``VirtualAddress``, ``VirtualSize``). ``Section::mapped_span`` returns the extent
that the loader will map, and ``contains_mapped_rva`` answers whether an RVA falls
inside it. A section with ``PointerToRawData == 0`` and ``SizeOfRawData == 0`` but a
non-zero ``VirtualSize`` is a *virtual-only* section, the signature of a packed
image (see :doc:`detect`).
