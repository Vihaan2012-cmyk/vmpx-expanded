vmpx-rebuild
============

Guarded PE rewriting: section append, entry and TLS redirects, VM-stub redirects,
conventional relayout, loader validation and import-table reconstruction.
*Depends on:* ``vmpx-pe``.

Every operation is a guarded transaction: the output is reparsed and checked
before it is returned.

Section mutation
----------------

``append_section`` (with ``AppendSection``, ``AppendedSection``, ``RebuiltPe``),
``next_section_rva``, ``set_entry_point`` and ``RebuildError``.

``next_section_rva(input)``, ``set_entry_point(input, entry_point)`` and
``append_section(input, AppendSection)`` take and return whole images as byte
slices. ``append_section`` adds one file-backed section to a PE32+ image without moving
existing bytes. It needs an unused 40-byte section-table slot before
``SizeOfHeaders`` and the first raw section, all-zero slack, no overlap with a data
directory, and no security directory. ``set_entry_point`` requires the target RVA
to be in an executable, file-backed section.

Reintegration
-------------

.. code-block:: rust

    pub fn reintegrate(
        conventional: &[u8],
        request: ReintegrationRequest<'_>,
    ) -> Result<(Vec<u8>, ReintegrationReport), RebuildError>;

    pub fn set_tls_callback(
        input: &[u8],
        index: usize,
        target: Rva,
    ) -> Result<Vec<u8>, RebuildError>;

    pub fn redirect_vm_stub(
        input: &[u8],
        stub_rva: Rva,
        target_rva: Rva,
    ) -> Result<Vec<u8>, RebuildError>;

See :doc:`/tool/pipeline/reintegration` for the request and report types and the
layout contract.

Layout
------

.. code-block:: rust

    pub fn relayout_conventional(image: &[u8]) -> Result<(Vec<u8>, RelayoutReport), LayoutError>;
    pub fn validate_for_loader(image: &[u8]) -> Result<LoaderValidationReport, LoaderValidationError>;
    pub fn is_conventional_layout(pe: &PeImage<'_>) -> bool;
    pub fn is_rva_layout(pe: &PeImage<'_>) -> bool;
    pub fn is_discardable(section: &Section) -> bool;
    pub fn normalize_characteristics(input: &[u8]) -> Result<Vec<u8>, LayoutError>;

``relayout_conventional`` converts an RVA-layout analysis image into a conventional
file. Every section keeps its ``VirtualAddress``, so no directory, relocation or
branch needs rewriting; only ``PointerToRawData`` and ``SizeOfRawData`` change.
Zero-fill sections keep no raw data. The tested invariant: the bytes the loader
maps are unchanged apart from the section header table. Limits: PE32+ only, at
most 96 sections, no security directory, no overlapping sections.

``validate_for_loader`` walks every directory the loader walks and requires the
referenced bytes to be present. See :doc:`/tool/pipeline/reintegration` for the
checks. Result types: ``RelayoutReport``, ``SectionLayout``,
``LoaderValidationReport`` (counts of imports, exports, TLS callbacks and
relocations) and the error types ``LayoutError`` and ``LoaderValidationError``.

Imports
-------

``serialize_import_table``, ``plan_import_reconstruction`` and
``install_import_directories``, with ``ImportModuleSpec``, ``ImportTarget``,
``ImportReconstructionPlan``, ``ImportSlotRemap``, ``SerializedImportTable``,
``SerializedDirectoryRange`` and ``ImportTableError``.

Application import-set reconstruction (SP3):

.. code-block:: rust

    pub fn bindings_from_modules(modules: &[ImportModule]) -> Vec<SlotBinding>;
    pub fn reduce_to_proven_imports(...) -> ProvenImportSet;
    pub fn plan_proven_imports(...) -> Result<ImportReconstructionPlan, ImportTableError>;

With ``SymbolRef`` (``Name`` or ``Ordinal``), ``SlotBinding``, ``ProvenImport``,
``ProvenModule``, ``ProvenImportSet``, ``UnresolvedSlot`` and ``UnresolvedReason``.
See :doc:`/tool/pipeline/imports-and-stripping`.
