vmpx-unpack
===========

Locates the ``PACKER_INFO`` table and restores LZMA-compressed sections into an
RVA-layout analysis image. *Depends on:* ``vmpx-pe``, ``lzma-rs``.

API
---

.. code-block:: rust

    pub fn unpack_to_mapped_image(
        data: &[u8],
        limits: UnpackLimits,
    ) -> Result<UnpackResult, UnpackError>;

    pub fn locate_packer_table(
        pe: &PeImage<'_>,
        limits: UnpackLimits,
    ) -> Result<PackerTable, UnpackError>;

Types
-----

.. code-block:: rust

    pub struct UnpackLimits { /* max_image_size: 2 GiB, max_packed_sections: 256 */ }

    pub enum PackerLayoutKind { LegacyPlaintext, EncodedRotatingKey }

    pub struct PackerEntry { pub source_rva: Rva, pub destination_rva: Rva }

    pub struct PackerTable {
        pub kind: PackerLayoutKind,
        pub file_offset: FileOffset,
        pub destination_key: Option<u32>,
        pub lzma_properties: [u8; 5],
        pub entries: Vec<PackerEntry>,
    }

    pub struct UnpackedBlock {
        pub section_index: usize,
        pub section_name: String,
        pub source_rva: Rva,
        pub destination_rva: Rva,
        pub section_virtual_size: u32,
        pub destination_capacity: u32,
        pub decompressed_size: u32,
    }

    pub struct UnpackReport {
        pub schema_version: u32,
        pub input_size: u64,
        pub output_size: u64,
        pub packed_sections: usize,
        pub packer_table: PackerTable,
        pub blocks: Vec<UnpackedBlock>,
        pub in_memory_reparsed: bool,
        pub disk_reparsed: bool,
        pub warnings: Vec<String>,
    }

    pub struct UnpackResult { pub image: Vec<u8>, pub report: UnpackReport }

``UnpackError`` has four variants, each prefixed in its message: ``Pe``,
``Locate``, ``Decompress { block, section, message }`` and ``Reconstruct``.

Algorithm
---------

The table search, the rotating-key scheme, the LZMA header synthesis and the image
reconstruction are described in :doc:`/tool/pipeline/unpacking`, and the format
itself in :doc:`/vmprotect/packing`.
