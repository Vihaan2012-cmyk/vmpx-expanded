vmpx-detect
===========

Structural VMProtect evidence scoring and image-layout classification.
*Depends on:* ``vmpx-pe``.

API
---

.. code-block:: rust

    pub fn detect_vmprotect(pe: &PeImage<'_>) -> ProtectionReport;
    pub fn va_owner(pe: &PeImage<'_>, va: Va) -> Option<usize>;

``va_owner`` returns the index of the section that owns a virtual address.

Types
-----

.. code-block:: rust

    pub enum DetectionVerdict {
        NoStrongEvidence,
        Suspicious,
        VmprotectLikely,
    }

    pub enum ImageLayout {
        ConventionalFile,
        PackedVirtualSections,
        RvaMappedAnalysisImage,
        Ambiguous,
    }

    pub struct DetectionEvidence {
        pub id: &'static str,
        pub weight: u8,
        pub statement: String,
    }

    pub struct ExecutableRegion {
        pub section_index: usize,
        pub section_name: String,
        pub start: Rva,
        pub size: u32,
        pub entropy: Option<f64>,
        pub owns_entry_point: bool,
    }

    pub struct ProtectionReport {
        pub schema_version: u32,
        pub verdict: DetectionVerdict,
        pub confidence: u8,
        pub image_layout: ImageLayout,
        pub entry_point: Rva,
        pub entry_section: Option<usize>,
        pub initialized_virtual_only_sections: usize,
        pub rva_mapped_sections: usize,
        pub executable_regions: Vec<ExecutableRegion>,
        pub evidence: Vec<DetectionEvidence>,
        pub warnings: Vec<String>,
    }

Design
------

Detection is a weighted evidence model rather than a signature scanner. The
weights, thresholds and the entropy cut-off (7.2 bits per byte) are listed in
:doc:`/tool/cli/detect`. Entropy is the Shannon entropy of the section's raw bytes;
a section with no raw data has no entropy value.

Section names are weak evidence by design: ``.vmp0`` and ``.vmp1`` are the
conventional names, but a protector can rename them, so names alone never decide
the verdict, and a verdict reached without names carries a warning.
