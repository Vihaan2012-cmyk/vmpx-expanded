Crate Reference
===============

The workspace has seventeen crates; the first eleven are described below, and the
later additions are listed at the end of this page. Each library exposes its surface from
``src/lib.rs`` through ``pub use`` re-exports, ``pub fn`` items and public types;
the pages below list that surface with its responsibilities.

.. code-block:: text

    vmpx-pe ---+--> vmpx-detect
               +--> vmpx-unpack
               +--> vmpx-rebuild
               +--> vmpx-disasm ---> vmpx-vm ---+
                                                +--> vmpx-vmlift
    vmpx-recompile -----------------------------+

    vmpx-cli depends on all of the above and produces the vmpx binary;
    vmpx-ui drives that binary as a child process.

.. toctree::
   :maxdepth: 1

   pe
   detect
   disasm
   unpack
   vm
   vmlift
   recompile
   rebuild
   cli
   vtil-sys

``vmpx-ui`` has its own page, :doc:`/tool/ui`.

Crates added since this page was first written
----------------------------------------------

The workspace now has seventeen crates. The diagram and the pages above cover the
original eleven. The additions have no page of their own yet; their roles, taken from
each crate's documentation:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Crate
     - Role
   * - ``vmpx-emulate``
     - The contained dynamic-assist engine: an in-process x86-64 interpreter, PE loader,
       Win32 API stubs, OEP and dump detection, observation hooks and the
       ``ObservedFacts`` hand-off (:doc:`/vmprotect/devirtualization/dynamic-assist`)
   * - ``vmpx-mutation``
     - Shared types for mutation-mode deobfuscation (problem, candidate, cost, score),
       native CFG recovery and dead-term elimination
   * - ``vmpx-solvers``
     - The solver portfolio: linear and general MBA, opaque-predicate, substitution,
       dictionary and synthesis solvers
   * - ``vmpx-tournament``
     - The per-problem-class, cost-aware ELO ladder that ranks and routes solvers
   * - ``vmpx-verify``
     - The input-testing scorer and the SMT equivalence gate that promotes a candidate
       to the sound tier
   * - ``vmpx-demutate``
     - The tournament wired end to end, including a binary demutation pass and a
       synthetic benchmark

Dependency summary
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Crate
     - Depends on
     - Role
   * - ``vmpx-pe``
     - none (workspace)
     - PE/COFF parsing and address model
   * - ``vmpx-detect``
     - ``vmpx-pe``
     - VMProtect evidence and layout classification
   * - ``vmpx-disasm``
     - ``vmpx-pe``, ``iced-x86``
     - Decoding, CFG, IAT references, dispatch proofs, dangling references
   * - ``vmpx-unpack``
     - ``vmpx-pe``, ``lzma-rs``
     - PACKER_INFO and LZMA restore
   * - ``vmpx-vm``
     - ``vmpx-pe``, ``vmpx-disasm``, ``iced-x86``
     - The VM analysis engine
   * - ``vmpx-vmlift``
     - ``vmpx-vm``, ``vmpx-recompile``
     - Lifting, readiness, replacement construction, equivalence
   * - ``vmpx-recompile``
     - ``iced-x86``
     - Conservative AMD64 code generation
   * - ``vmpx-rebuild``
     - ``vmpx-pe``
     - Guarded PE rewriting and reintegration
   * - ``vmpx-cli``
     - all of the above, ``clap``, ``serde``, ``serde_json``, ``sha2``, ``anyhow``
     - The ``vmpx`` binary and ``clean`` pipeline
   * - ``vmpx-ui``
     - ``eframe``, ``rfd``, ``serde_json``, ``anyhow``
     - egui front-end
   * - ``vmpx-vtil-sys``
     - ``thiserror``
     - Feature-gated FFI to the VTIL bridge

Workspace-wide settings
-----------------------

.. code-block:: toml

    [workspace.package]
    version = "0.1.0"
    edition = "2024"
    license = "MIT"
    rust-version = "1.88"

    [workspace.lints.rust]
    unsafe_code = "forbid"

    [workspace.lints.clippy]
    all = "warn"
    pedantic = "warn"

``unsafe_code = "forbid"`` applies to every crate except ``vmpx-vtil-sys``, which
opts out (``unsafe_code = "allow"``) because it is the only crate that performs
FFI.
