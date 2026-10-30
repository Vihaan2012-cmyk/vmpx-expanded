Mapping Mechanisms to Tooling
=============================

Every VMProtect mechanism described in this part has a counterpart in the tool. The
table lists, for each, the page that explains it, the attack, the crate that implements
the attack and the command that exposes it.

.. list-table::
   :header-rows: 1
   :widths: 20 20 28 16 16

   * - VMProtect mechanism
     - Reference
     - Attack
     - Crate
     - Command
   * - Packed sections, ``PACKER_INFO``
     - :doc:`/vmprotect/packing`
     - Locate table, decode key, LZMA restore
     - :doc:`/tool/crates/unpack`
     - ``unpack``
   * - Section anatomy, entropy
     - :doc:`/vmprotect/layout`
     - Weighted structural evidence
     - :doc:`/tool/crates/detect`
     - ``detect``
   * - Entry stubs
     - :doc:`/vmprotect/vm/entry`
     - Scored discovery, instance grouping
     - :doc:`/tool/crates/vm`
     - ``vm-list``
   * - Bootstrap, register roles
     - :doc:`/vmprotect/vm/entry`
     - Role enumeration, concrete bootstrap run
     - :doc:`/tool/crates/vm`
     - ``clean``
   * - Rolling-key bytecode
     - :doc:`/vmprotect/vm/bytecode`
     - Keyed transform codecs
     - :doc:`/tool/crates/vm`
     - ``clean``
   * - Threaded relative dispatch
     - :doc:`/vmprotect/vm/dispatch`
     - Backward slice, dispatch codec, postlude proof
     - :doc:`/tool/crates/vm`, :doc:`/tool/crates/disasm`
     - ``clean``
   * - Handlers
     - :doc:`/vmprotect/vm/handlers`
     - Effect extraction and classification
     - :doc:`/tool/crates/vm`
     - ``clean``
   * - VM exit, native calls
     - :doc:`/vmprotect/vm/context-switch`
     - Exit proof, bridge recognition, call summaries
     - :doc:`/tool/crates/vm`
     - ``clean``
   * - Virtual flags
     - :doc:`/vmprotect/vm/context-switch`
     - ``vm_flags`` intrinsics, finite covers
     - :doc:`/tool/crates/vm`
     - ``clean``
   * - Mutation
     - :doc:`/vmprotect/mutation`
     - Effect-based slicing, symbolic folding
     - :doc:`/tool/crates/vm`
     - ``clean``
   * - Exception-driven flow
     - :doc:`/vmprotect/exceptions`
     - Stack sled recognition
     - :doc:`/tool/crates/vm`
     - ``clean``
   * - Checksum / memory protection
     - :doc:`/vmprotect/integrity`
     - Table checksum recognition and evaluation
     - :doc:`/tool/crates/vm`
     - ``clean``
   * - Timing checks
     - :doc:`/vmprotect/anti-analysis`
     - ``rdtsc`` intrinsic, neutralize
     - :doc:`/tool/crates/vm`
     - library
   * - Import protection
     - :doc:`/vmprotect/imports`
     - Resync scans, proven import set
     - :doc:`/tool/crates/disasm`, :doc:`/tool/crates/rebuild`
     - ``imports``
   * - Original entry point
     - :doc:`/vmprotect/overview`
     - Native continuation of the header entry
     - :doc:`/tool/crates/vm`
     - ``oep``
   * - Many virtualized functions
     - :doc:`/vmprotect/vm/instances`
     - Inventory, live-in, input domains
     - :doc:`/tool/crates/vmlift`, :doc:`/tool/crates/cli`
     - ``clean``
   * - Result code
     - :doc:`index`
     - Lift, compile, reintegrate
     - :doc:`/tool/crates/vmlift`, :doc:`/tool/crates/recompile`, :doc:`/tool/crates/rebuild`
     - ``clean``
