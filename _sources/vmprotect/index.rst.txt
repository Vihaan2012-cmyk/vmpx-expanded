Introduction
============

This part of the documentation is a reference on how VMProtect works and how each
of its mechanisms is attacked statically. It is written for reverse engineers and
malware analysts who need to understand protected code, and it doubles as the
specification that explains *why* the vmpx pipeline is shaped the way it is.

VMProtect is a commercial software protector for native Windows executables. It
combines several independent protections, which can be enabled separately:

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Protection
     - Idea
     - Page
   * - Packing
     - Compress the original sections and restore them at load time
     - :doc:`packing`
   * - Virtualization
     - Replace native code with bytecode for a private virtual machine
     - :doc:`vm/index`
   * - Mutation
     - Rewrite native code into a larger, equivalent, confusing form
     - :doc:`mutation`
   * - Import protection
     - Hide which APIs the program calls and where
     - :doc:`imports`
   * - Memory and file integrity
     - Detect modification of code and of the file
     - :doc:`integrity`
   * - Anti-debug and anti-VM
     - Detect analysis environments and tooling
     - :doc:`anti-analysis`

How to use this part
--------------------

* Start with :doc:`overview` for the anatomy of a protected binary, then
  :doc:`layout` for what it looks like on disk.
* :doc:`packing` and :doc:`vm/index` are the core of the reference.
* :doc:`devirtualization/index` describes how each mechanism is attacked, and maps
  every attack onto the vmpx component that implements it.

What the sources of this knowledge are
--------------------------------------

Two kinds of statement appear, and they are kept distinct.

*Grounded in vmpx's source.* Layouts the vmpx code recognises, structures it
recovers and invariants it refuses to assume: the ``PACKER_INFO`` table and the
rotating key, the entry-stub shape, the keyed transform codecs, the relative
threaded dispatch, the native-call bridge, the exception-sled mechanism, the
checksum recurrence. Where a page states such a fact it names the module that
reveals it.

*General knowledge of VMProtect.* The wider picture from public reverse-engineering
research and from the product's own documentation: protection options, the SDK
markers, the typical anti-debug and anti-VM techniques, the virtual instruction
set as reported in published devirtualization work. These are described at the
level of detail that is stable across builds, and are flagged as general knowledge
when it matters.

.. warning::

    VMProtect is closed source and changes between versions. The generated code is
    deliberately randomised per build and per protected region. Nothing here is a
    specification published by the vendor. Treat register names, section names
    and sizes as examples of what is commonly observed, never as fixed constants;
    the vmpx code makes the same point by never assuming "handler bytes,
    addresses, or fixed register allocation".
