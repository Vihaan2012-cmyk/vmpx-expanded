vmpx
====

vmpx is a static-analysis and recovery framework for Windows x64 binaries
protected with VMProtect. It is written in Rust, parses and analyses
protected files without ever executing them, and is built around one rule:
a result is reported as recovered only when it has passed verification.

The documentation is split in two. The first part documents the tool itself:
how to build it, every command-line subcommand and the JSON it emits, the
pipeline that takes a packed and virtualized image toward a clean one, and the
crates that implement each stage. The second part is a reference on how
VMProtect works (packing, the virtual machine, mutation, anti-analysis,
import protection) and how each of those mechanisms is attacked statically.

.. code-block:: console

    $ cargo build --release
    $ vmpx detect protected.dll
    $ vmpx unpack protected.dll analysis.bin --conventional loadable.dll
    $ vmpx vm-list analysis.bin
    $ vmpx clean protected.dll clean.dll --report report.json --experimental reintegrated.dll

.. note::

    vmpx recovers behaviour, not bytes. Virtualization is lossy: the original
    instruction encodings, function boundaries and padding are destroyed by the
    protector. A devirtualized output computes the same register state and
    effects; it is not the original compiler output and will not match a hash
    of the unprotected binary. See :doc:`tool/soundness`.

.. toctree::
   :caption: The Tool
   :maxdepth: 2

   tool/overview
   tool/getting-started
   tool/cli/index
   tool/pipeline/index
   tool/soundness
   tool/dll-support
   tool/crates/index
   tool/ui
   tool/testing

.. toctree::
   :caption: VMProtect Knowledge
   :maxdepth: 2

   vmprotect/index
   vmprotect/overview
   vmprotect/packing
   vmprotect/layout
   vmprotect/vm/index
   vmprotect/mutation
   vmprotect/anti-analysis
   vmprotect/integrity
   vmprotect/imports
   vmprotect/exceptions
   vmprotect/devirtualization/index
   vmprotect/glossary
   vmprotect/further-reading

.. toctree::
   :caption: Project
   :maxdepth: 2

   project/roadmap
   project/log
