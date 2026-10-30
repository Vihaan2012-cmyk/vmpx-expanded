The Virtual Machine
===================

Virtualization is VMProtect's central protection. A region of native code is
translated into *bytecode* for a processor that exists only inside the protected
file, and an *interpreter* for that bytecode is embedded alongside it. The original
x86-64 instructions are discarded.

.. code-block:: text

        native caller
             |  call fn
             v
        +------------------+         +--------------------------------------+
        | entry stub       |         |  VM (per instance)                   |
        |  push <seed>     |         |                                      |
        |  call bootstrap  |-------->|  bootstrap: save native state,       |
        +------------------+         |    build frame, load VIP/key         |
                                     |        |                             |
                                     |        v                             |
                                     |  dispatch --> handler --> dispatch   |
                                     |      ^             |                 |
                                     |      +-------------+  (threaded)     |
                                     |        |                             |
                                     |        v                             |
                                     |  VM exit: restore state, ret         |
                                     +--------------------------------------+
             ^                                          |
             +------------------------------------------+
                         back in native code

The vocabulary is borrowed from ordinary interpreters:

*Bytecode*
    The encoded virtual instructions. Each operand is encrypted with a rolling key.

*VIP*
    The *virtual instruction pointer*: the native register that points at the next
    byte of bytecode.

*VSP*
    The *virtual stack pointer*: the native register that addresses the VM's
    operand stack.

*Handler*
    A short fragment of native code that implements one virtual operation, then
    dispatches to the next.

*Dispatch*
    Fetching the next virtual opcode and jumping to the right handler.

*Context*
    The VM's private state: its virtual registers and saved native registers.

*Bootstrap*
    The native code that builds a VM frame when execution enters the VM.

The pages in this section describe each part, in the order execution meets them.

.. toctree::
   :maxdepth: 1

   architecture
   entry
   bytecode
   dispatch
   handlers
   context-switch
   instances

What makes VMProtect's VM hard
------------------------------

Interpreters are not unusual. What makes this one costly to analyse is that **every
constituent is randomised per build**:

* the mapping from virtual opcodes to handlers;
* the encoding of the bytecode and of each operand;
* which native registers play VIP, VSP, key and handler pointer;
* the *shape* of every handler, since each is mutated (:doc:`../mutation`);
* where handler fragments sit in the file, since they are scattered and joined by
  jumps.

There is therefore no stable signature. An analysis must recover the VM's structure
from its *behaviour*, which is the approach every stage of the vmpx pipeline takes
(:doc:`/tool/pipeline/vm-recovery`).
