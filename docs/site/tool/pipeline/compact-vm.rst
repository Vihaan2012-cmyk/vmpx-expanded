The VMProtect 3.11 Compact VM
=============================

*Crates:* :doc:`/tool/crates/vm`
*Background:* :doc:`/vmprotect/vm/dispatch`, :doc:`/vmprotect/vm/handlers`

The first real-sample target is a VMProtect 3.11 build using the "Compact"
virtual machine. Its VM is decoded end to end: every handler in the dispatch
table is classified, the bytecode stream of a function is decoded, and branches
are followed through a control-flow walk. This page records what that covers.

Result
------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Item
     - State
   * - Handlers classified
     - 256 of 256. No ``Unknown`` and no failed classification
   * - Dispatch
     - Table-indexed and keyless; modelled and decoded
   * - Linear instruction stream
     - Decoded, with the table base propagated through the stream
   * - Branches
     - Followed by a CFG walk; targets resolved by symbolic execution and
       proven input domains
   * - Native calls and checksum loops
     - Decoded as dedicated handler semantics

The classification is by effect, as everywhere else in vmpx: no handler bytes or
addresses are assumed, and a handler is given a semantic only when its
normalized shape is unambiguous and its value flow is proven
(:doc:`vm-recovery`).

Table-indexed keyless dispatch
------------------------------

Earlier work modelled threaded dispatch, where each handler computes the next
handler from an encrypted displacement and a rolling key. The Compact VM does
not use a key. Its dispatch is:

* the opcode is read raw from ``[vip]``, with no rolling-key transform;
* it indexes a static table of signed relative entries;
* ``handler = table_base + sext(entry[opcode])``.

The table base is the dispatch anchor. When a native ``lea`` establishes it
statically it is recorded in the codec (``TableIndexedDispatch``); otherwise the
decoder uses the dispatch state's handler as the base. Postlude-branch dispatch,
where the branch to the next handler sits after the handler's semantic work, is
recognised as well.

Keyless operand recovery
~~~~~~~~~~~~~~~~~~~~~~~~

With no key, operands are plain loads. A keyless operand is a raw ``mov`` or
``movzx`` of ``[vip]`` whose width equals the VIP advance, with the register
untouched until its first consumer and no interaction with the key. Flag capture
follows ``pushfq; pop reg; mov [vsp], reg``. Arithmetic on the key and handler
registers counts as semantic work when the VM is keyless. Virtual-register slot
indices may be 16 bits wide.

The handler semantic set
------------------------

The recovered semantics, in addition to the stack, memory and control families
listed in :doc:`vm-recovery`:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Family
     - Semantics
   * - Arithmetic
     - ``Add``, ``Sub``, ``Mul``, ``SignedMultiply`` (``rdx:rax = rax * operand``),
       ``UnsignedDivide``, including narrow-width variants
   * - Logic
     - ``Nor``, ``Nand`` (the VM's basis; ``not``, ``and``, ``or`` and ``xor``
       are built from them)
   * - Shifts and rotates
     - ``ShiftLeft``, ``ShiftRightLogical``, ``ShiftRightArithmetic``,
       ``RotateLeft``, ``RotateRight``, and the double shifts
       ``ShiftLeftDouble`` / ``ShiftRightDouble`` (``shld`` / ``shrd``)
   * - Machine state
     - ``Cpuid``, ``GsRelativeLoad`` (``gs:[pointer]``), ``PopFlags``,
       ``ReadTimestampCounter``
   * - Native bridge
     - ``NativeCall``, decoded from the keyless call-bridge handlers
   * - Integrity
     - ``TableChecksum``, the byte-at-a-time table checksum over program memory
   * - Control
     - ``ConditionalBranch``, ``UnconditionalBranch``, ``VmExit``

Stream decode and the CFG walk
------------------------------

``recover_and_decode_linear_vm_stream`` recovers each handler's codec on demand
and decodes a function's instructions up to the first control-flow boundary. For
a linear function that is the whole body, ending at ``VmExit``.

``recover_vm_cfg`` (module ``cfg.rs``) continues from there. It decodes the
entry block, symbolically executes it from a proven entry frame, and follows
each block's terminating branch to its successors, carrying the symbolic VM
state. Each block ends in one of four ways: ``VmExit``, ``Branch``,
``NativeCall`` or ``Stalled``. Edges are one of three kinds:

``Branch``
    The target simplified to a constant.
``GuardedBranch``
    Several targets, each proven reachable over an exhaustive input domain
    (:doc:`symbolic-execution`).
``NativeCallReturn``
    Control resumes after a proven native call.

A successor is followed only when its branch target is proven. Anything else
stops that path and is recorded as a ``CfgStall`` with the reason; no edge is
invented. A CFG is *complete* only when it has no stalls and no limit was hit
(blocks, visits, per-block visits, total instructions). Symbolic expressions use
structural sharing and simplification so that long functions stay within bounds.

Native calls
~~~~~~~~~~~~

Keyless native-call bridge handlers are decoded to a ``NativeCall`` whose call
frame (target and arguments) is recovered from the VM stack. The CFG walk
treats the call as a block boundary, proves the frame, and resumes at the
recorded continuation.

What it does not yet do
-----------------------

Decoding is complete for the handler set, but decoding is not lifting: three
decoded semantics (``TableChecksum``, the wide multiplies and divides, and
``ReadTimestampCounter``) have no verified native lowering, so a function using
them is decoded and reported but not turned into native code
(:doc:`function-lift`). The boundaries of what the walk can resolve are listed in
:doc:`/vmprotect/devirtualization/static-frontier`.
