Lifting Decoded Functions
=========================

*Crates:* :doc:`/tool/crates/vmlift`, :doc:`/tool/crates/recompile`
*Input:* :doc:`compact-vm`

This stage takes a decoded function (a linear stream, or a recovered CFG) and
produces a native replacement. It is the last step before reintegration, and it
shares the rule of every other stage: a function is lifted completely or not at
all.

Pipeline reached
----------------

.. code-block:: text

    detect -> bootstrap -> dispatch -> handlers -> stream decode -> CFG walk -> LIFT to native

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Function shape
     - State
   * - Linear
     - Fully devirtualized to native x86-64
   * - Branching
     - Decoded through the CFG, branch targets resolved by symbolic execution
       and proven input domains, then lifted by the CFG-level lift, including
       region (continuation) exits. The first real branching function (RVA
       ``0x126b02`` of the Compact sample) devirtualizes to native code
   * - Anything else
     - Refused with a reason (:doc:`/vmprotect/devirtualization/static-frontier`)

The lift is exercised from the library and its examples (``cfglift``,
``liftdiag``) and, with ``clean --experimental --decoded-lift``, is spliced into
a hybrid image (:doc:`reintegration`). It is opt-in because it is slow on large
inputs; a plain ``clean`` does not run it.

Semantic lowering
-----------------

Every decoded VM instruction first passes ``semantic_lowering``, which assigns
one of four outcomes:

``Computed``
    Symbolically executed to a pure expression that the back end computes.
    Covers dispatch, constants, virtual-register access, push and pop, ``Add``,
    ``Sub``, ``Nor``, ``Nand``, all shifts and rotates, and memory load/store.
``Passthrough``
    Re-emitted as the identical native instruction: ``Cpuid`` and
    ``GsRelativeLoad`` (marked as a machine-state read that static equivalence
    must trust), and ``ShiftLeftDouble``, ``ShiftRightDouble`` and ``PopFlags``.
``BlockExit``
    ``VmExit``, handled by the exit bridge.
``Refused``
    No verified lowering: ``TableChecksum``, ``Mul``, ``SignedMultiply``,
    ``UnsignedDivide`` and ``ReadTimestampCounter``. Control transfers also
    report as refused in the linear lift because they need the CFG lift.

The first instruction with no lowering stops the lift with its index and reason
(``first_unliftable``). Nothing is approximated.

Linear functions
----------------

``lift_decoded_linear`` executes the decoded stream symbolically from a proven
entry frame and recovers the native exit state from the VM-exit handler's
restore layout (``recover_stub_exit``). The exit must be a callee-style return:

* every architectural register is restored exactly once, and flags are restored;
* the return address is a stack slot the VM never wrote, at or above the entry
  stack pointer, so a native ``ret`` reads the same word;
* the stack unwind is 8-byte aligned.

The result goes through ``build_replacement_popping``: each exit value is
computed into its own spill slot, observable effects (stores and imported calls)
are replayed first, and one epilogue loads the registers and returns. The lift
records the assumptions it relies on as ``LiftAssumption`` values:
``FlagsVolatileAtReturn`` (flags are volatile across a Win64 call, so they may
be left clobbered) and ``VmStateSurvivesNativeCall``.

Branching functions
-------------------

``lift_decoded_cfg`` lifts a fully recovered CFG. Each block is symbolically
executed from the exact state its predecessor left, so the CFG is expanded into
a tree of paths and a join is handled by re-expanding the successor under each
incoming state rather than by merging states.

* A conditional VM branch becomes a native compare-and-jump on the symbolic
  branch target. It is lowered only when its targets resolve over a proven
  domain and every target is a recovered CFG edge.
* A block that ends in a native call becomes a real call through the symbolic
  ``[vsp]`` target, with up to four arguments. Memory around the call is handled
  by the call's memory summary; without one, the call is assumed not to write the
  VM stack or context, and only when neither the target nor any argument can
  carry a pointer into the VM stack.
* A memory read that precedes an external store or call on the same path is
  hoisted, with an alias check against that path's earlier stores, so a
  function that reads state and then writes can still be lifted. A read that
  cannot be shown unaffected by a barrier is refused.
* A VM exit that is not a callee return is a *region exit* (continuation exit):
  the VM wrote the return address, so the exit continues into another native or
  VM entry. The lift stages the continuation's stack words in the top of the
  frame, restores the entry flags, loads the exit registers and returns through
  the staged target, leaving those staged bytes on the stack
  (``ReturnRetaining``). Together with nested-entry edges in the CFG
  (:doc:`compact-vm`), this lets a function whose exit tail-continues into
  another virtualized entry be lifted as one unit. An exit whose flags are a stack
  pointer is still refused.
* Expansion is bounded (``max_expanded_blocks``).

The lift refuses the whole function on an incomplete CFG, a block that repeats on
one path (a loop), a branch whose targets are not proven recovered edges, a
condition or value the back end cannot lower, or any block with an unlowerable
instruction. No partial function is ever returned.

The back end
------------

The recompile IR gained the operations the new lowerings need: ``Cpuid``,
``GsLoad``, ``DoubleShift``, ``ReadFlags``, ``EntryStackPointer`` and
``ReturnPopping`` and ``ReturnRetaining`` forms of return. As before, unsupported operations and unsafe ABI
states are rejected, not approximated (:doc:`lifting-recompilation`).
