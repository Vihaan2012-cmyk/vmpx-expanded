Attacking the Virtual Machine
=============================

*Mechanism:* :doc:`/vmprotect/vm/index`
*Implemented by:* :doc:`/tool/pipeline/vm-recovery`, :doc:`/tool/pipeline/symbolic-execution`,
:doc:`/tool/pipeline/oep-recovery`, :doc:`/tool/pipeline/function-recovery`,
:doc:`/tool/pipeline/lifting-recompilation`

The strategy is to **model the VM exactly, then execute its bytecode symbolically**.
The model is not assumed; it is recovered from the VM's own code, one fact at a time,
and each fact is either proven or the process stops.

Step 1: find the entries
------------------------

Locate every ``push seed; call bootstrap`` stub and score it. A stub is not enough on
its own, so the evidence is the *chain* it leads to: many saved registers, flags,
a large aligned stack reservation, an indirect dispatch at the end. Group stubs that
share a bootstrap into instances.

*Output:* a list of entries with confidence, source and reachability, and a list of
VM instances (:doc:`/tool/cli/vm-list`).

Step 2: recover the roles
-------------------------

For each instance, work out which native register is the VIP, the key and the handler
pointer. The method is not to guess but to *enumerate* the architectural assignments
and keep those that produce a behaviourally valid dispatch slice. The bootstrap is
executed on a small concrete machine, with the seed placed where the entry stub left
it, to get the initial VIP, key and handler anchor.

*Output:* ``RecoveredEntryProfile`` with the roles and the bytecode direction.

Step 3: recover the dispatch codec
----------------------------------

For each handler, slice backwards from the terminal jump, extract the VIP-relative
loads as candidate codecs, and prove with a bounded search that every direct successor
reaches a dispatch through the handler register. The codec says *how to decode the
displacement*, and so how to find the next handler.

*Output:* ``RelativeDispatchCodec`` per handler, or a diagnostic.

Step 4: decode the bytecode
---------------------------

Simulate the decoder from the start of the bytecode. State is ``(VIP, key, handler)``;
each step decodes an optional operand and a displacement, moves to the next handler
and updates the key. Stop on any of the defined stop conditions rather than continuing
on a doubtful state.

*Output:* a linear stream of ``DecodedVmInstruction`` values per block.

Step 5: extract what each handler does
--------------------------------------

Extract the handler's *effects* from its semantic prefix: memory reads and writes by
region, the VSP adjustment, arithmetic, flags. Classify only when the normalized shape
is unambiguous, and require a def-use proof on top. Everything else stays ``Unknown``.

*Output:* ``VmHandlerSemantic`` per handler, with evidence.

Step 6: execute symbolically
----------------------------

Execute the decoded instructions over a bit-vector expression IR with an exact
byte-addressed memory model for the VM stack. Inputs are symbols named for the native
registers at entry (``native_rdx_initial``). Chain blocks across virtual branches and
native profile changes while carrying the full symbolic state. For a branch that depends
on an input, enumerate a *proven-exhaustive* finite domain and run each value as an
independent path (:doc:`/tool/pipeline/function-recovery`).

*Output:* for each path, a ``NativeExitState``: an expression for every register, the
flags and the return address, plus an ordered list of effects (stores, calls).

Step 7: prove the exit
----------------------

Verify that the last handler really is a VM exit (stack switch, every register popped,
``popfq``, ``ret``), then read the exact values from the terminal memory snapshot. If
the exit continues into another VM entry, compose its frame from the exact prior state
and continue; if it continues into native code, follow it with a bounded interpreter.

Step 8: lift and recompile
--------------------------

Turn the exit state into a native function: compute each register's expression into its
own spill slot, replay the effects, restore the flags, load every slot into its register
and return. Check that every needed lowering capability is implemented, and refuse
otherwise (:doc:`/tool/pipeline/lifting-recompilation`).

What the symbolic representation buys
-------------------------------------

Symbolic execution over an expression IR is the step that makes the rest tractable. It
is also what dissolves much of mutation: a junk sequence that adds a constant and then
subtracts it simply folds to the identity.

.. code-block:: text

    decoded virtual instructions          symbolic stack / registers
      LREG   r3          ; r3 holds rcx      stack: [native_rcx_initial]
      LCONST 0x10                            stack: [native_rcx_initial, 0x10]
      ADD                                    stack: [native_rcx_initial + 0x10, flags]
      SREG   r5          ; r5 maps to rax    rax = native_rcx_initial + 0x10

(The bytecode listing is schematic. Real output is a ``NativeExitState`` with an
expression per register.)

Control flow in the recovered semantics
---------------------------------------

Because branches are computed addresses, "control flow" in a virtualized function is
recovered by enumeration. For each reachable virtual branch with an input-dependent
target:

#. express the target as a function of the live-in registers;
#. check that every use of each register is of a form for which a finite cover is
   *provably* exhaustive (a low slice of at most 8 bits, or the add-status-flags of
   ``±reg + c``);
#. run each cover value as a separate path.

If any use is not of those forms (a jump table, a memory-dependent target, a loop on an
input, nested input branches, mixed widths) the function **stays a blocker**. The general
case is undecidable, and the implementation says so.
