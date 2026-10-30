Mutation
========

*Mutation* is VMProtect's second code transformation. Where virtualization replaces
native code with bytecode, mutation keeps the code native but rewrites it into a
much larger, much less readable form that does the same thing. It is applied in two
places: to regions the user marks for mutation, and to **everything the protector
itself emits**: the entry stubs, the bootstraps and the VM handlers. That second use
is what makes the VM hard to recognise.

The transformations
-------------------

The families below are those described in public analyses of the product and
visible in the structures vmpx handles. A mutated region typically uses several of
them at once.

Instruction substitution
~~~~~~~~~~~~~~~~~~~~~~~~

An instruction is replaced by an equivalent sequence. Examples, all with the same
effect on the register and a possibly different effect on the flags:

.. code-block:: nasm

    ; original
    add     eax, 1
    ; substituted
    sub     eax, -1
    ; or
    lea     eax, [rax+1]
    ; or
    not     eax
    neg     eax
    not     eax
    neg     eax
    ; ... and so on

Constant unfolding
~~~~~~~~~~~~~~~~~~

A literal is replaced by arithmetic that computes it:

.. code-block:: nasm

    ; original
    mov     eax, 0x12345678
    ; unfolded
    mov     eax, 0x6C9E2A5B
    xor     eax, 0x7EAA7C23
    add     eax, 0x1A3F0E08
    rol     eax, 5

Junk code and dead stores
~~~~~~~~~~~~~~~~~~~~~~~~~

Instructions whose results are never used are interleaved with real ones: arithmetic
into registers that are overwritten before being read, pushes and pops that cancel,
loads from memory whose value is discarded. They make the code longer and force an
analysis to compute liveness before it can tell what matters. Mutation often chooses
junk that *looks* meaningful, such as reads from the stack, so that a human cannot
dismiss it by eye.

Opaque predicates
~~~~~~~~~~~~~~~~~

A conditional branch whose outcome is fixed but non-obviously so:

.. code-block:: nasm

    mov     eax, [rsp+8]
    imul    eax, eax
    test    al, 2               ; bit 1 of a perfect square is always 0
    jnz     never_taken         ; so this jump is dead

A symbolic engine has to prove the predicate constant to remove the dead edge.

Register renaming and swapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The register allocation of an otherwise identical fragment is permuted, including
moves through temporary registers that serve no purpose. This is why the register
roles in a VM (VIP, VSP, key) change between instances.

Control-flow fragmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

A straight run of code is cut into fragments joined by unconditional jumps, and the
fragments are placed throughout the section in an arbitrary order:

.. code-block:: nasm

    fragment_a:
        mov     rax, [rbp]
        add     rax, 1
        jmp     fragment_b          ; somewhere else in the section
    ...
    fragment_b:
        mov     [rbp], rax
        jmp     fragment_c

Linear disassembly of the section yields nonsense, and a disassembly that follows
only fall-through finds tiny pieces. The effect is strongest on the VM handlers.

Mutation of stack discipline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Values are moved on and off the stack in indirect ways: pushed and popped to other
registers, stored with a negative offset and then the stack pointer adjusted, and so
on. This hides both the stack effect of a fragment and the data flow.

Ultra
-----

The "ultra" compilation type applies mutation *and* virtualization to a region: the
region is virtualized, and the VM that runs it is built from mutated handlers. In
practice every protected region of a modern build gets some of both, because the VM
code itself is always mutated.

What mutation does *not* change
-------------------------------

Mutation preserves semantics. This is what an attacker relies on. However confused
the form, the **observable behaviour of a region** is the same: the same register
values on exit, the same memory writes, the same calls. A semantic analysis that
ignores the form and reasons about effects can, in principle, see through any
mutation. The attack surface is therefore:

* the cost of getting there: the analysis must handle large amounts of junk without
  state explosion;
* mutations that preserve semantics only under assumptions the analysis cannot see,
  such as an opaque predicate that depends on memory the analysis treats as unknown.

How vmpx handles it
-------------------

vmpx does not try to *undo* mutation. It never matches a mutated pattern. It reasons
about what a fragment *does*:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mutation
     - Handling
   * - Junk code, dead stores
     - The backward slice keeps only instructions that define live-out registers
       (VIP, key, handler) or flags, so junk drops out
   * - Register renaming
     - Register roles are *recovered* per instance by enumerating role assignments
       and keeping those that form a valid dispatch slice
   * - Fragmentation
     - Direct-jump fragment chains are traced and linearized by reachability
   * - Dispatch via ``jmp reg`` or ``push reg; ret``
     - Both accepted: "a mutation, not a semantic difference"
   * - Duplicate saved-register slots
     - Legal in the context scatter; the last write wins
   * - Constant unfolding, substitution
     - Symbolic execution over a bit-vector expression IR folds them away; the
       expression folder recognises idioms like ``not(not(x)+a)+b``
   * - Mutated VM exit
     - Tolerated if each register's *final* definition is its ``pop``
   * - Opaque predicates
     - Handled when they reduce to a constant under the expression simplifier;
       otherwise the path is explored, and it is a blocker if it cannot be
       resolved

See :doc:`devirtualization/attacking-mutation` for a longer treatment.
