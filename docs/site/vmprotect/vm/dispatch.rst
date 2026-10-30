Dispatch
========

*Dispatch* is the step between two handlers: work out which handler runs next and
jump to it. In a conventional interpreter this is a table lookup. In VMProtect it is
engineered to resist exactly that kind of analysis.

Two dispatch styles
-------------------

Table dispatch
~~~~~~~~~~~~~~

The classical style, used by simple interpreters and reported for earlier
VMProtect versions: the next opcode byte indexes a table of handler addresses.

.. code-block:: nasm

    movzx   eax, byte [rsi]
    inc     rsi
    jmp     qword [table + rax*8]

The weakness is obvious: once the table is located, the handler for each opcode is
known.

Threaded dispatch with relative displacement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The style vmpx models, and the one its recovery algorithms and fixtures are built
around. Every handler ends
by *computing the address of the next handler itself*, from the current handler
address and an encrypted displacement fetched from the bytecode:

.. code-block:: nasm

    ; illustrative
    mov     eax, [rsi]          ; encrypted displacement
    add     rsi, 4
    xor     eax, ebx            ; ... decode (see "Bytecode and the Rolling Key")
    neg     eax
    xor     ebx, eax            ; update key
    movsxd  rax, eax            ; sign-extend
    add     rdi, rax            ; handler pointer += displacement
    jmp     rdi                 ; "threaded": each handler dispatches itself

There is no central dispatch loop and no table. The "next handler" is the running
sum of decoded displacements. The anchor of that sum comes from the bootstrap, and
a handler may reset it.

The terminal jump can take either of two forms, and vmpx accepts both:

.. code-block:: nasm

    jmp     rdi                 ; indirect jump through a register
    ; or
    push    rdi
    ret                         ; the same effect through the stack

Which one a given VM instance uses is a *mutation*, not a semantic difference.

Why scattered code defeats linear disassembly
---------------------------------------------

Handlers are not laid out contiguously. A handler is split into **fragments**, each
ending in an unconditional jump to the next fragment, and the fragments of different
handlers are interleaved throughout the section. Reading the file in address order
gives a meaningless mix. vmpx therefore never reads linearly: it traces *chains*
of direct-jump fragments (``trace_direct_jump_fragments``) and linearizes each handler
by control-flow reachability.

Recovering the dispatch
-----------------------

The relevant vmpx pipeline, in order:

#. **Backward slice.** Starting from the handler's terminal jump, keep only the
   instructions that define the live-out registers (VIP, key, handler pointer) or
   flags. Memory accesses are kept; calls and exception-like transfers are barriers.
   Jumps that merely connect fragments are marked structural.
#. **Candidate codecs.** Every VIP-relative load in the slice is treated
   independently as a possible *relative-dispatch codec*. Candidates that fail a
   check keep a diagnostic and are not used.
#. **Postlude proof.** A bounded search shows that every direct-flow successor of the
   handler reaches a dispatch through the handler register without clobbering it
   (defaults: 50,000 states, stack depth 256).
#. **Symbolic decode.** The codec, applied to the actual bytes at the actual VIP,
   gives the operand and the displacement; the displacement gives the next handler.
   Iterate.

The procedure never needs to know which opcode a handler "is". It needs to know how
the handler computes the next handler, and what the handler does to the VM state.
Those two questions are separated: the first by the dispatch codec, the second by the
effect extractor (:doc:`handlers`).

A subtlety: calls and flags inside handlers
-------------------------------------------

Handlers occasionally call out to native code (for APIs, for ``cpuid``/``rdtsc``) or
use native arithmetic to compute flags. A conservative slice must treat a call as a
barrier that retains all earlier definitions, because the callee may read any
register. This is one reason slices of mutated handlers stay larger than their
semantics suggest, and why vmpx's slicer prefers to over-keep than to under-keep.
