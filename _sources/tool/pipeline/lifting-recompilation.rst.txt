Lifting and Recompilation
=========================

*Crates:* :doc:`/tool/crates/vmlift`, :doc:`/tool/crates/recompile`

Once a path has been executed symbolically to a native exit, its meaning is a set
of expressions. Lifting turns that into a compact semantic routine; recompilation
turns the routine into x86-64 machine code.

Lifting
-------

``lift_profile_chain`` converts a successfully executed profile chain into a
``LiftedSemanticRoutine``. In the words of the source, "VM bytecode, handler
addresses, rolling keys, and dispatch mechanics are absent; only expressions,
ordered external effects, context outputs, and control targets remain".

* The effects are ``MemoryWrite``, ``MemoryHavoc`` and ``NativeCall``.
* Each routine is marked *path-specific*. There is no control-flow-graph merge
  yet: a function with several paths is several routines.
* ``lift_symbolic_block`` and ``lift_profile_path_family`` cover single blocks and
  whole input-domain families.
* ``collect_path_effects`` gathers the stores and imported calls of a path so
  they can be replayed.

Readiness gate
~~~~~~~~~~~~~~

``audit_instructions`` and ``audit_profile_chain`` compute the *lowering
capabilities* a routine needs (from ``WidthAwareIntegerBackend`` to
``NativeExitBridge``), compare them with the implemented set, and report
``ready_for_codegen`` only when nothing is missing. The implemented capabilities
are the width-aware integer, unary, shift, resize, memory and branch backends,
plus stack SSA lowering. Unknown semantics are reported as issues, and a routine
containing ``TableChecksum`` is not codegen-ready because that capability is
required but not implemented.

``audit_entry_effect_liveness`` is the effect-liveness audit. For each path it
reports how many VM-stack writes, dead-allocation writes and dead-allocation calls
can be removed because nothing can observe them, and how many observable writes,
observable calls and memory havocs must be preserved (``PathEffectLivenessReport``).

The replacement function
------------------------

A recovered entry path ends with an exact symbolic value for every native
register. ``build_replacement`` lowers each value into the project IR:

#. Each value is computed through scratch registers into **its own spill slot**.
#. A single epilogue loads every slot into its destination register. This is safe
   because each load reads a slot and writes a register, so the order of loads
   never matters.
#. Observable effects (stores and imported calls) are replayed *before* the
   epilogue.
#. Flag state is restored via ``RestoreFlags``, because a caller may branch on
   flags the routine set.
#. When the exit continues into the protector's own native continuation, the
   function ends with a tail ``jmp`` to it after restoring registers.

Imports in replacements
~~~~~~~~~~~~~~~~~~~~~~~

Fresh-allocation intrinsics are lowered to a real imported call through the IAT
slot; any intrinsic that cannot be resolved this way is refused. The
``vm_flags_<op>_wN`` intrinsics are rewritten to plain arithmetic.

Constant dispatch
~~~~~~~~~~~~~~~~~

``compile_constant_dispatch`` (configured by ``ConstantDispatchConfig``: input
register ``rdx`` and scratch register ``r10`` by default) compiles the simple case
in which each input value maps to a constant result. More generally, one VM entry
can behave differently per input value, so each finite-domain path gets its own
compiled body and ``emit_domain_dispatcher`` selects among them (see
:doc:`reintegration`).

The back end
------------

``vmpx-recompile`` is a small width-aware integer IR: add, sub, mul, xor, and,
or; not, neg; shifts and rotates; resize; load and store; spill; Win64 call;
compare; branch; return; tail jump; flags restore. It validates ABI facts before
encoding with ``iced-x86`` in a two-pass branch resolve:

* register assignment, with one virtual register per physical register;
* Win64 stack alignment and shadow space.

The back end's principle is that "unsupported operations and unsafe ABI states
are rejected instead of approximated". The set of operations is small because
each is supported only when its lowering can be shown correct.

The domain dispatcher
~~~~~~~~~~~~~~~~~~~~~

``emit_domain_dispatcher(domain, arms, fallback, dispatcher_va)`` hand-encodes

.. code-block:: nasm

    cmp  reg, imm_0
    je   body_0
    cmp  reg, imm_1
    je   body_1
    ...
    jmp  fallback

so that no frame setup clobbers incoming registers. It is a naked trampoline:
``compile()`` emits a Win64 frame, which a dispatcher must not. The ``je``/``jmp``
targets are rel32 from the dispatcher's own location, which is why dispatchers are
compiled at their final addresses. It is unit-tested by re-decoding the emitted
bytes and asserting each target resolves to the intended absolute address.

Where VTIL fits
---------------

The optional VTIL bridge (:doc:`/tool/crates/vtil-sys`) is the future home of an
optimizing pass between lifting and recompilation. The ``vtil`` stage in
``CleanReport`` is currently a stub, and the native library is linked only with
the ``vtil`` Cargo feature. Any optimization pass must pass the differential
verification described in :doc:`verification` before it can be on by default.
