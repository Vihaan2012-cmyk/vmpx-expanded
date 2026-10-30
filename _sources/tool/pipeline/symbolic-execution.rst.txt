Symbolic Execution
==================

*Crate:* :doc:`/tool/crates/vm`

Decoded VM instructions are executed over a bit-vector expression IR. The result
of executing a complete path is a *native exit state*: an exact symbolic value
for every native register, the flags and the return address at the moment the
VM hands control back to native code.

*Modules:* ``semantic_exec.rs``, ``expr.rs``, ``symbolic_domain.rs``,
``profile_explore.rs``, ``profile_paths.rs``, ``path_graph.rs``,
``image_concretize.rs``.

The expression IR
-----------------

``Expr`` is a bit-vector expression with widths of 8, 16, 32 and 64 bits and an
exact byte-addressed memory model for the VM stack and external stores.
Operations are grouped as ``BinaryOp``, ``UnaryOp`` and ``ResizeOp``. Defaults
for one linear block are 4,096 instructions, 1 MiB of tracked memory and
200,000 expression nodes. Errors name the instruction and VIP, and the executor
"never guesses a missing operation".

A store to an address that could alias tracked memory discards precise memory
(*havoc*). Unknown native calls also havoc tracked memory.

Inputs are symbols named after the entry value of a register, for example
``native_rdx_initial``. The stack cursor is ``vsp_initial``.

Chaining blocks: the profile explorer
-------------------------------------

The profile explorer chains blocks. It follows same-profile virtual branches and
native profile changes while carrying the full symbolic state, and "never guesses
constraints or merges decoder states". Its stop reasons are:

* ``SegmentLimit``, ``TotalInstructionLimit``
* ``DecoderBoundary``, ``ControlFlowBoundary``
* ``UnsupportedTransition``, ``SymbolicTransitionTarget``
* ``AmbiguousProfileRoles``, ``SymbolicExecutionFailure``
* ``VmExit``, the successful outcome

Resolving branches with finite input domains
--------------------------------------------

Branches are resolved by *finite input domains*: the caller supplies explicit
constant sets for symbolic inputs, and every combination is run as an
independent path (``explore_vm_profile_paths``). Missing domains leave an
incomplete result.

For a DLL entry, the pipeline uses one input, ``native_rdx_initial``, over 0 to 3
(the four ``DllMain`` reason codes; see :doc:`/tool/dll-support`). An
executable entry has a single invocation path, with no input enumerated.

For other functions the domain is *inferred*, and proven exhaustive, by
:doc:`function-recovery`.

The path graph assembler merges two nodes only when decoder state, profile
roles, direction, the complete symbolic entry state and block length are all
equal.

Image-backed loads
------------------

Loads from the immutable image can be concretized. ``recover_modulo_image_targets``
enumerates every residue of a 32-bit unsigned-division remainder to find every
image-backed target it selects. Callers must separately prove the referenced
bytes are immutable.

Native transitions, native calls and VM exit
--------------------------------------------

*Modules:* ``native_call.rs``, ``native_call_verify.rs``, ``call_effects.rs``,
``native_api.rs``, ``vm_exit.rs``, ``native_exit_state.rs``,
``native_vm_chain.rs``, ``native_continuation.rs``.

Native-call bridge
~~~~~~~~~~~~~~~~~~

A handler returns to native code to call an API, then resumes the VM. The
recognizer proves the bridge shape by control-flow dominance and
register/memory effects: load the call target into a register, save the native
``RSP``, call, store the return value, restore the VSP, restore the saved VM
registers, and complete a *selector* that picks one of several exhaustively
proven outcomes before the threaded dispatch. A decoded selector with no proven
outcome is an error.

Call summaries
~~~~~~~~~~~~~~

A call target is matched against parsed imports (lookup values, IAT RVAs and
preferred-base IAT VAs are all considered; duplicates are never collapsed). An
unknown target has an ``Opaque`` memory effect, forcing a havoc. The trusted
registry is deliberately narrow: ``LocalAlloc`` and ``GlobalAlloc`` (only with
proven fixed flags, since ``MOVEABLE`` would return a handle), ``HeapAlloc`` and
``CoTaskMemAlloc``. Every dynamic call gets a unique *epoch* id, so two
identical calls are never assumed to return the same value.

VM exit
~~~~~~~

The epilogue is proven by requiring a VSP-to-native-RSP switch, restoration of
every non-RSP general-purpose register from the stack, a final ``POPFQ``, no
non-stack memory access and a native ``RET``. Junk produced by register
mutation is tolerated only if the final definition of each register is its
corresponding ``POP``.

Native exit state
~~~~~~~~~~~~~~~~~

The exact register, flags and return-address values are read from the terminal
symbolic memory snapshot. A restore-frame byte missing from the snapshot is an
error.

VM-to-VM chaining
~~~~~~~~~~~~~~~~~

If the exit ``RET`` targets another VM entry stub, exploration continues by
composing the next entry's frame, call continuation, flags, registers and
saved-stack cursor from the exact prior exit state.

Native continuation
~~~~~~~~~~~~~~~~~~~

After an exit that returns into ordinary native code, a bounded
concrete/affine interpreter (defaults: 100,000 instructions, 64 KiB stack,
4 KiB C strings) follows it with a symbolic stack origin, records calls
(resolving imports) and C-string arguments, and fails closed on unmodeled
writes or unknown branches.

Timestamp counter
~~~~~~~~~~~~~~~~~

``ReadTimestampCounter`` is a recognized handler (an ``rdtsc`` pushed as a split
dword). It is represented as an opaque ``read_timestamp_counter_*`` intrinsic.
The code does not assert its purpose; the :doc:`verification` stage uses it as
evidence for a timing check.

Exception flow
--------------

*Module:* ``exception_flow.rs``.

For a handler classified as a VM memory load, the external (non-VM-stack,
non-context) read sites are collected as *fault sites*. Each is matched against
the image's x64 exception table: a covering ``RUNTIME_FUNCTION`` whose
``UNWIND_INFO`` names a language-specific exception handler.

Each such language handler is analysed as a *stack sled builder*. The
recognizer requires it to load ``CONTEXT.Rsp`` (offset ``0x98``) through the
context argument in ``R8``, then copy a continuation seed and a validated
``ret`` gadget through a cursor to build a bounded virtual-unwind stack.
Classification is by register/memory effects and control flow, not by bytes or
addresses.

Three continuation sources are modelled: a non-zero seed, cursor memory, and a
zero-seed unwind in which the stack is filled with ``RET`` gadgets and control
unwinds through the original native frame to the caller's saved return address.
There is no dispatch address on that path.

The mechanism means a faulting read inside a handler can be an *intended*
control-flow path: execution continues through an exception handler rather than
falling through. The code reports whether all external reads route through
language handlers and whether continuations are recovered, and refuses to claim a
continuation without an exact pre-fault state. See
:doc:`/vmprotect/exceptions`.

Checksum and integrity handlers
-------------------------------

*Modules:* ``checksum.rs``, ``checksum_handler.rs``.

The VM has a ``TableChecksum`` handler, described in the source as a "recovered
byte-at-a-time table checksum over program memory". The contract
(``TableChecksumMetadata``) is:

* a 32-bit accumulator seeded from a constant or from the VM stack, XORed with
  ``initial_xor``;
* a pointer and length taken from VM-stack offsets, walked with a signed
  ``pointer_stride``;
* per byte, a table index ``(accumulator & 0xFF) XOR input_byte`` into a
  256-entry 32-bit table at a fixed virtual address;
* the accumulator is shifted (logical left or right by ``shift_bits``) and XORed
  with the table entry and an optional constant;
* after the loop an optional ``NOT``, then ``final_xor``;
* the result lands at a recovered VM-stack offset after a ``vsp_delta``.

Only that exact recurrence is supported: ``validate()`` demands a 32-bit
accumulator, a 256-entry table of 32-bit entries, a shift below 32 and a
non-zero stride. The shape matches a CRC-32 style computation.

*Recognition.* The analyzer finds a counted 32-bit byte-table loop in the native
handler and proves that both the zero-length path and the loop-exit path converge
on the same ordinary threaded-dispatch tail. Constants and register choices come
from data flow; "no handler addresses, table contents, or mutation bytes are
signatured".

*Evaluation.* With a concrete pointer, length and table, the symbolic executor
computes the checksum exactly by reading the analysed image's bytes (bounded by
the memory limit). Otherwise it emits an opaque intrinsic,
``table_checksum_low_byte_xor_recurrence_w32``, carrying the accumulator, pointer,
length, table, shift, stride and a memory-snapshot epoch.

*Lifting.* ``TableChecksum`` is a required lowering capability that is not in the
implemented set, so a routine containing it is not reported codegen-ready.

.. note::

    A checksum over program memory depends on the bytes it reads. Because the
    analysis evaluates it against the analysed image, a rebuilt image whose bytes
    differ would not reproduce the original checksum value if the checked range
    changed. The code keeps checksum results symbolic or image-derived rather than
    assuming they are constants for the output. See :doc:`/vmprotect/integrity`.
