Generalized Function Recovery
=============================

*Crates:* :doc:`/tool/crates/vm`, :doc:`/tool/crates/vmlift`, :doc:`/tool/crates/cli`

SP1 handled the module entry and TLS callbacks. A real protected program also
has virtualized routines scattered through its code: exported functions, helpers
and callbacks, each entered through its own ``push seed; call bootstrap`` stub.
Generalized recovery discovers every such routine and decides, one by one,
whether it can be recovered soundly.

The pipeline has four steps.

1. The VM-entry inventory
-------------------------

``build_vm_inventory`` merges observations of virtualized routines from every
discovery source:

* the executable-stub scan (:doc:`/tool/cli/vm-list`);
* the export-dependency walk, which follows every branch and direct call
  reachable from each non-forwarded export and records the distinct addresses
  that present an entry stub. Counting edges instead would report the same
  routine many times and overstate the work.

``merge_vm_inventory`` combines observations into one ``VmInventoryEntry`` per
RVA: the highest confidence wins, source labels are unioned and the
``reachable_from_export`` flag is OR-combined. The output is sorted by RVA and
deterministic. The inventory lands in ``CleanReport.vm_inventory``.

2. Live-in and calling-convention inference
-------------------------------------------

*Module:* ``live_in.rs`` in :doc:`/tool/crates/vmlift`.

A replacement needs to know which architectural registers (and stack slots) are
inputs. ``infer_live_in`` is the deliberately small core: given a routine's
register accesses in execution order, a register is live-in if its **first**
access is a read. A read-modify-write is a read event followed by a write event.
The result is sorted and deterministic.

``calling_convention`` adapts a recovered entry path family to that core. Path
effects are expressed over ``native_<reg>_initial`` symbols, so every such
symbol an effect expression mentions is a read of that register at entry. Writes
need no events, since they are already folded into the expressions. The result
is the union over all paths and also reports the stack cursor.

.. warning::

    The inferred set under-reports. It is for diagnostics and reporting only.
    Emission binds *all* architectural registers, and never drops an input on the
    strength of this set.

3. Input-closed functions
-------------------------

*Modules:* ``input_closed.rs`` in :doc:`/tool/crates/vm` and
``input_closed_functions.rs`` in :doc:`/tool/crates/cli`.

``is_input_closed`` is a conservative predicate: is the native exit a pure
function of the inputs a replacement will actually provide? An expression is
closed when it depends only on constants, registers named in the *allowed* set,
and pure intrinsics over such terms. Memory loads and opaque intrinsics (memory
epochs and snapshots, native-call results, fresh allocations, the timestamp
counter, unknown names) make an expression not closed. Unknown shapes are never
closed.

The exit is closed only if **every** value it carries is closed: each register,
``rflags`` and the return address. The allowed set is the full set the
replacement binds: every architectural register, ``vsp_initial`` and the cursor
register's rewritten name. It is never a subset inferred from observed use,
because "closed" must mean "depends on nothing the replacement does not
provide".

``recover_input_closed`` accepts an inventory entry only when:

* its confidence is at least 60;
* its input domain is empty;
* it has exactly one complete path;
* it has no native calls or unsupported effects, and no havocs or tainted memory
  in any retained symbolic state;
* the exit, its stores and its memory are all closed.

Collected effects (stores and calls) are carried with the exit so the replacement
replays them. An exit alone would silently drop them.

The executable entry
~~~~~~~~~~~~~~~~~~~~

A non-DLL executable's entry point (CRT start) is invoked once, not with a
``DllMain``-style reason, so its invocation domain is a single path. The same
empty-domain input-closed recovery applies (``decide_exe_entry``). On failure
the blocker reads "executable entry is not recoverable as an input-closed
single path"; on a successful recovery there is no blocker.

4. Branching functions and proven-exhaustive domains
----------------------------------------------------

*Module:* ``input_domain.rs`` in :doc:`/tool/crates/vm`.

When the first virtual branch depends on live-in registers, the routine needs a
finite set of input values so every branch can be enumerated. For the general
case this is undecidable, so vmpx implements a **decidable subset** and refuses
everything else.

A branch target is an expression over live-in registers. A finite set of register
values is a *proven exhaustive domain* only if the target's image over that set
equals its image over every possible value. That is established structurally,
never by sampling. Each input register must be observed only through:

* a low slice of at most ``max_exhaustive_bits`` bits (default 8), covered by
  enumerating all of its values (``DomainCertificate::Exhaustive``); or
* ``vm_flags_add_wN(±reg + c1, c2)`` status-flag words of one common width,
  which depend on the register only through the cell it falls in between a
  fixed set of cut points and its residue modulo 256
  (``DomainCertificate::AddFlagsCover``). This covers ``cmp``/``jcc`` and small
  switch ladders.

For widths above the exhaustive bound, the flags of ``y + k`` (with ``y = ±x +
off``) are determined by which cell of the cut set ``x`` lies in (zero, sign and
the carry/overflow thresholds, each with its neighbour so a reflected ``y`` is
also split) and by ``x mod 256`` (parity and auxiliary carry). The first
``min(cell size, 256)`` values of every cell therefore realize every attainable
flag tuple of every observation jointly. One value per distinct tuple is kept.

16-bit covers are verified against the exact 2\ :sup:`16` image. 32- and 64-bit
covers are proven exhaustive by derivation; the tests only sample them, which is
a process gap but not a soundness one.

``InputDomainLimits`` bounds the result: ``max_values_per_variable`` defaults to
4,096.

Refusals
~~~~~~~~

``infer_branch_input_domains`` returns an ``InputDomainRefusal`` and the entry
stays a blocker for:

* memory-dependent targets (``MemoryDependent``);
* jump tables, loops and nested input branches;
* direct arithmetic on an input, mixed widths, and ``nor``/``nand`` use;
* opaque or logical-flag intrinsics, and flags of two input-dependent operands;
* any cover larger than the limit (``TooManyValues``).

``recover_branching`` then recovers a routine only if the first branch's inputs
have a proven-exhaustive domain *and* every resulting path is closed. A recovered
branching function is reintegrated by redirecting its VM stub to a *value dispatcher*
that computes the selector and jumps to the matching path body
(:doc:`reintegration`).


.. _coverage-gate:

Coverage gate
-------------

*Modules:* ``coverage_gate.rs`` and ``clean_wiring.rs`` in :doc:`/tool/crates/cli`.

``decide_coverage`` is the strict gate that decides whether the OEP blocker and
the "VMProtect sections remain referenced" blocker may clear, and it is wired into
``clean`` (``analyze_clean_with_image``) through ``clean_wiring``. Every doubt keeps
its blocker:

* the OEP was not proven;
* an inventory entry is ``Resolved`` but was not recovered;
* an inventory entry is ``Unresolved``: it is treated as uncovered, so the gate
  can never clear by an entry vanishing;
* entries were dropped before reaching the inventory;
* dangling ``.vmp`` references remain (see :doc:`imports-and-stripping`).

What counts as *covered*
~~~~~~~~~~~~~~~~~~~~~~~~

The ``covered`` set the gate receives contains only VM-entry stubs that were
actually redirected to an emitted replacement:

* a single-body or branching function's stub counts only when ``redirect_vm_stub``
  succeeded (a function that was compiled but whose redirect failed is excluded);
* a header-entry or TLS group counts only when its redirect landed **and** its path
  family was proven domain-complete, because otherwise the dispatcher falls back to
  the original VM stub for unmatched inputs;
* nested continuations are never counted.

The gate's verdict is reported in ``CleanReport.coverage`` (:doc:`/tool/cli/reports`),
and ``coverage_blockers`` turns it into blockers: the OEP blocker stays unless the OEP
was proven; the reintegration blocker stays unless an image was produced; the
``.vmp`` blocker stays unless the gate clears *and* a reintegrated image exists *and*
the dangling-reference checker found no unresolved reference.

.. note::

    The whole clearing path is covered by unit tests of each helper, but, with no
    real VMProtect corpus in CI, it is unexercised end-to-end: no protected sample has
    been shown to drive every blocker to empty.
