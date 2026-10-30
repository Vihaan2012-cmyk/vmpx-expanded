Devirtualization
================

*Devirtualization* is the recovery of native code from a virtualized function. This
section explains the problem, the strategy vmpx uses, and how each VMProtect
mechanism described in the previous pages is attacked.

The problem, precisely
----------------------

Given a protected binary ``P``, produce a binary ``P'`` such that for every input,
``P'`` has the same observable behaviour as ``P``, and ``P'`` does not contain the
protector's machinery. *Observable behaviour* means the register state,
memory writes and external calls at the function boundary.

What the problem is **not**: reconstructing the original source, or the original
compiler's bytes. Virtualization destroys that information. The information that
survives is the *semantics* of each function, and that is what a devirtualizer
recovers (see the note on the front page).

Three strategies
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Strategy
     - How
     - Trade-off
   * - Dynamic tracing
     - Run the program, record the executed native or VM instructions, and simplify
       the trace
     - Sees only the paths that ran; must defeat every anti-debug and anti-VM check
       to run at all
   * - Pattern matching
     - Recognise the handlers of a known VM version and translate bytecode with a
       fixed table
     - Brittle: every build randomises the handlers; breaks with each release
   * - Static semantic recovery
     - Decode the bytecode by simulating the VM's own decoder, execute it
       symbolically and recover what each path does
     - Needs a precise model of the VM; sound but incomplete: refuses when a fact is
       unproven

vmpx implements the third. Each choice that follows is a consequence.

The static semantic pipeline
----------------------------

.. code-block:: text

    packed file
        |  attacking-packing           PACKER_INFO + LZMA   -> analysis image
        v
    analysis image
        |  attacking-the-vm            entries, roles, codecs, handlers
        v
    decoded VM paths
        |  (symbolic execution)        bytecode -> expressions over the inputs
        v
    native exit states
        |  OEP / functions / lift      semantics -> native replacement code
        v
    replacement image
        |  imports, stripping          conventional PE, no protector sections
        v
    clean binary

The properties that make this tractable are the same properties the previous pages
identified as VMProtect's design constraints:

* the VM is **deterministic**: given a seed and the bytecode, decoding is a pure
  function, so it can be simulated exactly;
* the exit **pops every register from memory the VM controls**, so the native exit
  state is closed-form in the inputs;
* the **interface of each function survives**, so correctness has a precise
  definition;
* the machinery is **entirely static** to anyone willing to model it.

.. toctree::
   :maxdepth: 1

   attacking-packing
   attacking-the-vm
   attacking-mutation
   attacking-anti-analysis
   attacking-imports
   limits
   static-frontier
   tooling-map

How to read a result
--------------------

A devirtualization result is only as good as the proof behind it. The project's stated
aim is an honest per-function status in one of three states:

* **recovered**: decoded to a native exit state, lifted, and verified to be
  equivalent to the VM routine over the recovered input domain;
* **partial**: decoded, but some fact is unproven (a domain that is not proven
  exhaustive, an effect the lifter cannot represent);
* **blocked**: not recovered, with the reason in the report.

In the ``CleanReport`` this is read from ``vm_inventory`` against
``input_closed_functions``, ``virtualized_routines[*].issues`` and ``blockers``
(:doc:`/tool/cli/reports`). Sound output requires that nothing be blocked.

See :doc:`/tool/soundness`.
