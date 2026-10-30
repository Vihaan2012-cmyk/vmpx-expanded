Multiple VMs and Versions
=========================

Several VM instances per file
-----------------------------

A protected binary can contain **more than one VM**. Each instance has its own
bootstrap, its own handler set and its own register roles, and shares nothing with
the others at the level of code bytes. Different protected functions may be assigned
to different instances.

For an analyst this changes the problem in two ways:

* A handler classification learned from one instance **cannot be reused** on
  another. Every instance must be recovered from scratch.
* A single function may span instances: one VM exits to another VM's entry stub.
  The analysis has to follow the chain (:doc:`context-switch`).

vmpx records the structure explicitly. Entry stubs that reach the same
stack-reservation instruction share a bootstrap and are grouped into a
``VmInstance``:

.. code-block:: json

    {
      "id": 0,
      "bootstrap_rva": 4198400,
      "entry_rvas": [4202496, 4206592],
      "distinct_seeds": [2882400001, 3735928559]
    }

``vmpx vm-list`` prints the instances before the entries, and
``CleanReport.vm_instances`` carries them.

The unprotected program and the VM boundary
-------------------------------------------

A function can be protected at different granularities:

*A whole function.*
    The stub replaces the function's first bytes; its entire behaviour is virtual.

*A marked region.*
    The SDK markers bracket a block of code; the block becomes virtual and the rest
    of the function stays native.

*Several regions chained.*
    The exit of one region lands at the entry stub of another.

The common property is that **the function's interface survives**: callers see the
same address, the same calling convention and the same effect on registers and
memory. This is the invariant a devirtualizer uses to define correctness.

Versions
--------

VMProtect has evolved considerably across its major versions, and published work
distinguishes at least the 2.x and 3.x series. Without claiming a precise history:

* The **2.x** generation is generally described as using table-driven dispatch and
  simpler handlers.
* The **3.x** generation introduced the heavier mutation of handlers, the threaded
  relative dispatch with rolling-key operand encryption, and the NOR/NAND logic
  primitives. These are the features vmpx models.
* Releases within a series change details: register conventions, the flag
  materialization, the bootstrap shape, the section naming and the anti-debug
  checks.

vmpx is **not** built around a version number. Every structural fact it relies on is
established by analysing the instance in front of it, and a fact that cannot be
established is a refusal. That is the correct response to a protector whose purpose
is to vary.

Practical triage of an unknown sample
-------------------------------------

#. ``vmpx detect`` for the verdict, layout and the entropy map of executable regions.
#. If the layout is ``PackedVirtualSections``: ``vmpx unpack``.
#. ``vmpx vm-list`` on the unpacked image for the instances and entries.
#. ``vmpx oep`` to see whether the header entry is a VM entry and where it hands off.
#. ``vmpx clean --report`` for the full picture and the list of blockers.
