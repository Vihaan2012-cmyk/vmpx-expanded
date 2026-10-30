Dynamic Assist: Emulation First
===============================

:doc:`static-frontier` lists what a purely static devirtualizer must refuse:
unbounded input branches, environment reads, re-entrant continuations and packers
whose unpacking is itself computed at run time. A recorded run supplies exactly
those facts. This page describes the design (called SP7) for obtaining them
without ever running a hostile sample on real hardware.

.. note::

    This is a design. Nothing on this page is implemented. It covers **user-mode**
    malware only. Kernel-mode samples (drivers, boot code, anything that needs a
    privileged device model) are out of the current scope.

The flow: unpack and observe
----------------------------

.. code-block:: text

    packed sample
        |  detonate in a contained software emulator
        v
    observed execution: unpacked image, memory writes, API calls, branch inputs
        |  dump the unpacked image
        v
    unpacked PE  ->  vmpx static devirtualizer  ->  hybrid or clean PE

The emulator runs the sample until it has unpacked itself (or a step and time
budget is spent) and the image is dumped from emulated memory. That image is the
input to the existing static pipeline (:doc:`/tool/pipeline/unpacking` and the
stages after it). The run also yields side facts the static model can use: the
concrete values of environment reads, the real continuation of a protector exit,
and inputs that exercise a branch that static analysis could not enumerate
(:doc:`static-frontier`).

Dynamic assist does not replace static recovery. The devirtualizer still decodes,
executes symbolically and lifts; the run only supplies what the binary does not
contain.

Why emulation first
-------------------

A sample run on real hardware, or in a conventional virtual machine, has an escape
surface: the CPU, the firmware, the hypervisor and every emulated device are code
the sample can attack. A software emulator that models only user-mode instruction
execution and a thin operating-system API layer has no real hardware behind it.
There is no device to exploit and no driver to load, so the surface reduces to the
emulator's own implementation, which is small, memory-safe if written in Rust, and
fully under the analyst's control.

Emulation also serves the analysis directly. Every memory access, branch and API
call passes through the emulator, so observation is complete and does not depend on
hooks the sample can detect or remove. Time, randomness and the environment are
whatever the emulator says they are, which makes a run repeatable.

Containment layers
------------------

No single layer is trusted. Each assumes the one inside it has failed.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Layer
     - What it does
   * - Minimal device model
     - The emulator implements user-mode CPU semantics and the API surface the
       sample touches, and nothing else: no disk or network devices, no
       privileged instructions, no interrupt controller. Unimplemented behaviour
       stops the run instead of being guessed.
   * - Process sandbox
     - The emulator itself runs under an OS sandbox with no ambient authority
       (seccomp-style syscall filtering and a mandatory access control policy
       where an operating system supports them). A compromised emulator can still
       do almost nothing.
   * - Virtual machine, when used
     - If a full VM is needed for compatibility, it gets seccomp and mandatory
       access control on the hypervisor process, no shared folders or clipboard,
       and no device it does not need.
   * - Fake network
     - Network calls are answered by a simulated network that records requests and
       replies with canned data. Nothing leaves the host.
   * - Disposable
     - Every run starts from a clean snapshot and is destroyed afterwards. Nothing
       the sample writes persists, and runs share no state.
   * - Air gap for the worst samples
     - Samples judged most dangerous run on a machine with no network path to
       anything, and only the unpacked image and the run log are carried out,
       as inert data.

The trust tier: observed, not proven
------------------------------------

A fact from a run is evidence about one execution. It is not a proof about every
execution. vmpx therefore keeps a separate tier for it:

* **proven** facts come from static analysis and pass verification. They are the
  only facts the sound path (``clean`` output) may rely on;
* **observed** facts come from an emulated run. They may seed the static model (a
  domain for an otherwise unbounded input, a resolved nested entry, the value of an
  environment read) but are labelled as observed wherever they are used.

Output that depends on an observed fact is quarantined in the same way
``--experimental`` output is today: it is never written as the sound ``OUTPUT``, the
report names each observed fact it used, and the blockers list stays honest. Where
an observed value later contradicts a static result, the run wins for diagnosis and
the static result is reported as blocked, not silently patched.

The honest limit
----------------

No containment is 100%. Emulators have bugs, sandboxes have escapes, and a sample
that detects emulation can refuse to unpack at all, so a clean-looking run is not
evidence of a benign sample. The design does not rest on any layer being perfect.
It aims at **zero blast radius**: assume the emulator will eventually be
compromised and arrange that doing so gains the sample nothing: no credentials, no
network, no persistent storage, no second target and no path back to the analyst's
machine. A run that cannot be made safe on those terms is not run.
