Dynamic Assist: Observe, Then Feed Back
=======================================

:doc:`static-frontier` lists what a purely static devirtualizer must refuse:
unbounded input branches, environment reads, runtime-initialized globals,
re-entrant continuations, import protection and packers whose unpacking is itself
computed at run time. A recorded run supplies exactly those facts. This page
describes the dynamic-assist engine (called SP7) that obtains them without ever
running a hostile sample on real hardware, and states plainly how far it gets.

.. note::

    The engine is implemented (crate ``vmpx-emulate`` and its bridges into the
    static model) and covers **user-mode** samples only. Kernel-mode samples
    (drivers, boot code, anything that needs a privileged device model) are out of
    scope. Everything it produces lives on a labelled **experimental** tier. None of
    it is written as the sound ``OUTPUT`` of ``clean``.

The flow: observe, then feed back
---------------------------------

.. code-block:: text

    packed or protected module
        |  detonate in the contained software emulator
        v
    observed facts: unpacked image, resolved imports, runtime globals,
                    recursion depth, VM branch targets
        |  dump the unpacked image          |  feed facts to the static side
        v                                   v
    unpacked PE  ->  vmpx static devirtualizer (seeded, labelled)  ->  experimental PE

The emulator runs the module until it reaches its original entry point (or a step
budget is spent) and the image is dumped from emulated memory. That image is the
input to the static pipeline (:doc:`/tool/pipeline/unpacking` and the stages after
it). The same run yields side facts the static model can use. Dynamic assist does
not replace static recovery: the devirtualizer still decodes, executes symbolically
and lifts; the run only supplies what the binary does not contain.

The engine
----------

``vmpx-emulate`` is an in-process software interpreter for x86-64 user-mode code.
Every instruction is decoded and applied to an in-process CPU and memory model.
There is no host pointer the guest can reach, no syscall it can issue, no file,
process or network path, and no foreign-function call.

* **Exact or stop.** An instruction executes only with fully modelled semantics.
  Flags the architecture leaves undefined are tracked, and reading one stops the run
  (a deterministic undefined-flag mode exists for detonation). A faulting instruction
  leaves state untouched. Unmodelled behaviour ends the run; it is never guessed.
* **Loader and environment.** The module is mapped at its preferred base with
  section permissions, TEB, PEB and loader-list stubs, a stack, and an import table
  of sentinel thunks. Module export tables are modelled faithfully (with an offline
  reference of system export names) so a protector that walks exports directly sees
  consistent data. Time, ``rdtsc`` and ``cpuid`` are deterministic.
* **Win32 surface.** The minimum API surface a protector startup needs is emulated
  (memory, loader and heap APIs, ntdll syscall stubs),
  and a ``GetProcAddress`` call records the binding it resolves.
* **Network closed.** A denylist covering the Winsock, WinINet, WinHTTP, DNS and
  URL-download surfaces is checked before any API is modelled, and a run that
  resolves one stops. The workspace builds with cargo forced offline.

Facts, and the bridges that use them
------------------------------------

What a run records is an ``ObservedFacts`` bundle. Every entry carries an ``Observed``
provenance tag so it cannot be mistaken for a proven fact.

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Observed fact
     - Gap it addresses
     - How the static side uses it (experimental only)
   * - Unpacked image and OEP
     - Packers and protector unpacking computed at run time
     - Detected by write-then-execute; the memory image is dumped and rebuilt into a
       loadable PE (``clean --dump-pe``, ``generic-unpack``)
   * - Resolved imports
     - Import protection: bindings exist only after the protector's loader runs
     - ``GetProcAddress`` bindings feed experimental import reconstruction
       (``clean --dynamic``)
   * - Runtime-initialized globals
     - A branch fed by a null-looking read of a pointer set during initialization
     - Concrete values seed the symbolic state's reads, labelled, to resolve the stall
   * - Recursion depth
     - A recursive function that the lift cannot expand to a finite tree
     - A concrete depth observed under one input bounds an unroll
   * - VM branch targets
     - Computed or data-dependent targets static analysis cannot enumerate
     - Observed targets are offered to the CFG walker as a bridge for those branches

Each bridge is opt-in, runs only on the experimental path, and names every observed
fact it used in the report. The sound gate is never relaxed. The emulator also
provides a differential harness that can run a lift and the original side by side
for comparison.

Packers other than VMProtect
----------------------------

The same engine backs ``generic-unpack`` (:doc:`/tool/cli/generic-unpack`).
``detect`` classifies UPX, ASPack, PECompact, MPRESS and FSG; the named-section
packers are routed to a bounded detonate, OEP, dump and rebuild path. A real UPX
stub has been run to its original entry point end to end. The other routed packers
have not been validated on real samples, and FSG, found by a low-confidence
heuristic, is never routed.

The trust tier: observed, not proven
------------------------------------

A fact from a run is evidence about one execution. It is not a proof about every
execution. vmpx therefore keeps a separate tier for it:

* **proven** facts come from static analysis and pass verification. They are the
  only facts the sound path (``clean`` output) may rely on;
* **observed** facts come from an emulated run. They may seed the static model but
  are labelled as observed wherever they are used.

Output that depends on an observed fact is quarantined the way ``--experimental``
output is: never written as the sound ``OUTPUT``, with the report naming each
observed fact used and the blockers list left honest. Where an observed value
contradicts a static result, the run wins for diagnosis and the static result is
reported as blocked, not silently patched.

The honest limit
----------------

Dynamic assist does not close the static gap. What stops it:

* **A run needs a driver.** A virtualized function called by the host application is
  reached only if something calls it with inputs. Observing host-called virtualized
  functions needs a host-API driver that supplies the caller's side, and that is not
  general.
* **One run, one path.** Observed branch targets and depths come from the inputs the
  run happened to use. Other inputs may take other branches, and an observed fact
  says nothing about them.
* **Irreducible dependence.** Some branches depend on the environment or on external
  input (time, files, the network, user input). No emulation can decide them, and
  vmpx does not invent an answer.
* **Modelling gaps.** The ISA and the Win32 surface are modelled one case at a time. A
  sample that needs an unmodelled instruction or API stops, and a sample that detects
  emulation can refuse to unpack at all. A clean-looking run is therefore not
  evidence that a sample is benign.
* **No promotion.** Observed results never become sound results. The 313 of 814
  functions recovered on the measured large module (:doc:`static-frontier`) are static
  and proven; dynamic assist is not counted in that figure.

Containment
-----------

No containment is 100%. Emulators have bugs, and the design does not rest on any one
layer being perfect. The layers, and which of them are enforced by the code:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Layer
     - Status
   * - In-process interpreter
     - Enforced by construction. No native execution, host syscall, file, process,
       network or FFI path exists in the emulator crate
   * - Minimal device model
     - Enforced. Only user-mode CPU semantics and the API surface the sample
       touches are modelled; unimplemented behaviour stops the run
   * - Network denylist and offline build
     - Enforced. Networking APIs stop a run before modelling; cargo is forced offline
   * - Bounded runs
     - Enforced. Every detonation has a step budget, and ``generic-unpack`` uses a
       fixed modest bound so a hostile stub cannot run away
   * - OS sandbox, disposable VM, air gap
     - Operational practice, not enforced by the tool. Untrusted samples should still
       be handled on a machine with no network path, with only inert output carried
       out

The aim is **zero blast radius**: assume the emulator will eventually be compromised
and arrange that doing so gains the sample nothing, with no credentials, no network,
no persistent storage and no path back to the analyst's machine. A run that cannot be
made safe on those terms is not run.
