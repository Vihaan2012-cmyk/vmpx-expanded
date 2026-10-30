The Recovery Pipeline
=====================

The ``clean`` command chains the libraries into one pipeline. This section
describes each stage: what it takes in, what it proves, and what it refuses.

.. code-block:: text

    input PE
       |
       v
    detect ---------- verdict, layout
       |
       v
    unpack ---------- RVA-layout analysis image            (PackedVirtualSections only)
       |
       v
    VM recovery ----- entries, bootstrap, handler codecs
       |
       v
    symbolic exec --- native exit states per path
       |
       +--> OEP decision
       +--> function inventory, live-in, input domains
       |
       v
    lift + recompile  native replacement bodies
       |
       v
    reintegrate ----- append .vmpx0, redirect entry / TLS / VM stubs
       |
       v
    relayout + loader validation  --> proof gate --> OUTPUT

.. toctree::
   :maxdepth: 1

   unpacking
   vm-recovery
   symbolic-execution
   oep-recovery
   function-recovery
   lifting-recompilation
   reintegration
   imports-and-stripping
   verification

Sub-projects
------------

The pipeline was built as a sequence of sub-projects, each with its own design
spec under ``docs/superpowers/specs/``. The stages above map to them as
follows.

.. list-table::
   :header-rows: 1
   :widths: 10 40 50

   * - SP
     - Scope
     - Pipeline stages
   * - SP1
     - Native reintegration and image emission
     - :doc:`lifting-recompilation`, :doc:`reintegration`
   * - SP2
     - OEP recovery and generalized virtualized-function recovery
     - :doc:`oep-recovery`, :doc:`function-recovery`
   * - SP3
     - Application import-set reconstruction
     - :doc:`imports-and-stripping`
   * - SP4
     - Dangling ``.vmp`` reference resolution and stripping
     - :doc:`imports-and-stripping`
   * - SP5
     - Optimizing recompilation (VTIL), differential verification,
       ``--neutralize``
     - :doc:`verification`

Design rules shared by every stage
----------------------------------

Fail closed
    Ambiguity is kept in a report and yields no result, rather than a guess.
Static only
    No stage runs the target. Anti-debug never fires.
Fixture first
    Every component has synthetic fixtures as its CI oracle; local samples are
    an out-of-band behavioural check.
Equivalence preserving
    No transformation may change observable behaviour, and where one could, it
    ships with a verification check.
Opt-in behaviour changes
    ``--experimental`` (gate-relaxed emission) and ``--neutralize`` (anti-debug
    stripping) each weaken a guarantee. Both are off by default, labelled in
    output and report, and never alter the semantics of the sound ``OUTPUT``.
