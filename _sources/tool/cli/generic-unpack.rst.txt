generic-unpack
==============

.. program:: vmpx generic-unpack

Recognizes a common non-VMProtect packer and, for a supported one, attempts a
bounded, contained unpack to the original entry point (OEP), writing the rebuilt
PE. It is the command-line face of the packer front-end: the classification is the
same one ``detect`` reports (:doc:`detect`), and the unpack runs in the contained
emulator described in :doc:`/vmprotect/devirtualization/dynamic-assist`.

The sample is never executed natively. The run is a pure in-process software
interpretation with no host syscall, filesystem, FFI or network path.

Synopsis
--------

.. code-block:: console

    vmpx generic-unpack <INPUT> <OUTPUT>

Options
-------

.. option:: INPUT

   The PE to classify and unpack.

.. option:: OUTPUT

   Where the rebuilt PE is written, only on success. The command refuses to overwrite
   the input, directly or through an alias, and refuses an output that already exists.
   The written file is reparsed from disk.

What it does
------------

#. Parses the input and classifies the packer (UPX, ASPack, PECompact, MPRESS or
   FSG) from structural evidence, printing the classification and its confidence.
#. Routes the named-section packers (UPX, ASPack, PECompact, MPRESS) to a bounded
   detonation. These stubs self-unpack at run time through a write-then-execute
   handoff, which the emulator's OEP detector catches.
#. On reaching the OEP, dumps the emulated image and rebuilds a loadable PE from it.

The run is capped at a fixed step budget (2,000,000 instructions) so that a hostile
or malformed stub cannot run away.

Outcomes
--------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Outcome
     - Behaviour
   * - No packer detected
     - Prints that there is nothing to route. Exit ``0``, nothing written
   * - Detected, not routed
     - FSG is found by a low-confidence heuristic and is not trusted to drive a run.
       The command reports it as experimental and unsupported and refuses to
       fabricate output. Exit ``0``, nothing written
   * - OEP reached, PE rebuilt
     - Writes ``OUTPUT``, prints the OEP and step count, and reparses the file
   * - OEP not reached within the bound
     - Fails closed with an error. Nothing written
   * - Detonation could not start, or rebuild failed
     - Error naming the reason. Nothing written

Text output
-----------

.. code-block:: text

    [packer] detected UPX (confidence 90)
      <evidence statement>
    [packer] detected UPX; reached OEP 0x... in N steps (bound 2000000)
    [packer] wrote N bytes to OUTPUT - OBSERVED artifact from one contained run; UNVALIDATED on real samples offline
    [verify] rebuilt output reparsed successfully from disk

The ``[packer]`` lines and their wording are printed by the command itself, including
the trailing provenance label on the written file. There is no ``--json`` form.

What the result is, and is not
------------------------------

The output is an **observed artifact**: the memory state of one contained run,
rebuilt into a PE. It is not sound static output and is not passed through the proof
gate. Its limits are stated plainly:

* Real UPX has been validated end to end in the emulator (the stub runs to its OEP
  and the dump rebuilds). ASPack, PECompact and MPRESS use the same path but have not
  been validated on real samples, so a failure to reach the OEP, or a stop on an
  unmodelled instruction, is an expected outcome rather than a bug to hide.
* A packer that checks for an emulated environment, or needs an unmodelled instruction
  or API, ends the run. Nothing is guessed.
* Imports in the rebuilt image reflect what the run resolved. They are not
  reconstructed or proven.

This is separate from ``unpack``, which statically decompresses VMProtect's own
packer without running anything (:doc:`unpack`). The unpacked or rebuilt image can be
fed back to ``detect``, ``vm-list`` and ``clean``.
