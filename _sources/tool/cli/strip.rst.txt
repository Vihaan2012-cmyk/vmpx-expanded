strip
=====

.. program:: vmpx strip

Removes named sections from a PE, writing the result only when the sound
dangling-reference checker approves and the rewritten image reparses and validates
for the Windows loader.

Synopsis
--------

.. code-block:: console

    vmpx strip <INPUT> <OUTPUT> --remove <SECTION> [--remove <SECTION> ...] [--drop-directory <INDEX> ...]

Options
-------

.. option:: INPUT

   The PE to rewrite.

.. option:: OUTPUT

   Where the rewritten PE is written, only on success. The command refuses to
   overwrite the input, directly or through an alias, and refuses an output that
   already exists.

.. option:: --remove SECTION

   A section name to remove. Repeat for several; every section with that name is
   removed. At least one is required, and the command fails with "nothing selected"
   otherwise.

.. option:: --drop-directory INDEX

   A data-directory index to drop. Repeat for several. A directory may be dropped
   only if it lies wholly inside a removed section.

Behaviour
---------

If any reference from the surviving image into a removed section would be left
dangling, the command refuses, prints ``[strip] refused: <reason>`` and writes
nothing. On success it prints a summary and confirms that the output was reparsed
and validated from disk. This is the same check ``clean --experimental --strip``
applies to the reintegrated image (:doc:`clean`).

Stripping is a removal primitive, not a recovery step. Removing the protector's
sections from a binary whose virtualized functions have not all been replaced leaves
the remaining ones with nothing to run on, which is exactly what the reference check
is there to catch. See :doc:`/tool/pipeline/imports-and-stripping`.
