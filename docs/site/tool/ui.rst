The vmpx-ui Front End
=====================

``vmpx-ui`` is an egui/eframe desktop application that drives the ``vmpx`` command
line and renders its reports. It contains no analysis logic: all analysis is done
by the CLI binary, and the UI only builds argument lists, runs the process and
displays the output.

.. code-block:: console

    $ cargo run -p vmpx-ui --release

Using it
--------

#. Choose an input binary, either by typing a path or with the native file dialog
   (``rfd``).
#. Pick how to launch the CLI: a built binary (default
   ``target/release/vmpx.exe``) or ``cargo run`` through the workspace.
#. Click an action: **Detect**, **Info**, **Verify**, **Imports**, **VM list**,
   **OEP** or **Clean**.
#. Read the result. JSON-producing actions are run with ``--json`` and shown as a
   collapsible **Summary** (structured, with colour-coded status) plus the **Raw
   JSON**. Actions without JSON output show their text output.

Options
~~~~~~~

``no_scan``
    Passes ``--no-scan`` to **VM list**.

``experimental``
    For **Clean**, passes ``--experimental <path>`` with the output path given in
    the UI. The result shows the blockers list and the experimental flag; see
    :ref:`experimental-output`.

What it shows
-------------

For **Clean**, the summary surfaces the fields of the ``CleanReport``
(:doc:`cli/reports`): the protection verdict, the OEP result, the VM-entry
inventory, the reintegration metrics (``section_rva``, ``entries_redirected``,
``dispatchers``, single-body counts) and the blockers. Each run also shows the
exit code (green for ``0``), the elapsed time and the exact command line, so a
result in the UI can be reproduced in a terminal.

Design notes
------------

* The CLI is run as a child process on a worker thread; results return over an
  ``mpsc`` channel, and the window never blocks. A worker panic is caught and
  reported rather than freezing the interface.
* Because the UI consumes the same JSON the CLI serializes, it picks up new report
  fields without touching any devirtualization code.
* There is no display in continuous integration, so the UI is build-verified only;
  it is meant to be run on a desktop.

Dependencies
------------

``eframe`` 0.29 (with the ``glow`` backend and default fonts), ``rfd`` 0.15
without default features, ``serde_json`` and ``anyhow``.
