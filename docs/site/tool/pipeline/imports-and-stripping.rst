Imports and Stripping
=====================

*Crates:* :doc:`/tool/crates/rebuild`, :doc:`/tool/crates/disasm`

Two blockers remain even when every virtualized routine has a native replacement:
the import table still describes what the *protector* needed, and the protector
sections are still in the file. SP3 and SP4 address them. Both are implemented as
library components with their own tests; wiring them into ``clean`` so that the
blockers can clear is future work (see :doc:`/project/roadmap`).

Application import-set reconstruction (SP3)
-------------------------------------------

*Module:* ``import_reconstruct.rs``

A protected binary's import table is a protector-era view: it includes whatever
the packer stub, the VM and the integrity checks needed, mixed with what the
application itself imports. Reconstruction reduces it to the slots that have a
*proven direct native reference*, then builds a conventional import directory.

Soundness rule
~~~~~~~~~~~~~~

``reduce_to_proven_imports`` takes the slot bindings and the set of referenced
slots. A slot becomes an application import only when it is referenced **and**
carries exactly one unambiguous identity. Every other referenced slot is reported
as unresolved, with a reason:

``NoBinding``
    The referenced slot has no binding at all.
``AmbiguousSymbol``
    A binding exists but its symbol identity is ambiguous (for example both or
    neither of name and ordinal present).
``ConflictingBindings``
    The same slot was bound to differing identities.
``InvalidIdentity``
    A zero slot RVA, or an empty module or symbol name.

Bound slots with no proven reference are counted as ``unreferenced_slots`` and
dropped. The evidence for "referenced" comes from the :doc:`/tool/cli/imports`
scan, which is itself conservative: absence of a reference does not prove an
import unused, which is why *dropping* an import is gated on the whole-binary
coverage proof and not on this reduction alone.

Building the table
~~~~~~~~~~~~~~~~~~

``plan_proven_imports`` serializes a proven set into a conventional import
directory plan, via ``plan_import_reconstruction`` and ``serialize_import_table``:

* a null-terminated descriptor array;
* separate import-name tables, one contiguous IAT, aligned hint/name records and
  the DLL name strings;
* inputs are semantic specifications. No original layout RVAs or raw thunk values
  are trusted, and names are restricted to non-empty ASCII;
* each resolved source IAT slot maps to **exactly one** rebuilt slot
  (``ImportSlotRemap``), so a later stage can retarget references without name
  matching;
* duplicate source slots and symbols without a single name or ordinal identity are
  rejected, and an empty set yields ``EmptyTable``.

A missing hint is serialized as zero, since loader resolution is by name.
``install_import_directories`` sets optional-header directories 1 (import) and 12
(IAT) and checks that the reparsed import list equals the serialized one.

Dangling ``.vmp`` references (SP4)
----------------------------------

*Module:* ``dangling.rs`` in :doc:`/tool/crates/disasm`

Stripping the protector sections is only sound if nothing live still points into
them. ``collect_dangling_refs`` finds every remaining reference into a ``.vmp*``
section that is not covered by a native replacement. An empty result is the
precondition for stripping.

Targets are classified (``classify_target``):

``Outside``
    Not in a protector section.
``Covered``
    In a protector section but exactly the entry RVA of a routine that now has a
    native replacement.
``Dangling``
    In a protector section and not covered.

References are tagged with the construct that holds them (``DanglingKind``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Kind
     - Where the reference lives
   * - ``EntryPoint``
     - ``AddressOfEntryPoint``
   * - ``TlsCallback``, ``TlsDirectory``
     - TLS callback array entries and the TLS directory fields
   * - ``Export``
     - An exported function address
   * - ``DataDirectory``
     - A data directory pointing into a protector section
   * - ``Relocation``
     - A base relocation whose target is in a protector section (``HIGHLOW`` and
       ``DIR64`` entries)
   * - ``UnwindInfo``, ``UnwindHandler``
     - x64 exception metadata and language-specific handlers
   * - ``Branch``, ``Call``, ``RipRelative``
     - Instruction operands found by walking code from every root
   * - ``Unresolved``
     - A reference or code region that could not be resolved. **Always blocks
       stripping.**

Like every other stage, the checker fails closed: a region it cannot analyse is an
``Unresolved`` reference, not a silent omission.

Strict coverage decision
------------------------

``decide_coverage`` combines the pieces. It may clear the two blockers only when
the OEP is proven, every inventory entry is both resolved and recovered, nothing
was dropped from the inventory, and no dangling reference remains. See
:doc:`function-recovery`.
