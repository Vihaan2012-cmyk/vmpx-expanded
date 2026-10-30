Import Protection
=================

A program's imports are the plainest description of what it does: the list of
operating-system functions it calls. An unprotected PE publishes them in the import
directory, and a disassembler turns every ``call qword ptr [rip+disp]`` into a named
call. *Import protection* removes that description.

What is hidden
--------------

Three separate things can be hidden, and a protector can choose among them.

The table
    The import directory (descriptors, name tables, thunks) is replaced or emptied.
    A tool that lists a file's imports sees only what the *protector* needs, such as
    a handful of functions in the system DLLs, not what the program needs.

The call sites
    The code that used to say "call through slot *k* of the IAT" no longer does.

The binding
    Which slot corresponds to which API, and when that is resolved.

Typical mechanics
-----------------

Public analyses describe the following arrangement. Details differ per version and
per build.

*Call-site rewriting.*
    An indirect call through an IAT slot,
    ``call qword ptr [rip+disp]`` (six bytes), is replaced by a direct call into a
    protector thunk followed by a one-byte pad: ``call thunk; nop``. Indirect jumps
    through the IAT (tail-call thunks, ``jmp qword ptr [rip+disp]``) are treated the
    same way. The program therefore has no direct reference to the IAT.

*The thunk.*
    Each thunk (or a shared resolver it jumps to) computes the real API address and
    transfers to it, typically with an idiom that leaves the stack in the state the
    API expects: the address is pushed and a ``ret`` jumps to it, so the API returns
    straight to the original call site. The address is found by hashing, by name
    lookup in the loaded modules' export tables, or by decrypting a stored value, so
    that no plain pointer to the API exists anywhere in the file.

*Lazy or eager resolution.*
    The protector may resolve all APIs in its initialization (and write them into a
    private table), or resolve each on first call and cache it. A dump taken before
    all calls have happened is incomplete under the lazy scheme.

*Imports used by the VM.*
    The VM's native-call bridge (:doc:`vm/context-switch`) calls APIs by loading a
    target into a register, so APIs called from virtualized code reach the OS
    through the bridge rather than through a thunk.

Duplicate and aliased imports
-----------------------------

An import protected this way can appear several times under different identities,
because each rewritten site may get its own thunk. A symbol imported by name in one
DLL and by ordinal in another, and a symbol that is imported twice, are all
"duplicate identity groups". vmpx **preserves duplicates as evidence** and never
collapses them: collapsing would assume what the analysis is trying to prove.

How vmpx inventories imports
----------------------------

The ``imports`` subcommand counts modules, named versus ordinal symbols and duplicate
groups, then looks for the *direct* native references to exact IAT slots with two
resynchronizing decode scans (:doc:`/tool/cli/imports`):

* one over the ranges named by the exception table (``RUNTIME_FUNCTION`` entries give
  function boundaries even when the code is obfuscated);
* one over the remaining executable bytes.

Resynchronizing means decoding from many independent starting points so that
instruction boundaries lost to junk bytes are recovered. A reference found this way
is a **decoded candidate**, not a proven reachable use, and the report says so:

* absence of a direct reference does not prove an import unused;
* duplicates are preserved as evidence;
* exception metadata alone does not prove reachability;
* independent-start findings are candidates, not proven.

Reconstructing the application's import set
-------------------------------------------

The goal of import reconstruction is to replace the protector-era import view with the
**application's real imports**: only the slots the program's own code uses, each with
exactly one name or ordinal, in a conventional import directory. vmpx's rule
(:doc:`/tool/pipeline/imports-and-stripping`):

* a slot becomes an application import only when it is *referenced* and carries
  *exactly one unambiguous identity*;
* every other referenced slot is reported as unresolved, with a reason (``NoBinding``,
  ``AmbiguousSymbol``, ``ConflictingBindings``, ``InvalidIdentity``);
* each resolved source slot maps to exactly one rebuilt slot so references can be
  retargeted without matching names.

The remaining hard part, resolving **indirect** use (a protector thunk or a call
inside virtualized code) to a concrete API name, is why the pipeline still lists
"application import set is not proven independently reconstructed" as a blocker.
The pieces that exist today are the evidence scan, the soundness-preserving
reduction and the table serializer; connecting thunk and bridge resolution to them
is the open work.
