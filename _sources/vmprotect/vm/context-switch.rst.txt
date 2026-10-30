Crossing the Boundary
=====================

The VM does not live in isolation. It is entered from native code, calls back into
native code to use the operating system, computes flags with native instructions,
and eventually returns. Each crossing is a hazard for the VM's designer and a
foothold for the analyst.

Entering
--------

See :doc:`entry`. In summary: the entry stub pushes a seed and calls the bootstrap,
which saves every register and the flags, builds a frame, scatters the saved values
into the context, decodes the seed and dispatches.

Leaving: the VM exit
--------------------

The exit sequence reverses the bootstrap. In sequence:

#. A virtual-register-to-stack sequence arranges the final register values on the VM
   stack in the order the exit will pop them.
#. The stack pointer is **switched from the VM stack to the native stack**.
#. Every non-``RSP`` general-purpose register is restored from the stack by a
   ``pop`` (the order is arbitrary and mutated, but each register's final
   definition is its corresponding ``pop``).
#. ``popfq`` restores the flags.
#. ``ret`` returns to the address now at the top of the native stack.

vmpx proves a handler is a VM exit by requiring all of: a VSP-to-native-``RSP``
switch, restoration of every non-``RSP`` register from the stack, a final ``popfq``,
no non-stack memory access, and a native ``ret``. Junk from register mutation is
tolerated only if the *final* definition of each register is its ``pop``.

Where does the ``ret`` go?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Usually to the instruction following the entry stub's call site, so the function
returns to its caller as if it had run natively. But the return address is simply
the value on the native stack, which the VM can have *rewritten*. It can therefore
return to:

* the original caller (a normal virtualized function);
* the **original entry point** of the program (the protector's initialization
  virtualizes the setup and then hands off to the unprotected program; this is the
  basis of vmpx's OEP recovery, :doc:`/tool/pipeline/oep-recovery`);
* **another VM entry stub** (a chain of virtualized regions); vmpx continues
  exploration by composing the next entry's frame from the exact prior exit state; or
* arbitrary **native code** (a nested continuation), which vmpx follows with a
  bounded interpreter.

The exit state
~~~~~~~~~~~~~~

Because the exit pops every register from memory the VM controlled, the machine
state on exit is fully determined by the contents of the VM stack at that moment. A
symbolic executor that tracks that memory exactly obtains, for every register, the
flags and the return address, a closed-form expression over the inputs. That
expression set is vmpx's ``NativeExitState``.

Calling native code from the VM
-------------------------------

A virtualized function that calls ``WriteFile`` cannot do so from inside the
interpreter without leaving it. The **native-call bridge** handler does:

#. load the call target (an API address) into a register;
#. save the native ``RSP`` and switch to it;
#. push arguments per the Win64 ABI and ``call``;
#. store the return value back into the VM context;
#. restore the VSP and the saved VM registers;
#. run a *selector* that decides which of several outcomes applies (for instance
   whether the call returned normally or the target was an internal one); and
#. resume threaded dispatch.

vmpx proves this shape from control-flow dominance and register/memory effects. A
decoded selector with no proven outcome is an error. The call's target is matched
against the parsed imports (by lookup value, IAT RVA, or preferred-base IAT VA) to
name the API. An unknown target is an ``Opaque`` memory effect that forces a *havoc*
of tracked memory. A narrow, trusted registry gives meaning to a few allocation
APIs: ``LocalAlloc``/``GlobalAlloc`` (only with proven fixed flags, since a
``MOVEABLE`` allocation returns a handle), ``HeapAlloc`` and ``CoTaskMemAlloc``.
Each call receives a unique *epoch* so that two identical calls are never assumed to
return the same value.

Virtual flags
-------------

The VM does not model the flags register. To obtain the flags of an addition it
**executes a native add** inside the handler and captures ``RFLAGS`` with ``pushfq``.
Flag values then travel on the VM stack like any other value, and conditional
branching masks them: the carry flag is bit 0, parity bit 2, adjust bit 4, zero bit
6, sign bit 7, overflow bit 11.

This is easy for a human and awkward for an analysis, because a naive symbolic
executor sees an opaque ``pushfq``. vmpx encodes the result as intrinsic expressions,
``vm_flags_<op>_w<N>``, which carry the operation and width and are rewritten to plain
arithmetic at code-generation time. They are also what makes the finite-domain
analysis of :doc:`/tool/pipeline/function-recovery` possible: a compare-then-branch is a
function of ``vm_flags_add_wN(±reg + c1, c2)``, and a bounded set of register values
realizes every flag combination.

Native transitions between VM profiles
--------------------------------------

An exit that lands at another VM instance's entry stub is a *native profile change*:
the next VM has a different bootstrap, different register roles and a different
bytecode. vmpx's profile explorer carries the full symbolic state across, and
re-runs role recovery for the new profile. It "never guesses constraints or merges
decoder states": if the new profile's roles are ambiguous it stops with
``AmbiguousProfileRoles`` instead.
