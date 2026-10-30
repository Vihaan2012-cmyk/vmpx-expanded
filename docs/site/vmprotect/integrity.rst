Integrity Checks
================

The "memory protection" option makes the protected program verify, while it runs, that
its own code has not been modified. It is the defence that makes the other protections
**expensive to remove**: patching a byte, setting a software breakpoint, or rebuilding
the file changes the bytes that a checksum covers.

The checksum handler
--------------------

The check is implemented inside the VM, as a dedicated handler that vmpx calls
``TableChecksum`` and describes as a "recovered byte-at-a-time table checksum over
program memory". Its structure is a classic table-driven CRC:

.. code-block:: nasm

    ; illustrative: one byte per iteration
    loop:
        movzx   edx, byte [rsi]          ; next input byte
        xor     dl, al                   ; combine with the low byte of the accumulator
        shr     eax, 8                   ; shift accumulator
        xor     eax, [table + rdx*4]     ; fold in the table entry
        add     rsi, stride
        dec     rcx
        jnz     loop

vmpx's ``TableChecksumMetadata`` records the exact recurrence that it supports:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Element
     - Contract
   * - Accumulator
     - 32-bit, seeded from a constant or from the VM stack, then XORed with
       ``initial_xor``
   * - Input
     - A pointer and a length taken from VM-stack offsets, walked with a signed
       ``pointer_stride``
   * - Per byte
     - Table index ``(accumulator & 0xFF) XOR input_byte`` into a 256-entry 32-bit
       table at a fixed virtual address
   * - Update
     - Accumulator shifted logically left or right by ``shift_bits``, then XORed with
       the table entry and an optional constant
   * - Finish
     - An optional ``NOT``, then ``final_xor``
   * - Result
     - Stored at a recovered VM-stack offset after a ``vsp_delta``

The shape matches CRC-32. ``validate()`` demands a 32-bit accumulator, a 256-entry
table of 32-bit entries, a shift below 32 and a non-zero stride, and rejects anything
else: only that exact recurrence is supported.

What is checked
---------------

The covered range is a property of the protected program. Commonly it includes the
code sections (so software breakpoints and patches are detected), the protector's own
runtime, and possibly the headers. The *consumer* of the result varies:

* compare against an expected value and branch to a failure path;
* feed the value into a later decryption, so a modified file silently decrypts
  garbage. This is the stealthy variant: there is no comparison to find;
* store it to be checked again later.

SDK-level integrity
~~~~~~~~~~~~~~~~~~~

The application can also ask for a check explicitly. ``VMProtectIsValidImageCRC``
returns whether the loaded image's CRC matches the one the protector recorded; the
protected program may call it and decide for itself what to do.

How a static recovery recognises it
-----------------------------------

The analyzer finds a **counted 32-bit byte-table loop** in the native handler and
proves that both the zero-length path and the loop-exit path converge on the same
ordinary threaded-dispatch tail. Constants and register choices come from data flow.
Nothing is matched by bytes: "no handler addresses, table contents, or mutation bytes
are signatured".

With a concrete pointer, length and table, the symbolic executor **computes the
checksum exactly** by reading the analysed image's bytes. Otherwise it emits an opaque
intrinsic, ``table_checksum_low_byte_xor_recurrence_w32``, carrying the accumulator,
pointer, length, table, shift, stride and a memory-snapshot epoch.

Why it matters for a rebuilt binary
-----------------------------------

A checksum depends on the bytes it reads. The analysis evaluates it against the
*analysed* image. A rebuilt image whose bytes differ in the checked range would not
reproduce the original value. Consequences:

* **Appending is safer than patching.** The reintegration design adds a new section
  and changes header fields rather than overwriting protector bytes, which avoids the
  common case of a checksum over ``.vmp`` code. Overwriting an entry stub with a jump
  (``redirect_vm_stub``) edits existing code, which is why it is confined to the
  experimental image (:doc:`/tool/pipeline/reintegration`).
* **Do not assume a constant.** The code keeps checksum results symbolic or
  image-derived instead of assuming a constant for the output.
* **The real fix is removal.** Stripping the protector's sections removes the checksum
  handlers along with the data they protect. This is why section stripping is a late,
  separately gated step (:doc:`/tool/pipeline/imports-and-stripping`).
* **Neutralization is the other option.** Rewriting a guard that depends on the
  checksum intrinsic to the non-detecting path makes the result debuggable at the
  price of equivalence (:doc:`/tool/pipeline/verification`).

Lifting status
--------------

``TableChecksum`` is a required lowering capability that is not in the implemented
set, so a routine containing it is not reported ready for code generation. It can be
*recognised* and *evaluated*, but not yet *compiled back*.
