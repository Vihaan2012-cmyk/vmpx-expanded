# "proven null memory read, requires exception-flow analysis" — investigation and verdict

Date: 2026-10-31
Branch base: `sp2-oep-generalized` (HEAD 8201b32)
Scope: investigate-first. No pipeline changes; this document + evidence only.
Owned files touched for recovery: none (fail-closed behaviour is correct).

## TL;DR

For the bigcorpus functions that stall with

```
symbolic VM execution failed at instruction N (<vip>, MemoryLoad):
proven null memory read of K byte(s) requires exception-flow analysis
```

the stall label is a **misnomer**. These are **not** exception-driven-dispatch
faults. The null read is the VM dereferencing a **global pointer slot that is zero
in the static protected image** and only acquires its real value at runtime (CRT
init / DllMain / another virtualized routine). There is **no exception
infrastructure in the image to resolve a continuation from**, so the post-fault
target is **not statically determinable**.

**Verdict: needs-runtime (defer to dynamic). Fail closed is correct. Do not
fabricate a continuation.** No Phase-2 fix was implemented.

## Reproducers

`E:/VMPX/testcorpus/bigcorpus/protected/sws.virt.dll`
- `sws_chars` (0x8c020): faults at VM instr 550, native read site RVA 0xBC366, 4 bytes.
- `sws_opcode` (0x84dc): faults at VM instr 549, native read site RVA 0x3A09, 4 bytes.

Driver: `cargo run -q --release -p vmpx-vm --example cfgdiag -- <sample> <rva>`.

## What the null read actually is

The faulting VM handler is a plain `MemoryLoad`:

```
; memory-load handler (sws_chars, entry 0x1800BC361)
mov  rax, qword ptr [rbp]     ; pop pointer operand off the VM operand stack (rbp = VSP)
mov  r10d, dword ptr [rax]    ; <-- fault site 0xBC366: external 4-byte read from [rax]
```

`rax` is the pointer on top of the VM operand stack. The symbolic executor proves
that pointer is the **constant 0**, so `mov r10d,[0]` is a provable null read and
`load_external` refuses to invent a value (`semantic_exec.rs:671`).

**Provenance of the 0 (instrumented run):** across the whole ~550-instruction
execution the only image-sourced read is a single **qword at RVA 0x3B900**, whose
static bytes are all zero. Every other external write in the run targets the VM
operand stack (`vsp_initial + 64/72`), never 0x3B900. So the sequence is: the VM
loads `qword [0x3B900]` (= 0) as a pointer, then a later `MemoryLoad` dereferences
that 0. The address `0x3B900` is embedded as a **VM-bytecode immediate** (there is
**no native xref** to 0x3B900 anywhere in the executable sections — confirmed by a
full capstone scan), which is why the global never appears in a `mov [rip+...]`.

`0x3B900` lives in `.gdo` (chars `0x60000020`: CODE | EXECUTE | READ, **not
writable**), has **no base relocation**, and is **not** in the IAT (IAT is at
0xf5000). Nothing in the static image writes it. Therefore its runtime value is
state produced by *other* code during program execution — not knowable from static
analysis of this one function on the protected image.

## Why it is not exception flow (decisive evidence)

The existing `exception_flow` module (and `docs/site/vmprotect/exceptions.rst`)
models the real VMProtect trick: a `MemoryLoad` reads a deliberately-invalid
address, the fault is caught by a **.pdata language handler** that builds a stack
sled and redirects `CONTEXT.Rip`. That machinery keys off
`RUNTIME_FUNCTION` + `UNWIND_INFO` (`UNW_FLAG_EHANDLER`).

Running that analysis on the sws fault sites (via the handler recovery +
`analyze_vm_memory_exception_coverage`) reports, for both handlers:

```
fault_sites=[0xBC366]  all_fault_sites_have_language_handler=false
  rep: fault site 0xBC366 has no covering runtime-function record
fault_sites=[0x3A09]   all_fault_sites_have_language_handler=false
  rep: fault site 0x3A09 has no covering runtime-function record
```

Direct PE inspection explains why and generalizes it:

- `sws.virt.dll`: **EXCEPTION data directory size = 0** (no `.pdata`, zero
  `RUNTIME_FUNCTION`s), **TLS size = 0**, relocations only 0x244 bytes.
- **Every** binary in `testcorpus/bigcorpus/protected/` (190+ images) has
  **EXCEPTION size 0 and TLS size 0**.
- Imports contain the standard MSVC-CRT `Rtl*` exception symbols
  (`RtlLookupFunctionEntry`, `RtlUnwindEx`, `RtlVirtualUnwind`,
  `SetUnhandledExceptionFilter`) but **no `AddVectoredExceptionHandler` /
  `RemoveVectoredExceptionHandler`**. With an empty exception directory,
  `RtlLookupFunctionEntry` finds nothing in-module, so table-based SEH cannot
  dispatch a fault inside this code.

So there is no SEH/VEH mechanism that could catch a fault at 0xBC366/0x3A09. A real
null deref there would simply crash. Since both static paths through the function
reach this read, the only consistent conclusion is that at genuine runtime the
global at 0x3B900 is **non-zero** — i.e. the read does not fault at runtime; it
faults only because the static image has not yet been initialized.

The `exception_flow` stack-sled machinery is therefore **not applicable** to this
corpus (it correctly fails closed), and it is not wrong — it targets a different
class of VMProtect images (those that actually carry `.pdata` language handlers),
which are simply absent here.

## Static-vs-runtime verdict

**Not statically resolvable.** Resolving the fault requires the runtime value of
the global pointer slot (here 0x3B900), which is established by code outside the
analyzed function (CRT startup / DllMain / another VM routine) and is absent from
the static protected image. There is no statically-provable continuation (no
language handler, no VEH, no registered handler) to follow, so the sound action is
to keep the stall. Guessing any continuation would be fabrication.

## What a dynamic engine must observe

To close this class of stall, the dynamic/runtime engine must capture, for the
faulting `MemoryLoad`:

1. the **concrete runtime contents of the global pointer slot** the VM loads (for
   sws: `qword [0x18003B900]`), after the module's load-time initialization has
   run (DllMain / CRT global constructors), and
2. feed that value back into the symbolic state as the proven pointer so recovery
   can continue past the load.

If (contrary to this corpus) an image *does* carry `.pdata` with a language
handler covering the read site, the existing `exception_flow` stack-sled path
applies instead, and the dynamic observation needed there is the
fault→handler→`CONTEXT.Rip` transfer (seed/cursor live on the **native** stack at
`CONTEXT.Rsp + offset`, which the VM symbolic state does not model — see
`resolve_stack_sled_continuation`).

## Note on the diagnostic wording

The error string "requires exception-flow analysis" over-commits to one cause. For
these images the precise cause is "reads a global pointer slot that is zero in the
static image (runtime-initialized)". Refining the message would not resolve any
stall or change completion counts, so it was left unchanged under the lean /
no-fabrication guidance; recorded here for the next iteration.
