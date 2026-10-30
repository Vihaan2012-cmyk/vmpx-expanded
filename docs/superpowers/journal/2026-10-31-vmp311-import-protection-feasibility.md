# VMProtect 3.11 Import Protection — Static Resolvability (Phase 1 findings)

Date: 2026-10-31
Reproducer: `E:/VMPX/testcorpus/protected/speck.improt.dll`
(VMProtect Ultimate 3.11.0 build 2835, project Options=0x40200, `<Procedures>` empty — import protection + packing only, **no functions virtualized**)

## Verdict

**NEEDS RUNTIME.** VMProtect 3.11 import protection on this corpus is **not** statically
resolvable into a proven call-site -> (module, name) binding. The hidden import identities
are produced by the protector's virtualized loader at run time; nothing in the static image
encodes the hidden identities in a form that can be decoded without emulating that loader.

`derive_referenced_slots()` returning `BTreeSet::new()` is therefore **correct** for this
input, and `IMPORT_BLOCKER` must stay. No code was written in Phase 2 (soundness: never
fabricate an import name).

## The surviving import surface (7 symbols, 2 KERNEL32 descriptors)

IAT dir rva=0xED000 size=0x48 (exactly 7 qword slots + null terminator):

| slot rva | name | role |
|---|---|---|
| 0xED000 | GetSystemTimeAsFileTime | seed / entropy |
| 0xED010 | GetModuleHandleA | module base lookup |
| 0xED018 | LoadLibraryA | load target module |
| 0xED020 | GetProcAddress | **resolve hidden API by name** |
| 0xED028 | HeapAlloc | allocate runtime thunk table |
| 0xED030 | HeapFree | free runtime thunk table |
| 0xED038 | ExitProcess | tamper response |

This is precisely a dynamic import-resolver surface: GetModuleHandleA/LoadLibraryA +
GetProcAddress resolve the hidden set, HeapAlloc/HeapFree hold the resolved thunks in
runtime-allocated memory (there is no static section for them — see below).

## Ground truth and what is hidden

- Original unprotected input `bigcorpus/dll/speck.dll`: **0 imports** (freestanding cipher,
  2 sections, 5 exports, 0x200 bytes of code). The application imports nothing.
- `bigcorpus/protected/speck.virt.dll` (import protection OFF): **105 imports / 1 module** —
  these are VMProtect's **own runtime** imports, left visible (VirtualProtect, VirtualAlloc,
  CreateToolhelp32Snapshot, OpenThread, Thread32First/Next, SuspendThread, WriteProcessMemory,
  TerminateProcess, CreateEventA, ... — the anti-debug / loader API surface).
- `speck.improt.dll` (import protection ON): the same runtime needs those APIs, but only the
  7-symbol resolver surface survives; the other ~98 are hidden.

**Soundness consequence specific to this reproducer:** the hidden imports are
protector-resident, not application imports. The application import set is empty. Attributing
any of the hidden set to the application (to "close" IMPORT_BLOCKER) would be wrong. For a real
target that genuinely imports APIs, those would be hidden among the same ~98 and resolved by
the same runtime loader — the verdict and mechanism are unchanged.

## Why static resolution fails (evidence by address)

1. **Names are not plaintext.** Of the 105 virt.dll names, only the 7 survivors appear as
   ASCII in `speck.improt.dll`; the other 98 do not appear anywhere in the raw bytes.
2. **No static thunk table for the hidden set.** The IAT section is exactly 0x48 bytes
   (7 slots). There is no reserved static space for ~98 resolved pointers; they live in
   HeapAlloc'd memory populated at load time.
   - The one contiguous 96-pointer relocated run at rva 0x30E40 is a red herring: it points
     into 0x30D14..0x30E28, which is the **MSVC CRT `__unDName` operator table**
     (`operator`, `__cdecl`, `__stdcall`, `__fastcall`, ` new`, ` delete`, ...), not import
     descriptors.
3. **No native reference to the resolver slots.** An alignment-independent byte scan for
   `call/jmp [rip+d]` (FF15/FF25) and `mov/lea r64,[rip+d]` (48/4C 8B/8D 05) across the whole
   image found **zero** references into [0xED000,0xED040). The existing CLI resync agrees:
   `vmpx imports` reports "0 IAT slots have direct native references". The loader reaches the
   IAT through VM-interpreted / computed addressing, not a decodable native operand.
4. **The startup is virtualized.** Entry 0xFAC20 = `push 0x51EED12C; call 0x1800FADCD` — the
   classic VMProtect VM bootstrap (24-stub swarm all target dispatcher 0xFADCD). The import
   resolver runs inside the VM, under per-build mutation/encryption.

The hidden identities, if present at all, are encrypted descriptors consumed by a per-build
polymorphic, VM-ized decryptor; recovering them requires either reversing that per-build
decryptor or observing resolution at run time. Neither is a static structural proof.

## What SP7 (the dynamic engine) must do to resolve these imports

1. Map the image and run (or faithfully emulate) VMProtect's virtualized startup from entry
   0xFAC20 through dispatcher 0xFADCD, with a loader environment providing the 7 resolver
   APIs (GetModuleHandleA, LoadLibraryA, GetProcAddress, HeapAlloc/HeapFree).
2. Intercept the loader's resolution calls: for each hidden import, capture the
   `(HMODULE from GetModuleHandleA/LoadLibraryA, name/ordinal passed to GetProcAddress)` pair
   and the heap address of the thunk slot the resolved pointer is written to.
3. Build the proven map: resolved-pointer-slot (heap VA, later a rebuilt IAT slot) ->
   (module, name/ordinal), taken directly from the observed GetProcAddress arguments — a
   structural observation, not an inference.
4. Correlate each hidden call site (native `call/jmp` through the heap thunk, or the VM
   "call external" handler operand) to its thunk slot, yielding call-site -> identity.
5. Hand the proven, non-ambiguous subset to `reduce_to_proven_imports`/the installer. Anything
   resolved through a computed/indirect path that cannot be tied to an observed GetProcAddress
   argument stays unresolved and keeps IMPORT_BLOCKER.

Only observed GetProcAddress arguments (or an equivalently structural emulated observation)
constitute proof; a name-hash guess against assumed system-DLL export tables is environment
-dependent and is not acceptable as a proven identity.

## Phase 2

Not performed — Phase 1 verdict is NEEDS RUNTIME. No resolver code written; no fabricated
imports. `derive_referenced_slots` and `IMPORT_BLOCKER` left unchanged.
