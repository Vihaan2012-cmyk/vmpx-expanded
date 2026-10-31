# SP7 dynamic engine — design

Status: active (2026-10-31). The static frontier's high-value avenues are exhausted;
three capability gaps are **confirmed to require runtime observation** (import
protection, runtime-initialized global pointers, recursion depth). SP7 builds the
dynamic engine to close them. It is the project's finish line.

## Non-negotiable constraints
1. **OFFLINE, CONTAINED, ALWAYS.** The emulator (`vmpx-emulate`) is a pure in-process
   software interpreter: no host pointer, syscall, file, process, or **network** path
   exists, and none may ever be added. Unmodeled behavior STOPS the run, never guessed.
   Never execute a protected sample in anything with network access. (User hard rule,
   companion to avoid-rm-rf.)
2. **Observed ≠ proven (trust tier).** A run yields *observed* facts. Observed facts may
   only SEED the static model on the quarantined/experimental path, labeled `Observed`,
   with every used fact named in the report. They are NEVER written as sound `OUTPUT`.
   Where observed contradicts static, the run wins for diagnosis and the static result is
   reported blocked, not silently patched. (= dynamic-assist.rst trust tier.)
3. **Dynamic ASSISTS static, does not replace it.** The run supplies only what the binary
   does not contain; the static devirtualizer still does the recovery.

## Foundation (already real, from the survey)
`vmpx-emulate` Layer 1: `load()` maps a `vmpx_pe::PeImage` at preferred base with section
perms, TEB/PEB/LDR stubs (`gs:`), stack, and an import-sentinel IAT (`STUB_REGION`
0x20_0000_0000; `import_stubs[stub_va]=(dll,func)`; `resolve_stub`). `Emulator` single-steps
iced-x86 with rollback-on-stop; `Hook{on_mem,on_flow}` fires BEFORE commit, `Stop::Hook`
leaves state intact so the caller can patch and resume. `Memory` = permissioned sparse
pages (dump source + W^X). `differential` oracle runs an export/RVA to the return sentinel
and compares two targets (SP5 verification, to check observed-seeded lifts against the
original). Containment confirmed by construction.

## The cross-cutting enabler (prerequisite for D2/D3/D5)
The `Hook` trait cannot mutate `Cpu`/`Memory` in-callback, and `TraceEntry` carries only
`{rip,mnemonic}`. **WP-D1** fixes both:
- Extend the observation callback to receive `&mut Cpu, &mut Memory` (e.g.
  `on_flow(&mut self, &FlowEvent, &mut Cpu, &mut Memory) -> HookAction`), OR fold a
  dispatch sink into `Emulator`. Preserve the rollback-on-stop guarantee.
- Richer observation record: per-step effective-address / operand / value / VSP, behind a
  cheap opt-in (don't bloat the hot path when unused).
- An `ObservedFacts` bundle type (the hand-off format to static): `{ imports:
  Vec<(slot_rva, module, symbol|ordinal)>, globals: Vec<(va, bytes)>, loop_depths:
  Vec<(loop_head_rva, depth)>, oep: Option<rva>, … }`, every entry carrying the `Observed`
  label.

## Work packages (distinct crate/file ownership)
- **WP-D1 enabler** (`vmpx-emulate/src/{emu.rs, lib.rs}` + new `observe.rs`): Hook/trace
  extension + `ObservedFacts`. LANDS FIRST; D2/D3/D5 depend on it.
- **WP-D2 Layer 2 API stubs + import bridge** (`vmpx-emulate/src/winapi.rs` new): dispatch
  on `FlowEvent{Call|Jmp, to∈STUB_REGION}` → `resolve_stub` → emulate the minimum Win32
  surface (VirtualAlloc/Protect/Free, HeapAlloc/Free/GetProcessHeap, GetModuleHandleA/W,
  LoadLibraryA/W, GetProcAddress, GetSystemTimeAsFileTime, IsDebuggerPresent→0,
  CheckRemoteDebuggerPresent→0), deterministic fake HMODULE/proc addresses, read string
  args from guest memory, set RAX, pop return. Capture each GetProcAddress
  `(module,symbol)→resolved-slot` into `ObservedFacts.imports`. Args in RCX/RDX/R8/R9.
- **WP-D3 Layer 4 dump/OEP + global-capture bridge** (`vmpx-emulate/src/dump.rs` new):
  OEP/unpack detection (write-then-execute via `on_mem` + fetch-into-dirtied-page, or
  `run_until`), `Memory`→PE snapshot for `vmpx-rebuild`; after load-time init, capture the
  concrete contents of runtime-initialized global slots into `ObservedFacts.globals`.
- **WP-D4 Layer 3 ISA widening** (`vmpx-emulate/src/{emu.rs exec, alu.rs}`): add the
  instructions real VMProtect startup needs — `mul/imul/div/idiv`, `adc/sbb`,
  `rol/ror/rcl/rcr`, `bt/bts/btr/bsf/bsr`, `setcc`, `cmov`, `xchg`, `rep movs/stos`,
  `pushfq/popfq`, needed SSE moves — each with EXACT flag semantics (alu.rs style) or stop.
  Parallelizable with D1 (different concern; coordinate emu.rs with D1).
- **WP-D5 Layer 5 VM observation + concolic recursion depth** (`vmpx-emulate/src/
  observe.rs`): recognize the VM dispatcher address, record handler traces; derive bounded
  recursion depth from back-edge counts under a concrete input → `ObservedFacts.loop_depths`.
- **WP-D6 static feedback wiring** (`vmpx-cli/src/clean_wiring.rs::derive_referenced_slots`
  from observed imports; `vmpx-vm` symbolic-state seeding from observed globals, labeled;
  recursion lift seeded by observed depth): consume `ObservedFacts` on the experimental
  path, label everything `Observed`, verify seeded lifts with the `differential` oracle,
  never touch the sound gate. Closes `IMPORT_BLOCKER` (with observed, quarantined bindings)
  and the null-read / recursion stalls on the experimental tier.

## Build order
D1 + D4 first (D4 parallel). Then D2, D3, D5 (all need D1). Then D6 (needs the
`ObservedFacts` format + D2/D3/D5 producing them). Verify every observed-seeded lift with
the differential oracle before reporting it, always labeled, never in the sound `OUTPUT`.

## Done = the three bridges work end-to-end on the real samples
(a) import protection: observed GetProcAddress bindings reconstruct the IAT for a protected
sample (experimental, labeled); (b) a runtime-global null-read (sws qword[0x18003B900])
resolves from the captured concrete value; (c) a recursive function (rmath_fib) lifts under
an observed concrete depth. All contained, offline, observed-labeled.
