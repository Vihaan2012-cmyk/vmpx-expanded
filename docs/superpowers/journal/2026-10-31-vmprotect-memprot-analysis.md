# VMProtect 3.11 Memory Protection — static characterization and sound removal path

Date: 2026-10-31
Branch base: `sp2-oep-generalized`
Scope: investigate-first. No pipeline changes; this document + evidence only.

## TL;DR

- **Structural delta:** Memory Protection adds **~144 KiB (0x24000)** of *plaintext*
  protector code/data, almost entirely into the first protector section
  (+0x2092A) and a little into the entry-owning protector section (+0x3330). It
  adds **no new data directories, no TLS, no load-config, no exceptions**. Layout
  stays `ConventionalFile`.
- **At-rest encryption: NO.** The memprot sections are plaintext (entropy ~5.81),
  raw bytes present on disk, and the original `.text`/`.rdata` are preserved
  **byte-identical** to the unprotected input. At-rest encryption/compression is a
  *separate* option ("Pack the Output File", 0x100), which alone produces
  `PackedVirtualSections` + entropy 7.64. Static reading of the image is fully
  possible.
- **Where the check lives:** native protector code in the memprot section,
  dispatched through the protector's **own bootstrap VM** (the entry stub swarm),
  reachable **only from the module entry point** (DllMain) — there is no TLS or
  other load-time trigger in a memprot-only binary. The exact checksummed region
  and mismatch behaviour are **runtime-only** observable (region descriptors are
  materialized inside the bootstrap VM at run time).
- **checksum_handler.rs / neutralize.rs do NOT cover it.** Those operate on the
  *user-function* VM lift (a bytecode-threaded table-checksum loop, recovered-guard
  `Expr`s). Memprot's check is native/bootstrap-VM code that the devirt pipeline
  never lifts. Different mechanism.
- **Recommended sound path:** **(b) strip the protector section post-devirt** is the
  only statically-sound *removal*. **(a) append-only reintegration already
  neutralizes memprot-only** binaries in practice — not because the check tolerates
  appended sections, but because entry/TLS redirection means the memprot bootstrap
  **never runs** (sole trigger is the entry point; imports are conventional). **(c)
  does not apply.** **(d) runtime/SP7 is needed only** to confirm what the check
  covers *if* the protector must remain live (e.g. combined with import
  protection).

---

## 1. Structural delta (memprot vs non-memprot)

Reproducers are VMProtect_Con 3.11 output over the same input
(`E:/VMPX/testcorpus/bigcorpus/dll/speck.dll`, a 2-section export-only DLL, 2048 B).
Options isolate a single feature over the 0x40000 "strip-debug" base:

| file | Options | feature over base | layout |
|------|---------|-------------------|--------|
| `speck.pack.dll`   | 262400 (0x40100) | pack output         | `PackedVirtualSections` |
| `speck.improt.dll` | 262656 (0x40200) | import protection   | `ConventionalFile` |
| `speck.memprot.dll`| 263168 (0x40400) | **memory protection** | `ConventionalFile` |
| `speck.full.dll`   | 263936 (0x40700) | all three           | — |

`memprot` vs `improt` is the cleanest isolation of the memprot footprint (both are
`ConventionalFile`, both differ from base only by one bit).

### `vmpx info` sections

`speck.memprot.dll` — image 0x14F000, entry_rva **0x11C184**, 6 sections:

```
 0 .text    rva=0x1000    vsize=0x156    raw_size=0x200     chars=0x60000020  (R-X)  [original]
 1 .rdata   rva=0x2000    vsize=0xAA     raw_size=0x200     chars=0x40000040  (R--)  [original]
 2 .5x:     rva=0x3000    vsize=0x10A321 raw_size=0x10A400  chars=0x60000020  (R-X)  protector #1
 3 .a.8     rva=0x10E000  vsize=0x350    raw_size=0x400     chars=0xC0000040  (RW-)  IAT
 4 .`Ew     rva=0x10F000  vsize=0x3ED10  raw_size=0x3EE00   chars=0x60000020  (R-X)  protector #2 (entry)
 5 .reloc   rva=0x14E000  vsize=0x244    raw_size=0x400     chars=0x42000040
directories: export 0x2000, import 0x14D338, base_relocation 0x14E000, iat 0x10E000
tls=0  load_config=absent  exception=0  security=absent  debug=absent
```

`speck.improt.dll` — image 0x12B000, entry_rva 0xFAC20, 6 sections, same shape:
protector #1 `.kU\` vsize 0xE99F7, protector #2 `.Z}U` vsize 0x3B9E0.

### The delta

| region | improt | memprot | delta |
|--------|--------|---------|-------|
| protector #1 vsize | 0xE99F7 | 0x10A321 | **+0x2092A (+133,418 B)** |
| protector #2 vsize | 0x3B9E0 | 0x3ED10 | **+0x3330 (+13,104 B)** |
| SizeOfImage | 0x12B000 | 0x14F000 | **+0x24000** |
| file size | 1,204,736 | 1,352,192 | **+147,456 B (~144 KiB)** |

Memory Protection therefore adds ~144 KiB **inside the existing protector
sections** — no new section, no new directory, no TLS/load-config/exception/
security. The bulk (+133 KiB) lands in the large protector section (the CRC engine
plus its region/descriptor tables), ~13 KiB in the entry-owning section (the setup
glue invoked from DllMain).

## 2. Entropy / at-rest encryption (part 1 + part 4)

`vmpx info`/`detect` section entropy:

- memprot `.5x:` = **5.8087**, `.`Ew` = **5.8143** (plaintext x86 range)
- improt `.kU\` = 5.7496, `.Z}U` = 5.7198
- pack `."uc` = **7.6392** (compressed/encrypted), layout `PackedVirtualSections`,
  entropy "unavailable" for the raw-less sections.

Byte comparison (verified):

```
orig .text  == memprot .text   : True
orig .rdata == memprot .rdata   : True
memprot .text == improt .text   : True
```

**Conclusion (part 4): memprot does NOT encrypt section bytes at rest.** Sections
are file-backed, plaintext, and the original code/data survive byte-for-byte.
On-disk encryption/compression is the orthogonal *pack* option, not memprot.
Memprot's protection is a **runtime, in-memory** integrity check (and, per
VMProtect's design, runtime page re-encryption / anti-dump that leaves the
at-rest file plaintext). Static reading of the protected image is unobstructed.

## 3. Where the integrity check lives, and how far static goes (part 2)

Facts established statically:

- Entry point `0x11C184` lies inside the entry-owning protector section `.`Ew`
  (0x10F000..0x14DD10). For a DLL, `AddressOfEntryPoint` is DllMain — i.e. the
  memprot runtime initializes at load, from the entry point.
- `detect` reports a **VM entry-stub swarm** (24 push-imm/call stubs, common target
  **0x1121D7**, also in `.`Ew`), and 6/6 probed stubs decode into coherent VM
  streams. The protector bootstrap — including memprot setup — is itself
  **virtualized** by the protector's own bootstrap VM.
- There is **no TLS callback, no load-config, no exception directory** in the
  memprot binary. The **entry point is the sole load-time trigger** for memprot.

Static tracing boundary: because the memprot setup is dispatched through the
bootstrap VM and its protected-region descriptors are decrypted/materialized at run
time, the precise bytes it checksums (headers only? code sections? whole mapped
image?) and the mismatch disposition (silent corruption vs. termination) are **not
reliably recoverable by static analysis** of this binary. That part is honestly
**dynamic-only** (SP7). What *is* certain statically: the check is **native /
bootstrap-VM protector code**, physically contained in the protector sections, and
only reached by executing the module entry.

## 4. Does checksum_handler.rs / neutralize.rs cover it? (part 3)

**No — different mechanism.**

- `crates/vmpx-vm/src/checksum_handler.rs` recognizes a **user-VM** construct: a
  counted W32 byte-table checksum **VM handler** — a bytecode-threaded dispatch loop
  that reads a pointer/length off the VSP, walks a 256×W32 table, and converges on
  one relative-dispatch tail. It is matched *inside the lifted VM*, keyed on VSP/VIP/
  handler register roles.
- `crates/vmpx-vm/src/neutralize.rs` rewrites **recovered-guard `Expr`s** whose
  condition depends on the in-tree `table_checksum_low_byte_xor_recurrence_w32`
  intrinsic (or RDTSC / `IsDebuggerPresent`-style call results). It acts on guards
  produced by devirtualizing **user functions**.

Memprot's integrity check is **not** a lifted user-VM handler and produces no such
`Expr`/intrinsic: it is native code (virtualized by the *bootstrap* VM, which the
devirt pipeline does not lift) that runs before/around the user payload. Neither
pass can see or rewrite it. They would only ever apply if a *user* function were
virtualized and itself contained a table-checksum guard — unrelated to the
packer-level memprot engine.

## 5. Recommendation — sound removal path

### (b) Strip the protector section post-devirt — PRIMARY sound removal

`vmpx-pe::strip_sections` + its `Checker` dangling-ref pass are exactly the right
instrument and are **sound by construction**: the checker refuses the strip if *any*
retained structure (entry point, imports, exports, TLS, load-config, relocations,
exceptions, relocated pointers) statically references the removed section. Once
devirt/unpack has reconstructed the real OEP and the live code into retained/
appended sections, the protector sections (`.5x:`, `.`Ew`, the protector IAT) are
dead; removing them physically deletes the memprot engine, so the check **cannot
run** — the strongest possible guarantee. For the memprot-only shape this is
tractable: imports are conventional (loader-resolvable) and there is no TLS/
load-config to re-root into the protector.

Caveat: `strip_sections` is today a **library primitive** in `vmpx-pe`; it is **not
wired into the `clean` pipeline** (clean.rs only calls `reintegrate`). Turning (b)
into a shipping capability is a wiring task: after reintegration, select the
protector sections, prove liveness is gone via the existing checker, and strip. The
checker already enforces soundness, so this is low-risk — but it is a real pipeline
change and out of scope for this investigate-first pass.

### (a) Append-only reintegration — already neutralizes memprot-only, for a specific reason

The current pipeline (`clean.rs:1469` → `vmpx_rebuild::reintegrate`) appends a
`.vmpx0` section and **redirects the entry point and TLS callbacks** to the
reconstructed native code. For a memprot-only binary this **already defeats
memprot**, but the correct reasoning is "**the check never executes**", not "the
check tolerates an appended section":

- the sole memprot trigger is the entry point (no TLS/load-config — section 3);
- redirecting `AddressOfEntryPoint` to the reconstructed DllMain bypasses the
  protector bootstrap entirely;
- imports are conventional, so bypassing the protector does not break the IAT.

This is contingent on those three facts. It is **not** a general proof that memprot
ignores appended sections. If the protector bootstrap must stay live (e.g. memprot
combined with **import protection**, where the real IAT is only populated by the
protector at run time, or a build that roots memprot in a TLS callback), then the
check *does* run and whether the append survives depends on **what it covers** —
which is the dynamic-only question of part 2.

### (c) Neutralize the guard — does NOT apply

Per section 4, memprot's check is never lifted into the guard/`Expr` form that
`neutralize.rs` consumes. (c) is a non-starter unless the protector bootstrap VM is
itself lifted through the same pipeline, which it is not.

### (d) Dynamic engine / SP7 — needed only for the residual coverage question

SP7 is **not** required to remove memprot on the strip path (b) or on the
entry-redirect path (a) for memprot-only binaries. It **is** required to answer, for
the protector-stays-live case, exactly **which bytes the integrity check covers**
(headers / section table / code / whole image) and its **mismatch behaviour**, so
that append-only output can be *proven* safe when the bootstrap cannot be bypassed.
Concretely, SP7 must: run the protector entry under the engine, observe the
bootstrap VM's reads to recover the protected-region descriptor set and the compare/
dispatch on mismatch, and report whether the appended `.vmpx0` section and the
rewritten PE header/section-table fall inside the checksummed range.

## Recommendation summary

| approach | verdict for memprot |
|----------|---------------------|
| (a) append-only reintegration | **Works for memprot-only today** — because entry/TLS redirect means the check never runs (sole trigger is the entry; imports conventional). Not a general append-tolerance guarantee. |
| (b) strip protector section post-devirt | **Primary sound removal.** `strip_sections` + dangling-ref checker is sound by construction; deletes the engine so the check cannot run. Needs pipeline wiring (library primitive today). |
| (c) neutralize guard | **N/A** — memprot's check is native/bootstrap-VM, never lifted to a guard `Expr`. |
| (d) dynamic / SP7 | **Only for the residual coverage question** when the protector must stay live (e.g. + import protection). Says exactly what the check covers and the mismatch action. |

Bottom line for the user's directive ("removing prot like memprot"): the sound
static answer is **(b) strip the protector section after devirt**, backed by the
existing dangling-ref checker; **(a)** is the pragmatic path that already holds for
memprot-only binaries because the engine is bypassed, not tolerated; **memprot does
not encrypt at rest**, so none of this is blocked by unreadable bytes; and SP7 is
reserved for proving append-safety only when the protector cannot be bypassed.

## Evidence / reproduction

```
vmpx detect E:/VMPX/testcorpus/protected/speck.memprot.dll   # ConventionalFile, entropy ~5.81
vmpx detect E:/VMPX/testcorpus/protected/speck.pack.dll      # PackedVirtualSections, entropy 7.64
vmpx info   E:/VMPX/testcorpus/protected/speck.memprot.dll
vmpx info   E:/VMPX/testcorpus/protected/speck.improt.dll
vmpx info   E:/VMPX/testcorpus/bigcorpus/dll/speck.dll
```
Original `.text`/`.rdata` vs memprot/improt: byte-identical (verified).
