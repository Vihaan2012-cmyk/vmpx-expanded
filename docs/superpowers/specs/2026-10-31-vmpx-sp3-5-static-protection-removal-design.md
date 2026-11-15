# SP3–SP5 static protection removal — design

Status: active (2026-10-31). Goal: make `vmpx` ingest a VMProtect 3.11 binary with
**every file-level protection enabled** (packing, import protection, memory
protection / integrity, resource protection, relocations stripped, anti-debug) and
statically strip the protection back to a sound, loadable, parse-equivalent PE —
devirtualization of function bodies is the separate VM/mutation track. Dynamic
loader-stub emulation (self-decrypting / anti-dump images) is **deferred to the
dynamic engine (SP7)** per the static-first mandate; everything here is static.

## Current ground truth (survey 2026-10-31)

The protection-removal stack is **sound, tested library primitives gated behind
honest proof-blockers**, not stubs. What exists and works:

- `vmpx-unpack::unpack_to_mapped_image` — static LZMA depack of the `PACKER_INFO`
  table (plaintext + rotating-key), rebuilds the SizeOfImage memory image, bounded,
  reparses. REAL — but never yet run on a genuinely packed sample.
- `vmpx-rebuild` — `serialize_import_table` / `plan_import_reconstruction` /
  `install_import_directories` / `reduce_to_proven_imports`; `relayout_conventional`
  (PE32+), `validate_for_loader`, `reintegrate` (append R+X section + redirects).
  REAL.
- `vmpx-pe::strip_sections` — section removal with a dangling-static-reference
  checker (entry/imports/exports/TLS/load-config/debug/relocs/exception), on-disk
  compaction, header/checksum recompute, reparse+revalidate. REAL but **not wired
  into the CLI/clean pipeline**.
- `vmpx-vm::checksum_handler` — proves a counted CRC-32 byte-table integrity loop by
  data-flow; emits `table_checksum_...` intrinsic or an exact value when concrete.
  REAL (recognition/evaluation); integrity is **not removed**.
- `vmpx-vm::{anti_debug_idioms,neutralize}` — recognizes PEB/heap idioms and the
  checksum/RDTSC/IsDebuggerPresent constructs; `neutralize()` rewrites guards.
  REAL library, **recognition-only in the pipeline, no `--neutralize` CLI**.
- `vmpx-detect` — structural classifier (`ImageLayout`, `ProtectionMode`). No
  file-level option bitfield decode (the bitfield is compile-time, absent in the
  binary — classification stays structural).

### The blockers (in `vmpx-cli/src/clean_wiring.rs`) and what closes each
- `IMPORT_BLOCKER` — `derive_referenced_slots()` returns `BTreeSet::new()` by design.
  Closed by **WP2**: resolve VMProtect import call-sites (thunk + VM native-call
  bridge) to concrete `module!symbol`, producing a *proven* referenced-IAT-slot set.
- `VMP_BLOCKER` — protector sections remain. Closed by **WP3**: wire `strip_sections`
  to remove protector/VM sections once provably dangling-reference-free.
- `REINTEGRATION_BLOCKER` — emitted code needs relocations. Closed by **WP4**:
  additive base-reloc generation for appended code.
- anti-debug `applied:false` — closed by **WP5**: `--neutralize` CLI + guard/provenance
  extraction feeding `neutralize()`.

## Soundness rules (non-negotiable, all WPs)
1. Never fabricate. A slot/section/guard is acted on only with structural proof;
   otherwise it is reported unresolved and the blocker stays. Fail closed.
2. Every emitted PE must reparse from disk and pass `validate_for_loader`.
3. Two tiers: a proof-gated sound `output`, and loudly-labelled `--experimental`.
4. Assumptions are labels (`AssumesCleanEnvironment`, `PreferredImageBase`, …), never
   silent.
5. Append-only reintegration stays the default (does not trip integrity over existing
   code); stripping is the alternative that removes the integrity machinery itself.

## Work packages (distinct crate/file ownership — no two agents share a file)
- **WP1 corpus** (testcorpus/*.ps1, docs): the Options bitfield map + a
  full-protection corpus (pack / import-prot / memprot / resources / reloc-strip /
  anti-debug, each in isolation and combined) across diverse DLLs. Prerequisite for
  WP2/WP3 validation.
- **WP2 imports** (`vmpx-rebuild/src/import_reconstruct.rs` + a new
  `import_resolve.rs`; `vmpx-disasm` call-site decoding; `clean_wiring.rs::derive_referenced_slots`):
  resolve protected import call-sites to names; feed the proven set; close
  `IMPORT_BLOCKER`.
- **WP3 strip** (`vmpx-pe/src/strip.rs` extensions; new `vmpx-cli/src/strip_wiring.rs`;
  `clean.rs` call-site): compute the provably-removable protector section set and
  invoke `strip_sections`; close `VMP_BLOCKER`. New CLI `strip` subcommand lives in a
  dedicated module to avoid `main.rs` churn.
- **WP4 reloc** (`vmpx-rebuild/src/reloc.rs` new; `reintegrate.rs`): generate a base
  relocation block for appended code; close `REINTEGRATION_BLOCKER`.
- **WP5 neutralize** (`vmpx-vm/src/neutralize.rs` guard extraction; new
  `vmpx-cli/src/neutralize_wiring.rs`): extract conditional guards + import-name
  provenance, wire `--neutralize`.
- **WP6 unpack validation** (`vmpx-unpack`): run on real packed samples from WP1, fix
  whatever real LZMA/table layouts break; add a non-ignored fixture.

CLI subcommand/flag additions each go in their own `*_wiring.rs` module and are
registered with a single small enum-arm edit, staged so only one agent edits
`main.rs` at a time.
