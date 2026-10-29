# SP2 Task 3 report: per-function live-in inference

## Built
New `crates/vmpx-vmlift/src/live_in.rs`, re-exported from `lib.rs`:
- `RegisterAccess { register: X64NativeRegister, kind: AccessKind::{Read,Write} }` (+ `read()/write()` ctors).
- `infer_live_in(&[RegisterAccess]) -> BTreeSet<X64NativeRegister>`: registers whose first access is a Read. Sorted/deterministic via register Ord.
- `calling_convention(family, frame) -> EntryCallingConvention { live_in, stack_cursor }`: thin adapter. For each path, `collect_path_effects`; every `native_<reg>_initial` symbol mentioned by a store address/value or call argument is a Read event; union over paths. `stack_cursor` = `frame.cursor_architectural_register`.

## Event representation (why)
A flat ordered list of (register, Read|Write). Simplest faithful form; read-modify-write = Read then Write. The existing effects are symbolic expressions over initial-register symbols, so writes are already folded in and every initial-symbol mention is by construction a read-before-write; the wrapper therefore only emits Reads. The pure function still handles Writes so it is correct for any ordered trace.

## TDD
11 unit tests (empty, read-before-write, write-then-read, read-only, untouched/write-only, repeated reads, later write doesn't hide earlier read, earlier write shields, multi-register independence, order matters, sorted/deterministic). Tests and impl were written together; RED was demonstrated afterwards by mutating the read check to `if false`: 8 of 11 failed; restored => 11/11 GREEN.

## Verification
`cargo test -p vmpx-vmlift`: pass. `cargo test --workspace`: 264 passed, 0 failed. `cargo build --workspace` clean. No clippy findings in live_in.rs (pre-existing warnings only in examples/emitcheck.rs).

## Proven vs deferred
Proven: `infer_live_in` semantics (unit tests). Compiles/links: the wrapper. Deferred: wrapper behavior on real samples is observed out-of-band (no corpus, no unit test per brief). Task 5 wiring (per-entry inputs in reintegration) not done; existing pipeline behavior unchanged (clean.rs untouched).

## Concerns
- Wrapper only inspects stores and call arguments. Registers that influence only the exit state/return value or control flow are not seen, so `live_in` may under-report; the brief allows binding more than live-in, and Task 5 should not DROP inputs based on this alone without extending to exit-state expressions.
- `stack_cursor` register's own `native_<reg>_initial` is the stub's entry value (see replacement_inputs); it may appear in `live_in` as well; Task 5 must keep the existing rewrite handling.
- `cargo fmt --all` reformats unrelated files in the repo (clean.rs, reintegrate.rs, ui); I reverted those and did not commit them.
