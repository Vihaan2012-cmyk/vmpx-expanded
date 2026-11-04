# vmpx-expanded

An expansion of [vmpx](https://github.com/skyboundfs/vmpx) — static analysis and
recovery tooling for VMProtect-protected Windows binaries, written in Rust — aimed
at taking the pipeline all the way to behavior-preserving devirtualized output.

The original project recovers a great deal (unpacking, detection, PE modelling,
disassembly, VM handler discovery, symbolic execution, native continuations) but
deliberately refuses to emit a devirtualized binary until the original entry point
and application import set are proven. This fork implements that remaining work in
sound, independently testable stages, and now recovers control-flow graphs and
raises virtualized functions to C-like pseudocode.

Everything here is **static**. The protected sample is never executed natively.

## Where it stands

Measured 2026-11-04 on one VMProtect 3.11-protected x64 binary with 814 discovered
VM entry stubs, under `--serve-ro`:

| | count |
|---|---|
| VM entry stubs discovered | 814 |
| **N1** — CFG-complete under any labeled assumption | **581** |
| **N2** — complete with an *empty* assumption set | **248** |
| entries raised to devirtualized C | **385** |
| of those, passing the IR round-trip self-check | **38** |

Read those last two rows together, because the gap is the point: of the 385
functions emitted, 38 are confirmed by the round-trip check and **346 fail only
because the checker cannot evaluate an unknown** (an opaque intrinsic such as a
timestamp read or an untracked memory load). Zero fail from an actual
disagreement. So the decompiler does not currently contradict its own IR
anywhere — but most of its output is unverified, and we label it that way rather
than counting it as proven.

This is one sample, not a benchmark. `--serve-ro` serves read-only image bytes
under a labeled assumption, which raises N1 and suppresses N2; the sound-mode
figures move accordingly and are not quoted here because they are not current.

## The three tiers

The report is deliberately three numbers rather than one, because "recovered" is
not a single property:

- **N1 — CFG-complete.** The VM control-flow graph closed with no stalls, under
  an explicitly *labeled* set of assumptions. Every assumption is named in the
  output (`AssumesImageProtectionStable`,
  `AssumesExternalStoresDisjointFromVmStack`, `AssumesCrossVmSectionContinuation`,
  and so on), so a result is never silently conditional.
- **N2 — PROVEN.** The same, with an **empty** assumption set. Nothing was taken
  on trust. N2 is a subset of N1 and is the number that actually carries weight.
- **N3 — proven-equivalent.** Recovered semantics checked against the recompiled
  replacement. Not currently wired at the PE level.

Assumptions are *conceded*, never assumed away. Closing a loop by shifting the
VM stack base, for example, concedes frame privacy, so such an entry can reach N1
and is barred from N2 by construction.

## Soundness discipline

The parts worth knowing before trusting any output:

- **Fail closed.** Every recovery step refuses rather than guessing. A stall is
  recorded with its cause and the entry stays incomplete.
- **Labeled, not silent.** Anything unproven becomes a named assumption attached
  to that entry's result.
- **Three self-checks** per emitted function: an IR round-trip (the structured
  program against its own IR on random vectors), a C-text re-evaluation under real
  C typing and UB rules, and a linkage check against the independently unrolled
  recovered paths.
- **Honest widening.** Where abstract interpretation must generalize, it widens to
  an explicit unknown rather than narrowing to a convenient value, and the
  resulting imprecision shows up as a refusal downstream instead of as output.

## Build

```bash
cargo build --release
cargo run -p vmpx-cli -- --help
```

Useful entry points:

```bash
# tiered report over every discovered VM entry
vmpx report <image> --all --threads 8

# raise recovered functions to C-like pseudocode
vmpx decompile <image> --rva 0x...

# discover VM entry stubs and bootstrap profiles
vmpx vm-list <image>
```

`report` and `decompile` write nothing and execute nothing; they analyse.

## Crates

`vmpx-pe`, `vmpx-disasm`, `vmpx-detect`, `vmpx-unpack` cover the file and
instruction layers. `vmpx-vm` holds the VM model, handler/codec recovery,
symbolic execution and CFG recovery; `vmpx-vmlift` lifts a recovered CFG to IR;
`vmpx-decompile` raises IR to C; `vmpx-verify` and `vmpx-solvers` provide the
equivalence machinery; `vmpx-mutation`, `vmpx-demutate` and `vmpx-tournament`
address native mutation; `vmpx-rebuild`, `vmpx-recompile` and `vmpx-emulate`
cover image emission and contained emulation. `vmpx-cli` is the front end.

## Design documents

Stage designs live in [`docs/superpowers/specs/`](docs/superpowers/specs/) —
native reintegration, generalized OEP recovery, the mutation engine, and the
dynamic engine.

MIT licensed, as the upstream project.
