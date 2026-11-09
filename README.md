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

The default path is **static**. A contained software emulator supplies the
separate Observed tier, and that is the only thing that runs the code. **The
protected sample is never executed natively.**

## Where it stands

Measured 2026-11-08 on one VMProtect 3.11-protected x64 binary, sweeping all 814
discovered VM entry stubs under `--serve-ro` with
`VMPX_FOLLOW_CROSS_VMP_SECTION=1` and `VMPX_INSTRUCTION_LIMIT=2048`:

| | count |
|---|---|
| VM entry stubs discovered | 814 |
| **N1** — CFG-complete under any labeled assumption | **699** (85.9%) |
| **N2** — complete with an *empty* assumption set | **253** (31.1%) |
| incomplete | 113 |
| errors / panics | 2 / 0 |

This is the first sweep to reach 814/814. Earlier runs stopped near 800 because a
single entry spun without ever returning, and the per-entry wall-clock budget did
not bound it; the cause was an input-domain scan with no visited set, walking a
hash-consed expression DAG as a tree.

Of the 699, **253 carry no assumption and 446 rest on
`AssumesExternalStoresDisjointFromVmStack`**, which is labeled unproven. That
split matters more than the headline: N1 can always be raised by widening the
assumption set, so N2 is the number that carries weight.

Every one of the 113 incomplete entries is a *semantic* stall:

```
incomplete entries by stop reason: {"BlockVisits": 5, "stall": 108}
```

Zero `WallClock` and zero `InstructionLimit` — the time-and-budget axis is
closed, so no increase in any budget will convert them.

### What blocks the remaining 113

By each entry's first stall:

| blocker | entries | what it needs |
|---|---|---|
| handler has no recovered specification | 36 | prove a conditional predicate opaque before crossing it |
| imprecise external-memory store | 31 | semantics for the address-computation intrinsics |
| proven-null read | 25 | a post-init snapshot that reaches the slot |
| branch target depends on unproven input | 18 | memory precision (96 of 98 are reads of havoc'd memory) |
| VM-stack-stride loop | 1 | — |

Two things are worth knowing before picking one up.

**Three of the four are the same problem.** The imprecise store, the unproven
branch input and the null read are all memory modelling — 74 of 113 entries.
Only the handler class is about control flow.

**The cheap fixes are wrong, not weak.** Each of these classes has had several
plausible fixes tried and reverted, and in every case the result was an
incorrect CFG rather than a conservatively labeled one:

- Walking a handler trace past a conditional branch (either one arm, or both
  arms requiring agreement) recovers postlude shapes that no execution performs.
  The branch is sometimes a junk predicate and sometimes a guard the recogniser
  reads, and the trace alone does not separate them.
- Widening the external-store disjointness gates, or bounding an unsupported
  address node by its own width, buys nothing: those store addresses are
  `f(frame_pointer)` for an opaque `f`, which has no provable range relative to
  the VM stack at all.

Each attempt and its measurement is recorded in the code at the place it was
made, so the next reader meets it rather than repeating it.

Not re-measured since 2026-11-04 and so quoted as of that date: **385** entries
raised to C, of which **38** pass the IR round-trip self-check. The gap is the
point — 346 fail only because the checker cannot evaluate an unknown (an opaque
intrinsic such as a timestamp read, or an untracked load), and **zero** fail from
an actual disagreement. The decompiler does not contradict its own IR anywhere,
but most of its output is unverified and is labeled that way rather than counted
as proven.

This is one sample, not a benchmark. `--serve-ro` serves read-only image bytes
under a labeled assumption, which raises N1 and suppresses N2; without it N1 on
this sample falls to roughly half.

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
vmpx report <image> --all --threads 16 --serve-ro

# raise recovered functions to C-like pseudocode
VMPX_DECOMPILE_SERVE_RO=1 vmpx decompile <image> --rva 0x...

# discover VM entry stubs and bootstrap profiles
vmpx vm-list <image>
```

`report` and `decompile` write nothing and execute nothing; they analyse.

There is also a desktop front end, `vmpx-ui`, which drives the CLI and renders
its JSON reports — pick a binary, tick the flags, set the environment, run:

```bash
cargo run --release -p vmpx-ui
```

### Settings whose absence costs results silently

Several knobs are not implied by anything, and leaving one out does not produce
an error — just a worse number:

- **`report` needs `--serve-ro`.** Without it N1 on the sample above falls from
  658 to 359. `VMPX_DECOMPILE_SERVE_RO` does *not* apply to `report`.
- **`decompile` needs `VMPX_DECOMPILE_SERVE_RO=1`.** Without it the CFG reads as
  incomplete and you get one line saying so instead of code. There is no
  `--serve-ro` flag on `decompile`.
- **`--dynamic` needs `--volatility-snapshot`.** Without the snapshot it takes
  the bare-detonation branch, which never finishes the multi-billion-step init
  and so observes nothing — reporting a silent zero rather than failing.
- **No snapshot exists by default, and no subcommand writes one.** Build it with
  the `xpl_init_steps` example and `XPL_SNAPSHOT_OUT`; it takes a few minutes.
  Raise `XPL_STACK_KB` while doing so — at the 8 MiB default the init overflows
  the guest stack and the snapshot captures a truncated state.
- **The accelerators are on by default and the init needs them.** The protector
  spends its init decompressing with LZMA, and `XPL_NOACCEL=1` / `XPL_NOLZMA=1`
  turn off the folds that make that finish.
- **`VMPX_INSTRUCTION_LIMIT` is a multiplier, not an absolute**, and `report`'s
  base is 512 while `decompile`'s is 100000.

### Emulator throughput, and why a step is not the unit

The init is one routine. Profiling the first TLS callback puts **98.3% of its
steps inside a single 0x4000-byte window**, spread over 963 distinct addresses —
VMProtect's LZMA decompressor, flattened and padded with `jmp +0` and flag-only
junk. Three loop folds (`accelerate_checksums`, plus `accelerate_literals` for
the literal decoder) run the recognized loops natively.

Two things about measuring this are easy to get wrong, and both were got wrong
before they were measured:

- **Steps are not comparable across configurations.** A fold retires one step
  for an arbitrary number of guest instructions, so the steps/s figure *falls*
  when a fold starts working. On a 400M-step init the folded run does ~1,021M
  guest instructions in 3.5s where the unfolded run does 400M in 2.0s — 292M
  against 197M guest instructions per second. Compare decompressed output per
  wall-clock second, which is what `bytes=` in the fold line reports.
- **Recognition is probed at every block start, so a MISS has to be cheap.**
  Each recognizer originally walked the loop before testing anything, which cost
  more than the folding bought: the literal fold measured +22% progress per step
  and 35% *slower* in wall-clock. Rejecting on the first instruction's opcode,
  capping the pre-match padding chase, and a 32KiB negative filter in front of
  the shape caches took the same configuration from 145k to 544k output bytes/s
  with byte-identical output.

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

## Legal and ethical use

This software is intended exclusively for:

- Defensive malware analysis in isolated laboratory VMs
- Academic and classroom instruction in reverse engineering
- Incident-response work where the analyst has lawful possession of the sample
- Security research published openly

You must not use this tool:

- Against software you do not own or do not have written permission to analyse
- To defeat licensing or DRM on commercial software
- To distribute unpacked malware binaries to parties who would not otherwise
  have lawful access

Nothing here is a grant of permission to analyse a particular binary. Lawful
possession and, where it applies, written authorisation are the user's
responsibility.

MIT licensed, as the upstream project.
