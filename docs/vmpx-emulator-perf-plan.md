# vmpx emulator performance plan

How to make the contained emulator faster — prioritized by payoff-per-effort for *our*
workload (VMProtect VM interpretation: flag-heavy, branch-heavy, self-modifying), not generic
emulation.

## Where we are

- Interpreter, ~16M → 155M → **~270M steps/s HOOKLESS** (pure execution, no hooks) via:
  decode-block cache, superinstruction fusion, hook-interest elision. ~17×.
- **But any real target runs at ~97–100M/s**, because servicing Windows APIs (`GetSystemInfo`,
  `VirtualAlloc`, …) *requires* the WinApiHook, which fires on every call/branch — and that is
  unavoidable for a sample that calls into the OS. Measured on the `.xpl` init: ~97M/s even with
  a "bare" `detonate()`. So **270M is a benchmark ceiling, not the working speed**; the API hook
  sets the real floor at ~100M, and the observation hook (branch watching) comes *on top* of that.
- **Snapshot/restore** already removes the single biggest practical cost — the one-time ~27B
  `.xpl` init is checkpointed and restored in ~1s, so it's never re-paid. For the actual
  workflow, this dwarfs any raw-speed gain.

## Principle: measure before optimizing

The workload is specific; profile it, don't guess. **Build a cheap profiler first** (per-handler
/ per-mnemonic cycle attribution, or sampled rip buckets) and run it on two representative
targets: the `.xpl` init, and a VM-function detonation. Every item below is a *hypothesis* until
the profile confirms the hot path. No optimization ships without a measured before/after and a
green differential (correctness is non-negotiable).

A benchmark harness (`xpl_dllmain_test`-style: fixed steps, report steps/s) is the acceptance gate.

## Tier 1 — raw-speed wins (days each, scoped, certain-ish)

### 1. Lazy flags *(best bang for the buck)*
VM obfuscation drowns in flag work — the trace shows `Clc`/`Cmc`/`Test`/`Cmp`/`Xor` as the top
mnemonics (~20K each per 400k steps). If we eagerly compute all six flags (CF/OF/SF/ZF/PF/AF)
after each arithmetic op but most are immediately overwritten or never read, that's wasted.
**Defer flag computation:** store `(op, operands)` and materialize a flag only when consumed
(a `jcc`/`setcc`/`adc`/read). Classic interpreter win, aimed exactly at the hot path.
- Expected: **1.3–1.8×** on VM code. Effort: ~days. Risk: must match x86 flag semantics bit-for-bit
  — gated by the emulator test suite + the differential. `resolve_undefined_flags` already models
  undefined-flag behaviour; lazy flags must preserve it.

### 2. Narrow the observation hook *(recovers the 270M for instrumented runs)*
The 270M→100M drop is `VmObserveHook` firing on every branch. We only need **dispatcher
re-entry**, not every `jmp`. Narrow the hook's interest to the dispatcher VAs (we already discover
them), or sample. This is the one that helps the runs that *matter* (branch observation).
- Expected: toward **2×** on observing runs. Effort: low. Risk: low (must not miss a dispatch edge).

### 3. Dispatch + hot-path audit
Confirm the decoded-block fast path is branch-minimal and dispatch is direct-threaded (not a
switch). Tighten the memory fast path (the TLB exists; verify the common aligned load/store
skips the slow path). Profile-driven micro-opt.
- Expected: **1.1–1.3×**. Effort: low–moderate.

## Tier 2 — parallelism (the big *practical* speedup)

A single emulation is a sequential dependency chain and **cannot** be split across cores. But the
work that takes hours is embarrassingly parallel:

### 4. Parallel fork-from-snapshot exploration *(the dynamic-engine win)*
For the ~126 runtime branches: a work queue of (input/branch) targets, N worker processes each
restoring the shared checkpoint and driving one target (concolic). Scales ~linearly with cores.
This — not raw interpreter speed — is how the branch hunt goes fast. The concolic engine already
has the loop; formalize the queue + N workers + result merge.
- Expected: **~N×** (N = cores). Effort: moderate. The snapshot makes it cheap (restore ~1s/worker).

### 5. Census parallelism (already have `--jobs`)
3,500 independent functions → N processes. Already supported; tune jobs against z3's ~3 GB/job RAM
(cores-vs-memory bound), pin the shared z3 binary so no worker rebuilds. Scales the corpus with cores.

## Tier 3 — the ceiling (weeks, and partly defeated here)

### 6. Hybrid block JIT/DBT
Translate hot **non-self-modifying** basic blocks to host code (QEMU-style), keep the interpreter
for self-modifying regions. The catch: **VMProtect is aggressively self-modifying** (it
`WriteProcessMemory`s its own code — literally the loop we're debugging), so JIT'ing the *VM* is
mostly defeated by invalidation. The real target would be the **plugin's own code after unpack**
(stable), not the VM dispatcher.
- Expected: multi-× on *plugin* execution, ~0 on the VM unpacker. Effort: weeks (codegen, block
  chaining, SMC invalidation). **Only worth it if, after Tier 1–2, plugin-execution time still
  dominates.** Do not start here.

## Recommendation / order

1. **Profiler + benchmark harness** (half a day) — nothing else without this.
2. **Lazy flags** (Tier 1.1) — the one raw win worth doing now; helps every run.
3. **Narrow observation hook** (Tier 1.2) — recovers speed for the runs we actually instrument.
4. **Parallel fork-from-snapshot** (Tier 2.4) — the dynamic-engine throughput win; uses all cores.
5. Re-profile. Only consider the JIT (Tier 3) if plugin execution is still the bottleneck.

## What not to do

- Don't JIT the self-modifying VM unpacker (SMC defeats it).
- Don't micro-optimize blind — profile first.
- Don't trade away determinism or observability for speed; those are the product, not the overhead.
- Don't chase raw interpreter speed as the headline — snapshot + parallelism already make the
  *workflow* fast; raw speed is a secondary, scoped win (lazy flags).
