# "loop changes the VM stack pointer" — recursion-as-loop investigation and verdict

Date: 2026-10-31
Branch base: `sp2-oep-generalized` (HEAD 29027a6)
Scope: investigate-first. No pipeline changes; this document + evidence only.
Owned files touched for recovery: none (fail-closed behaviour is correct).

## TL;DR

The self-recursive bigcorpus functions (`rmath_fib/ack/binom/pow_`,
`rmisc_hanoi/part/arr_max`, and their `.repK` variants) stall with

```
loop cannot be closed: loop changes the VM stack pointer
```

VMProtect compiles the recursion **not** as a native `call` back into the VM
entry, but as a **single bytecode loop that manages an explicit push-down stack
on the VM stack**. Each loop iteration pushes a fresh activation frame
(decrementing VSP by a **proven constant −64 bytes**) and the return phase pops
frames back off. The loop-widening machinery refuses because `a.vsp != r.vsp`.

The constant VSP delta is **not** the real obstacle. The real obstacle is that
the loop is an **unbounded push-down stack**: the set of live tracked memory
cells **grows by one frame (~48–64 bytes) every iteration and never reaches a
fixpoint**, the recursion depth is **data-dependent on symbolic input**, and the
return phase **reads every saved frame back** (so frames cannot be dropped or
havoc'd). Phase 2's proposed remedy — carry a constant-delta VSP and relocate a
*fixed window* of loop-carried cells — does not apply, because the live cell set
is not a fixed window; it is a monotonically growing stack.

**Verdict: genuinely unbounded / needs-runtime. Fail closed is correct. No
Phase-2 fix was implemented.** An empirical experiment (below) proves the
abstract state does not converge even when the VSP guard is lifted.

## Reproducers

`E:/VMPX/testcorpus/bigcorpus/protected/rmath.virt.dll`,
`.../rmisc.virt.dll` (+ `.rep1/2/3`). Export → VM-entry RVAs (from
`out/<mod>.report.json`):

| function       | entry RVA   | loop head(s)                 |
|----------------|-------------|------------------------------|
| rmath_fib      | `0xd5fc7`   | `0x1800F0E57`, `0x18005FFB0` |
| rmath_pow_     | `0x2aca0`   | `0x1800EF4CE`                |
| rmath_ack      | `0xcbd70`   | `0x18006A86A`                |
| rmath_binom    | `0x1d640`   | `0x1800DC873`                |
| rmisc_hanoi    | `0xc9a03`   | `0x18000DF34`                |
| rmisc_part     | `0xc07b4`   | `0x18009F629`                |
| rmisc_arr_max  | `0xba316`   | `0x1800A6184`                |

Driver: `cargo run -q --release -p vmpx-vm --example cfgdiag -- <sample> <rva>`.

Source (`E:/VMPX/testcorpus/bigcorpus/src/rmath.c`), e.g.:

```c
u32 rmath_fib(u32 n) { n &= 15; return n<2 ? n : rmath_fib(n-1) + rmath_fib(n-2); }
```

(Note: `rmath_gcd` is **tail**-recursive, lifted by VMProtect as an iterative
loop with no net per-iteration VSP change, and already completes. Only the
genuinely recursive functions stall here.)

## How the VM encodes recursion

- **Zero native calls.** `cfgdiag` reports `native_call_blocks=0` for every
  recursive function. There is no `call` back into the export's VM entry. The
  recursion is inlined into one VM program.
- The program is a loop. For `rmath_fib` the CFG has a back-edge pair
  `0x1800F0E57 ⇄ 0x18005FFB0` (and another `0x1800CED74 ⇄ 0x18001E7D8`). The
  loop body pushes a VM activation frame, "calls" (continues into the body with
  a decremented VSP), and on the other branch pops a frame and continues the
  return.
- Saved native registers (`native_rbp_initial`, `native_rdi_initial`,
  `native_rflags_initial`, …) and prior-frame values are spilled into the VM
  stack at VSP-relative byte addresses — this is a software-managed call stack,
  not the native one.

### The per-iteration VSP delta IS a proven constant (−64)

Captured at the loop heads (VSP expressions, reduced):

```
block 0x1800F0E57:  a.vsp = vsp_initial
                    r.vsp = ((vsp_initial + 120) - 40) + (-144) = vsp_initial - 64
block 0x18005FFB0:  a.vsp = ((vsp_initial+120)-40) + (-136)     = vsp_initial - 56
                    r.vsp = (reduces to)                           vsp_initial - 120
```

Both loop heads show a **constant −64-byte** VSP decrement per iteration. So the
condition Phase 2 asks for ("per-iteration VSP delta is a proven constant") is
met. It is not sufficient.

### Why a constant delta does not make it liftable: the live state is unbounded

Captured tracked-cell counts across successive widen attempts at the loop heads
(context slots are a fixed 24 at offsets 0..184 step 8; the **memory** cell set
is what grows):

```
WIDEN at 1800f0e57: a.mem=216 r.mem=216
WIDEN at 18005ffb0: a.mem=216 r.mem=264     (+48 = one pushed frame)
WIDEN at 1800f0e57: a.mem=216 r.mem=272
WIDEN at 18005ffb0: a.mem=264 r.mem=328     (+64)
WIDEN at 1800f0e57: a.mem=272 r.mem=336
WIDEN at 18005ffb0: a.mem=328 r.mem=392     (+64)
WIDEN at 1800f0e57: a.mem=336 r.mem=400     ... climbs without bound
```

Each iteration adds a fresh activation frame at **new** VSP-relative addresses,
and the stored values nest one level deeper per frame (e.g. a cell holds
`vm_flags_sub_w64(vsp_initial+120, 40)`, the frame below holds
`(vsp_initial+120)-40`, …). There is:

1. **No fixed window to relocate.** The live set is `{initial frame} ∪ {all
   pushed frames so far}`, growing by ~48–64 bytes per iteration — not a
   translating fixed-size window. Re-expressing cells relative to the moving VSP
   makes the *active* frame invariant but leaves the accumulated saved frames
   growing.
2. **Data-dependent, unbounded depth.** Recursion depth is a function of the
   symbolic input (`n`). The `&15` mask that bounds it at runtime is a VM
   operation the static domain sees only symbolically; no static bound exists
   without a recursive-procedure summary.
3. **Saved frames stay live.** The return phase reads every saved frame back, so
   the deep stack cannot be soundly dropped or havoc'd to bound the state.

This is a genuine **push-down system**. A finite abstract-interpretation state
with the current exact byte-addressed memory model has no fixpoint for it.

## Decisive experiment: lifting the VSP guard does not help

To rule out "the VSP check is merely over-conservative," the guard in
`widen_state` (`crates/vmpx-vm/src/cfg.rs`) was temporarily replaced (behind an
env flag, **reverted afterwards — no committed code change**) so that a differing
VSP widens to a fresh TOP (havoc) register and memory/context generalization
proceeds. Result on `rmath_fib`:

- The memory-cell set still **grows monotonically and never converges**
  (216 → 264 → 272 → 328 → 336 → 392 → 400 → … ; memory epoch climbed past 2500).
- Because VSP is now TOP, VSP-relative load addresses become havoc-dependent, so
  the stall simply **migrates** to:
  `unresolved branch target: ... memory_load_epoch_NNNN (untracked memory read):
  vm_flags_sub_w32 operand is not an affine view of one register`.
- The CFG remains `complete=false`.

So lifting the guard does not lift the recursion; it only relabels the failure.
The guard is a correct, early fail-closed for an abstract state that cannot
converge.

## What a dynamic engine would observe

With a concrete input (say `fib(10)`), the recursion depth is concrete and
bounded, every pushed frame lives at a concrete address, and each VSP-relative
load resolves to a concrete byte. A concolic/dynamic executor would unroll the
loop exactly `depth` times and terminate. The unbounded-ness is strictly an
artefact of static symbolic input; it is exactly the information a dynamic run
supplies.

## Verdict and recommendation

- **Static feasibility: no** (for the generic engine). Soundly lifting this
  needs either (a) a **dynamic/concolic** pass that supplies the concrete
  recursion depth, or (b) a **recursive-procedure-summary / push-down**
  abstraction that recognises the loop as a call, summarises the callee, and
  applies the summary at the recursive site — a research-grade re-architecture,
  not a loop-widening extension.
- **Fail closed is correct.** Keep the `widen_state` VSP guard as-is. Do not
  fabricate a recursion depth or a synthetic bound.
- No Phase-2 fix was implemented, per the investigate-first contract.
