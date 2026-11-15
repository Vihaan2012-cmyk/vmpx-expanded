# vmpx-expanded

An expansion of [vmpx](https://github.com/skyboundfs/vmpx) — static analysis and
recovery tooling for VMProtect-protected Windows binaries, written in Rust — aimed
at taking the pipeline all the way to clean, behavior-preserving devirtualized output.

The original project recovers a great deal (unpacking, detection, PE modelling,
disassembly, VM handler discovery, symbolic execution, native continuations) but
deliberately refuses to emit a devirtualized binary until the original entry point
and application import set are proven. This fork plans and implements that remaining
work in sound, independently testable stages.

The roadmap and the full design of the first stage (native reintegration + image
emission) live in
[`docs/superpowers/specs/`](docs/superpowers/specs/2026-10-29-vmpx-sp1-native-reintegration-design.md).

## Capabilities

Recovery is reported in **honest tiers**, never as a single pass/fail — every
completed function is labeled with the assumptions its recovery depended on:

- **N1 — CFG-complete:** every path through the virtualized function reaches a VM
  exit, under any labeled assumption.
- **N2 — proven:** N1 with an *empty* assumption set.
- **N3 — proven-equivalent:** bit-exact against a ground-truth build via the SMT
  equivalence gate (`diff-original`).
- **Experimental / observed tiers:** completions that depend on a single
  detonation (observed branch targets, recursion depths, globals) or on an
  unproven assumption; counted and labeled separately, never folded into N2.

What the engine recovers from a VMProtect-protected image:

- **VM handler discovery and dispatch-codec recovery** — the per-handler
  bytecode-fetch / key-schedule, recovered structurally (never guessed).
- **Symbolic VM execution** with cross-profile transitions, nested VM entries
  (`pushfq`- and `push`-led re-entry stubs), and cut-point **loop segmentation**
  for input-dependent loops.
- **VM↔native SSE/FP chains** — VMProtect runs floating-point natively around the
  integer VM; the lifter follows the VM→native→VM hand-off and models Win64 calls.
- **Anti-analysis, handled soundly (not just detected):** CPUID / RDTSC /
  timestamp and PEB/TEB/API-return environment gates are resolved by *both-target
  over-approximation* (no path dropped, no guard rewritten); the init-time
  rewrite of declared-read-only `.vmp` pages is handled by a volatility veto plus
  post-init **snapshot serving**.
- **Ground-truth differential (`diff-original`):** a function recovered from the
  protected binary is compared, path by path, against the same function lifted
  from an unprotected build, and proven bit-exact under an explicit
  calling-convention policy — a false *proven* is treated as catastrophic, so the
  gate fails closed on any doubt.

### Commands

```bash
cargo build --release

# Tiered static report over discovered VM entries.
cargo run -p vmpx-cli -- report <image> --all --threads 4

# Serve read-only VM tables / post-init snapshot bytes (labeled, opened tier),
# and run the contained-emulator detonation + observed-tier re-sweep.
cargo run -p vmpx-cli -- report <image> --serve-ro \
    --volatility-snapshot <post-init-snapshot> --dynamic

# Ground-truth differential against an unprotected build.
cargo run -p vmpx-cli -- diff-original <protected> <original>
```

Emitting a clean, fully devirtualized binary (sound OEP + import reconstruction,
then image emission) remains the goal the staged roadmap is working toward; the
analysis, recovery, and equivalence machinery above is what exists today.

```bash
cargo run -p vmpx-cli -- --help
```

MIT licensed, as the upstream project.
