# Contributing

Thanks for your interest. This is a Rust workspace; everything builds with a
stable toolchain.

## Build and test

```bash
cargo build --release
cargo test --release                      # whole workspace
cargo test --release -p vmpx-vm           # a single crate (faster)
cargo run  --release -p vmpx-cli -- --help
```

Optional but recommended for iteration speed:

```bash
# cache compilations across builds
export RUSTC_WRAPPER=$(command -v sccache)
```

Before opening a PR:

```bash
cargo fmt --all
cargo clippy --all-targets --release
cargo test  --release
```

## The one rule that matters: soundness

This project's value is that its output is *trustworthy*, so the bar for any
change touching recovery, lifting, emission, or the emulator is high:

- **Fail closed.** Refuse rather than guess. A step that cannot prove something
  records a stall with its cause; it does not emit a convenient value.
- **Label, don't assume silently.** Anything taken on trust becomes a *named*
  assumption on that entry's result. An entry is only *proven* with an empty
  assumption set.
- **No inexact transform.** A simplification, fold, or rewrite that is not
  meaning-preserving is a bug, even in a display/readable path. Keep the existing
  round-trip / re-evaluation self-checks passing.
- **Observed ≠ proven.** Anything derived from the contained emulator is labeled
  *observed* (one execution), never presented as proof.

A PR that trades soundness for a bigger headline number will not be merged.

## Layout

| area | crates |
|---|---|
| file / instruction layers | `vmpx-pe`, `vmpx-disasm`, `vmpx-detect`, `vmpx-unpack` |
| VM model, recovery, CFG | `vmpx-vm` |
| lift CFG → IR → C/Rust | `vmpx-vmlift`, `vmpx-decompile` |
| native mutation | `vmpx-mutation`, `vmpx-solvers`, `vmpx-tournament`, `vmpx-verify` |
| image emission, emulation | `vmpx-rebuild`, `vmpx-recompile`, `vmpx-emulate` |
| front ends | `vmpx-cli`, `vmpx-ui` |

Stage designs live under [`docs/`](docs/). Tool overviews:
[vmpx-expanded](vmpx-expanded.md), [tigress-expanded](tigress-expanded.md).

## Scope and conduct

Keep changes focused and reviewable. Be respectful in issues and reviews. By
contributing you agree your work is licensed under the repository's MIT license.
