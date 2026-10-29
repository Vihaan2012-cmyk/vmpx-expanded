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

```bash
cargo build --release
cargo run -p vmpx-cli -- --help
```

MIT licensed, as the upstream project.
