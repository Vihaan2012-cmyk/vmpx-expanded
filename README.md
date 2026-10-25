# vmpx

Static analysis and recovery tooling for VMProtect-protected Windows binaries,
written in Rust. PE parsing, unpacking, VM handler discovery, lifting, and
rebuilding, split into crates under `crates/` with a single `vmpx` CLI.

This is mostly unfinished work. Unpacking, detection, PE modelling and
disassembly are solid. The devirtualization side gets decently far: handler
recovery, symbolic execution, native continuations and exception classification
all run on a real sample. It still refuses to emit a devirtualized binary
because the original entry point and the application import set are not proven.

```bash
cargo build --release
cargo run -p vmpx-cli -- --help
```

MIT licensed.
