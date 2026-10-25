# vmp-re

Static analysis and recovery tooling for VMProtect-protected Windows binaries,
written in Rust. Covers PE parsing, unpacking, VM handler discovery, lifting and
rebuilding, split into crates under `crates/` with a single `vmpx` CLI.

This is mostly unfinished work. A few stages are solid (PE model, unpacking,
detection, disassembly), the devirtualization and rebuild side is partial and
still refuses to emit output when it cannot prove the result. Decent progress,
not a working unpacker.

```bash
cargo build --release
cargo run -p vmpx-cli -- --help
```

MIT licensed.
