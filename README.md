# vmpx

Static analysis and recovery tooling for VMProtect-protected Windows binaries,
written in Rust. PE parsing, unpacking, VM handler discovery, lifting, and
rebuilding, split into crates under `crates/` with a single `vmpx` CLI.

This is mostly unfinished work. Unpacking, detection, PE modelling and
disassembly are solid. The devirtualization side gets decently far: handler
recovery, symbolic execution, native continuations and exception classification
all run on a real sample. It still refuses to emit a devirtualized binary
because the original entry point and the application import set are not proven.

What does work end to end:

```bash
cargo build --release
cargo run -p vmpx-cli -- unpack packed.exe mapped.bin --conventional clean.exe
```

`--conventional` writes the unpacked image in normal file layout, which the
Windows loader and objdump both accept. It is a layout transform, not a
devirtualization: protector sections and VM handlers are still in there.

```bash
cargo run -p vmpx-cli -- clean packed.exe out.exe --report report.json
```

`clean` runs the whole pipeline and writes a report. It refuses to write
`out.exe` until the remaining proofs land.

MIT licensed.
