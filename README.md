# vmpx

Static analysis and recovery tooling for VMProtect-protected Windows binaries,
written in Rust. PE parsing, unpacking, VM handler discovery, lifting, and
rebuilding, split into crates under `crates/` with a single `vmpx` CLI.

This is mostly unfinished work. Unpacking, detection, PE modelling, disassembly
and the PE relayout stage are solid and verified against a real sample.
Devirtualization gets decently far on that sample: handler recovery, symbolic
execution, native continuations, TLS entry analysis and exception classification
all run, but native replacement is not finished.

## Unpacking to a loadable DLL

```bash
cargo build --release
cargo run -p vmpx-cli -- unpack packed.exe mapped.bin --conventional clean.dll
```

`mapped.bin` is the RVA-layout analysis image, byte-identical to the VMPStatic
reference dump. `clean.dll` is the same image in normal file layout, which
objdump and the Windows loader accept.

The layout stage is proven, not assumed: every mapped byte is identical between
the two files, and the only differing bytes are `SizeOfRawData` and
`PointerToRawData` in the section headers. `unpack` also walks every directory
the loader consumes (imports, exports, TLS, relocations, entry point) and fails
if any of them is not backed by file data.

What this is not: the protector sections are still present and still execute.
On the reference sample the entry point and the first TLS callback are VM code,
and three of the five exports still branch into `.vmp2` at runtime. Only the
unpacking and layout stages are complete.

## Whole-pipeline report

```bash
cargo run -p vmpx-cli -- clean packed.exe out.exe --report report.json
```

`clean` runs unpacking, VM entry discovery, finite-domain path recovery, native
continuation analysis, TLS entry analysis, exception continuation classification
and per-export protector dependency. It writes the report, emits the
conventional-layout image via `--reconstructed`, and still refuses to write
`out.exe` because the entry point and import set are not yet proven native.

## Status on the reference sample

- unpacking: byte-identical to the reference mapped image
- conventional layout: verified, loader-valid, identical loaded image
- VM entries: header entry and TLS callback both recover to proven VM exits
- TLS callback: 14 segments, 2330 operations, all four DLL reasons resolved
- entry point: process-attach and detach reach VM exits; thread notifications
  stop at a proven null read through an encrypted `.vmp3` handler
- exports: `XPluginStart` and `XPluginStop` are VM-free;
  `XPluginEnable`, `XPluginDisable` and `XPluginReceiveMessage` are not

MIT licensed.
