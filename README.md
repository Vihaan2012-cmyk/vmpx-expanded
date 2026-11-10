<h1 align="center">expanded</h1>

<p align="center">
  <b>Static, behaviour-preserving recovery of protected x64 Windows binaries.</b><br>
  Strip the protection, recover the real program — without ever running the sample.
</p>

<p align="center">
  <code>Rust</code> · <code>static-first</code> · <code>fail-closed</code> · the sample is never executed natively
</p>

---

A family of static deobfuscators, one per protector. Each takes a protected
binary and reconstructs a clean, behaviour-preserving program — recovering the
entry point, imports, and control flow, and lifting virtualized or mutated code
back to readable form. Nothing here executes the protected sample; the only
component that runs code is a contained software emulator that produces a
separate, clearly-labeled *observed* result.

## Tools

| tool | target | status | docs |
|---|---|---|---|
| **vmpx-expanded** | VMProtect 3.x (x64 PE) | ✅ available | [vmpx-expanded.md](vmpx-expanded.md) |
| **tigress-expanded** | [Tigress](https://tigress.wtf/) C obfuscator | 🧭 planned | [tigress-expanded.md](tigress-expanded.md) |

**[vmpx-expanded](vmpx-expanded.md)** is the complete tool today: it strips the
protector, recovers the OEP and the encrypted import table, relocates mutated
native code, and devirtualizes the guards — in one command.

```sh
vmpx auto protected.dll clean.exe --dir out --log
```

**[tigress-expanded](tigress-expanded.md)** is the planned next target. Because
Tigress is a *source-to-source* obfuscator, it comes with something VMProtect
never gives us — a ground-truth oracle — which makes most of it a lower-risk
target than the VMProtect work. See its page for the difficulty assessment.

## Principles (shared by every tool)

- **Static-first.** The protected sample is never executed natively. Analysis is
  offline; the contained emulator is the only thing that runs code, and its
  output is labeled *observed*, never *proven*.
- **Fail closed.** Every step refuses rather than guessing. A stall is recorded
  with its cause; the entry stays incomplete.
- **Labeled, not silent.** Anything unproven becomes a named assumption attached
  to that result. A function counts as *proven* only when it closes with an
  **empty** assumption set.
- **Self-checked emission.** Recovered functions carry round-trip and
  re-evaluation checks before they are trusted.

See [`SECURITY.md`](SECURITY.md) for intended use and reporting, and each tool's
page for its full pipeline and soundness discipline.

## Build

```bash
cargo build --release
cargo run -p vmpx-cli -- --help
```

## Legal and ethical use

Intended exclusively for defensive malware analysis, academic instruction,
incident response on lawfully-held samples, and openly-published security
research. **Do not** use it against software you do not own or lack written
permission to analyse, or to defeat licensing/DRM on commercial software. Lawful
possession and, where it applies, written authorisation are the user's
responsibility. Full terms in each tool's page and in [`SECURITY.md`](SECURITY.md).

MIT licensed.
