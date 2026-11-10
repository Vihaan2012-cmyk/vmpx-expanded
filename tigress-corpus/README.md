# tigress-corpus

The ground-truth corpus for [tigress-expanded](../tigress-expanded.md). It runs
[Tigress](https://tigress.wtf/) over a seed of C programs across a matrix of
obfuscation configs, compiles the clean and obfuscated sources, runs both on
fixed inputs, and records a `(clean, obfuscated, expected-output)` triple for
each. That triple is the oracle the deobfuscator is scored against — the thing
VMProtect never gives us.

## Why it exists

Tigress is source-to-source, so correctness is *measurable*: we own the original
program, so we know the right answer for every input. The harness manufactures
labelled inputs in bulk and marks each one `matched` (the obfuscated binary is
behaviourally identical to the original) so later work can hill-climb against a
real number instead of an assumption.

## Layout

```
tigress-corpus/
  build_corpus.py     the harness: obfuscate → compile → run → diff → manifest
  seeds/              hand-written C programs, one target function each
    arith.c           mix()      — integer/bit MBA, for Virtualize + EncodeArithmetic
    control.c         classify() — nested control flow / switch, for Flatten
    loop_hash.c       hash()     — FNV-1a loop, for loops under every transform
  out/                generated (git-ignored): binaries + manifest.json
```

## Setup

Two external tools, neither vmpx-specific:

- **Tigress** — not on `PATH` by default. Install from <https://tigress.wtf/>
  and either put `tigress` on `PATH` or pass `TIGRESS=/path/to/tigress`.
- **A C compiler** — `gcc` or `clang`. On Windows, run the harness under WSL, or
  point `ENV` at your MinGW toolchain (the harness itself is pure Python 3).

The harness fails closed: if either tool is missing it prints exactly what to
install and exits, rather than writing a half-built corpus.

## Run

```sh
python build_corpus.py
# or override any of:
TIGRESS=/opt/tigress/tigress CC=clang ENV=x86_64:Linux:Gcc:4.6 OUT=./out python build_corpus.py
```

It writes `out/manifest.json` — one record per `(seed, config, argv)` with the
clean/obfuscated binary paths, the expected return code and stdout, and
`matched`. The closing line reports how many triples preserved behaviour; any
`[DIVERGENCE]` line means a transform changed the program's output and is a bug
in the config, not in the deobfuscator.

The config matrix (`CONFIGS` in [build_corpus.py](build_corpus.py)) starts
narrow — single transforms plus a couple of combos — and is meant to be widened
once it runs clean against your Tigress build.

## What comes next

The manifest is the input to the deobfuscation pass. The engine that does the
lifting already exists on the VMProtect side (`vmpx-vm`, `vmpx-solvers`,
`vmpx-tournament`); the roadmap in [tigress-expanded.md](../tigress-expanded.md)
tracks what transfers and in what order.

See [CONTRIBUTING.md](CONTRIBUTING.md) to add seeds or configs.
