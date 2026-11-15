# Contributing to tigress-expanded

This is the [Tigress](https://tigress.wtf/) side of the project — a static
deobfuscator built on the same engine as [vmpx-expanded](../README.md), validated
against the ground-truth [corpus](README.md). The repo-wide rules in the
[top-level CONTRIBUTING](../CONTRIBUTING.md) apply here too; this file covers the
parts specific to the Tigress work.

## The soundness bar is the same

Tigress gives us an oracle VMProtect never did — we own the original program, so
we can *measure* correctness. That raises the bar rather than lowering it: a
recovery that passes the corpus on the sampled inputs is validated on those
inputs, not proven. Keep the distinction honest.

- **Fail closed.** A pass that cannot recover a transform records a stall with
  its cause; it does not emit a plausible-looking guess.
- **Measured, not asserted.** Any claim that a transform is handled comes with
  corpus numbers — how many `(seed, config)` triples it recovers, on how many
  input vectors — not a headline.
- **Label the gap.** Where recovery is input-sampled rather than proven
  equivalent, say so on that result, the same way the VMProtect side labels an
  assumption.

## Adding a seed

A seed is a small, deterministic C program with one target function and a `main`
that prints a function of `argv` (so output is reproducible and diffable).

1. Drop `seeds/<name>.c` next to the existing three. Keep it self-contained —
   libc only, no I/O beyond `printf`, output determined solely by `argv`.
2. Register it in `SEED_SPEC` in [build_corpus.py](build_corpus.py): the target
   function name and a list of fixed `argv` vectors (include the no-arg default).
3. Run the harness and confirm every `matched` is true before committing — a
   seed whose own obfuscation diverges is a broken seed.

Aim each seed at a transform the matrix exercises: arithmetic/MBA, control flow,
loops, literals, opaque predicates.

## Adding a config

Configs live in `CONFIGS` in [build_corpus.py](build_corpus.py) — a name mapped
to the Tigress transform args for a target function. Start from a single
transform, confirm it compiles and preserves behaviour across all seeds, then add
combinations. A config that makes Tigress or the C compiler fail is skipped with a
reason printed; a config that changes program output prints `[DIVERGENCE]` and
must be fixed, not committed.

## Before a PR

```sh
python build_corpus.py          # must end with no [DIVERGENCE] lines
```

If your change also touches the shared Rust engine, run that crate's tests too —
see the [top-level CONTRIBUTING](../CONTRIBUTING.md).

By contributing you agree your work is licensed under the repository's MIT license.
