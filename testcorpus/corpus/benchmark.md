# SP6 Mutation Benchmark — ground-truth oracle dataset (WP12)

The mutation solver tournament (SP6, see
`docs/superpowers/specs/2026-10-30-vmpx-sp6-mutation-engine-design.md`, §2/§5/§7)
is scored by **differential input-testing against ground truth**: for each
candidate simplification, thousands of concrete inputs are run through the
original and the candidate and the outputs compared. This directory holds the
**clean-side ground truth** — the sound input→output oracle every candidate is
measured against.

This WP provides the *clean side only*. The **obfuscated-side** expressions
(lifted from mutation-protected native code) are produced later by the
native-lift work (**WP5 — native CFG recovery**). A benchmark instance is a
pair `(clean expr, obfuscated expr)`; WP12 pins down the clean expr and its
oracle, WP5 supplies the obfuscated expr, and the tournament checks that a
solver's simplification of the obfuscated side agrees with this oracle on every
input.

## Files

| File | Role |
|------|------|
| `gen-corpus.py` | Generator. Emits `N` known functions `f000…` as well-defined unsigned expressions over params `a,b,c` at width 32/64. Writes `gen_manifest.json` (abstract exprs) and `gen_corpus.c` (compilable). Deterministic for a given `(N, seed)`. |
| `gen_manifest.json` | Clean-side source of truth: `{ "fNNN": {"expr", "params", "width"}, … }`. The `expr` is the exact clean expression; consumed by `gen-oracle.py`. |
| `gen-oracle.py` | **Ground-truth evaluator.** Reads the manifest, evaluates every function over the standard input set with exact unsigned width-W semantics, writes `gen_oracle.json`. This is the sound oracle. |
| `gen_oracle.json` | The input→output oracle table (schema below). |
| `gen_corpus.c` | One exported C function per manifest entry; the input to `build-gen.ps1`. |
| `build-gen.ps1` | Compiles `gen_corpus.c` and drives VMProtect protection (virt / **mutation** / ultra). The protect step needs VMProtect and is the user's step. |

## Generate the corpus at scale

```sh
# 1. Generate N known functions (clean side). Deterministic; pick a seed.
python gen-corpus.py 2000 --seed 1234567
#    -> gen_manifest.json, gen_corpus.c

# 2. Build the sound oracle from the manifest.
python gen-oracle.py
#    -> gen_oracle.json
```

```powershell
# 3. Compile + protect in mutation mode (user's step; needs VMProtect).
#    Produces the obfuscated native binary the tournament/WP5 lift from.
./build-gen.ps1 -Mode mutation
```

`gen_oracle.json` is the **input-testing ground truth**: the tournament compares
each candidate's output against `cases[].out` for the matching `in` tuple and
requires 1.0 agreement for a candidate to be eligible (see spec §4, the
`Scorer`). It does **not** encode the obfuscated form — that comes from WP5.

## `gen_oracle.json` schema

```jsonc
{
  "fNNN": {
    "width":  32 | 64,            // unsigned bit width
    "params": ["a", ...],         // 1–3 params, in order
    "cases": [
      { "in": [v0, v1, ...], "out": v }, // in[] matches params[]; all unsigned ints
      ...
    ]
  },
  ...
}
```

All integers are unsigned (`0 .. 2^width - 1`). `out` is the function evaluated
on `in` with exact unsigned width-W semantics.

## Input set composition (per function)

Built from a per-width **interesting-value set** and combined deterministically
(seed derived from the function name, so the oracle is reproducible):

- **Interesting values** (94 distinct at width 32, 190 at width 64): `0`, `1`,
  small `2,3`, `MAX`/all-ones (`2^w-1`), the sign boundary `2^(w-1)` and
  `±1` around it, every single-bit `2^k`, and every `2^k-1` / `2^k+1`.
- **all-params-equal** — `(v,v,…)` for each interesting value.
- **one-param-swept** — one param over every interesting value while the others
  are held at each baseline in `{0, 1, MAX}` (multi-param functions).
- **edge×edge cross terms** — random tuples drawn from the interesting set
  (default 64), to exercise simultaneous edge values across params.
- **uniform-random full-width tuples** — default 64, over the whole `[0, 2^w)`
  range.

Duplicate tuples are removed. Typical sizes: 1-param/w32 ≈ 158 cases;
3-param/w32 ≈ 1059; 1-param/w64 ≈ 254; 3-param/w64 ≈ 2019. Counts of the random
components are tunable (`--random`, `--edge-random`).

## Soundness

The oracle matches C unsigned semantics **exactly** at the declared width:

- `+ - *` reduce mod `2^width`; `& | ^` are bitwise; `~x == x ^ mask`; unary
  `-x == (2^width - x) mod 2^width`.
- Shifts are logical; `<<` is reduced mod `2^width`. The result of **every**
  operation (not just the final one) is masked to `2^width`, so compositions
  like `(a << 4) >> 2` are modelled correctly (e.g. width-8, `a=0xFF` → `0x3C`,
  which a final-only mask would get wrong as `0xFC`).
- `gen-oracle.py` parses the C expression with a hand-written recursive-descent
  parser using **C** operator precedence (not Python's, which differs for the
  bitwise/shift operators) and applies the per-op width mask.
- The generator emits only well-defined unsigned ops (`+ - * & | ^ ~ << >>`)
  with shift amounts in `[0, width)` (constant, or a param masked to
  `width-1`), so there is no C undefined behaviour. In `gen_corpus.c` every
  numeric literal is suffixed (`u` / `ull`) so pure-literal subexpressions and
  constant shifts are computed at width-W unsigned — otherwise C's default
  `int`-typed literals would diverge from the oracle on width-64 functions.

**Validation.** `gen_corpus.c` was compiled with gcc (`-O2` and `-O0`) and every
oracle case was re-evaluated by calling the real compiled function via ctypes:
44393/44393 cases matched at both optimization levels for a 50-function corpus.
A wrong oracle would mis-score every solver, so this native cross-check is the
primary soundness gate; re-run it whenever the generator or evaluator changes.
