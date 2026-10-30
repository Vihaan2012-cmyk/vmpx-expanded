#!/usr/bin/env python3
"""Ground-truth evaluator for the SP6 mutation benchmark (WP12).

Reads gen_manifest.json (each function's clean source `expr` over params
`a`,`b`,`c` at a declared width) and emits gen_oracle.json: the input->output
oracle table that the solver tournament scores candidate simplifications
against by differential input-testing.

SOUNDNESS
---------
The C expressions are evaluated with exact *unsigned* semantics at the declared
width. Every operation's result is reduced mod 2^width (the unsigned type's
modulus), which is how C defines unsigned arithmetic, so wraparound, `~`, and
shifts are modelled exactly:

  a + b, a - b, a * b   -> (x) & mask          (mask = 2^width - 1)
  ~a                    -> (~x) & mask  == x ^ mask
  -a  (unary)           -> (-x) & mask  == (2^width - x) mod 2^width
  a << s                -> (x << s) & mask      (logical)
  a >> s                -> x >> s                (logical; operands are masked,
                                                  non-negative, so Python >> is
                                                  a logical shift)

Masking at *every* node — not only at the end — matters: e.g. for width 8,
`(a << 4) >> 2` with a=0xFF is ((0xFF<<4)&0xFF)>>2 = 0xF0>>2 = 0x3C, which a
final-only mask would get wrong (0xFC). A hand-written recursive-descent parser
is used (not Python `eval`) because C operator precedence differs from Python's
for the bitwise/shift operators, and because we must apply the width mask per
operation.

The generator only emits well-defined unsigned ops (+ - * & | ^ ~ << >>) with
shift amounts in [0,width), so no case is undefined.

INPUT SET (per function, deterministic)
---------------------------------------
Structured edge cases built from a per-width "interesting value" set
(0, 1, MAX/all-ones, sign boundary 2^(w-1) and +-1, every single-bit 2^k, and
2^k +- 1) combined as:
  * all-params-equal, for each interesting value;
  * one-param-swept over every interesting value while the others are held at
    each baseline in {0, 1, MAX};
  * random tuples drawn from the interesting set (edge x edge cross terms);
plus uniform-random full-width tuples. Deterministic per function (seed derived
from the function name), so the oracle is reproducible.

Usage:  python gen-oracle.py [--manifest PATH] [--out PATH]
                             [--random N] [--edge-random N] [--seed S]
"""
import argparse
import hashlib
import json
import os
import random


# --------------------------------------------------------------------------
# Expression evaluator: recursive-descent parser with C precedence, every node
# result masked to the function width (exact unsigned semantics).
# --------------------------------------------------------------------------

class Parser:
    # precedence (low -> high), C unsigned subset
    _BIN_LEVELS = [
        ["|"],
        ["^"],
        ["&"],
        ["<<", ">>"],
        ["+", "-"],
        ["*"],
    ]

    def __init__(self, text, width):
        self.s = text
        self.i = 0
        self.width = width
        self.mask = (1 << width) - 1
        self.toks = self._tokenize(text)
        self.p = 0

    @staticmethod
    def _tokenize(s):
        toks = []
        i, n = 0, len(s)
        two = {"<<", ">>"}
        while i < n:
            c = s[i]
            if c.isspace():
                i += 1
                continue
            if i + 1 < n and s[i:i + 2] in two:
                toks.append(s[i:i + 2])
                i += 2
                continue
            if c in "()+-*&|^~":
                toks.append(c)
                i += 1
                continue
            if c.isdigit():
                j = i
                if s[i] == "0" and i + 1 < n and s[i + 1] in "xX":
                    j = i + 2
                    while j < n and s[j] in "0123456789abcdefABCDEF":
                        j += 1
                    toks.append(("num", int(s[i:j], 16)))
                else:
                    while j < n and s[j].isdigit():
                        j += 1
                    toks.append(("num", int(s[i:j], 10)))
                i = j
                continue
            if c.isalpha() or c == "_":
                j = i
                while j < n and (s[j].isalnum() or s[j] == "_"):
                    j += 1
                toks.append(("id", s[i:j]))
                i = j
                continue
            raise ValueError("bad char %r in %r" % (c, s))
        return toks

    def _peek(self):
        return self.toks[self.p] if self.p < len(self.toks) else None

    def _next(self):
        t = self.toks[self.p]
        self.p += 1
        return t

    def parse(self, env):
        self.env = env
        v = self._bin(0)
        if self.p != len(self.toks):
            raise ValueError("trailing tokens in %r" % self.s)
        return v & self.mask

    def _bin(self, level):
        if level >= len(self._BIN_LEVELS):
            return self._unary()
        ops = self._BIN_LEVELS[level]
        left = self._bin(level + 1)
        while self._peek() in ops:
            op = self._next()
            right = self._bin(level + 1)
            left = self._apply(op, left, right)
        return left

    def _apply(self, op, a, b):
        m = self.mask
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        elif op == "*":
            r = a * b
        elif op == "&":
            r = a & b
        elif op == "|":
            r = a | b
        elif op == "^":
            r = a ^ b
        elif op == "<<":
            r = a << (b & (self.width - 1))
        elif op == ">>":
            # operands are already masked (>=0, < 2^width) -> logical shift
            r = a >> (b & (self.width - 1))
        else:
            raise ValueError("bad op %r" % op)
        return r & m

    def _unary(self):
        t = self._peek()
        if t == "~":
            self._next()
            return (~self._unary()) & self.mask
        if t == "-":
            self._next()
            return (-self._unary()) & self.mask
        if t == "+":
            self._next()
            return self._unary()
        return self._primary()

    def _primary(self):
        t = self._next()
        if t == "(":
            v = self._bin(0)
            close = self._next()
            if close != ")":
                raise ValueError("expected ) in %r" % self.s)
            return v & self.mask
        if isinstance(t, tuple):
            kind, val = t
            if kind == "num":
                return val & self.mask
            if kind == "id":
                return self.env[val] & self.mask
        raise ValueError("unexpected token %r in %r" % (t, self.s))


def make_eval(expr, width):
    p = Parser(expr, width)

    def ev(values):  # values: dict param->int
        p.p = 0
        return p.parse(values)
    return ev


# --------------------------------------------------------------------------
# Input set construction.
# --------------------------------------------------------------------------

def interesting_values(width):
    mask = (1 << width) - 1
    vals = set()
    vals.update([0, 1, 2, 3, mask])          # 0, 1, small, all-ones/MAX
    sign = 1 << (width - 1)                   # sign boundary
    vals.update([sign - 1, sign, sign + 1])
    for k in range(width):                    # single-bit / 2^k
        bit = 1 << k
        vals.add(bit)
        vals.add((bit - 1) & mask)            # 2^k - 1
        vals.add((bit + 1) & mask)            # 2^k + 1
    return sorted(v & mask for v in vals)


def build_inputs(params, width, n_random, n_edge_random, seed):
    mask = (1 << width) - 1
    np = len(params)
    iv = interesting_values(width)
    baselines = [0, 1, mask]
    rng = random.Random(seed)
    tuples = []
    seen = set()

    def add(tpl):
        if tpl not in seen:
            seen.add(tpl)
            tuples.append(tpl)

    # all-params-equal over every interesting value
    for v in iv:
        add(tuple([v] * np))

    # one param swept over all interesting values, others at each baseline
    if np > 1:
        for pos in range(np):
            for base in baselines:
                for v in iv:
                    t = [base] * np
                    t[pos] = v
                    add(tuple(t))

    # edge x edge cross terms: random tuples drawn from the interesting set
    for _ in range(n_edge_random):
        add(tuple(rng.choice(iv) for _ in range(np)))

    # uniform random full-width tuples
    for _ in range(n_random):
        add(tuple(rng.randint(0, mask) for _ in range(np)))

    return tuples


def func_seed(base_seed, name):
    h = hashlib.sha256(("%d:%s" % (base_seed, name)).encode()).hexdigest()
    return int(h[:16], 16)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(here, "gen_manifest.json"))
    ap.add_argument("--out", default=os.path.join(here, "gen_oracle.json"))
    ap.add_argument("--random", type=int, default=64,
                    help="uniform-random full-width tuples per function")
    ap.add_argument("--edge-random", type=int, default=64,
                    help="random tuples drawn from the interesting-value set")
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    oracle = {}
    for name in sorted(manifest):
        entry = manifest[name]
        width = entry["width"]
        params = entry["params"]
        expr = entry["expr"]
        ev = make_eval(expr, width)
        inputs = build_inputs(params, width, args.random, args.edge_random,
                              func_seed(args.seed, name))
        cases = []
        for tpl in inputs:
            env = dict(zip(params, tpl))
            out = ev(env)
            cases.append({"in": list(tpl), "out": out})
        oracle[name] = {"width": width, "params": params, "cases": cases}

    with open(args.out, "w") as f:
        json.dump(oracle, f, separators=(",", ":"), sort_keys=True)

    total = sum(len(v["cases"]) for v in oracle.values())
    print("evaluated %d functions, %d total cases" % (len(oracle), total))
    print("  oracle: %s" % args.out)


if __name__ == "__main__":
    main()
