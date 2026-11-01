#!/usr/bin/env python3
"""A380 census runner: thin subprocess orchestrator over `vmpx diff-original`.

For each function in a manifest ([{symbol, demangled, bucket}, ...]) or, with
--enumerate, every export of the protected DLL, run one bounded
`vmpx diff-original <protected> <original> --function <symbol>` and classify the
verdict. Writes census.md (bucket x verdict table, totals, ranked fix-list),
census.json (machine-readable) and one raw log per function.

Static only: diff-original never executes the sample. The vmpx binary must be
built with `--features smt-z3,smt-z3-bundled` or every function TIMEOUTs.

Matching is by export name: `symbol` must be the export name in BOTH DLLs
(`demangled` is display-only).
"""
import argparse, collections, json, os, re, struct, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

VERDICT_RE = re.compile(r"^(\S.*?) \[0x([0-9a-f]+)\]: (PROVEN|REFUTED|TIMEOUT)(.*)$")
SUMMARY_RE = re.compile(r"PROVEN (\d+) \| REFUTED (\d+) \| TIMEOUT (\d+)")
CEX_RE = re.compile(r"^\s+COUNTEREXAMPLE .*$")
REASON_RE = re.compile(r"^\s+reason: (.*)$")

# (class, regex on reason) in priority order; hint = what to fix next.
CLASSES = [
    ("FP-not-supported", r"(?i)\b(xmm\d*|ymm\d*|st\d|x87|sse\d*|avx\d*|float\w*|fp|fpu|mxcsr)\b|unmodeled (instruction|mnemonic)", "model FP/SSE cells in lifter + comparator"),
    ("store-aliasing", r"may alias an earlier store|stores outside its frame|load/store ordering", "symbolic memory: model store/load ordering and aliasing"),
    ("input-dependent-branch", r"depends on the function's inputs|branch `[^`]*` reads flag", "multi-path native unroll: fork on input-dependent branches"),
    ("tree-lift-refused", r"protected recovery/unroll|stream recovery|entry frame not recovered|cfg recovery|no VM-exit|no callee-return path|depends on the VM stack root|recovery failed", "VM-side recovery/lift of the protected function"),
    ("original-not-modelable", r"original CFG|original not modelable|did not return within|unmodeled condition code|was not lifted|reads `[^`]*` before it is defined|stack is unbalanced", "native-side lift of the original function"),
    ("solver-timeout", r"Timeout\(|Unknown\(|wall-clock", "solver budget / simplifier: expression too large for z3"),
    ("modeling-gap", r"modeling gap|ModelingGap", "equivalence-gate modeling gap"),
    ("strict-gates", r"strict gates", "environment-gate arms (only with --strict-gates)"),
]
HINTS = {c: h for c, _, h in CLASSES}
HINTS.update({"not-in-original": "symbol missing from original DLL: fix manifest/matching",
              "not-in-protected": "symbol missing from protected DLL: fix manifest/matching",
              "not-virtualized": "export is not a push-seed;call VM stub: not a virtualized function",
              "no-output": "diff-original produced no verdict line (crash/OOM?): inspect raw log",
              "other": "unclassified reason: inspect raw log"})


def classify(reason):
    for name, rx, _ in CLASSES:
        if re.search(rx, reason):
            return name
    return "other"


def pe_exports(path):
    d = open(path, "rb").read()
    pe = struct.unpack_from("<I", d, 0x3C)[0]
    nsec = struct.unpack_from("<H", d, pe + 6)[0]
    osz = struct.unpack_from("<H", d, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", d, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    erva, esz = struct.unpack_from("<II", d, dd)
    secs = [struct.unpack_from("<IIII", d, opt + osz + 40 * i + 8) for i in range(nsec)]

    def off(rva):
        for vs, va, rs, rp in secs:
            if va <= rva < va + max(vs, rs):
                return rva - va + rp
        raise ValueError(hex(rva))

    if not erva:
        return []
    e = off(erva)
    nnames, _, names_rva = struct.unpack_from("<III", d, e + 24)
    out = []
    for i in range(nnames):
        nrva = struct.unpack_from("<I", d, off(names_rva) + 4 * i)[0]
        o = off(nrva)
        out.append(d[o:d.index(b"\0", o)].decode("latin1"))
    return out


def run_one(args, entry, control):
    sym = entry["symbol"]
    for label, exports in (("protected", args.prot_exports), ("original", args.orig_exports)):
        if sym not in exports:
            print(f"[skip] {entry.get('bucket','')}/{sym}: not exported by {label}", flush=True)
            return dict(entry, control=control, seconds=0, rc=0, verdict="SKIPPED", cls=f"not-in-{label}",
                        reason=f"symbol not exported by the {label} DLL", counterexample="",
                        modulo_env_gates=False, backends="")
    cmd = [args.vmpx, "diff-original", args.protected, args.original, "--function", sym,
           "--timeout-ms", str(args.timeout_ms)]
    if control:
        cmd.append("--control")
    if args.strict_gates:
        cmd.append("--strict-gates")
    cmd += args.extra
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=args.wall_timeout_s)
        out, rc, wall_hit = p.stdout, p.returncode, False
        err = p.stderr
    except subprocess.TimeoutExpired as ex:
        out = (ex.stdout.decode("utf8", "replace") if isinstance(ex.stdout, bytes) else ex.stdout) or ""
        err, rc, wall_hit = "", -1, True
    res = dict(entry, control=control, seconds=round(time.time() - t0, 1), rc=rc,
               verdict=None, reason="", counterexample="", modulo_env_gates=False, backends="")
    lines = out.splitlines()
    for ln in lines:
        if ln.startswith("backends:"):
            res["backends"] = ln.split(";")[0]
    for i, ln in enumerate(lines):
        m = VERDICT_RE.match(ln)
        if m and m.group(1) == sym:
            res["verdict"] = m.group(3)
            res["modulo_env_gates"] = "modulo environment gates" in m.group(4)
            res["detail"] = m.group(4).strip(" |")
            for nxt in lines[i + 1:]:
                if VERDICT_RE.match(nxt) or nxt.startswith("===="):
                    break
                r = REASON_RE.match(nxt)
                if r:
                    res["reason"] = r.group(1)
                if CEX_RE.match(nxt) and not res["counterexample"]:
                    res["counterexample"] = nxt.strip()
            break
    if res["verdict"] is None:
        for ln in lines:
            if ln.startswith(sym + ":"):
                res["verdict"] = "SKIPPED"
                res["reason"] = ln[len(sym) + 1:].strip()
                res["cls"] = "not-in-original" if "no export" in ln else "not-virtualized"
        if res["verdict"] is None:
            res["verdict"] = "TIMEOUT"
            res["reason"] = "wall-clock timeout (killed)" if wall_hit else f"no verdict line (rc={rc}) {err.strip()[-200:]}"
            res["cls"] = "solver-timeout" if wall_hit else "no-output"
    elif res["verdict"] == "TIMEOUT":
        res["cls"] = classify(res["reason"])
    elif res["verdict"] == "REFUTED":
        res["cls"] = "REFUTED"
        if not res["reason"]:
            res["reason"] = res["counterexample"] or "refuted"
    else:
        res["cls"] = "PROVEN"
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sym)[:100]
    tag = "control" if control else "run"
    with open(os.path.join(args.out, "logs", f"{tag}.{safe}.log"), "w", encoding="utf8") as f:
        f.write("$ " + " ".join(cmd) + "\n" + out + ("\n--- stderr ---\n" + err if err else ""))
    print(f"[{tag}] {res['verdict']:8} {res['seconds']:7.1f}s {entry.get('bucket','')}/{sym} {res['cls']}", flush=True)
    return res


def report(args, results, mode_control):
    L = []
    title = "A380 census (NEGATIVE CONTROL)" if mode_control else "A380 census"
    L += [f"# {title}", "",
          f"- protected: `{args.protected}`", f"- original:  `{args.original}`",
          f"- budget: {args.timeout_ms} ms/cell, {args.wall_timeout_s}s wall/function, jobs={args.jobs}"
          + (", strict-gates" if args.strict_gates else ""),
          f"- backends: `{next((r['backends'] for r in results if r['backends']), 'n/a')}`", ""]
    skipped = [r for r in results if r["verdict"] == "SKIPPED"]
    scored = [r for r in results if r["verdict"] != "SKIPPED"]
    cnt = collections.Counter(r["verdict"] for r in scored)
    M = len(scored)
    if mode_control:
        L += [f"## Totals: {cnt['REFUTED']} REFUTED / {cnt['PROVEN']} PROVEN / {cnt['TIMEOUT']} TIMEOUT of {M}",
              "Control passes when nothing is PROVEN (soundness) and at least one is REFUTED (the harness surfaces "
              "counterexamples). TIMEOUT is allowed: a mismatched pair can exceed the solver budget.", ""]
    else:
        L += [f"## Totals: {cnt['PROVEN']} PROVEN / {cnt['REFUTED']} REFUTED / {cnt['TIMEOUT']} TIMEOUT of {M}"
              + (f" ({len(skipped)} skipped: unmatched/non-virtualized)" if skipped else ""), ""]
        env = sum(r["modulo_env_gates"] for r in scored if r["verdict"] == "PROVEN")
        if env:
            L += [f"PROVEN is *modulo environment gates* for {env} of {cnt['PROVEN']} functions "
                  "(gate-failure/spin arms excluded; rerun with --strict-gates to count them as unproven).", ""]
    if cnt["REFUTED"] and not mode_control:
        L += ["**CRITICAL: REFUTED functions present (protected != original). See fix-list.**", ""]
    buckets = sorted({r.get("bucket", "") for r in results})
    L += ["## Bucket x verdict", "", "| bucket | PROVEN | REFUTED | TIMEOUT | skipped | total |", "|---|---:|---:|---:|---:|---:|"]
    tot = collections.Counter()
    for b in buckets:
        c = collections.Counter(r["verdict"] for r in results if r.get("bucket", "") == b)
        L.append(f"| {b or '(none)'} | {c['PROVEN']} | {c['REFUTED']} | {c['TIMEOUT']} | {c['SKIPPED']} | {sum(c.values())} |")
        tot.update(c)
    L.append(f"| **all** | {tot['PROVEN']} | {tot['REFUTED']} | {tot['TIMEOUT']} | {tot['SKIPPED']} | {sum(tot.values())} |")
    L.append("")
    if not mode_control:
        bad = [r for r in results if r["cls"] != "PROVEN"]
        groups = collections.defaultdict(list)
        for r in bad:
            groups[r["cls"]].append(r)
        order = sorted(groups, key=lambda k: (k != "REFUTED", -len(groups[k]), k))
        L += ["## Ranked fix-list (REFUTED first, then by function count)", ""]
        if not order:
            L += ["Nothing to fix: every function PROVEN.", ""]
        for rank, k in enumerate(order, 1):
            g = groups[k]
            hint = "CRITICAL: semantic divergence, fix before anything else" if k == "REFUTED" else HINTS.get(k, "")
            L += [f"### {rank}. {k} ({len(g)} fn) - {hint}", ""]
            for r in sorted(g, key=lambda r: (r.get("bucket", ""), r["symbol"])):
                name = r.get("demangled") or r["symbol"]
                L.append(f"- `{name}` [{r.get('bucket','')}] {r['reason'][:240]}")
                if r["counterexample"]:
                    L.append(f"  - {r['counterexample'][:400]}")
            L.append("")
    else:
        L += ["## Control results", ""]
        for r in results:
            ok = {"REFUTED": "OK (REFUTED as expected)", "TIMEOUT": "inconclusive (allowed)"}.get(r["verdict"], "UNSOUND/HARNESS PROBLEM")
            L.append(f"- `{r['symbol']}`: {r['verdict']} -> {ok}")
            if r["counterexample"]:
                L.append(f"  - {r['counterexample'][:300]}")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("protected")
    ap.add_argument("original")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="JSON list of {symbol, demangled, bucket}")
    g.add_argument("--enumerate", action="store_true", help="self-test: every export of the protected DLL")
    ap.add_argument("--bucket", default="all", help="bucket label for --enumerate")
    ap.add_argument("--vmpx", default=os.environ.get("VMPX_BIN", "vmpx"), help="z3-enabled vmpx binary")
    ap.add_argument("--out", default="census_out")
    ap.add_argument("--timeout-ms", type=int, default=60000, help="per-cell solver budget")
    ap.add_argument("--wall-timeout-s", type=int, default=900, help="hard kill per function")
    ap.add_argument("--jobs", type=int, default=1, help="parallel functions (z3 is memory-heavy; keep 1-2)")
    ap.add_argument("--strict-gates", action="store_true")
    ap.add_argument("--control", action="store_true", help="negative control (each fn vs the next export's original): must yield no PROVEN and >=1 REFUTED")
    ap.add_argument("--only", action="append", default=[], help="restrict to these symbols")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args passed to diff-original")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    if args.manifest:
        entries = json.load(open(args.manifest, encoding="utf8"))
        for e in entries:
            e.setdefault("demangled", e["symbol"])
            e.setdefault("bucket", "")
    else:
        entries = [dict(symbol=s, demangled=s, bucket=args.bucket) for s in pe_exports(args.protected)]
    args.prot_exports = set(pe_exports(args.protected))
    args.orig_exports = set(pe_exports(args.original))
    if args.only:
        entries = [e for e in entries if e["symbol"] in args.only]
    if args.limit:
        entries = entries[: args.limit]
    print(f"{len(entries)} functions, jobs={args.jobs}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        results = list(ex.map(lambda e: run_one(args, e, args.control), entries))

    md = report(args, results, args.control)
    stem = "control" if args.control else "census"
    open(os.path.join(args.out, stem + ".md"), "w", encoding="utf8").write(md)
    json.dump(results, open(os.path.join(args.out, stem + ".json"), "w", encoding="utf8"), indent=1)
    print("\n" + md)
    c = collections.Counter(r["verdict"] for r in results)
    if args.control:
        return 0 if c["PROVEN"] == 0 and c["REFUTED"] > 0 else 2
    return 1 if c["REFUTED"] else 0


if __name__ == "__main__":
    sys.exit(main())
