#!/usr/bin/env python3
"""Select a hard-weighted, diverse 3000-4000 function set from the A380 plugin
for the devirt validation corpus. Reads syms_all.txt (nm dump, v0 Rust mangling),
filters to user crates, classifies into logic/physics/math/other buckets,
weights by difficulty, and does diverse weighted sampling per bucket/crate.

No rustfilt available (no network to install it) -> approximate demangling:
best-effort identifier-chunk extraction from the v0 string, not a full parser.
Crate classification instead uses the mangling's own length-prefixed identifier
encoding (every path segment is "<decimal length><chars>"), which is exact.

Export-safety filters (added after the first rebuild attempt surfaced real
link failures): only 'T' (global/COFF-external) symbols are candidates --
't' (local) symbols have no cross-object identity and can never be exported,
by any mechanism. Symbols whose exact mangled name maps to more than one
address (unmerged per-translation-unit local duplicates) and compiler-
internal ".llvm.<hash>" thunks are excluded too. With these three filters, a
.def file listing the full candidate pool linked clean on the first try
(0 "symbol not defined" errors across 24,718 names).
"""
import json, os, random, re, sys

SYMS_PATH = sys.argv[1] if len(sys.argv) > 1 else r"E:\VMPX\a380_src\syms_all.txt"
# Optional: a plain-text list of names actually present in the DLL's real PE
# export table (one per line). GNU ld's --export-all-symbols, empirically,
# does NOT export every global ('T') symbol in a project this size with many
# dependency rlibs (~26% export rate observed) -- the exact cause wasn't
# pinned down (both exported and non-exported symbols are IMAGE_SYM_CLASS_
# EXTERNAL at the COFF level, so it isn't a weak-vs-strong linkage split).
# Rather than depend on understanding that ld heuristic, filter candidates
# to names confirmed present in the real export table, so selection can only
# ever pick names that are already guaranteed export-safe.
REAL_EXPORTS_PATH = sys.argv[2] if len(sys.argv) > 2 else None
OUT_DIR = r"E:\VMPX\testcorpus\a380"
SEED = 20261101
TARGET_TOTAL = 3500

# Real workspace crate names for this plugin (from Cargo.toml scan), a320* excluded
# (different aircraft, not expected in this DLL).
USER_CRATES = {
    "a380_systems", "fbw_a380_systems", "systems",
    "ai_fo", "charts", "cids", "cpcs_extra", "css_cascade", "datalink",
    "deep_electrical", "deep_systems", "efb_map", "efb_pages", "fap_pages",
    "fms", "fms_data", "fms_host", "fms_lnav", "fms_perf", "fms_vnav",
    "fuel_mgmt", "fws", "fws_data", "fws_ecl", "gpws", "hosts_misc",
    "mfd_host", "mfd_native", "mfd_pages_a", "mfd_pages_b", "mfd_pages_c",
    "mfd_pages_d", "mfd_tree", "oans", "oans_oracle", "oit_native",
    "oit_pages", "rmp", "row_rop", "soundscape", "state_snapshot",
    "terrain_map", "wxr_extras",
    # NOTE: the workspace crate "ops" is deliberately excluded -- its 3-char
    # name is indistinguishable by this heuristic from core::ops backrefs
    # (B<n>_3ops...), which appear in nearly every generic iterator symbol.
}
LOGIC_CRATES = {
    "fms", "fms_data", "fms_host", "fms_lnav", "fms_perf", "fms_vnav",
    "fws", "fws_data", "fws_ecl", "mfd_host", "mfd_native", "mfd_pages_a",
    "mfd_pages_b", "mfd_pages_c", "mfd_pages_d", "mfd_tree", "oit_native",
    "oit_pages", "gpws", "tcas", "charts", "efb_pages", "efb_map",
    "fap_pages", "oans", "oans_oracle", "datalink", "ai_fo", "cids",
}
PHYSICS_CRATES = {"fuel_mgmt", "deep_electrical", "deep_systems"}
PHYSICS_TOKENS = ["hydraulic", "pneumatic", "engine", "electrical", "thermal", "adirs", "sensors"]
MATH_TOKENS = ["uom", "typenum", "nalgebra"]

# "tcas" wasn't in the Cargo.toml scan under that exact match (it is: crates/tcas) -- present above.

LEN_TOKEN_RE_CACHE = {}


def token_hits(mangled, token):
    """True if <len(token)><token> appears as a mangling path segment. v0
    length-prefixed identifiers have no trailing delimiter, so the only
    reliable anchor is on the left: the decimal length must not itself be
    the tail of a longer number (lookbehind excludes e.g. "19hydraulic"
    matching the "9hydraulic" needle)."""
    needle = f"{len(token)}{token}"
    pat = LEN_TOKEN_RE_CACHE.get(needle)
    if pat is None:
        # v0 crate-roots are always introduced right after a "C" tag or a
        # disambiguator's trailing "_" -- require that, or short common
        # workspace crate names (e.g. "ops") collide with std path segments
        # like core::ops that share the same length prefix by coincidence.
        pat = re.compile(r"(?:^|[C_])" + re.escape(needle))
        LEN_TOKEN_RE_CACHE[needle] = pat
    return pat.search(mangled) is not None


def classify_user_crate(mangled):
    for c in USER_CRATES:
        if token_hits(mangled, c):
            return c
    return None


def classify_bucket(mangled, crate):
    if crate in LOGIC_CRATES:
        return "logic"
    if crate in PHYSICS_CRATES or any(token_hits(mangled, t) for t in PHYSICS_TOKENS):
        return "physics"
    if any(t in mangled for t in MATH_TOKENS):  # substring ok, these are rare/distinctive
        return "math"
    return "other"


def approx_demangle(mangled):
    """Best-effort readable fragment: last couple of plausible identifier chunks
    (length-prefix decoded), not a real v0 parser. Falls back to raw mangled."""
    chunks = []
    i, n = 0, len(mangled)
    while i < n:
        if mangled[i].isdigit():
            j = i
            while j < n and mangled[j].isdigit():
                j += 1
            length = int(mangled[i:j])
            if 0 < length <= 64 and j + length <= n:
                cand = mangled[j:j + length]
                if cand[:1].isalpha() or cand[:1] == "_":
                    if all(c.isalnum() or c == "_" for c in cand):
                        chunks.append(cand)
                        i = j + length
                        continue
            i = j if j > i else i + 1
        else:
            i += 1
    if not chunks:
        return mangled[:80]
    tail = chunks[-4:] if len(chunks) > 4 else chunks
    return "::".join(tail)


def generic_complexity(mangled):
    return mangled.count("I") + mangled.count("N")


def main():
    rows = []
    with open(SYMS_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.split(None, 2)
            if len(parts) != 3:
                continue
            addr_s, typ, mangled = parts
            mangled = mangled.rstrip("\n")
            if not mangled.startswith("_R"):
                continue
            rows.append((int(addr_s, 16), typ, mangled))
    rows.sort(key=lambda r: r[0])
    print(f"total symbols: {len(rows)}", file=sys.stderr)

    # Local ('t') symbols aren't required to be name-unique across object
    # files -- the same monomorphization can get an independent, unmerged
    # local copy per translation unit, same exact mangled name at two
    # different addresses. A linker .def EXPORTS entry needs one address per
    # name, so any name that isn't globally unique here can't be a safe
    # export target regardless of which bucket picked it.
    name_addrs = {}
    for addr, typ, mangled in rows:
        name_addrs.setdefault(mangled, set()).add(addr)
    ambiguous_names = {m for m, a in name_addrs.items() if len(a) > 1}
    print(f"ambiguous (non-unique-address) names: {len(ambiguous_names)}", file=sys.stderr)

    # size = gap to next DISTINCT address (handles ICF-merged duplicate addresses)
    addrs = [r[0] for r in rows]
    n = len(rows)
    real_exports = None
    if REAL_EXPORTS_PATH:
        with open(REAL_EXPORTS_PATH, encoding="utf-8") as f:
            real_exports = {line.strip() for line in f if line.strip()}
        print(f"real export filter: {len(real_exports)} names", file=sys.stderr)

    candidates = []
    for i, (addr, typ, mangled) in enumerate(rows):
        if real_exports is not None and mangled not in real_exports:
            continue
        if typ != "T":
            # 't' (lowercase) is local/file-scope linkage at the COFF level --
            # it has no external symbol identity to export, with or without
            # --export-all-symbols or a .def entry. Only 'T' (global) is
            # ever export-able.
            continue
        if mangled in ambiguous_names:
            continue
        if "." in mangled:
            # compiler-internal thunks (drop_glue.llvm.<hash> etc.) -- not
            # meaningful devirt targets, and "." is awkward/unsafe in a
            # linker .def EXPORTS entry. A handful also duplicate verbatim
            # across translation units pre-link, which a .def can't export
            # twice either way.
            continue
        crate = classify_user_crate(mangled)
        if crate is None:
            continue
        j = i + 1
        while j < n and addrs[j] == addr:
            j += 1
        size = (addrs[j] - addr) if j < n else 4096
        if size <= 0:
            size = 16
        size = min(size, 1 << 20)  # cap absurd outliers (e.g. last symbol in a section)
        bucket = classify_bucket(mangled, crate)
        candidates.append(dict(addr=addr, typ=typ, mangled=mangled, crate=crate,
                                size=size, bucket=bucket))

    print(f"user-crate candidates: {len(candidates)}", file=sys.stderr)
    from collections import Counter
    print("by bucket:", Counter(c["bucket"] for c in candidates), file=sys.stderr)
    print("by crate (top 10):", Counter(c["crate"] for c in candidates).most_common(10), file=sys.stderr)

    import math as _m
    BUCKET_MULT = {"logic": 3.0, "physics": 1.6, "math": 1.4, "other": 1.0}
    for c in candidates:
        base = _m.log2(c["size"] + 1)
        cx = min(generic_complexity(c["mangled"]), 10)
        c["weight"] = base * BUCKET_MULT[c["bucket"]] * (1 + 0.15 * cx)

    rng = random.Random(SEED)
    BUCKET_FRACTION = {"logic": 0.40, "physics": 0.30, "math": 0.10, "other": 0.20}
    by_bucket = {}
    for c in candidates:
        by_bucket.setdefault(c["bucket"], []).append(c)

    selected = []
    for bucket, frac in BUCKET_FRACTION.items():
        pool = by_bucket.get(bucket, [])
        quota = round(TARGET_TOTAL * frac)
        if not pool:
            continue
        by_crate = {}
        for c in pool:
            by_crate.setdefault(c["crate"], []).append(c)
        crates = list(by_crate.keys())
        # proportional-to-sqrt(count) allocation, capped at 20% of bucket quota/crate
        weights = {cr: _m.sqrt(len(by_crate[cr])) for cr in crates}
        wsum = sum(weights.values())
        cap = max(1, round(quota * 0.20))
        alloc = {}
        remaining = quota
        order = sorted(crates, key=lambda cr: -weights[cr])
        for cr in order:
            want = round(quota * weights[cr] / wsum)
            got = min(want, cap, len(by_crate[cr]), remaining)
            alloc[cr] = got
            remaining -= got
        # distribute any leftover (rounding, caps) to crates with spare candidates
        for cr in order:
            if remaining <= 0:
                break
            spare = len(by_crate[cr]) - alloc[cr]
            if spare > 0:
                take = min(spare, remaining, cap - alloc[cr]) if alloc[cr] < cap else 0
                if take > 0:
                    alloc[cr] += take
                    remaining -= take
        for cr, k in alloc.items():
            if k <= 0:
                continue
            pool_cr = by_crate[cr]
            # Efraimidis-Spirakis weighted sampling without replacement
            keyed = sorted(pool_cr, key=lambda c: -(rng.random() ** (1.0 / max(c["weight"], 1e-6))))
            selected.extend(keyed[:k])

    # Top-up: some buckets (e.g. math) may be too small to fill their quota.
    # Pull additional weighted picks from the full remaining pool to reach the
    # 3000-4000 target, capping any single crate at 25% of the target so the
    # huge fbw_a380_systems "other" pool can't swallow the top-up.
    selected_ids = {id(c) for c in selected}
    from collections import Counter as _Counter
    crate_counts = _Counter(c["crate"] for c in selected)
    crate_cap = round(TARGET_TOTAL * 0.25)
    if len(selected) < TARGET_TOTAL:
        remaining_pool = [c for c in candidates if id(c) not in selected_ids
                          and crate_counts[c["crate"]] < crate_cap]
        keyed = sorted(remaining_pool,
                        key=lambda c: -(rng.random() ** (1.0 / max(c["weight"], 1e-6))))
        for c in keyed:
            if len(selected) >= TARGET_TOTAL:
                break
            if crate_counts[c["crate"]] >= crate_cap:
                continue
            selected.append(c)
            crate_counts[c["crate"]] += 1

    print(f"selected total: {len(selected)}", file=sys.stderr)
    print("selected by bucket:", Counter(c["bucket"] for c in selected), file=sys.stderr)
    print("selected distinct crates:", len(set(c["crate"] for c in selected)), file=sys.stderr)

    manifest = []
    for c in selected:
        manifest.append({
            "symbol": c["mangled"],
            "demangled": approx_demangle(c["mangled"]),
            "typ": c["typ"],
            "addr": f"0x{c['addr']:x}",
            "size": c["size"],
            "crate": c["crate"],
            "bucket": c["bucket"],
            "weight": round(c["weight"], 3),
        })
    manifest.sort(key=lambda m: m["addr"])

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "a380_selection.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(OUT_DIR, "a380_selection_addrs.txt"), "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(m["addr"] + "\n")

    print(f"wrote {len(manifest)} functions -> a380_selection.json / a380_selection_addrs.txt")


if __name__ == "__main__":
    main()
