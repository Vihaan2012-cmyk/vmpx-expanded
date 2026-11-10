#!/usr/bin/env python3
"""Build the tigress-expanded ground-truth corpus.

For every seed x every obfuscation config: run Tigress to obfuscate it, compile
the clean seed and the obfuscated source, run BOTH on fixed inputs, and record a
(clean_binary, obfuscated_binary, expected_output, matched) triple. The whole
point is the oracle: `matched == True` means the obfuscated binary is behaviourally
identical to the original, so later the deobfuscator can be scored against the
clean output for real.

Run it anywhere Tigress + a C compiler exist (Linux/macOS native; Windows via WSL
or MinGW). Nothing here is vmpx-specific -- it just manufactures labelled inputs.

    python build_corpus.py                 # uses ./seeds -> ./out
    TIGRESS=/path/to/tigress CC=clang ENV=x86_64:Linux:Gcc:4.6 python build_corpus.py
"""
import json, os, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = HERE / "seeds"
OUT = Path(os.environ.get("OUT", HERE / "out"))
TIGRESS = os.environ.get("TIGRESS", "tigress")
CC = os.environ.get("CC", shutil.which("gcc") or shutil.which("clang") or "cc")
ENV = os.environ.get("ENV", "x86_64:Linux:Gcc:4.6")  # adjust to your Tigress/toolchain

# Per-seed: the function(s) to obfuscate, and the fixed input vectors (argv).
SEED_SPEC = {
    "arith.c":     {"funcs": "mix",      "inputs": [[], ["305419896", "1450741096"], ["0", "0"]]},
    "control.c":   {"funcs": "classify", "inputs": [[], ["42"], ["7"], ["100"]]},
    "loop_hash.c": {"funcs": "hash",     "inputs": [[], ["hello"], ["a longer test string"]]},
}

def tf(*xforms):
    """Flatten a list of (Transform, extra-args...) into Tigress CLI tokens."""
    out = []
    for x in xforms:
        out += [f"--Transform={x[0]}"] + list(x[1:])
    return out

# The config matrix. Each entry: name -> a function(funcs)->tigress-transform-args.
# Start narrow and exact; widen once the harness runs clean.
CONFIGS = {
    "virtualize_switch":   lambda f: tf(("Virtualize", f"--Functions={f}", "--VirtualizeDispatch=switch")),
    "virtualize_direct":   lambda f: tf(("Virtualize", f"--Functions={f}", "--VirtualizeDispatch=direct")),
    "virtualize_indirect": lambda f: tf(("Virtualize", f"--Functions={f}", "--VirtualizeDispatch=indirect")),
    "flatten":             lambda f: tf(("Flatten", f"--Functions={f}")),
    "encode_arith":        lambda f: tf(("EncodeArithmetic", f"--Functions={f}")),
    "encode_literals":     lambda f: tf(("EncodeLiterals", f"--Functions={f}")),
    "opaque":              lambda f: tf(("InitOpaque", "--Functions=main"), ("AddOpaque", f"--Functions={f}")),
    "flatten+arith":       lambda f: tf(("Flatten", f"--Functions={f}"), ("EncodeArithmetic", f"--Functions={f}")),
    "virt+flatten":        lambda f: tf(("Flatten", f"--Functions={f}"), ("Virtualize", f"--Functions={f}", "--VirtualizeDispatch=switch")),
}

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)

def compile_to(src, exe):
    r = run([CC, str(src), "-O1", "-w", "-o", str(exe)])
    return r.returncode == 0, r.stderr

def output_of(exe, argv):
    try:
        r = run([str(exe), *argv])
        return r.returncode, r.stdout
    except Exception as e:
        return None, f"<run error: {e}>"

def preflight():
    missing = []
    if not shutil.which(TIGRESS) and not Path(TIGRESS).exists():
        missing.append(f"Tigress ('{TIGRESS}'): install from https://tigress.wtf/ and set TIGRESS=/path/to/tigress")
    if not shutil.which(CC) and not Path(CC).exists():
        missing.append(f"C compiler ('{CC}'): install gcc or clang, or set CC=")
    if missing:
        print("Cannot build the corpus yet -- missing tooling:\n  - " + "\n  - ".join(missing))
        print("\nTigress is a source-to-source obfuscator (not on PATH by default). On Windows,\n"
              "run this under WSL or point ENV at your MinGW toolchain. Everything else is ready.")
        sys.exit(2)

def main():
    preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    exe = lambda p: p.with_suffix(".exe" if os.name == "nt" else "")
    manifest, ok, skipped = [], 0, 0
    for seed, spec in SEED_SPEC.items():
        src = SEEDS / seed
        if not src.exists():
            continue
        base = OUT / Path(seed).stem
        clean_exe = exe(base.with_name(base.name + "_clean"))
        cok, cerr = compile_to(src, clean_exe)
        if not cok:
            print(f"[skip] {seed}: clean build failed\n{cerr}"); continue
        for cfg_name, cfg in CONFIGS.items():
            obf_c = base.with_name(f"{base.name}.{cfg_name}.c")
            cmd = [TIGRESS, f"--Environment={ENV}", *cfg(spec["funcs"]), f"--out={obf_c}", str(src)]
            tr = run(cmd)
            if tr.returncode != 0 or not obf_c.exists():
                print(f"[skip] {seed} / {cfg_name}: tigress failed\n  {' '.join(cmd)}\n  {tr.stderr.strip()[:300]}")
                skipped += 1; continue
            obf_exe = exe(base.with_name(f"{base.name}.{cfg_name}"))
            ook, oerr = compile_to(obf_c, obf_exe)
            if not ook:
                print(f"[skip] {seed} / {cfg_name}: obf build failed\n{oerr[:300]}"); skipped += 1; continue
            for argv in spec["inputs"]:
                rc_c, out_c = output_of(clean_exe, argv)
                rc_o, out_o = output_of(obf_exe, argv)
                matched = (rc_c == rc_o) and (out_c == out_o)
                manifest.append({
                    "seed": seed, "config": cfg_name, "argv": argv,
                    "clean": str(clean_exe), "obfuscated": str(obf_exe),
                    "expected_rc": rc_c, "expected_stdout": out_c,
                    "matched": matched,
                })
                ok += 1
                if not matched:
                    print(f"[DIVERGENCE] {seed} / {cfg_name} argv={argv}: obfuscation changed behaviour!")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    mism = sum(1 for m in manifest if not m["matched"])
    print(f"\nwrote {len(manifest)} triples ({ok} runs, {skipped} configs skipped) -> {OUT/'manifest.json'}")
    print(f"behaviour preserved: {len(manifest)-mism}/{len(manifest)}" + ("  (!! divergences above)" if mism else ""))

if __name__ == "__main__":
    main()
