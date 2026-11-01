#!/usr/bin/env python3
"""Protect the selected A380 functions with VMProtect (Ultra + confirmed-strongest
protections). Mirrors bigcorpus/protect_all.py's drop-overlap-and-retry loop.

*** STILL BLOCKED, DIFFERENT (deeper) CAUSE -- see the de-risk report ***
The DLL has been rebuilt with a surgical .def file so every selected function
IS a confirmed-real PE export (llvm-readobj: 3500/3500 present) -- the
originally diagnosed blocker (no export at all) is fixed. VMProtect_Con
STILL reports every one of them "not found", including a short plain-ASCII
.def alias pointing at the exact same RVA an export-table entry resolves
under its real name. Isolated control experiments (tiny synthetic C and
Rust DLLs, up to ~58MB and 3000 exports) show dllexport-attribute exports,
.def exports, long/short names, and export-table scale all work fine in
general -- so this is specific to these functions inside this particular
binary, not the export mechanism. Leading unconfirmed hypothesis: VMProtect
cross-references .pdata (x64 unwind-info) to find safe function boundaries,
and something about this binary's .pdata (huge, heavily templated/
monomorphized Rust code) defeats that lookup for everything except the 5
original FFI entry points. This script runs as designed; it will mark 0
until that is resolved.
"""
import json, os, re, subprocess, sys, time

ROOT = r"E:\VMPX\testcorpus\a380"
CON = r"E:\VMPX\testcorpus\vmprotect\install\VMProtect_Con.exe"
ORIG_DLL = r"E:\VMPX\a380_src\source\plugin\target\release\fbw_a380_systems.dll"

# Options bitmask, empirically confirmed bit-by-bit against this VMProtect 3.11
# build (see de-risk report): strip-debug-info(0x40000) | pack(0x100) |
# import-protection(0x200) | memory-protection(0x400). Resource-protection and
# anti-debug/anti-VM bit positions were NOT independently isolated (no GUI
# access to confirm) -- they are left OUT here rather than guessed.
OPTIONS_ALL_CONFIRMED = 0x40000 | 0x100 | 0x200 | 0x400  # 263936
COMPILATION_ULTRA = 2


def make_project(dll, manifest, proj_path, out_dll, options, batch_tag):
    procs = "\n".join(
        f'            <Procedure MapAddress="{m["symbol"]}" Options="0" CompilationType="{COMPILATION_ULTRA}" />'
        for m in manifest)
    xml = (f'<?xml version="1.0" encoding="UTF-8" ?>\n<Document Version="2">\n'
           f'    <Protection InputFileName="{dll}" Options="{options}" VMCodeSectionName=".???" '
           f'VMComplexity="20" OutputFileName="{out_dll}">\n'
           f'        <Procedures>\n{procs}\n        </Procedures>\n    </Protection>\n</Document>\n')
    with open(proj_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(xml)


def protect(dll, manifest, tag, options=OPTIONS_ALL_CONFIRMED, max_attempts=12, timeout=1800):
    proj = os.path.join(ROOT, "proj", f"a380.{tag}.vmp")
    out = os.path.join(ROOT, "protected", f"a380.{tag}.dll")
    log = os.path.join(ROOT, "logs", f"a380.{tag}.log")
    names = [m["symbol"] for m in manifest]
    dropped = []
    t0 = time.time()
    for attempt in range(max_attempts):
        make_project(dll, [m for m in manifest if m["symbol"] in names], proj, out, options, tag)
        r = subprocess.run([CON, proj], capture_output=True, text=True, timeout=timeout)
        with open(log, "w") as f:
            f.write(r.stdout + "\n" + r.stderr + f"\nrc={r.returncode}\n")
        if os.path.exists(out) and os.path.getsize(out) > 0 and "Compilation completed" in r.stdout:
            not_found = re.findall(r'\[Warning\] Function "([^"]+)" not found', r.stdout)
            overlap = re.search(r"\[Error\] (\S+)\.[0-9A-Fa-f]+: Address is already used by function", r.stdout)
            if overlap and overlap.group(1) in names:
                names.remove(overlap.group(1))
                dropped.append(overlap.group(1))
                continue
            return dict(status="ok", time=time.time() - t0, dropped=dropped,
                        not_found=not_found, marked=len(names) - len(not_found),
                        requested=len(manifest), out_size=os.path.getsize(out),
                        rc=r.returncode)
        m2 = re.search(r"\[Error\] (\S+)\.[0-9A-Fa-f]+: Address is already used by function", r.stdout)
        if m2 and m2.group(1) in names:
            names.remove(m2.group(1))
            dropped.append(m2.group(1))
            continue
        return dict(status=f"FAIL rc={r.returncode}", time=time.time() - t0, dropped=dropped,
                    stdout_tail=r.stdout[-2000:])
    return dict(status="FAIL max_attempts", time=time.time() - t0, dropped=dropped)


def main():
    manifest = json.load(open(os.path.join(ROOT, "a380_selection.json"), encoding="utf-8"))
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    tag = sys.argv[2] if len(sys.argv) > 2 else f"derisk{batch_size}"
    batch = manifest[:batch_size]

    print(f"protecting batch '{tag}' ({len(batch)} functions, Ultra, Options={OPTIONS_ALL_CONFIRMED})")
    result = protect(ORIG_DLL, batch, tag)
    print(json.dumps(result, indent=1))

    status_path = os.path.join(ROOT, "protect_status.json")
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}
    status[tag] = result
    json.dump(status, open(status_path, "w"), indent=1)


if __name__ == "__main__":
    main()
