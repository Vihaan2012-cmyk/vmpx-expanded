#!/usr/bin/env python3
"""Protect the selected A380 functions with VMProtect (Ultra + confirmed-strongest
protections). Mirrors bigcorpus/protect_all.py's drop-overlap-and-retry loop.

*** KNOWN BLOCKER (see the de-risk report) ***
VMProtect_Con's <Procedure MapAddress="..."> resolves ONLY against names the
tool already recognizes at load time (confirmed: the file's real PE exports).
Patching in NEW export entries (add_exports.py) produces a structurally valid
export table (verified with llvm-readobj) but VMProtect still reports those
names "not found" -- even when reusing an address that resolves correctly
under its original export name. So, as built today, this script will mark
0 of the ~3500 selected internal functions; it runs the full batch anyway so
the attempt, timing, and output are on record. It is otherwise ready to use
once the input DLL has a real export table covering the selection (e.g. a
rebuild with the mingw linker's --export-all-symbols).
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


def build_export_patched_dll(manifest, out_dll):
    sys.path.insert(0, os.path.dirname(__file__))
    import add_exports
    list_path = out_dll + ".explist.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for m in manifest:
            # export name must be unique+identifier-safe; the mangled symbol
            # itself always is (v0 mangling is ASCII letters/digits/underscore).
            f.write(f"{m['addr'][2:]} {m['symbol']}\n")
    sys.argv = ["add_exports.py", ORIG_DLL, out_dll, list_path]
    add_exports.main()
    return out_dll


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

    exp_dll = os.path.join(ROOT, "protected", f"fbw_a380_systems.{tag}.exports.dll")
    print(f"patching exports for {len(batch)} functions -> {exp_dll}")
    build_export_patched_dll(batch, exp_dll)

    print(f"protecting batch '{tag}' ({len(batch)} functions, Ultra, Options={OPTIONS_ALL_CONFIRMED})")
    result = protect(exp_dll, batch, tag)
    print(json.dumps(result, indent=1))

    status_path = os.path.join(ROOT, "protect_status.json")
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}
    status[tag] = result
    json.dump(status, open(status_path, "w"), indent=1)


if __name__ == "__main__":
    main()
