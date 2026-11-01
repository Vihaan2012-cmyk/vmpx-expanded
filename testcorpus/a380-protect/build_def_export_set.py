#!/usr/bin/env python3
"""Drop-and-retry .def-based export build: start from the full export-safe
candidate pool (T-type, unambiguous, non-dot, user-crate), try a surgical
.def link, drop whatever ld reports as not-defined, retry. Mirrors
bigcorpus/protect_all.py's drop-overlap-and-retry loop, applied to the build
step instead of VMProtect. Ends with a confirmed real-export name list that
select_a380.py can filter against reliably."""
import os, re, subprocess, sys, time

ROOT = r"E:\VMPX\testcorpus\a380"
PLUGIN_DIR = r"E:\VMPX\a380_src\source\plugin"
DEF_PATH = os.path.join(ROOT, "a380_pool.def")
POOL_PATH = os.path.join(ROOT, "export_candidate_pool.txt")


def write_def(names):
    with open(DEF_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("EXPORTS\n")
        for n in names:
            f.write("    " + n + "\n")


def main():
    names = set()
    with open(POOL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.add(line)
    print(f"starting pool: {len(names)}")

    def_posix = DEF_PATH.replace("\\", "/")
    max_attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for attempt in range(max_attempts):
        write_def(sorted(names))
        t0 = time.time()
        r = subprocess.run(
            ["cargo", "rustc", "-p", "fbw_a380_systems_xp", "--lib", "--release",
             "--features", "js", "--", "-C", f"link-arg={def_posix}"],
            cwd=PLUGIN_DIR, capture_output=True, text=True, timeout=600,
            env={**os.environ, "RUSTC_WRAPPER": "sccache"})
        dt = time.time() - t0
        ok = r.returncode == 0
        print(f"attempt {attempt}: names={len(names)} rc={r.returncode} time={dt:.1f}s ok={ok}")
        if ok:
            print(f"SUCCESS with {len(names)} names after {attempt + 1} attempt(s)")
            with open(os.path.join(ROOT, "def_confirmed_exports.txt"), "w", encoding="utf-8") as f:
                for n in sorted(names):
                    f.write(n + "\n")
            return
        missing = set(re.findall(r"cannot export (\S+): symbol not defined", r.stderr + r.stdout))
        if not missing:
            print("FAIL: no 'cannot export' pattern found, dumping tail of output")
            print((r.stderr + r.stdout)[-3000:])
            return
        print(f"  dropping {len(missing)} unresolved names")
        names -= missing
    print("FAIL: exhausted attempts")


if __name__ == "__main__":
    main()
