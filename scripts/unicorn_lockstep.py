#!/usr/bin/env python3
"""Unicorn lockstep bisection for the vmpx software emulator.

Loads a `start.snap` (VMPXSNP1) + `vmpx_trace.bin` produced by
`VMPX_UNICORN=<dir>` (detonate.rs `unicorn_lockstep_dump`) and finds the first
step at which the vmpx interpreter's register state diverges from Unicorn
(QEMU) ground truth. A divergence on a normal in-image instruction is a vmpx
emulator bug; a divergence at a synthetic/stub address is the API-hook boundary
(vmpx emulates Win32 there, Unicorn cannot), not a bug.

usage: python unicorn_lockstep.py <dir-with-start.snap-and-vmpx_trace.bin>
"""
import sys, struct, os
from unicorn import *
from unicorn.x86_const import *

PAGE = 4096
REC = 18 * 8  # bytes per trace record: rip + 16 regs + rflags

REGMAP = [
    UC_X86_REG_RAX, UC_X86_REG_RCX, UC_X86_REG_RDX, UC_X86_REG_RBX,
    UC_X86_REG_RSP, UC_X86_REG_RBP, UC_X86_REG_RSI, UC_X86_REG_RDI,
    UC_X86_REG_R8, UC_X86_REG_R9, UC_X86_REG_R10, UC_X86_REG_R11,
    UC_X86_REG_R12, UC_X86_REG_R13, UC_X86_REG_R14, UC_X86_REG_R15,
]
RNAMES = ["rax","rcx","rdx","rbx","rsp","rbp","rsi","rdi",
          "r8","r9","r10","r11","r12","r13","r14","r15"]
XMM = [getattr(__import__("unicorn.x86_const", fromlist=["x"]), f"UC_X86_REG_XMM{i}") for i in range(16)]


def parse_snap(path):
    d = open(path, "rb").read()
    assert d[:8] == b"VMPXSNP1", "bad magic"
    o = 8
    regs = list(struct.unpack_from("<16Q", d, o)); o += 128
    rip, rflags, undef, gs_base = struct.unpack_from("<4Q", d, o); o += 32
    xmm = [int.from_bytes(d[o + 16 * i:o + 16 * i + 16], "little") for i in range(16)]; o += 256
    mxcsr = struct.unpack_from("<I", d, o)[0]; o += 4
    npages = struct.unpack_from("<Q", d, o)[0]; o += 8
    pages = []
    for _ in range(npages):
        base = struct.unpack_from("<Q", d, o)[0]; o += 8
        perm = d[o]; o += 1
        data = d[o:o + PAGE]; o += PAGE
        pages.append((base, perm, data))
    return dict(regs=regs, rip=rip, rflags=rflags, gs_base=gs_base,
                xmm=xmm, mxcsr=mxcsr, pages=pages)


def uc_prot(p):
    r = 0
    if p & 1: r |= UC_PROT_READ
    if p & 2: r |= UC_PROT_WRITE
    if p & 4: r |= UC_PROT_EXEC
    return r or UC_PROT_READ  # a 0-perm page still needs to exist


def build_uc(snap):
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    # Coalesce contiguous same-perm pages into bigger maps (far fewer map calls).
    pages = sorted(snap["pages"])
    i = 0
    while i < len(pages):
        base, perm, _ = pages[i]
        j = i + 1
        while j < len(pages) and pages[j][0] == pages[j - 1][0] + PAGE and pages[j][1] == perm:
            j += 1
        size = (j - i) * PAGE
        uc.mem_map(base, size, uc_prot(perm))
        blob = b"".join(p[2] for p in pages[i:j])
        uc.mem_write(base, blob)
        i = j
    for i, r in enumerate(snap["regs"]):
        uc.reg_write(REGMAP[i], r)
    uc.reg_write(UC_X86_REG_RIP, snap["rip"])
    uc.reg_write(UC_X86_REG_RFLAGS, snap["rflags"] | 0x2)
    uc.reg_write(UC_X86_REG_GS_BASE, snap["gs_base"])
    for i, x in enumerate(snap["xmm"]):
        try: uc.reg_write(XMM[i], x)
        except Exception: pass
    try: uc.reg_write(UC_X86_REG_MXCSR, snap["mxcsr"])
    except Exception: pass
    return uc


def load_trace(path):
    """Full trace as an (N,18) uint64 array: [exec_rip, 16 regs(after), rflags]."""
    import numpy as np
    a = np.fromfile(path, dtype="<u8")
    a = a[: (a.size // 18) * 18].reshape(-1, 18)
    return a, a.shape[0]


def rec_from_file(path, i):
    with open(path, "rb") as f:
        f.seek(i * REC)
        v = struct.unpack("<18Q", f.read(REC))
    return v[0], v[1:17], v[17]


def main():
    d = sys.argv[1]
    snap = parse_snap(os.path.join(d, "start.snap"))
    tpath = os.path.join(d, "vmpx_trace.bin")
    tr, n = load_trace(tpath)
    exec_rips = tr[:, 0]
    print(f"snapshot: {len(snap['pages'])} pages, entry rip={snap['rip']:#x}, gs_base={snap['gs_base']:#x}")
    print(f"trace: {n} steps; forward-scanning RIP+GP-regs (rep-safe, code-hook driven)")

    uc = build_uc(snap)
    state = {"step": 0, "div": None, "regdiv": None}

    pagemap = {b: d for b, _, d in snap["pages"]}
    def snap_bytes(_snap, addr, nb):
        out = bytearray()
        for j in range(nb):
            a = addr + j
            pg = pagemap.get(a & ~0xFFF)
            out.append(pg[a & 0xFFF] if pg else 0)
        return bytes(out)

    # Neutralize rdtsc (expected nondeterministic) by overwriting Unicorn's
    # rax/rdx to the vmpx-recorded value right after each rdtsc executes, so only
    # a genuine downstream divergence is reported. rdtsc rips scanned from image.
    import numpy as np
    RDTSC_RIPS = set()
    uniq = np.unique(exec_rips)
    for r in uniq.tolist():
        pg = snap_bytes(snap, int(r), 2)
        if pg == b"\x0f\x31":
            RDTSC_RIPS.add(int(r))
    print(f"rdtsc rips: {[hex(x) for x in RDTSC_RIPS]}")

    def hook(uc, address, size, _):
        i = state["step"]
        if i >= n:
            uc.emu_stop(); return
        # Unicorn fires UC_HOOK_CODE once per `rep` ITERATION (same rip), while
        # vmpx collapses the whole rep into ONE step. Skip the continuations.
        if i > 0 and address == int(exec_rips[i - 1]) and address != int(exec_rips[i]):
            return
        if address != int(exec_rips[i]):
            state["div"] = (i, address, int(exec_rips[i]))
            uc.emu_stop(); return
        # If the previous step was an rdtsc, overwrite Unicorn's host TSC with the
        # vmpx-recorded value so the expected rdtsc difference is neutralized.
        if i > 0 and int(exec_rips[i - 1]) in RDTSC_RIPS:
            uc.reg_write(UC_X86_REG_RAX, int(tr[i - 1][1]))
            uc.reg_write(UC_X86_REG_RDX, int(tr[i - 1][3]))
        # Compare GP registers to the PREVIOUS step's post-state (== current pre-state).
        if i > 0:
            vals = uc.reg_read_batch(REGMAP)
            row = tr[i - 1]
            for k in range(16):
                if vals[k] != int(row[1 + k]):
                    state["regdiv"] = (i, k, int(row[1 + k]), vals[k])
                    uc.emu_stop(); return
        state["step"] = i + 1

    uc.hook_add(UC_HOOK_CODE, hook)
    fault = None
    try:
        uc.emu_start(snap["rip"], 0, count=0)
    except UcError as e:
        fault = e

    i = state["step"]
    if state["regdiv"] is not None:
        si, k, vmpx_v, uc_v = state["regdiv"]
        buggy_exec = int(tr[si - 1][0])
        print(f"\nFIRST REGISTER DIVERGENCE at step {si} — reg {RNAMES[k]}")
        print(f"  instruction that produced it: exec_rip={buggy_exec:#x}")
        print(f"  {RNAMES[k]}  vmpx={vmpx_v:#018x}   unicorn(truth)={uc_v:#018x}")
        # Decode that instruction + show full reg/flag context before and after.
        try:
            from capstone import Cs, CS_ARCH_X86, CS_MODE_64
            md = Cs(CS_ARCH_X86, CS_MODE_64)
            code = bytes(uc.mem_read(buggy_exec, 16))
            ins = next(md.disasm(code, buggy_exec), None)
            print(f"  insn: {code[:ins.size].hex() if ins else code[:8].hex()}  "
                  f"{(ins.mnemonic+' '+ins.op_str) if ins else '?'}")
        except Exception as e:
            print(f"  insn bytes: {bytes(uc.mem_read(buggy_exec,8)).hex()} (capstone: {e})")
        # All regs differing at this step, + the pre-state (trace[si-2]).
        row_after = tr[si - 1]
        vals = uc.reg_read_batch(REGMAP)
        print("  regs AFTER this instruction (vmpx vs unicorn):")
        for kk in range(16):
            v, u = int(row_after[1 + kk]), vals[kk]
            print(f"    {RNAMES[kk]:<4} vmpx={v:#018x}  unicorn={u:#018x}{'   <<< DIFF' if v!=u else ''}")
        print(f"  rflags AFTER vmpx={int(row_after[17]):#x}  unicorn={uc.reg_read(UC_X86_REG_RFLAGS):#x}")
        return
    if state["div"] is None and state["regdiv"] is None:
        print(f"\nNO DIVERGENCE — vmpx matched Unicorn (rip+GP regs) through step "
              f"{i} of {n}{' (reached end)' if i>=n else f', Unicorn stopped early: {fault}'}. "
              f"With rdtsc neutralized this means the TSC VALUE is the sole cause.")
        return
    if state["div"] is None and fault is not None:
        # Unicorn faulted where vmpx kept going: divergence at the current step.
        exec_rip = int(exec_rips[i]) if i < n else 0
        print(f"\nUNICORN FAULTED at step {i}: {fault}")
        print(f"  vmpx next exec_rip = {exec_rip:#x}  uc_rip={uc.reg_read(UC_X86_REG_RIP):#x}")
        _classify(d, tpath, snap, i, exec_rip, uc, uc.reg_read(UC_X86_REG_RIP))
        return

    i, uc_rip, want_rip = state["div"]
    print(f"\nFIRST RIP DIVERGENCE at step {i}")
    print(f"  vmpx went to {want_rip:#x}")
    print(f"  unicorn went to {uc_rip:#x}")
    _classify(d, tpath, snap, i, want_rip, uc, uc_rip)


FLAGBITS = [(0,"CF"),(2,"PF"),(4,"AF"),(6,"ZF"),(7,"SF"),(10,"DF"),(11,"OF")]


def _classify(d, tpath, snap, i, want_rip, uc, uc_rip=None):
    prev_exec, prev_regs, prev_rflags = rec_from_file(tpath, i - 1)
    print(f"  prev step exec_rip = {prev_exec:#x}  (the branch whose direction forked)")
    # The branch does not change GP regs or flags, so uc's state == pre/post-branch.
    uc_rflags = uc.reg_read(UC_X86_REG_RFLAGS)
    print(f"  rflags  vmpx={prev_rflags:#x}  unicorn={uc_rflags:#x}")
    diff = prev_rflags ^ uc_rflags
    named = [n for b, n in FLAGBITS if diff & (1 << b)]
    print(f"  FLAG BITS DIFFERING: {named if named else '(none in CF/PF/AF/ZF/SF/DF/OF)'}")
    # GP registers (should match if it is purely a flag bug).
    regdiff = []
    for k in range(16):
        uv = uc.reg_read(REGMAP[k])
        if uv != prev_regs[k]:
            regdiff.append((RNAMES[k], prev_regs[k], uv))
    if regdiff:
        print("  GP REGISTERS DIFFERING (vmpx vs unicorn):")
        for n, v, u in regdiff:
            print(f"    {n:<4} vmpx={v:#018x}  unicorn={u:#018x}")
    else:
        print("  GP registers all match -> pure FLAG miscomputation upstream.")
    # Disassemble the branch and the two targets.
    try:
        from capstone import Cs, CS_ARCH_X86, CS_MODE_64
        md = Cs(CS_ARCH_X86, CS_MODE_64)
        for label, addr in [("branch", prev_exec), ("vmpx-target", want_rip), ("uc-target", uc_rip)]:
            if addr is None:
                continue
            code = bytes(uc.mem_read(addr, 16))
            ins = next(md.disasm(code, addr), None)
            txt = f"{ins.mnemonic} {ins.op_str}" if ins else "?"
            print(f"  {label:12} {addr:#x}: {code[:8].hex():<16} {txt}")
    except Exception as e:
        for label, addr in [("branch", prev_exec), ("vmpx-target", want_rip), ("uc-target", uc_rip)]:
            if addr is not None:
                print(f"  {label:12} {addr:#x}: {bytes(uc.mem_read(addr,8)).hex()}")


if __name__ == "__main__":
    main()
