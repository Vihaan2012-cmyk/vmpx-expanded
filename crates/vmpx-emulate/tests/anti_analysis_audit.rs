//! Env-gated audit of a real sample's init against the anti-debug/anti-VM
//! surface. `VMPX_AUDIT_SAMPLE=<pe> cargo test -p vmpx-emulate --release
//! --test anti_analysis_audit -- --ignored --nocapture`.
//!
//! Reports every API / NT info class / `cpuid` / `rdtsc` the sample issues, and
//! every read of the anti-debug fields (PEB.BeingDebugged, NtGlobalFlag,
//! ProcessHeap flags, KUSER_SHARED_DATA), plus how DllMain ended.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use vmpx_emulate::{
    Cpu, FlowEvent, Hook, HookAction, Interest, LoadOptions, MemEvent, Memory, WinApiHook, load,
};
use vmpx_pe::PeImage;

const KUSD: u64 = 0x7FFE_0000;

/// Delegates to the real `WinApiHook`, additionally counting reads of the
/// TEB / PEB / process-heap pages and `KUSER_SHARED_DATA` by field.
struct Watch {
    inner: WinApiHook,
    teb: u64,
    reads: Rc<RefCell<BTreeMap<String, u64>>>,
}

impl Hook for Watch {
    fn interest(&self) -> Interest {
        let mut i = self.inner.interest();
        i.read = [(KUSD, KUSD + 0x1000), (self.teb, self.teb + 0x2000)];
        i
    }
    fn on_mem(&mut self, ev: &MemEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        let a = ev.addr;
        let label = if (KUSD..KUSD + 0x1000).contains(&a) {
            Some(format!("KUSER_SHARED_DATA+{:#x}", a - KUSD))
        } else if (self.teb..self.teb + 0x400).contains(&a) {
            Some(format!("TEB+{:#x}", a - self.teb))
        } else if (self.teb + 0x400..self.teb + 0x800).contains(&a) {
            Some(format!("PEB+{:#x}", a - self.teb - 0x400))
        } else if (self.teb + 0x1000..self.teb + 0x2000).contains(&a) {
            Some(format!("ProcessHeap+{:#x}", a - self.teb - 0x1000))
        } else {
            None
        };
        if let Some(l) = label {
            *self.reads.borrow_mut().entry(l).or_insert(0) += 1;
        }
        self.inner.on_mem(ev, cpu, mem)
    }
    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        self.inner.on_flow(ev, cpu, mem)
    }
}

#[test]
#[ignore = "needs VMPX_AUDIT_SAMPLE"]
fn audit_real_sample_init() {
    let path = std::env::var("VMPX_AUDIT_SAMPLE").unwrap();
    let max: u64 = std::env::var("VMPX_AUDIT_MAXSTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3_000_000_000);
    let data = std::fs::read(&path).unwrap();
    let pe = PeImage::parse(&data).unwrap();
    const WATCHED: &[&str] = &[
        "IsDebuggerPresent", "CheckRemoteDebuggerPresent", "OutputDebugString", "DebugBreak",
        "NtQueryInformationProcess", "NtQuerySystemInformation", "NtSetInformationThread",
        "GetThreadContext", "SetThreadContext", "SystemFirmwareTable", "GetModuleHandle",
        "LoadLibrary", "CloseHandle", "GetTickCount", "QueryPerformance", "UnhandledException",
        "SetUnhandledExceptionFilter", "GetUserName", "GetComputerName", "NtClose",
        "IsProcessorFeaturePresent", "GetSystemInfo", "GlobalMemoryStatus", "EnumProcess",
        "CreateToolhelp32Snapshot", "RegOpenKey", "RegQueryValue", "FindWindow",
    ];
    for m in &pe.report().imports {
        for e in &m.entries {
            if let Some(n) = &e.name {
                if WATCHED.iter().any(|w| n.contains(w)) {
                    eprintln!("static import (anti-analysis-capable): {}!{n}", m.name);
                }
            }
        }
    }
    let mut img = load(&pe, &LoadOptions::default()).unwrap();
    let entry = img.entry.unwrap();
    let base = img.image_base;
    let sentinel = img.return_sentinel();
    let teb = img.gs_base;
    let inner = WinApiHook::from_image(&img);
    let audit = inner.audit();
    let reads = Rc::new(RefCell::new(BTreeMap::new()));
    let mut emu = img.prepare(entry, &[base, 1, 0]).unwrap();
    emu.hook = Some(Box::new(Watch { inner, teb, reads: Rc::clone(&reads) }));
    emu.resolve_undefined_flags = true;
    emu.record_trace = false;
    // The loader runs TLS callbacks (PROCESS_ATTACH) before DllMain.
    let rsp0 = emu.cpu.regs[4];
    // The callback array is re-read from guest memory each step, as the loader does.
    let cb_array = pe.report().tls.as_ref().map_or(0, |t| t.address_of_callbacks.0);
    for i in 0..16u64 {
        let Some(cb) = (cb_array != 0)
            .then(|| emu.mem.read_uint(cb_array + 8 * i, 8).ok())
            .flatten()
            .filter(|&v| v != 0)
        else {
            break;
        };
        let cb = &cb;
        emu.cpu.rip = *cb;
        emu.cpu.regs[4] = rsp0;
        emu.cpu.regs[1] = base;
        emu.cpu.regs[2] = 1;
        emu.cpu.regs[8] = 0;
        let before = emu.steps();
        let stop = emu.run_until(sentinel, max);
        eprintln!(
            "TLS[{i}] rva {:#x}: stop={stop:?} steps={} rax={:#x}",
            cb - base,
            emu.steps() - before,
            emu.cpu.regs[0]
        );
        if stop != vmpx_emulate::Stop::Reached {
            let mut buf = [0u8; 16];
            if emu.mem.read(emu.cpu.rip, &mut buf).is_ok() {
                let ins = iced_x86::Decoder::with_ip(64, &buf, emu.cpu.rip, 0).decode();
                eprintln!("  stopped at: {ins}");
            }
            break;
        }
    }
    emu.cpu.rip = entry;
    emu.cpu.regs[4] = rsp0;
    emu.cpu.regs[1] = base;
    emu.cpu.regs[2] = 1;
    emu.cpu.regs[8] = 0;
    let stop = emu.run_until(sentinel, max);
    eprintln!(
        "DllMain: stop={stop:?} steps={} rax={:#x} rip={:#x}",
        emu.steps(),
        emu.cpu.regs[0],
        emu.cpu.rip
    );
    eprintln!("-- API / NT anti-analysis probes (clean answer, count) --");
    for (k, n) in audit.borrow().iter() {
        eprintln!("  {k}  x{n}");
    }
    eprintln!("-- reads of anti-debug state (field, count) --");
    for (k, n) in reads.borrow().iter() {
        eprintln!("  {k}  x{n}");
    }
    eprintln!("-- instructions --");
    for ((leaf, sub), n) in &emu.audit.cpuid {
        eprintln!("  cpuid leaf={leaf:#x} sub={sub:#x}  x{n}");
    }
    eprintln!("  rdtsc x{}  rdtscp x{}", emu.audit.rdtsc, emu.audit.rdtscp);
    for (rip, step) in &emu.audit.tsc_sites {
        eprintln!("  tsc @ rva {:#x} step={step}", rip.wrapping_sub(base));
    }
}
