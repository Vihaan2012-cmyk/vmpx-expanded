//! Decrypt VMProtect `.svmpstr` strings by emulating each decrypt call.
//! usage: decrypt_svmpstr <pe> <pairs.txt>
//! pairs.txt: `<decryptor_va_hex> <record_va_hex>` per line (# comments ok).
//! Each line: set rcx=record, call the decryptor, read the plaintext at rax.
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use vmpx_emulate::{
    Cpu, DEFAULT_SYSTEM_MODULES, LoadOptions, Memory, SysDllDb, WinApiHook, Xplm,
    export_reference_from_imports, load,
};
use vmpx_pe::PeImage;

fn bytes_at(mem: &Memory, addr: u64, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    for i in 0..n {
        if mem.read(addr + i as u64, &mut out[i..i + 1]).is_err() {
            out.truncate(i);
            break;
        }
    }
    out
}

fn cstr(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

fn wstr(b: &[u8]) -> String {
    let mut u = Vec::new();
    for pair in b.chunks_exact(2) {
        let c = u16::from_le_bytes([pair[0], pair[1]]);
        if c == 0 {
            break;
        }
        u.push(c);
    }
    String::from_utf16_lossy(&u)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read(&args[1]).expect("pe");
    let pairs: Vec<(u64, u64)> = std::fs::read_to_string(&args[2])
        .expect("pairs")
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let d = u64::from_str_radix(it.next()?, 16).ok()?;
            let r = u64::from_str_radix(it.next()?, 16).ok()?;
            Some((d, r))
        })
        .collect();
    let max_steps: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50_000_000);
    // WARMUP=<steps>: run the TLS callbacks + DllMain(PROCESS_ATTACH) first so the
    // protector's init-populated tables (the thing the decryptor null-reads cold)
    // are set up, THEN decrypt from that warmed memory. 0 = cold (no init).
    let warmup: u64 = std::env::var("WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(0);

    let names = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
    let pe = PeImage::parse(&raw).unwrap();
    let rep = pe.report();
    let mut img = load(&pe, &LoadOptions { stack_size: 8 << 20 }).unwrap();
    let sentinel = img.return_sentinel();
    let base = img.image_base;
    let gs = img.gs_base;
    let initial_rsp = img.initial_rsp;
    let entry = img.entry.unwrap();
    let tls: Vec<u64> = rep.tls.as_ref().map(|t| t.callbacks.iter().map(|v| v.0).collect()).unwrap_or_default();

    let mut reference = export_reference_from_imports(&pe);
    let db = SysDllDb::new();
    let mut mods: BTreeSet<String> = DEFAULT_SYSTEM_MODULES.iter().map(|m| (*m).to_owned()).collect();
    // Common app DLLs (the activation HTTP stack + usual deps) so their exports
    // are mapped when the app LoadLibrary's them and resolves imports.
    for m in [
        "wininet.dll", "ws2_32.dll", "winhttp.dll", "advapi32.dll", "shell32.dll",
        "ole32.dll", "oleaut32.dll", "shlwapi.dll", "gdi32.dll", "crypt32.dll",
        "version.dll", "msvcrt.dll", "ucrtbase.dll", "wintrust.dll", "psapi.dll",
        "iphlpapi.dll", "userenv.dll", "secur32.dll", "dnsapi.dll", "bcrypt.dll",
    ] {
        mods.insert(m.to_owned());
    }
    mods.extend(reference.keys().cloned());
    for m in mods {
        if let Some(n) = db.exports(&m) {
            reference.insert(m, n);
        }
    }
    let mut winapi = WinApiHook::from_image(&img).with_export_reference(reference);
    if let Some(n) = db.syscall_numbers("ntdll.dll") {
        winapi = winapi.with_syscall_numbers(n);
    }
    winapi = winapi.with_xplm(Rc::new(RefCell::new(Xplm::new())));

    // Pre-populate PEB->Ldr so the protector's import resolution (which walks the
    // loaded-module list + export tables directly) finds real module bases instead
    // of an empty list -> a null -> `call 0`. This is what detonate does; without
    // it init faults at `call rax`=0. VMPX_NO_LDR disables.
    if std::env::var_os("VMPX_NO_LDR").is_none() {
        let mut modules: Vec<String> =
            DEFAULT_SYSTEM_MODULES.iter().map(|m| (*m).to_owned()).collect();
        modules.extend(rep.imports.iter().map(|m| m.name.to_lowercase()));
        winapi.preload_ldr(&mut img.memory, img.gs_base, &modules);
    }

    let facts = winapi.facts();
    let book = winapi.address_book();
    let audit = winapi.audit();
    let soi = u64::from(rep.optional.size_of_image);
    let mut emu = img.prepare(entry, &[]).unwrap();
    emu.hook = Some(Box::new(winapi));
    emu.accelerate_checksums = true;
    emu.accelerate_literals = true;
    emu.resolve_undefined_flags = true;
    emu.record_trace = std::env::var_os("TRACE").is_some();

    // Warm up: run each init phase to its return (or until the warmup budget runs out).
    if warmup > 0 {
        let mut phases: Vec<(String, u64)> = tls.iter().enumerate().map(|(i, cb)| (format!("tls{i}"), *cb)).collect();
        phases.push(("DllMain".into(), entry));
        let mut spent = 0u64;
        for (label, f) in &phases {
            emu.cpu = Cpu::new(*f);
            emu.cpu.gs_base = gs;
            emu.cpu.regs[4] = initial_rsp - 8;
            let _ = emu.mem.poke(initial_rsp - 8, &sentinel.to_le_bytes());
            emu.cpu.regs[1] = base; // hinstDLL
            emu.cpu.regs[2] = 1; // DLL_PROCESS_ATTACH
            emu.cpu.regs[8] = 0; // reserved
            let budget = warmup.saturating_sub(spent);
            if budget == 0 { break; }
            // Slice so a long unpack logs progress (rip) instead of going silent.
            let slice = 500_000_000u64;
            let phase_start = emu.steps();
            let s = loop {
                let left = warmup.saturating_sub(spent);
                if left == 0 { break vmpx_emulate::Stop::StepLimit; }
                let before = emu.steps();
                let st = emu.run_until(sentinel, slice.min(left));
                spent += emu.steps() - before;
                if !matches!(st, vmpx_emulate::Stop::StepLimit) || spent >= warmup {
                    break st;
                }
                println!("[warmup {label}] ...{spent} steps rip={:#x}", emu.cpu.rip);
                let _ = std::io::Write::flush(&mut std::io::stdout());
            };
            println!("[warmup {label}] ran {} steps stop={s:?} rip={:#x}", emu.steps() - phase_start, emu.cpu.rip);
            if emu.record_trace && matches!(s, vmpx_emulate::Stop::Fault { .. }) {
                let n = emu.trace.len();
                for te in emu.trace.iter().skip(n.saturating_sub(12)) {
                    println!("      {:#x} {:?} ea={:?} val={:?}", te.rip, te.mnemonic, te.ea, te.value);
                }
                let rl: String = (0..16).map(|r| format!(" {}={:#x}", names[r], emu.cpu.regs[r])).collect();
                println!("      regs:{rl}");
                break;
            }
        }
        println!("[warmup] total {spent} steps");
        // The authoritative import table = IAT slots that hold a known import
        // stub pointer. Scan the whole image for qwords that are address-book
        // keys (resolved stubs). This finds the APP's IAT, not just what the
        // export-walk enumerated.
        let b = book.borrow();
        let section_of = |slot: u64| -> &str {
            let rva = u32::try_from(slot - base).unwrap_or(0);
            rep.sections.iter()
                .find(|s| s.virtual_address.0 <= rva && rva < s.virtual_address.0 + s.virtual_size.max(s.size_of_raw_data))
                .map_or("?", |s| s.name.as_str())
        };
        let mut iat: Vec<(u64, String, String)> = Vec::new();
        let mut addr = base;
        while addr + 8 <= base + soi {
            if let Ok(q) = emu.mem.read_uint(addr, 8) {
                if let Some(k) = b.get(&q) {
                    iat.push((addr, k.module.clone(), k.name.clone()));
                }
            }
            addr += 8;
        }
        let distinct: BTreeSet<String> = iat.iter().map(|(_, m, n)| format!("{m}!{n:?}")).collect();
        println!("=== IAT slots filled: {} ({} distinct imports) ===", iat.len(), distinct.len());
        let mut by_sec: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        for (slot, _, _) in &iat {
            *by_sec.entry(section_of(*slot).to_owned()).or_default() += 1;
        }
        println!("    slots by section: {by_sec:?}");
        // Emit the import sidecar for the app's IAT (retained, non-protector
        // sections) in the auto `bind_slots` format: `<rva:08x>\t<module>\t<func>`.
        // Func is the bare export name, or `#N` for an ordinal import.
        if let Ok(path) = std::env::var("SIDECAR_OUT") {
            const APP_SECS: &[&str] = &[".text", ".rdata", ".data", "_RDATA", ".rodata", ".data1"];
            let mut lines: BTreeSet<String> = BTreeSet::new();
            for (slot, module, name) in &iat {
                if !APP_SECS.contains(&section_of(*slot)) {
                    continue;
                }
                // API-set contract names (api-ms-win-*/ext-ms-win-*) are virtual
                // DLLs; the loader won't accept them as an import-table module
                // name. They forward to kernelbase.dll, which exports the real
                // function, so bind against that host.
                let module = if module.starts_with("api-ms-win-") || module.starts_with("ext-ms-win-")
                {
                    "kernelbase.dll"
                } else {
                    module.as_str()
                };
                let rva = u32::try_from(slot - base).unwrap_or(0);
                lines.insert(format!("{rva:08x}\t{module}\t{name}"));
            }
            let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
            std::fs::write(&path, &body).expect("sidecar write");
            println!("    sidecar: {} app slot(s) -> {path}", lines.len());
        }
        let mut by_mod: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        for d in &distinct {
            let m = d.split('!').next().unwrap_or("?").to_owned();
            *by_mod.entry(m).or_default() += 1;
        }
        println!("    distinct imports by module: {by_mod:?}");
        if std::env::var_os("DUMP_IMPORTS").is_some() {
            for d in &distinct {
                println!("IMPORT {d}");
            }
        }
        let _ = &facts;
        // What modules/APIs the protector actually invoked (reveals missing
        // preloads, e.g. the HTTP-activation DLLs, and LoadLibrary targets).
        let a = audit.borrow();
        println!("=== audit (API calls, top 60) ===");
        let mut rows: Vec<(&String, &u64)> = a.iter().collect();
        rows.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        for (k, c) in rows.iter().take(60) {
            println!("    {c:>6}  {k}");
        }
    }

    for (dec, rec) in pairs {
        emu.cpu = Cpu::new(dec);
        emu.cpu.gs_base = gs;
        emu.cpu.regs[4] = initial_rsp - 8;
        let _ = emu.mem.poke(initial_rsp - 8, &sentinel.to_le_bytes());
        emu.cpu.regs[1] = rec; // rcx = record
        let before = emu.steps();
        let stop = emu.run_until(sentinel, max_steps);
        let ran = emu.steps() - before;
        let rax = emu.cpu.regs[0];
        let regline: String = (0..16).map(|r| format!(" {}={:#x}", names[r], emu.cpu.regs[r])).collect();
        let sample = bytes_at(&emu.mem, rax, 64);
        let hex: String = sample.iter().take(24).map(|b| format!("{b:02x}")).collect();
        println!("dec={dec:#x} rec={rec:#x} stop={stop:?} ran={ran} rax={rax:#x}");
        println!("    regs:{regline} rip={:#x}", emu.cpu.rip);
        println!("    bytes@rax: {hex}");
        println!("    ascii: {:?}", cstr(&sample));
        println!("    wide : {:?}", wstr(&sample));
    }
}
