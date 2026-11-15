//! Emulator throughput and dispatch mix.
//! usage: throughput <mapped.bin> <max_steps>
//!
//! Sets the guest up the way the detonation does (import stubs, export
//! reference, XPLM, undefined-flag resolution), runs DllMain(PROCESS_ATTACH),
//! then reports steps/sec and where the steps went: whole blocks vs single
//! steps, fast path vs generic `exec`, and which mnemonics forced the generic
//! path.
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use vmpx_emulate::{LoadOptions, Stop, WinApiHook, Xplm, load};
use vmpx_pe::PeImage;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, path, steps] = &args[..] else {
        eprintln!("usage: throughput <mapped.bin> <max_steps>");
        std::process::exit(2);
    };
    let max_steps: u64 = steps.parse().expect("max_steps");
    let bytes = std::fs::read(path).expect("read input");
    let pe = PeImage::parse(&bytes).expect("parse pe");

    let mut img = load(&pe, &LoadOptions::default()).expect("load");
    let base = img.image_base;
    let entry = img.entry.expect("entry point");

    let mut reference = vmpx_emulate::export_reference_from_imports(&pe);
    let db = vmpx_emulate::SysDllDb::new();
    let mut modules: std::collections::BTreeSet<String> =
        vmpx_emulate::DEFAULT_SYSTEM_MODULES.iter().map(|m| (*m).to_owned()).collect();
    modules.extend(reference.keys().cloned());
    for module in modules {
        if let Some(exports) = db.exports(&module) {
            reference.insert(module, exports);
        }
    }
    let mut winapi = WinApiHook::from_image(&img).with_export_reference(reference);
    if let Some(numbers) = db.syscall_numbers("ntdll.dll") {
        winapi = winapi.with_syscall_numbers(numbers);
    }
    winapi = winapi.with_xplm(Rc::new(RefCell::new(Xplm::new())));

    // DllMain(hinst, DLL_PROCESS_ATTACH, NULL).
    let mut emu = img.prepare(entry, &[base, 1, 0]).expect("prepare");
    emu.hook = Some(Box::new(winapi));
    emu.resolve_undefined_flags = true;
    emu.record_trace = false;
    emu.accelerate_checksums = std::env::var("ACCEL").is_ok_and(|v| v == "1");

    // Sample rip at slice boundaries. Slicing leaves the block path intact, so
    // the rate stays representative while showing whether the run is spinning.
    const SLICE: u64 = 10_000;
    let mut samples: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    let start = Instant::now();
    let mut stop = Stop::StepLimit;
    while emu.steps() < max_steps {
        let budget = SLICE.min(max_steps - emu.steps());
        let s = emu.run(budget);
        *samples.entry(emu.cpu.rip).or_default() += 1;
        if !matches!(s, Stop::StepLimit) {
            stop = s;
            break;
        }
    }
    let secs = start.elapsed().as_secs_f64();

    let done = emu.steps();
    #[allow(clippy::cast_precision_loss)]
    let rate = done as f64 / secs.max(f64::MIN_POSITIVE);
    println!("stop={stop:?} steps={done} wall={secs:.2}s rate={:.2}M steps/s", rate / 1e6);

    let p = &emu.prof;
    let dispatched = p.fast + p.slow;
    #[allow(clippy::cast_precision_loss)]
    let pct = |n: u64| if dispatched == 0 { 0.0 } else { 100.0 * n as f64 / dispatched as f64 };
    println!("fast={} ({:.1}%)  slow={} ({:.1}%)", p.fast, pct(p.fast), p.slow, pct(p.slow));
    println!(
        "blocks={} block_ops={} ({:.1}% of steps)  single_steps={}",
        p.blocks,
        p.block_ops,
        pct(p.block_ops),
        p.single
    );
    #[allow(clippy::cast_precision_loss)]
    {
        println!(
            "fused_jumps={} ({:.1}% of steps)",
            p.fused_jumps,
            pct(p.fused_jumps)
        );
        let pair_share = if p.jcc_total == 0 {
            0.0
        } else {
            100.0 * p.cmp_jcc_pairs as f64 / p.jcc_total as f64
        };
        println!(
            "jcc={} cmp+jcc pairs={} ({pair_share:.1}% of jcc, {:.1}% of steps)",
            p.jcc_total,
            p.cmp_jcc_pairs,
            pct(p.cmp_jcc_pairs)
        );
    }
    println!(
        "folded={} folded_iterations={} copies={} copied_bytes={} declined={}",
        p.folded, p.folded_iterations, p.copies_folded, p.copied_bytes, p.fold_declined
    );
    #[allow(clippy::cast_precision_loss)]
    if p.blocks > 0 {
        println!("mean ops per block run: {:.1}", p.block_ops as f64 / p.blocks as f64);
    }
    if p.slow > 0 {
        println!("top slow-path mnemonics (share of all dispatched steps):");
        for (index, count) in p.top_slow(24) {
            let name = iced_x86::Mnemonic::try_from(index)
                .map_or_else(|_| format!("#{index}"), |m| format!("{m:?}"));
            println!("  {count:>12}  {:>5.2}%  {name}", pct(count));
        }
    }
    if matches!(stop, Stop::StepLimit) {
        println!("note: hit the step limit; rip={:#x}", emu.cpu.rip);
    }
    let mut hot: Vec<(u64, u64)> = samples.into_iter().collect();
    hot.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    let total: u64 = hot.iter().map(|(_, n)| *n).sum();
    println!("
distinct rip samples: {} over {total} slice boundaries", hot.len());
    // How concentrated is the run? A window tally separates "spinning in one
    // loop" from "sweeping broadly", which need opposite fixes.
    let hottest = hot.first().map_or(0, |(rip, _)| *rip);
    let near: u64 = hot
        .iter()
        .filter(|(rip, _)| rip.abs_diff(hottest) <= 0x10000)
        .map(|(_, n)| *n)
        .sum();
    #[allow(clippy::cast_precision_loss)]
    {
        println!(
            "samples within 64KiB of the hottest address: {near}/{total} ({:.1}%)",
            100.0 * near as f64 / total as f64
        );
    }
    println!("hottest sample addresses (rva from base {base:#x}):");
    #[allow(clippy::cast_precision_loss)]
    for (rip, count) in hot.iter().take(12) {
        println!(
            "  {count:>8}  {:>5.1}%  rip={rip:#x}  rva={:#x}",
            100.0 * *count as f64 / total as f64,
            rip.wrapping_sub(base)
        );
    }
}
