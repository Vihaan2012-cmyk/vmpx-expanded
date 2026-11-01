// User-function devirt coverage: measures the EXPORTED virtualized functions
// (export -> [jmp ->] `push <seed>; call <wrapper>` stub), not the VMProtect
// runtime scaffold that `discover_vm_entries` / `xpl_coverage` sweeps.
// Static only. Per-entry wall-clock cap via XPL_ENTRY_TIMEOUT_SECS (default 60).
//
// Usage: xpl_user_coverage IMAGE [INSTRUCTION_LIMIT]

use std::collections::BTreeSet;
use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use iced_x86::{Code, Decoder, DecoderOptions, FlowControl, Mnemonic};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    EntryProfileRecoveryLimits, VmCfgAssumption, VmCfgLimits, VmDiscoveryLimits,
    analyze_vm_entry_at, recover_vm_cfg,
};

const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;

enum Outcome {
    Complete { assumptions: Vec<VmCfgAssumption> },
    Incomplete { causes: Vec<String> },
    NoEntry,
    Error(String),
    Panic,
    Timeout,
}

struct Counts {
    blocks: usize,
    edges: usize,
    instructions: usize,
}

struct Result {
    outcome: Outcome,
    counts: Option<Counts>,
}

// Resolve an export RVA to its `push seed; call wrapper` stub. Returns
// (stub RVA, wrapper RVA, hops) or a skip reason.
fn resolve_stub(pe: &PeImage<'_>, data: &[u8], export: u32) -> std::result::Result<(u32, u32, u32), String> {
    let mut rva = export;
    let mut hops = 0;
    loop {
        let off = pe
            .rva_to_file_offset(Rva(rva))
            .map_err(|e| format!("unmapped {rva:#x}: {e}"))?
            .0 as usize;
        let end = (off + 32).min(data.len());
        let mut dec = Decoder::with_ip(if pe.optional.kind == vmpx_pe::PeKind::Pe32Plus { 64 } else { 32 }, &data[off..end], u64::from(rva), DecoderOptions::NONE);
        let first = dec.decode();
        if first.flow_control() == FlowControl::UnconditionalBranch
            && first.near_branch_target() != 0
            && matches!(first.code(), Code::Jmp_rel32_32 | Code::Jmp_rel32_64 | Code::Jmp_rel8_32 | Code::Jmp_rel8_64)
        {
            hops += 1;
            if hops > 2 {
                return Err("jmp chain too long".into());
            }
            rva = first.near_branch_target() as u32;
            continue;
        }
        if first.mnemonic() == Mnemonic::Push && first.immediate_kind_is_imm() {
            let second = dec.decode();
            if second.flow_control() == FlowControl::Call && second.near_branch_target() != 0 {
                return Ok((rva, second.near_branch_target() as u32, hops));
            }
            return Err(format!("push imm at {rva:#x} not followed by direct call ({:?})", second.mnemonic()));
        }
        return Err(format!("starts with {:?} at {rva:#x}, not push-seed;call", first.mnemonic()));
    }
}

trait ImmKind {
    fn immediate_kind_is_imm(&self) -> bool;
}
impl ImmKind for iced_x86::Instruction {
    fn immediate_kind_is_imm(&self) -> bool {
        use iced_x86::OpKind;
        matches!(
            self.op0_kind(),
            OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64
                | OpKind::Immediate8to16 | OpKind::Immediate8to32 | OpKind::Immediate8to64
                | OpKind::Immediate32to64
        )
    }
}

fn normalize(reason: &str) -> String {
    let bytes = reason.as_bytes();
    let mut out = String::with_capacity(reason.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'0' && i + 1 < bytes.len() && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
            i += 2;
            while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                i += 1;
            }
            out.push('#');
        } else if c.is_ascii_digit() {
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            out.push('#');
        } else {
            out.push(c as char);
            i += 1;
        }
    }
    out
}

fn recover_one(pe: &PeImage<'_>, stub: u32, instruction_limit: usize) -> Result {
    let Some(candidate) = analyze_vm_entry_at(pe, Rva(stub), VmDiscoveryLimits::default()) else {
        return Result { outcome: Outcome::NoEntry, counts: None };
    };
    let limits = VmCfgLimits {
        entry: EntryProfileRecoveryLimits { instruction_limit, ..EntryProfileRecoveryLimits::default() },
        ..VmCfgLimits::default()
    };
    match recover_vm_cfg(pe, &candidate, limits) {
        Err(e) => Result { outcome: Outcome::Error(normalize(&e.to_string())), counts: None },
        Ok(cfg) => {
            let counts = Some(Counts {
                blocks: cfg.blocks.len(),
                edges: cfg.edges.len(),
                instructions: cfg.total_instructions(),
            });
            let outcome = if cfg.all_paths_reach_vm_exit() {
                Outcome::Complete { assumptions: cfg.assumptions.iter().copied().collect() }
            } else {
                let mut causes: Vec<String> = cfg.stalls.iter().map(|s| normalize(&s.reason)).collect();
                causes.sort();
                causes.dedup();
                if let Some(hit) = cfg.limit_hit {
                    causes.push(format!("limit {hit:?}"));
                }
                if causes.is_empty() {
                    causes.push("(no stall, no vm-exit block)".to_owned());
                }
                Outcome::Incomplete { causes }
            };
            Result { outcome, counts }
        }
    }
}

fn tier(a: &[VmCfgAssumption]) -> &'static str {
    let mut rank = 0;
    for x in a {
        rank = rank.max(match x {
            VmCfgAssumption::ObservedGlobal
            | VmCfgAssumption::ObservedRecursionDepth
            | VmCfgAssumption::ObservedBranchTarget => 3,
            VmCfgAssumption::AssumesExternalStoresDisjointFromVmStack => 2,
            VmCfgAssumption::PreferredImageBase | VmCfgAssumption::OverApproximatedEnvironmentGate => 1,
        });
    }
    ["PROVEN", "BENIGN", "UNPROVEN-ASSUMPTION", "OBSERVED"][rank]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_path = args.get(1).expect("usage: xpl_user_coverage IMAGE [INSTRUCTION_LIMIT]");
    let instruction_limit: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(100_000);
    let timeout = Duration::from_secs(
        std::env::var("XPL_ENTRY_TIMEOUT_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(60),
    );
    let t0 = Instant::now();

    let bytes = std::fs::read(image_path).unwrap_or_else(|e| panic!("read {image_path}: {e}"));
    let input_pe = PeImage::parse(&bytes).expect("input is not a supported PE");
    let virtual_only = input_pe
        .sections
        .iter()
        .filter(|s| {
            s.pointer_to_raw_data.0 == 0
                && s.size_of_raw_data == 0
                && s.virtual_size != 0
                && s.characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA == 0
        })
        .count();
    let mapped: Vec<u8> = if virtual_only >= 3 {
        println!("[load] packed -> unpacking in-process");
        vmpx_unpack::unpack_to_mapped_image(&bytes, vmpx_unpack::UnpackLimits::default())
            .expect("unpack failed")
            .image
    } else {
        bytes.clone()
    };
    // Leaked so per-entry worker threads are 'static and a timed-out one can be abandoned.
    let mapped: &'static [u8] = Box::leak(mapped.into_boxed_slice());
    let pe: &'static PeImage<'static> = Box::leak(Box::new(PeImage::parse(mapped).expect("reparse")));

    let exports = pe.report().exports;
    let symbols = exports.map(|e| e.symbols).unwrap_or_default();
    println!("[image] {image_path}: {} exports", symbols.len());

    // stub RVA -> export names
    let mut stubs: Vec<(u32, u32, u32, Vec<String>)> = Vec::new();
    let mut skipped = 0usize;
    for sym in &symbols {
        let name = sym.name.clone().unwrap_or_else(|| format!("#{}", sym.ordinal));
        if sym.forwarder.is_some() {
            println!("  skip {name} ({:#x}): forwarder", sym.rva.0);
            skipped += 1;
            continue;
        }
        match resolve_stub(pe, mapped, sym.rva.0) {
            Ok((stub, wrapper, hops)) => {
                if let Some(s) = stubs.iter_mut().find(|s| s.0 == stub) {
                    s.3.push(name);
                } else {
                    stubs.push((stub, wrapper, hops, vec![name]));
                }
            }
            Err(why) => {
                println!("  skip {name} ({:#x}): not a virtualized user function: {why}", sym.rva.0);
                skipped += 1;
            }
        }
    }
    println!("[resolve] {} user stubs from {} exports ({skipped} skipped)", stubs.len(), symbols.len());

    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let mut results: Vec<Result> = Vec::new();
    for (stub, wrapper, hops, names) in &stubs {
        let t = Instant::now();
        let (tx, rx) = mpsc::channel();
        let stub_rva = *stub;
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(move || {
                let r = panic::catch_unwind(AssertUnwindSafe(|| recover_one(pe, stub_rva, instruction_limit)));
                let _ = tx.send(r.unwrap_or(Result { outcome: Outcome::Panic, counts: None }));
            })
            .expect("spawn");
        let r = rx.recv_timeout(timeout).unwrap_or(Result { outcome: Outcome::Timeout, counts: None });
        let (label, detail) = match &r.outcome {
            Outcome::Complete { assumptions } => ("COMPLETE", format!("tier={} {assumptions:?}", tier(assumptions))),
            Outcome::Incomplete { causes } => ("STALL", causes.join(" | ")),
            Outcome::NoEntry => ("NO-ENTRY", String::new()),
            Outcome::Error(m) => ("ERROR", m.clone()),
            Outcome::Panic => ("PANIC", String::new()),
            Outcome::Timeout => ("TIMEOUT", format!(">{}s", timeout.as_secs())),
        };
        let c = r.counts.as_ref().map_or_else(
            || "-".to_owned(),
            |c| format!("blocks={} edges={} instrs={}", c.blocks, c.edges, c.instructions),
        );
        println!(
            "  stub {stub:#x} (wrapper {wrapper:#x}, {hops} jmp hop) {} -> {label} {c} [{:.1}s] {detail}",
            names.join(","),
            t.elapsed().as_secs_f64()
        );
        results.push(r);
    }
    panic::set_hook(prev_hook);

    let n = results.len();
    let complete = results.iter().filter(|r| matches!(r.outcome, Outcome::Complete { .. })).count();
    let mut tiers: [usize; 4] = [0; 4];
    let mut proven_empty = 0usize;
    for r in &results {
        if let Outcome::Complete { assumptions } = &r.outcome {
            let t = tier(assumptions);
            tiers[["PROVEN", "BENIGN", "UNPROVEN-ASSUMPTION", "OBSERVED"].iter().position(|x| *x == t).unwrap()] += 1;
            if assumptions.is_empty() {
                proven_empty += 1;
            }
        }
    }
    let stalls = results.iter().filter(|r| matches!(r.outcome, Outcome::Incomplete { .. })).count();
    let timeouts = results.iter().filter(|r| matches!(r.outcome, Outcome::Timeout)).count();
    let other = n - complete - stalls - timeouts;
    let _ = BTreeSet::<u8>::new();
    println!("\n==== USER-FUNCTION COVERAGE: {image_path}");
    println!("exports {} | virtualized user stubs {n} | non-virtualized/skipped {skipped}", symbols.len());
    println!("complete {complete}/{n} | proven(empty-assumptions) {proven_empty}/{n}");
    println!(
        "tiers: PROVEN {} BENIGN {} UNPROVEN-ASSUMPTION {} OBSERVED {} | stalls {stalls} timeouts {timeouts} other(err/panic/no-entry) {other}",
        tiers[0], tiers[1], tiers[2], tiers[3]
    );
    println!("wall {:.1}s", t0.elapsed().as_secs_f64());
    std::process::exit(0);
}
