//! `vmpx diff-original`: ground-truth differential. Each virtualized user
//! function recovered from the protected binary is compared, by the
//! equivalence gate, against the same function lifted statically from the
//! unprotected original. Nothing is executed.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use iced_x86::{Code, Decoder, DecoderOptions, FlowControl, Mnemonic, OpKind};
use vmpx_mutation::cfg_recovery::{CfgOptions, recover_native_cfg_pe_with};
use vmpx_pe::{PeImage, Rva};
use vmpx_verify::differential::{AbiPolicy, DiffOutcome, diff_against_original, execute_native_function};
use vmpx_verify::equiv::{Budget, CellVerdict, EquivVerdict, Method, Verifier};

pub(crate) struct DiffOptions {
    pub protected: PathBuf,
    pub original: PathBuf,
    pub only: Vec<String>,
    pub exhaustive_bits: u32,
    pub timeout_ms: u32,
    pub instruction_limit: usize,
    pub strict_gates: bool,
    pub max_blocks: usize,
    pub max_paths: usize,
    /// Negative control: compare each recovered function against the NEXT
    /// export's original instead of its own. Every verdict must be REFUTED.
    pub control: bool,
}

pub(crate) fn run(o: DiffOptions) -> Result<()> {
    std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(move || diff(&o))
        .context("spawn")?
        .join()
        .map_err(|_| anyhow::anyhow!("worker panicked"))?
}

fn is_push_imm(i: &iced_x86::Instruction) -> bool {
    i.mnemonic() == Mnemonic::Push
        && matches!(
            i.op0_kind(),
            OpKind::Immediate8
                | OpKind::Immediate16
                | OpKind::Immediate32
                | OpKind::Immediate64
                | OpKind::Immediate8to16
                | OpKind::Immediate8to32
                | OpKind::Immediate8to64
                | OpKind::Immediate32to64
        )
}

/// Export -> (jmp ->)* `push seed; call wrapper` stub RVA.
pub(crate) fn resolve_stub(pe: &PeImage<'_>, data: &[u8], export: u32) -> std::result::Result<u32, String> {
    let mut rva = export;
    for _ in 0..3 {
        let off = pe
            .rva_to_file_offset(Rva(rva))
            .map_err(|e| format!("unmapped {rva:#x}: {e}"))?
            .0 as usize;
        let end = (off + 32).min(data.len());
        let bits = if pe.optional.kind == vmpx_pe::PeKind::Pe32Plus { 64 } else { 32 };
        let mut dec = Decoder::with_ip(bits, &data[off..end], u64::from(rva), DecoderOptions::NONE);
        let first = dec.decode();
        if matches!(
            first.code(),
            Code::Jmp_rel32_32 | Code::Jmp_rel32_64 | Code::Jmp_rel8_32 | Code::Jmp_rel8_64
        ) {
            rva = first.near_branch_target() as u32;
            continue;
        }
        if is_push_imm(&first) {
            let second = dec.decode();
            if second.flow_control() == FlowControl::Call && second.near_branch_target() != 0 {
                return Ok(rva);
            }
        }
        return Err(format!("not a push-seed;call stub at {rva:#x}"));
    }
    Err("jmp chain too long".into())
}

fn verdict_word(o: &DiffOutcome) -> &'static str {
    match o.verdict {
        EquivVerdict::Proven => "PROVEN",
        EquivVerdict::Refuted => "REFUTED",
        EquivVerdict::Timeout => "TIMEOUT",
    }
}

fn methods(o: &DiffOutcome) -> String {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    if let Some(report) = &o.report {
        for (_, v) in report.paths.iter().flat_map(|p| &p.cells) {
            if let CellVerdict::Proven(m) = v {
                let key = match m {
                    Method::Structural => "structural".to_owned(),
                    Method::Solver(n) => n.clone(),
                };
                *counts.entry(key).or_default() += 1;
            }
        }
    }
    counts.iter().map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join(" ")
}

fn diff(o: &DiffOptions) -> Result<()> {
    let original_bytes = std::fs::read(&o.original).context("read original")?;
    let original = PeImage::parse(&original_bytes).context("original is not a supported PE")?;
    let original_exports: BTreeMap<String, u32> = original
        .report()
        .exports
        .map(|e| e.symbols)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|s| Some((s.name?, s.rva.0)))
        .collect();

    let protected_bytes = std::fs::read(&o.protected).context("read protected")?;
    let mapped = crate::equiv_gate::load_pe(&protected_bytes)?;
    let protected = PeImage::parse(&mapped).context("protected image failed reparse")?;

    #[allow(unused_mut)]
    let mut verifier = Verifier::new(Budget {
        exhaustive_bits: o.exhaustive_bits,
        timeout_ms: o.timeout_ms,
    });
    #[cfg(feature = "smt-z3")]
    {
        verifier = verifier
            .with_backend(Box::new(vmpx_verify::sweep::SweepBackend::new(Box::new(
                vmpx_verify::smt::Z3Backend,
            ))))
            .with_backend(Box::new(vmpx_verify::smt::Z3Backend));
    }
    let policy = AbiPolicy::win64();
    println!("protected: {}", o.protected.display());
    println!("original:  {}", o.original.display());
    println!(
        "backends: {:?}; budget: exhaustive<= {} bits, {} ms/cell; ABI: Win64 (compare rax + callee-saved; volatile regs, flags, return slot excluded)",
        verifier.backend_names(),
        o.exhaustive_bits,
        o.timeout_ms
    );

    let (mut proven, mut refuted, mut timeout, mut total) = (0, 0, 0, 0);
    let symbols = protected.report().exports.map(|e| e.symbols).unwrap_or_default();
    let mut targets: Vec<(String, u32, u32)> = Vec::new();
    for sym in symbols {
        let Some(name) = sym.name.clone() else { continue };
        let Ok(stub) = resolve_stub(&protected, &mapped, sym.rva.0) else {
            println!("{name}: not a virtualized user function (no push-seed;call stub)");
            continue;
        };
        let Some(original_rva) = original_exports.get(&name) else {
            println!("{name}: no export of that name in the original");
            continue;
        };
        targets.push((name, stub, *original_rva));
    }
    for index in 0..targets.len() {
        let (name, stub, own) = &targets[index];
        if !o.only.is_empty() && !o.only.contains(name) {
            continue;
        }
        let original_rva = if o.control {
            targets[(index + 1) % targets.len()].2
        } else {
            *own
        };
        total += 1;
        let outcome = compare_one(o, &protected, &original, *stub, original_rva, &verifier, &policy);
        match outcome.verdict {
            EquivVerdict::Proven => proven += 1,
            EquivVerdict::Refuted => refuted += 1,
            EquivVerdict::Timeout => timeout += 1,
        }
        print_outcome(name, *stub, &outcome);
    }
    println!("\n==== GROUND-TRUTH DIFFERENTIAL: {total} functions | PROVEN {proven} | REFUTED {refuted} | TIMEOUT {timeout}");
    Ok(())
}

fn compare_one(
    o: &DiffOptions,
    protected: &PeImage<'_>,
    original: &PeImage<'_>,
    stub: u32,
    original_rva: u32,
    verifier: &Verifier,
    policy: &AbiPolicy,
) -> DiffOutcome {
    let timeout = |reason: String| DiffOutcome {
        verdict: EquivVerdict::Timeout,
        reason,
        report: None,
        paths_compared: 0,
        region_paths: 0,
        spin_pruned: 0,
        frame_stores_dropped: 0,
    };
    let cfg = match recover_native_cfg_pe_with(original, Rva(original_rva), CfgOptions::PROVEN) {
        Ok(cfg) => cfg,
        Err(e) => return timeout(format!("original CFG: {e}")),
    };
    let native = match execute_native_function(&cfg, o.max_blocks) {
        Ok(n) => n,
        Err(e) => return timeout(format!("original not modelable: {e}")),
    };
    let recovered = match crate::decoded_splice::unroll_entry_cfg(
        protected,
        Rva(stub),
        o.instruction_limit,
        o.max_blocks,
        o.max_paths,
    ) {
        Ok(r) => r,
        Err(e) => return timeout(format!("protected recovery/unroll: {e}")),
    };
    println!(
        "  [{stub:#x}] original: {} blocks executed ({} branches decided, {} dead-frame stores dropped); recovered: {} paths ({} forks, {} counted-loop branches decided)",
        native.blocks_executed,
        native.branches_decided,
        native.frame_stores_dropped,
        recovered.paths.len(),
        recovered.forks,
        recovered.branches_decided
    );
    diff_against_original(&native.semantics, &recovered, verifier, policy, o.strict_gates)
}

fn print_outcome(name: &str, stub: u32, o: &DiffOutcome) {
    let gate = if o.verdict == EquivVerdict::Proven && o.modulo_env_gates() {
        format!(
            " (modulo environment gates: {} gate-failure region arm(s) and {} spin arm(s) excluded)",
            o.region_paths, o.spin_pruned
        )
    } else {
        String::new()
    };
    println!(
        "{name} [{stub:#x}]: {}{gate} | {} callee-return path(s) compared, {} VM-frame stores dropped | {}",
        verdict_word(o),
        o.paths_compared,
        o.frame_stores_dropped,
        methods(o)
    );
    if !o.reason.is_empty() {
        println!("    reason: {}", o.reason);
    }
    if let Some(report) = &o.report {
        for (i, path) in report.paths.iter().enumerate() {
            if let Some((cell, cex)) = path.first_counterexample() {
                println!(
                    "    COUNTEREXAMPLE path {i} cell {cell:?}: recovered={:#x} original={:#x} at {:x?}",
                    cex.vm, cex.recovered, cex.assignment
                );
            }
        }
    }
}
