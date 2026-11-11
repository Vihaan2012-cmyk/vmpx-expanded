//! `vmpx decompile` (SPIKE): pseudocode from recovered IR. Static only.

use std::path::PathBuf;

use anyhow::{Context, Result};
use vmpx_mutation::cfg_recovery::{CfgOptions, recover_native_cfg_pe_with};
use vmpx_pe::{PeImage, Rva};

/// `--fold proven`: collapse a subtree to a constant when it is PROVEN constant
/// against the image -- the rdtsc-invariant reads (timing fold) and the `.vmp2`
/// checksum (image concretization). Anything unproven is left symbolic.
fn fold_proven_expr(e: &vmpx_vm::Expr, pe: &PeImage<'_>) -> vmpx_vm::Expr {
    let limits = vmpx_vm::ImageConcretizationLimits {
        max_expression_nodes: 400_000,
        max_checksum_bytes: 64 * 1024 * 1024,
    };
    let modulo = vmpx_vm::ModuloImageTargetLimits {
        concretization: limits,
        ..vmpx_vm::ModuloImageTargetLimits::default()
    };
    let timing = vmpx_vm::fold_timing_invariant_loads(pe, e, modulo)
        .map(|r| r.expression)
        .unwrap_or_else(|_| e.clone());
    vmpx_vm::concretize_with_immutable_image(
        pe,
        &timing,
        &std::collections::BTreeMap::new(),
        limits,
    )
    .map(|r| r.expression)
    .unwrap_or(timing)
}

/// Fold the rolled tree's exprs BEFORE lowering: lowering rewrites the image
/// reads from `memory_load_epoch_*` intrinsics (which the fold recognises) into
/// `Load` nodes (which it does not), so folding must happen here.
fn fold_rolled(n: &mut vmpx_vmlift::RNode, pe: &PeImage<'_>) {
    use vmpx_vmlift::RNode;
    match n {
        RNode::Stores { stores, next } => {
            for s in stores.iter_mut() {
                s.address = fold_proven_expr(&s.address, pe);
                s.value = fold_proven_expr(&s.value, pe);
            }
            fold_rolled(next, pe);
        }
        RNode::Exit(e) => {
            for v in e.registers.values_mut() {
                *v = fold_proven_expr(v, pe);
            }
            if let Some(c) = &mut e.continuation {
                *c = fold_proven_expr(c, pe);
            }
        }
        RNode::Fork { selector, arms, .. } => {
            *selector = fold_proven_expr(selector, pe);
            for (_, a) in arms.iter_mut() {
                fold_rolled(a, pe);
            }
        }
        RNode::Loop { phis, body, .. } => {
            for p in phis.iter_mut() {
                p.init = fold_proven_expr(&p.init, pe);
            }
            fold_rolled(body, pe);
        }
        RNode::BackEdge { next } => {
            for e in next.iter_mut() {
                *e = fold_proven_expr(e, pe);
            }
        }
        RNode::Dead(_) => {}
    }
}

pub(crate) struct Options {
    pub input: PathBuf,
    pub only: Vec<String>,
    pub rvas: Vec<u32>,
    pub native: bool,
    pub original: Option<PathBuf>,
    pub probe: bool,
    pub instruction_limit: usize,
    /// Output dialect: "c" (default) or "rust".
    pub dialect: String,
    /// Fold level: "named" (default) or "proven".
    pub fold: String,
}

/// Import slots from the image's own import directory, so the decompiler can
/// name an IAT address instead of printing it, and label what the import
/// reaches (network, licensing, anti-debug, dynamic loading).
fn image_symbols(pe: &PeImage<'_>) -> vmpx_decompile::symbols::Symbols {
    let report = pe.report();
    vmpx_decompile::symbols::Symbols::from_imports(
        report.optional.image_base.0,
        report.imports.iter().map(|module| {
            (
                module.name.clone(),
                module
                    .entries
                    .iter()
                    .map(|entry| (entry.iat_rva.0, entry.name.clone()))
                    .collect::<Vec<_>>(),
            )
        }),
    )
}

pub(crate) fn run(o: Options) -> Result<()> {
    std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(move || go(&o))
        .context("spawn")?
        .join()
        .map_err(|_| anyhow::anyhow!("worker panicked"))?
}

fn go(o: &Options) -> Result<()> {
    vmpx_decompile::readable::set_dialect(match o.dialect.as_str() {
        "" | "c" => None,
        d => Some(d.to_owned()),
    });
    let bytes = std::fs::read(&o.input).context("read input")?;
    if o.native {
        return native(o, &bytes);
    }
    let mapped = crate::equiv_gate::load_pe(&bytes)?;
    let pe = PeImage::parse(&mapped).context("image failed reparse")?;
    let symbols = pe.report().exports.map(|e| e.symbols).unwrap_or_default();
    let mut targets: Vec<(String, u32)> = Vec::new();
    for sym in symbols {
        let Some(name) = sym.name.clone() else {
            continue;
        };
        if !o.only.is_empty() && !o.only.contains(&name) {
            continue;
        }
        match crate::diff_original::resolve_stub(&pe, &mapped, sym.rva.0) {
            Ok(stub) => targets.push((name, stub)),
            Err(why) => println!("// {name}: not a virtualized user function ({why})"),
        }
    }
    for rva in &o.rvas {
        targets.push((format!("vm_{rva:x}"), *rva));
    }
    for (name, stub) in targets {
        println!("// ==== {name} [{stub:#x}]");
        match vm_one(o, &pe, &name, stub) {
            Ok(()) => {}
            Err(e) => println!("// {name}: {e}\n"),
        }
    }
    Ok(())
}

fn vm_one(o: &Options, pe: &PeImage<'_>, name: &str, stub: u32) -> std::result::Result<(), String> {
    if o.probe {
        crate::decoded_splice::with_entry_input(pe, Rva(stub), o.instruction_limit, |input| {
            let cfg = input.cfg;
            println!(
                "// cfg: entry={} blocks={} edges={} stalls={} complete={}",
                cfg.entry,
                cfg.blocks.len(),
                cfg.edges.len(),
                cfg.stalls.len(),
                cfg.is_complete()
            );
            for (k, b) in &cfg.blocks {
                println!(
                    "//  block {k} addr={:?} instrs={} exit={:?}",
                    b.address,
                    b.instructions.len(),
                    b.exit
                );
            }
            for e in &cfg.edges {
                println!("//  edge {} -> {} {:?}", e.from, e.to, e.kind);
            }
        })?;
        let rolled = crate::decoded_splice::with_entry_input(
            pe,
            Rva(stub),
            o.instruction_limit,
            |input| {
                let resolver = vmpx_vmlift::PeNestedEntries {
                    pe,
                    limits: vmpx_vm::EntryProfileRecoveryLimits {
                        instruction_limit: o.instruction_limit,
                        ..vmpx_vm::EntryProfileRecoveryLimits::default()
                    },
                };
                vmpx_vmlift::roll_decoded_cfg_with_nested(input, Some(&resolver), 200_000)
            },
        )??;
        println!(
            "// rolled: loops={} forks={} blocks_walked={}",
            rolled.loops, rolled.forks, rolled.straight_blocks
        );
        print_tree(&rolled.root, 0);
        return Ok(());
    }
    if std::env::var_os("VMPX_PROBE_UNROLLED").is_some() {
        // The rolled frontend refuses a store outside the VM's dead frame; the
        // unrolled one has no such guard. Reports which of the two can reach this
        // entry at all, so the size of an unrolled fallback is measurable before
        // anyone builds the lowering it would need.
        let rolled =
            crate::decoded_splice::with_entry_input(pe, Rva(stub), o.instruction_limit, |input| {
                let resolver = vmpx_vmlift::PeNestedEntries {
                    pe,
                    limits: vmpx_vm::EntryProfileRecoveryLimits {
                        instruction_limit: o.instruction_limit,
                        ..vmpx_vm::EntryProfileRecoveryLimits::default()
                    },
                };
                vmpx_vmlift::roll_decoded_cfg_with_nested(input, Some(&resolver), 200_000)
            })
            .map_err(|e| e.to_string())
            .and_then(|r| r.map_err(|e| e.to_string()));
        let unrolled = crate::decoded_splice::unroll_entry_cfg(
            pe,
            Rva(stub),
            o.instruction_limit,
            200_000,
            64,
        );
        println!(
            "// probe-unrolled {name}: rolled={} unrolled={}",
            match &rolled {
                Ok(_) => "ok".to_owned(),
                Err(e) => format!("ERR({})", e.chars().take(60).collect::<String>()),
            },
            match &unrolled {
                Ok(u) => format!("ok({} paths)", u.paths.len()),
                Err(e) => format!("ERR({})", e.chars().take(60).collect::<String>()),
            }
        );
        return Ok(());
    }
    let rolled =
        crate::decoded_splice::with_entry_input(pe, Rva(stub), o.instruction_limit, |input| {
            let resolver = vmpx_vmlift::PeNestedEntries {
                pe,
                limits: vmpx_vm::EntryProfileRecoveryLimits {
                    instruction_limit: o.instruction_limit,
                    ..vmpx_vm::EntryProfileRecoveryLimits::default()
                },
            };
            vmpx_vmlift::roll_decoded_cfg_with_nested(input, Some(&resolver), 200_000)
        })??;
    let mut rolled = rolled;
    if o.fold == "proven" {
        fold_rolled(&mut rolled.root, pe);
    }
    let low = vmpx_decompile::vm::lower(&rolled, name)?;
    let d = vmpx_decompile::decompile_with(&low.func, &image_symbols(pe))?;
    // The lifter records a note per path it lowered, so the same caveat arrives
    // once for every path that hit it -- 20 lines for 8 distinct notes on
    // vm_c447739. Say each one once, in the order first seen.
    let mut said = std::collections::HashSet::new();
    for note in &low.notes {
        if said.insert(note.as_str()) {
            println!("// note: {note}");
        }
    }
    print!("{}", d.text);
    println!(
        "// structured={} loops={} ifs={} gotos={} fallback={:?}
// IR round-trip (structured program vs its IR): {:?}
// C-text re-evaluation (C typing, UB checks) vs IR (checked, skipped): {:?}",
        d.structured, d.loops, d.ifs, d.gotos, d.fallback_reason, d.check, d.text_check
    );
    // linkage 1: the decompiled function against the unrolled (PROVEN) paths
    let unrolled =
        crate::decoded_splice::unroll_entry_cfg(pe, Rva(stub), o.instruction_limit, 200_000, 64)?;
    let mut targets = Vec::new();
    for path in unrolled.paths.iter().filter(|p| !p.region_exit) {
        if let Some(rax) = path
            .semantics
            .exit
            .registers
            .get(&vmpx_vm::X64NativeRegister::Rax)
        {
            targets.push(vmpx_decompile::vm::fix(rax));
        }
    }
    println!(
        "// linkage vs unrolled recovered paths ({} callee-return path(s)): {:?}",
        targets.len(),
        vmpx_decompile::linkage_vs_exprs(&low.func, &targets, 64)
    );
    if let Some(orig) = &o.original {
        let obytes = std::fs::read(orig).map_err(|e| e.to_string())?;
        let ope = PeImage::parse(&obytes).map_err(|e| e.to_string())?;
        let osym = ope
            .report()
            .exports
            .map(|e| e.symbols)
            .unwrap_or_default()
            .into_iter()
            .find(|s| s.name.as_deref() == Some(name))
            .ok_or("no such export in the original")?;
        let cfg = recover_native_cfg_pe_with(&ope, Rva(osym.rva.0), CfgOptions::PROVEN)
            .map_err(|e| e.to_string())?;
        let nf = vmpx_decompile::native::lower(&cfg, name)?;
        println!(
            "// linkage vs ORIGINAL native function (both lifted to IR): {:?}",
            vmpx_decompile::linkage_vs_func(&low.func, &nf, 64)
        );
    }
    println!();
    Ok(())
}

fn print_tree(n: &vmpx_vmlift::RNode, depth: usize) {
    use vmpx_vmlift::RNode;
    let pad = "  ".repeat(depth);
    let p = |e: &vmpx_vm::Expr, d| vmpx_verify::differential::pretty(&e.simplify(), d);
    match n {
        RNode::Exit(x) => {
            println!("// {pad}EXIT region={}", x.region);
            for (r, e) in &x.registers {
                println!("// {pad}  {r:?} = {}", p(e, 7));
            }
        }
        RNode::Stores { stores, next } => {
            println!("// {pad}STORES ({})", stores.len());
            for st in stores {
                println!(
                    "// {pad}  [{}] <- {} ({:?})",
                    p(&st.address, 7),
                    p(&st.value, 7),
                    st.width
                );
            }
            print_tree(next, depth);
        }
        RNode::Fork {
            selector,
            env,
            block,
            arms,
        } => {
            println!(
                "// {pad}FORK at {block} env={env} selector={}",
                p(selector, 9)
            );
            for (v, a) in arms {
                println!("// {pad} arm {v:#x}:");
                print_tree(a, depth + 2);
            }
        }
        RNode::Loop { header, phis, body } => {
            println!("// {pad}LOOP header={header} phis={}", phis.len());
            for ph in phis {
                println!(
                    "// {pad}  {} ({:?}) init={}",
                    ph.name,
                    ph.width,
                    p(&ph.init, 7)
                );
            }
            print_tree(body, depth + 1);
        }
        RNode::BackEdge { next } => {
            println!("// {pad}BACKEDGE");
            for (i, e) in next.iter().enumerate() {
                println!("// {pad}  next[{i}] = {}", p(e, 9));
            }
        }
        RNode::Dead(why) => println!("// {pad}DEAD {why}"),
    }
}

fn native(o: &Options, bytes: &[u8]) -> Result<()> {
    let pe = PeImage::parse(bytes).context("not a supported PE")?;
    // Explicit RVAs take precedence over the export scan: decompile exactly
    // those native functions (e.g. the mutated natives that `auto` relocates).
    if !o.rvas.is_empty() {
        for rva in &o.rvas {
            let name = format!("native_{rva:x}");
            println!("// ==== {name} [{rva:#x}]");
            // PROVEN_FP models call terminators (a `call` becomes Terminator::Call
            // with the clobber modeled), so the lift can follow past it.
            let cfg = match recover_native_cfg_pe_with(&pe, Rva(*rva), CfgOptions::PROVEN_FP) {
                Ok(c) => c,
                Err(e) => {
                    println!("// {name}: cfg recovery failed: {e}\n");
                    continue;
                }
            };
            match vmpx_decompile::native::lower(&cfg, &name)
                .and_then(|f| vmpx_decompile::decompile(&f))
            {
                Err(e) => println!("// {name}: {e}\n"),
                Ok(d) => {
                    print!("{}", d.text);
                    println!(
                        "// structured={} loops={} ifs={} gotos={} fallback={:?} roundtrip={:?} ctext={:?}\n",
                        d.structured, d.loops, d.ifs, d.gotos, d.fallback_reason, d.check, d.text_check
                    );
                }
            }
        }
        return Ok(());
    }
    let symbols = pe.report().exports.map(|e| e.symbols).unwrap_or_default();
    for sym in symbols {
        let Some(name) = sym.name.clone() else {
            continue;
        };
        if !o.only.is_empty() && !o.only.contains(&name) {
            continue;
        }
        let cfg = match recover_native_cfg_pe_with(&pe, Rva(sym.rva.0), CfgOptions::PROVEN) {
            Ok(c) => c,
            Err(e) => {
                println!("// {name}: cfg recovery failed: {e}\n");
                continue;
            }
        };
        match vmpx_decompile::native::lower(&cfg, &name).and_then(|f| vmpx_decompile::decompile(&f))
        {
            Err(e) => println!("// {name}: {e}\n"),
            Ok(d) => {
                print!("{}", d.text);
                println!(
                    "// structured={} loops={} ifs={} gotos={} fallback={:?} roundtrip={:?} ctext={:?}\n",
                    d.structured, d.loops, d.ifs, d.gotos, d.fallback_reason, d.check, d.text_check
                );
            }
        }
    }
    Ok(())
}
