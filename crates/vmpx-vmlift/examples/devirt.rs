// Probes whether a virtualized guard can be devirtualized to native code with
// its integrity checksum folded to a constant.
//
// The 14 guard references into the VM are tail calls whose returned value is
// `f(args, checksum(.vmp2 region))`. The checksummed regions are inside the
// REMOVED `.vmp2`, so a replacement cannot recompute the checksum -- but it does
// not need to: on a legitimate run the bytes are fixed, so the checksum is a
// constant, and `concretize_with_immutable_image` folds it by reading those
// bytes from the INTACT image. After folding, `build_replacement` can lower the
// exit register state to native x64.
//
// This reports, per entry: path count, whether concretization resolved every
// intrinsic (the checksum included), and whether the native build succeeded.
// It is a measurement; it writes nothing.
//
// Usage: devirt INTACT_IMAGE RVA [RVA ...]
//
// Run against the INTACT image -- the checksum regions live in `.vmp2`, which
// the strip removes.

use std::collections::{BTreeMap, BTreeSet};

use vmpx_pe::{PeImage, Rva};
use vmpx_recompile::{CodeAddress, PhysicalRegister};
use vmpx_vm::{
    EntryPathRecoveryLimits, Expr, ImageConcretizationLimits, VmDiscoveryLimits,
    VmProfileExplorerLimits, X64NativeRegister, analyze_vm_entry_at,
    concretize_with_immutable_image, recover_entry_paths,
};
use vmpx_vmlift::{ReplacementConfig, build_replacement};

fn physical_for(register: &X64NativeRegister) -> PhysicalRegister {
    use PhysicalRegister as P;
    use X64NativeRegister as R;
    match register {
        R::Rax => P::Rax,
        R::Rbx => P::Rbx,
        R::Rcx => P::Rcx,
        R::Rdx => P::Rdx,
        R::Rbp => P::Rbp,
        R::Rsi => P::Rsi,
        R::Rdi => P::Rdi,
        R::R8 => P::R8,
        R::R9 => P::R9,
        R::R10 => P::R10,
        R::R11 => P::R11,
        R::R12 => P::R12,
        R::R13 => P::R13,
        R::R14 => P::R14,
        R::R15 => P::R15,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let base = pe.optional.image_base.0;
    let next_rva = vmpx_rebuild::next_section_rva(&bytes).expect("next rva").0;

    let limits = EntryPathRecoveryLimits {
        chain: VmProfileExplorerLimits {
            max_segments: 64,
            max_instructions_per_segment: 4096,
            max_total_instructions: 65_536,
            ..Default::default()
        },
        ..Default::default()
    };
    let concretize_limits = ImageConcretizationLimits {
        max_expression_nodes: 400_000,
        max_checksum_bytes: 4 * 1024 * 1024,
    };

    let (mut ok, mut built, mut emitted_bytes) = (0_usize, 0_usize, 0_usize);
    for raw in &args[2..] {
        let rva = u32::from_str_radix(raw.trim_start_matches("0x"), 16).expect("rva");
        println!("=== {raw}");
        let Some(candidate) = analyze_vm_entry_at(&pe, Rva(rva), VmDiscoveryLimits::default())
        else {
            println!("  no VM entry at this address");
            continue;
        };
        let family = match recover_entry_paths(&pe, &candidate, &BTreeMap::new(), limits) {
            Ok(family) => family,
            Err(error) => {
                println!("  path recovery failed: {error}");
                continue;
            }
        };
        println!(
            "  {} path(s), domain_complete={}, all_exit={}",
            family.paths.len(),
            family.domain_complete,
            family.all_paths_reach_vm_exit
        );
        let [path] = family.paths.as_slice() else {
            println!("  not a single path -- env-gate fork, handled separately");
            continue;
        };
        println!("  chain stop: {:?}", path.chain.stop_reason);
        for d in path.chain.diagnostics.iter().take(2) {
            println!("    {d}");
        }
        let Some(exit) = path.native_exit.as_ref() else {
            println!("  path has no native exit (chain did not reach a VM exit)");
            continue;
        };
        ok += 1;

        // Concretize every exit register: fold the checksum and resolve image
        // reads against the intact image.
        let mut concretized: BTreeMap<X64NativeRegister, Expr> = BTreeMap::new();
        let mut resolved = 0_usize;
        let mut unresolved: BTreeSet<String> = BTreeSet::new();
        for (register, expr) in &exit.registers {
            match concretize_with_immutable_image(&pe, expr, &BTreeMap::new(), concretize_limits) {
                Ok(report) => {
                    resolved += report.resolved_intrinsics;
                    unresolved.extend(report.unresolved_intrinsics);
                    concretized.insert(*register, report.expression);
                }
                Err(error) => {
                    println!("  concretize failed for {register:?}: {error}");
                    concretized.insert(*register, expr.clone());
                }
            }
        }
        println!("  concretize: {resolved} intrinsic(s) folded, {} left", unresolved.len());
        for name in unresolved.iter().take(6) {
            println!("    unresolved: {name}");
        }

        let mut inputs: Vec<(String, PhysicalRegister)> = X64NativeRegister::ALL
            .iter()
            .map(|r| (r.symbolic_name().to_owned(), physical_for(r)))
            .collect();
        inputs.push(("vsp_initial".to_owned(), PhysicalRegister::Rbp));
        inputs.push(("native_return_address_initial".to_owned(), PhysicalRegister::Rbp));
        inputs.sort();
        inputs.dedup();
        let config = ReplacementConfig {
            inputs,
            max_expression_nodes: 400_000,
            max_spill_slots: 128,
            import_calls: BTreeMap::new(),
            continuation: None,
            effects: Vec::new(),
            effect_calls: Vec::new(),
            rflags: None,
            symbol_rewrites: BTreeMap::new(),
        };
        let va = base + u64::from(next_rva);
        match build_replacement(&concretized, &config, CodeAddress(va)) {
            Ok(replacement) => {
                built += 1;
                emitted_bytes += replacement.compiled.bytes.len();
                println!("  BUILT: {} byte(s) of native code", replacement.compiled.bytes.len());
            }
            Err(error) => println!("  build failed: {error}"),
        }
    }
    println!(
        "\n{} of {} entries reached a single-path exit; {built} built native code ({emitted_bytes} bytes)",
        ok,
        args.len() - 2
    );
}
