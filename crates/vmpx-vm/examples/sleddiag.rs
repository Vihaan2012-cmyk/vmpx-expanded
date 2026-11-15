//! What the exception-flow analysis actually recovers for a faulting handler.
//!
//! The null-read gate refuses on "no zero-seed-unwind continuation", and the
//! reason has to be read off the sled report rather than inferred: which fields
//! are classified, what the status is, and whether anything about the non-zero
//! path is available. Inferring it from the code has been wrong twice.
//!
//! usage: sleddiag <mapped.bin> <handler-rva-hex>
use std::{env, fs, process::ExitCode};

use vmpx_disasm::{DirectJumpTraceLimits, trace_direct_jump_fragments};
use vmpx_pe::{PeImage, Rva};

fn main() -> ExitCode {
    let arguments = env::args().collect::<Vec<_>>();
    let [_, path, rva] = arguments.as_slice() else {
        eprintln!("usage: sleddiag <mapped.bin> <handler-rva-hex>");
        return ExitCode::FAILURE;
    };
    let text = rva.strip_prefix("0x").or_else(|| rva.strip_prefix("0X")).unwrap_or(rva);
    let Ok(handler) = u32::from_str_radix(text, 16) else {
        eprintln!("invalid hexadecimal RVA");
        return ExitCode::FAILURE;
    };
    let handler = Rva(handler);
    let Ok(bytes) = fs::read(path) else {
        eprintln!("failed to read input");
        return ExitCode::FAILURE;
    };
    let Ok(pe) = PeImage::parse(&bytes) else {
        eprintln!("failed to parse the image");
        return ExitCode::FAILURE;
    };

    // Approximates the CFG walker's site set. The walker also excludes reads
    // based on the VIP, rolling-key and handler role registers, which are not
    // resolvable here, so this can report MORE language handlers than the
    // walker sees -- which is exactly how the over-broad set was spotted.
    let Ok(trace) = trace_direct_jump_fragments(&pe, handler, DirectJumpTraceLimits::default())
    else {
        eprintln!("handler trace failed");
        return ExitCode::FAILURE;
    };
    let mut sites: Vec<Rva> = trace
        .instructions()
        .filter(|decoded| {
            (0..decoded.instruction.op_count())
                .any(|operand| decoded.instruction.op_kind(operand) == iced_x86::OpKind::Memory)
                && decoded.instruction.mnemonic() != iced_x86::Mnemonic::Lea
                && !matches!(
                    decoded.instruction.memory_base().full_register(),
                    iced_x86::Register::RSP | iced_x86::Register::RBP | iced_x86::Register::None
                )
        })
        .map(|decoded| decoded.rva)
        .collect();
    sites.sort_unstable();
    sites.dedup();
    println!("handler {handler}: {} memory-reading site(s) in trace", sites.len());

    let report = match vmpx_vm::analyze_fault_site_exception_coverage(&pe, handler, sites) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("coverage failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("all_fault_sites_have_language_handler = {}", report.all_fault_sites_have_language_handler);
    println!("coverage records = {}", report.coverage.len());
    println!("continuation_recovered = {}", report.continuation_recovered);
    println!("\nstack sleds ({}):", report.stack_sled_handlers.len());
    for sled in &report.stack_sled_handlers {
        println!(
            "  handler={} status={:?} blocks={} confidence={}",
            sled.handler, sled.status, sled.cfg_blocks, sled.confidence
        );
        println!(
            "    context_rsp_field_offset={:?} seed={:?} cursor={:?}",
            sled.context_rsp_field_offset, sled.seed_context_offset, sled.cursor_context_offset
        );
        println!(
            "    unwind_boundary_delta={:?} return_gadget={:?} return_disposition={:?}",
            sled.unwind_boundary_delta, sled.return_gadget, sled.return_disposition
        );
        for line in sled.evidence.iter().take(6) {
            println!("    evidence: {line}");
        }
        for line in sled.diagnostics.iter().take(6) {
            println!("    diag: {line}");
        }
    }
    println!("\ncontinuations ({}):", report.continuation_resolutions.len());
    for resolution in &report.continuation_resolutions {
        println!(
            "  handler={} status={:?} source={:?} rva={:?}",
            resolution.handler, resolution.status, resolution.source, resolution.continuation_rva
        );
        for line in resolution.diagnostics.iter().take(3) {
            println!("    diag: {line}");
        }
    }
    ExitCode::SUCCESS
}
