use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    EntryProfileRecoveryLimits, LinearDecodeStop, RecoveredEntryStream, VmDiscoveryLimits,
    analyze_vm_entry_at, recover_entry_stream,
};

fn limits() -> EntryProfileRecoveryLimits {
    EntryProfileRecoveryLimits {
        instruction_limit: 100_000,
        ..EntryProfileRecoveryLimits::default()
    }
}

fn opcode_of(stream: &RecoveredEntryStream, index: usize) -> Option<u64> {
    let width = stream
        .bootstrap
        .codec
        .table
        .map_or(1, |table| table.opcode_width.bytes() as usize);
    let bytes = &stream.decode.decode.instructions.get(index)?.encoded_bytes;
    let tail = bytes.get(bytes.len().checked_sub(width)?..)?;
    Some(
        tail.iter()
            .rev()
            .fold(0, |acc, b| (acc << 8) | u64::from(*b)),
    )
}

fn report_stop(pe: &PeImage<'_>, stream: &RecoveredEntryStream) {
    let rva = |va| pe.optional.image_base.checked_rva_of(va).map_or(0, |r| r.0);
    let decode = &stream.decode.decode;
    let handler = decode.final_state.handler;
    let opcode = decode
        .instructions
        .len()
        .checked_sub(1)
        .and_then(|last| opcode_of(stream, last));
    println!(
        "stop: {:?} at vip={} next opcode={} handler RVA {:#x}",
        decode.stop_reason,
        decode.final_state.vip,
        opcode.map_or("?".to_owned(), |o| format!("{o:#04x}")),
        rva(handler)
    );
    let Some(recovery) = stream.decode.handler_recoveries.get(&handler) else {
        return;
    };
    println!(
        "  status={:?} semantic={:?} dispatch_cands={} operand_cands={} rejected_operands={}",
        recovery.status,
        recovery.handler_analysis.as_ref().map(|a| a.semantic),
        recovery.dispatch_analysis.candidates.len(),
        recovery.operand_analysis.candidates.len(),
        recovery.operand_analysis.rejected_sources.len(),
    );
    for line in recovery.diagnostics.iter().take(8) {
        println!("  diag: {line}");
    }
    for r in recovery.dispatch_analysis.rejected_sources.iter().take(4) {
        println!("  dispatch reject: {}: {}", r.encoded_load, r.reason);
    }
    if let Some(analysis) = &recovery.handler_analysis {
        for e in analysis.evidence.iter().take(6) {
            println!("  evidence: {e}");
        }
    }
}

fn single(pe: &PeImage<'_>, target: u32, show: usize) {
    let candidate = analyze_vm_entry_at(pe, Rva(target), VmDiscoveryLimits::default()).unwrap();
    let stream = match recover_entry_stream(pe, &candidate, limits()) {
        Ok(stream) => stream,
        Err(error) => {
            println!("entry stream recovery failed: {error}");
            return;
        }
    };
    let rva = |va| pe.optional.image_base.checked_rva_of(va).map_or(0, |r| r.0);
    let decode = &stream.decode.decode;
    println!(
        "bootstrap: vip={:?} handler={:?} vsp={:?} dir={:?} initial_vip={} table_anchor={} frame={:?}",
        stream.bootstrap.roles.vip,
        stream.bootstrap.roles.handler,
        stream.profile.effect_roles.vsp,
        stream.bootstrap.bytecode_direction,
        stream.bootstrap.initial_state.vip,
        stream.bootstrap.initial_state.handler,
        stream.native_frame.status,
    );
    println!(
        "decoded {} VM instructions; stop={:?}",
        decode.instructions.len(),
        decode.stop_reason
    );
    let mut histogram: BTreeMap<String, usize> = BTreeMap::new();
    let mut opcodes: BTreeMap<u64, (u32, String, usize)> = BTreeMap::new();
    for (index, instruction) in decode.instructions.iter().enumerate() {
        let semantic = format!("{:?}", instruction.semantic);
        *histogram.entry(semantic.clone()).or_default() += 1;
        // instruction N was dispatched by the opcode read at the end of N-1
        let opcode = index
            .checked_sub(1)
            .and_then(|previous| opcode_of(&stream, previous));
        if let Some(opcode) = opcode {
            opcodes
                .entry(opcode)
                .or_insert((rva(instruction.state_before.handler), semantic, 0))
                .2 += 1;
        }
        if index < show {
            println!(
                "{index:5} vip={} op={} h={:#x} {:?} w={:?} operand={:?} vsp={:?}",
                instruction.state_before.vip,
                opcode.map_or("boot".to_owned(), |o| format!("{o:#04x}")),
                rva(instruction.state_before.handler),
                instruction.semantic,
                instruction.operation_width,
                instruction.operand,
                instruction.vsp_delta,
            );
        }
    }
    println!("semantic histogram: {histogram:?}");
    println!("opcodes used (opcode -> handler RVA, semantic, count):");
    for (opcode, (handler, semantic, count)) in &opcodes {
        println!("  {opcode:#04x} -> {handler:#x} {semantic} x{count}");
    }
    if matches!(
        decode.stop_reason,
        LinearDecodeStop::MissingHandlerSpec | LinearDecodeStop::ControlFlowBoundary
    ) {
        report_stop(pe, &stream);
    }
}

fn scan(pe: &PeImage<'_>, bytes: &[u8], core: u32, max: usize) {
    // Entry stubs are `push imm32; call core`; sweep the raw image for them.
    let mut targets: BTreeSet<u32> = BTreeSet::new();
    for section in &pe.sections {
        let start = section.pointer_to_raw_data.0 as usize;
        let end = (start + section.size_of_raw_data as usize).min(bytes.len());
        for offset in start..end.saturating_sub(10) {
            if bytes[offset] != 0x68 || bytes[offset + 5] != 0xE8 {
                continue;
            }
            let rel = i32::from_le_bytes(bytes[offset + 6..offset + 10].try_into().unwrap());
            let rva = section.virtual_address.0 as usize + (offset - start);
            let call_target = (rva as i64 + 10 + i64::from(rel)) as u32;
            if call_target == core {
                targets.insert(rva as u32);
            }
        }
    }
    println!("entry stubs calling core {core:#x}: {}", targets.len());
    let mut stops: BTreeMap<String, usize> = BTreeMap::new();
    for target in targets.iter().take(max) {
        let Ok(offset) = pe.rva_to_file_offset(Rva(*target)) else {
            continue;
        };
        let b = &bytes[offset.0 as usize..(offset.0 as usize + 6).min(bytes.len())];
        if b.len() < 6 || b[0] != 0x68 || b[5] != 0xE8 {
            continue;
        }
        let Some(candidate) = analyze_vm_entry_at(pe, Rva(*target), VmDiscoveryLimits::default())
            .filter(|c| c.confidence >= 60)
        else {
            continue;
        };
        match recover_entry_stream(pe, &candidate, limits()) {
            Ok(stream) => {
                let decode = &stream.decode.decode;
                let handler = decode.final_state.handler;
                let semantic = stream
                    .decode
                    .handler_recoveries
                    .get(&handler)
                    .and_then(|r| r.handler_analysis.as_ref())
                    .map(|a| a.semantic);
                let opcode = decode
                    .instructions
                    .len()
                    .checked_sub(1)
                    .and_then(|last| opcode_of(&stream, last));
                let key = format!("{:?}/{semantic:?}", decode.stop_reason);
                *stops.entry(key.clone()).or_default() += 1;
                println!(
                    "{target:#x} n={} stop={key} opcode={} handler={:#x}",
                    decode.instructions.len(),
                    opcode.map_or("?".to_owned(), |o| format!("{o:#04x}")),
                    pe.optional
                        .image_base
                        .checked_rva_of(handler)
                        .map_or(0, |r| r.0),
                );
            }
            Err(error) => {
                *stops.entry(format!("error: {error}")).or_default() += 1;
                println!("{target:#x} error: {error}");
            }
        }
    }
    println!("stop summary: {stops:#?}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    if args[2] == "scan" {
        let core = u32::from_str_radix(args[3].trim_start_matches("0x"), 16).unwrap();
        let max = args
            .get(4)
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX);
        scan(&pe, &bytes, core, max);
        return;
    }
    let target = u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap();
    let show: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(60);
    single(&pe, target, show);
}
