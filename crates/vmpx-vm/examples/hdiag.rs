use std::collections::BTreeMap;
use std::fs;
use vmpx_disasm::{DirectJumpTraceLimits, RegisterDispatchProofLimits};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    HandlerCodecRecoveryConfig, HandlerEffectRoles, HandlerRegisterRoles, RecoveryStatus,
    recover_handler_codecs,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let image_base = pe.optional.image_base.0;
    let table_rva = 0x124B80_u32;
    let table_va = image_base + u64::from(table_rva);
    let table_off = pe.rva_to_file_offset(Rva(table_rva)).unwrap().0 as usize;

    let config = HandlerCodecRecoveryConfig {
        roles: HandlerRegisterRoles {
            vip: iced_x86::Register::RSI,
            rolling_key: iced_x86::Register::RCX,
            handler: iced_x86::Register::RBX,
        },
        bytecode_direction: vmpx_vm::BytecodeDirection::Ascending,
        effect_roles: Some(HandlerEffectRoles {
            vsp: iced_x86::Register::RBP,
            context_base: iced_x86::Register::RSP,
        }),
        live_out_registers: vec![
            iced_x86::Register::RBP,
            iced_x86::Register::RBX,
            iced_x86::Register::RSI,
            iced_x86::Register::RCX,
        ],
        live_out_rflags: 0,
        trace_limits: DirectJumpTraceLimits::default(),
        proof_limits: RegisterDispatchProofLimits::default(),
    };

    if let Ok(v) = std::env::var("HDIAG_RVA") {
        let rva = u32::from_str_radix(v.trim_start_matches("0x"), 16).unwrap();
        let r = recover_handler_codecs(&pe, Rva(rva), &config).unwrap();
        println!("status {:?}", r.status);
        println!("diagnostics {:#?}", r.diagnostics);
        println!("operand {:#?}", r.operand_analysis);
        println!("effects {:#?}", r.effects);
        println!("analysis {:#?}", r.handler_analysis);
        println!("flow {:#?}", r.value_flow);
        println!("checksum {:#?}", r.table_checksum);
        println!("native_call {:#?}", r.native_call);
        return;
    }
    let mut recovered = 0usize;
    let mut dispatch_ok = 0usize;
    let mut decode_spec = 0usize;
    let mut status_hist: BTreeMap<String, usize> = BTreeMap::new();
    let mut semantic_hist: BTreeMap<String, usize> = BTreeMap::new();
    let mut handler_rvas = std::collections::BTreeSet::new();

    for op in 0u32..256 {
        let entry = i32::from_le_bytes(
            bytes[table_off + (op as usize) * 4..table_off + (op as usize) * 4 + 4]
                .try_into()
                .unwrap(),
        );
        let handler_va = (table_va as i64 + i64::from(entry)) as u64;
        let handler_rva = (handler_va - image_base) as u32;
        // Distinct handlers only (table has no reuse per profile, but guard anyway).
        if !handler_rvas.insert(handler_rva) {
            continue;
        }
        let r = match recover_handler_codecs(&pe, Rva(handler_rva), &config) {
            Ok(r) => r,
            Err(e) => {
                *status_hist.entry(format!("Err:{e}")).or_default() += 1;
                continue;
            }
        };
        *status_hist.entry(format!("{:?}", r.status)).or_default() += 1;
        if r.status != RecoveryStatus::Failed {
            recovered += 1;
        }
        if r.dispatch.is_some() {
            dispatch_ok += 1;
        }
        if r.decode_spec.is_some() {
            decode_spec += 1;
        }
        if std::env::var("HDIAG_PER_OP").is_ok() {
            println!(
                "op{op} rva {handler_rva:#x} {:?} {:?} vip_from_vsp={:?} vsp_delta={:?}",
                r.status,
                r.handler_analysis.as_ref().map(|a| a.semantic),
                r.effects.as_ref().map(|e| e.loads_vip_from_vsp),
                r.effects.as_ref().map(|e| e.vsp_delta)
            );
        }
        if std::env::var("HDIAG_UNK").is_ok()
            && let (Some(a), Some(e)) = (r.handler_analysis.as_ref(), r.effects.as_ref())
            && format!("{:?}", a.semantic) == "Unknown"
        {
            let mem: Vec<String> = e
                .memory
                .iter()
                .map(|m| {
                    format!(
                        "{:?}{:+}/w{}{}{}",
                        m.region,
                        m.displacement,
                        m.width_bytes,
                        if m.reads { "R" } else { "" },
                        if m.writes { "W" } else { "" }
                    )
                })
                .collect();
            println!(
                "op{op} {handler_rva:#x} opw={:?} vsp={:?} mem={mem:?} ar={:?} fl={} pre={} ldvsp={} rs={:?} tsc={} diag={:?}",
                a.operand_width,
                e.vsp_delta,
                e.arithmetic_operations,
                e.captures_native_flags_to_vsp,
                e.pushes_pre_adjustment_vsp,
                e.loads_vsp_from_top,
                e.resize_operations
                    .iter()
                    .map(|x| (x.input_width.bits(), x.output_width.bits()))
                    .collect::<Vec<_>>(),
                e.timestamp_counter_reads.len(),
                e.diagnostics
            );
        }
        let sem = r
            .handler_analysis
            .as_ref()
            .map(|a| format!("{:?}", a.semantic))
            .unwrap_or_else(|| "none".to_string());
        *semantic_hist.entry(sem).or_default() += 1;
    }

    println!("distinct handlers: {}", handler_rvas.len());
    println!("recovered (status != Failed): {recovered}/256");
    println!("dispatch codec recovered: {dispatch_ok}");
    println!("decode_spec produced: {decode_spec}");
    println!("status histogram: {status_hist:?}");
    println!("semantic histogram: {semantic_hist:?}");
}
