use std::fs;
use vmpx_disasm::{DirectJumpTraceLimits, trace_direct_jump_fragments};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let image_base = pe.optional.image_base.0;
    let table_rva = 0x124B80_u32;
    let table_va = image_base + u64::from(table_rva);
    let table_off = pe.rva_to_file_offset(Rva(table_rva)).unwrap().0 as usize;
    let roles = HandlerRegisterRoles {
        vip: iced_x86::Register::RSI,
        rolling_key: iced_x86::Register::RCX,
        handler: iced_x86::Register::RBX,
    };
    let eroles = HandlerEffectRoles {
        vsp: iced_x86::Register::RBP,
        context_base: iced_x86::Register::RSP,
    };
    let mut hist = std::collections::BTreeMap::<String, usize>::new();
    for op in 0u32..256 {
        let o = table_off + op as usize * 4;
        let entry = i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let rva = ((table_va as i64 + i64::from(entry)) as u64 - image_base) as u32;
        let trace =
            trace_direct_jump_fragments(&pe, Rva(rva), DirectJumpTraceLimits::default()).unwrap();
        let Some(end) = trace
            .instructions()
            .find(|d| d.rva == Rva(0x11f148) || d.rva == Rva(0x11f12c))
        else {
            continue;
        };
        let slice = backward_slice_handler_trace(
            &trace,
            &HandlerSliceConfig {
                live_out_registers: vec![
                    iced_x86::Register::RBP,
                    iced_x86::Register::RBX,
                    iced_x86::Register::RSI,
                    iced_x86::Register::RCX,
                ],
                live_out_rflags: 0,
            },
        );
        let effects = extract_handler_effects_keyed(&trace, &slice, eroles, roles, end.rva, false);
        let a = classify_handler_effects(Rva(rva), &effects, None, Some(1));
        let proof = prove_handler_value_flow(&trace, &slice, &effects, &a, None, None, false);
        let sem = format!("{:?}/{:?}/{}", a.semantic, a.operation_width, proof.proven);
        if std::env::var("D").is_ok_and(|v| v.parse::<u32>() == Ok(op)) {
            println!("{effects:#?}\n{a:#?}\n{proof:#?}");
        }
        if std::env::var("V").is_ok() {
            println!("op{op} {rva:#x} {sem} delta={:?}", effects.vsp_delta);
        }
        *hist.entry(sem).or_default() += 1;
    }
    println!("{hist:#?}");
}
