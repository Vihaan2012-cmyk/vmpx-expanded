use std::fs;

use iced_x86::Register;
use vmpx_disasm::{
    DirectJumpTraceLimits, RegisterDispatchProofLimits, trace_direct_jump_fragments,
};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    BytecodeDirection, HandlerRegisterRoles, HandlerSliceConfig, analyze_table_checksum_handler,
    backward_slice_handler_trace, extract_relative_dispatch_codecs_with_postlude_direction,
};

#[test]
#[ignore = "large controlled mapped-image fixture"]
fn recognizes_ida_verified_p06_table_checksum_handler() {
    let bytes = fs::read(
        r"D:\vmp\testfilevmp\systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl",
    )
    .expect("controlled mapped-image fixture must exist");
    let pe = PeImage::parse(&bytes).expect("controlled fixture must parse");
    let roles = HandlerRegisterRoles {
        vip: Register::RSI,
        rolling_key: Register::R8,
        handler: Register::RDI,
    };
    let checksum = analyze_table_checksum_handler(
        &pe,
        Rva(252_427_640),
        roles,
        Register::RBP,
        BytecodeDirection::Ascending,
        &[Register::RBP, Register::RSI, Register::RDI, Register::R8],
        DirectJumpTraceLimits::default(),
        RegisterDispatchProofLimits::default(),
    )
    .expect("checksum handler analysis must decode");
    eprintln!("checksum analysis: {:#?}", checksum.report);
    assert!(checksum.recovered.is_some(), "{:#?}", checksum.report);
    for (label, entry) in [
        ("zero", Rva(252_003_124)),
        ("nonzero", Rva(252_158_214)),
        ("nonzero_loop_exit", Rva(252_039_302)),
    ] {
        let trace = trace_direct_jump_fragments(&pe, entry, DirectJumpTraceLimits::default())
            .expect("successor trace must decode");
        let slice = backward_slice_handler_trace(
            &trace,
            &HandlerSliceConfig {
                live_out_registers: vec![Register::RBP, Register::RSI, Register::RDI, Register::R8],
                live_out_rflags: 0,
            },
        );
        let dispatch = extract_relative_dispatch_codecs_with_postlude_direction(
            &pe,
            &trace,
            &slice,
            roles,
            BytecodeDirection::Ascending,
            RegisterDispatchProofLimits::default(),
        )
        .expect("successor dispatch extraction must complete");
        eprintln!(
            "{label}: {} fragments, {} instructions, stop {:?}, {} dispatch candidates, {:#?}",
            trace.fragments.len(),
            trace.decoded_instructions,
            trace.stop,
            dispatch.candidates.len(),
            dispatch,
        );
        for instruction in trace.instructions() {
            eprintln!("  {:#010X}: {}", instruction.rva.0, instruction.instruction);
        }
    }
}
