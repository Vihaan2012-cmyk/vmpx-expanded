use std::{fs, path::PathBuf};

use iced_x86::{InstructionInfoFactory, Mnemonic};
use vmpx_disasm::{DirectJumpTraceLimits, DirectJumpTraceStopKind, trace_direct_jump_fragments};
use vmpx_pe::{PeImage, Va};

#[test]
#[ignore = "large controlled mapped-image fixture"]
fn traces_ida_verified_fragmented_add_handler_to_dispatch() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../../testfilevmp/systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl",
    );
    let data = fs::read(fixture).expect("controlled mapped fixture must exist");
    let pe = PeImage::parse(&data).expect("controlled fixture must parse");
    let entry = pe
        .optional
        .image_base
        .checked_rva_of(Va(0x0003_7AAA_7BA9))
        .expect("IDA-verified handler VA belongs to the image");
    let trace = trace_direct_jump_fragments(
        &pe,
        entry,
        DirectJumpTraceLimits {
            max_fragments: 1_024,
            max_instructions: 50_000,
            max_instructions_per_fragment: 10_000,
        },
    )
    .expect("confirmed handler entry must decode");

    let mnemonics: Vec<_> = trace
        .instructions()
        .map(|decoded| decoded.instruction.mnemonic())
        .collect();
    let mut info_factory = InstructionInfoFactory::new();
    eprintln!(
        "fragments={} instructions={} stop={:?}",
        trace.fragments.len(),
        trace.decoded_instructions,
        trace.stop
    );
    for decoded in trace.instructions() {
        let summary = decoded.summary(&mut info_factory);
        eprintln!("{} {}", summary.rva, summary.text);
    }
    assert_eq!(trace.stop.kind, DirectJumpTraceStopKind::IndirectBranch);
    assert!(trace.fragments.len() > 1);
    assert!(mnemonics.contains(&Mnemonic::Add));
    assert!(mnemonics.contains(&Mnemonic::Pushfq));
}
