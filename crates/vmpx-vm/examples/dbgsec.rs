use std::fs;
use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{VmDiscoveryLimits, analyze_vm_entry_at, recover_bootstrap_dispatch};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    for t in [0xc45fcc2u32, 0xc4ea18d] {
        let c = analyze_vm_entry_at(&pe, Rva(t), VmDiscoveryLimits::default()).unwrap();
        let (_, b) = recover_bootstrap_dispatch(&pe, &c, DirectJumpTraceLimits::default()).unwrap();
        let b = b.unwrap();
        let first = b
            .codec
            .decode_direction(&pe, b.initial_state, b.bytecode_direction)
            .unwrap();
        println!(
            "{t:#x} initial vip={:?} first_after={:?}",
            b.initial_state.vip, first.state_after.vip
        );
        for va in [b.initial_state.vip, first.state_after.vip] {
            let rva = pe.optional.image_base.checked_rva_of(va);
            let sec = rva.and_then(|r| {
                pe.sections
                    .iter()
                    .find(|s| s.contains_mapped_rva(r))
                    .map(|s| (s.index, s.name.clone()))
            });
            println!("    va={va:?} rva={rva:?} section={sec:?}");
        }
        println!(
            "    sections: {:?}",
            pe.sections
                .iter()
                .map(|s| (s.index, s.name.clone()))
                .collect::<Vec<_>>()
        );
    }
}
