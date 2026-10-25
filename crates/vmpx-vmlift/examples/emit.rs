// Emits the native replacement for one VM exit state and appends it to a real
// PE image as an executable section, proving the emitted code can be placed.
use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_rebuild::{AppendSection, append_section, validate_for_loader};
use vmpx_recompile::{CodeAddress, PhysicalRegister};
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, VmDiscoveryLimits, VmEntrySource,
    VmProfileExplorerLimits, X64NativeRegister, discover_vm_entries, recover_entry_paths,
};
use vmpx_vmlift::{ReplacementConfig, build_replacement};

fn physical_for(register: &X64NativeRegister) -> PhysicalRegister {
    match register {
        X64NativeRegister::Rax => PhysicalRegister::Rax,
        X64NativeRegister::Rbx => PhysicalRegister::Rbx,
        X64NativeRegister::Rcx => PhysicalRegister::Rcx,
        X64NativeRegister::Rdx => PhysicalRegister::Rdx,
        X64NativeRegister::Rbp => PhysicalRegister::Rbp,
        X64NativeRegister::Rsi => PhysicalRegister::Rsi,
        X64NativeRegister::Rdi => PhysicalRegister::Rdi,
        X64NativeRegister::R8 => PhysicalRegister::R8,
        X64NativeRegister::R9 => PhysicalRegister::R9,
        X64NativeRegister::R10 => PhysicalRegister::R10,
        X64NativeRegister::R11 => PhysicalRegister::R11,
        X64NativeRegister::R12 => PhysicalRegister::R12,
        X64NativeRegister::R13 => PhysicalRegister::R13,
        X64NativeRegister::R14 => PhysicalRegister::R14,
        X64NativeRegister::R15 => PhysicalRegister::R15,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let discovery = discover_vm_entries(
        &pe,
        VmDiscoveryLimits {
            scan_executable_stubs: false,
            ..Default::default()
        },
    )
    .unwrap();
    let entry = discovery
        .entries
        .iter()
        .find(|e| e.source == VmEntrySource::PeEntryPoint)
        .unwrap();
    let domains = BTreeMap::from([(
        "native_rdx_initial".to_owned(),
        (0..=3).map(|r| Expr::constant(BitWidth::W64, r)).collect(),
    )]);
    let limits = EntryPathRecoveryLimits {
        chain: VmProfileExplorerLimits {
            max_segments: 32,
            max_instructions_per_segment: 2048,
            max_total_instructions: 8192,
            ..Default::default()
        },
        ..Default::default()
    };
    let family = recover_entry_paths(&pe, entry, &domains, limits).unwrap();
    let path = family
        .paths
        .iter()
        .find(|p| p.assignments.get("native_rdx_initial") == Some(&0))
        .expect("reason 0");
    let exit = path.native_exit.as_ref().expect("exit");

    let mut inputs: Vec<(String, PhysicalRegister)> = X64NativeRegister::ALL
        .iter()
        .map(|r| (r.symbolic_name().to_owned(), physical_for(r)))
        .collect();
    inputs.push(("vsp_initial".to_owned(), PhysicalRegister::Rbp));
    inputs.push((
        "native_return_address_initial".to_owned(),
        PhysicalRegister::Rbp,
    ));
    inputs.sort();
    inputs.dedup();

    // The replacement must be compiled at the address it will occupy.
    let next_rva = vmpx_rebuild::next_section_rva(&bytes).expect("next section rva");
    let va = pe.optional.image_base.0 + u64::from(next_rva.0);
    let config = ReplacementConfig {
        inputs,
        max_expression_nodes: 200_000,
        max_spill_slots: 64,
    };
    let replacement =
        build_replacement(&exit.registers, &config, CodeAddress(va)).expect("replacement");
    println!(
        "replacement: {} bytes at RVA {next_rva:?}",
        replacement.compiled.bytes.len()
    );

    let rebuilt = append_section(
        &bytes,
        AppendSection {
            name: *b".vmpx\0\0\0",
            data: &replacement.compiled.bytes,
            characteristics: 0x6000_0020,
        },
    )
    .expect("append");
    println!(
        "appended at RVA {:?}, output {} bytes",
        rebuilt.section.rva,
        rebuilt.bytes.len()
    );
    let validation = validate_for_loader(&rebuilt.bytes).expect("loader validation");
    println!("loader validation: {validation:?}");
    if let Some(out) = args.get(2) {
        fs::write(out, &rebuilt.bytes).expect("write");
        println!("wrote {out}");
    }
}
