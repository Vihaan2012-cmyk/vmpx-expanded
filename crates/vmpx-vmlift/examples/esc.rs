use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, VmDiscoveryLimits, VmEntrySource,
    VmProfileExplorerLimits, X64NativeRegister, discover_vm_entries, recover_entry_paths,
};

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
    for path in &family.paths {
        if path.assignments.get("native_rdx_initial") != Some(&1) {
            continue;
        }
        let exit = path.native_exit.as_ref().unwrap();
        let returns_to_caller =
            exit.return_address == Expr::register(BitWidth::W64, "native_return_address_initial");
        println!("returns_to_original_caller={returns_to_caller}");
        println!(
            "return_address deps={:?}",
            exit.return_address.intrinsic_dependencies()
        );
        let regs: Vec<&X64NativeRegister> = if returns_to_caller {
            vec![
                &X64NativeRegister::Rbx,
                &X64NativeRegister::Rbp,
                &X64NativeRegister::Rsi,
                &X64NativeRegister::Rdi,
                &X64NativeRegister::R12,
                &X64NativeRegister::R13,
                &X64NativeRegister::R14,
                &X64NativeRegister::R15,
                &X64NativeRegister::Rax,
            ]
        } else {
            X64NativeRegister::ALL.iter().collect()
        };
        for r in regs {
            if let Some(v) = exit.registers.get(r) {
                let deps = v.intrinsic_dependencies();
                if deps.iter().any(|d| d.starts_with("fresh_allocation")) {
                    println!("  {r:?} depends on fresh alloc: deps={deps:?}");
                }
            }
        }
        if exit
            .rflags
            .intrinsic_dependencies()
            .iter()
            .any(|d| d.starts_with("fresh_allocation"))
        {
            println!("  rflags depends on fresh alloc");
        }
    }
}
