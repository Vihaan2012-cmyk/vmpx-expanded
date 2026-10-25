use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_recompile::{CodeAddress, PhysicalRegister};
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, VmDiscoveryLimits, VmEntrySource,
    VmProfileExplorerLimits, X64NativeRegister, discover_vm_entries, recover_entry_paths,
};
use vmpx_vmlift::{ExpressionCodegenConfig, ExpressionInput, compile_expression};

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
        let Some(exit) = path.native_exit.as_ref() else {
            continue;
        };
        println!(
            "\n=== reason {:?}",
            path.assignments.get("native_rdx_initial")
        );
        // Map every symbolic native input to a distinct physical register.
        // Map every symbolic input the exit state can reference, including the
        // VM stack root, to a distinct physical register.
        // Inputs may only use registers that are not the result, temporary, or
        // shift-count register.
        let phys = [
            PhysicalRegister::Rbx,
            PhysicalRegister::Rsi,
            PhysicalRegister::Rdi,
            PhysicalRegister::R8,
            PhysicalRegister::R9,
            PhysicalRegister::R10,
            PhysicalRegister::R11,
            PhysicalRegister::R12,
            PhysicalRegister::R13,
            PhysicalRegister::R14,
            PhysicalRegister::R15,
            PhysicalRegister::Rdx,
            PhysicalRegister::Rbx,
            PhysicalRegister::Rsi,
            PhysicalRegister::Rdi,
            PhysicalRegister::R8,
            PhysicalRegister::Rbp,
        ];
        let mut names: Vec<String> = X64NativeRegister::ALL
            .iter()
            .map(|r| r.symbolic_name().to_owned())
            .collect();
        names.push("vsp_initial".to_owned());
        names.push("native_return_address_initial".to_owned());
        names.sort();
        names.dedup();
        let mut inputs: Vec<ExpressionInput> = Vec::new();
        for (name, p) in names.into_iter().zip(phys) {
            inputs.push(ExpressionInput::register(name, BitWidth::W64, p));
        }
        let config = ExpressionCodegenConfig {
            inputs: inputs.clone(),
            result_register: PhysicalRegister::Rax,
            temporary_register: PhysicalRegister::Rbp,
            shift_count_register: PhysicalRegister::Rcx,
            max_spill_slots: 64,
            max_expression_nodes: 200_000,
        };
        let mut compiled = 0;
        let mut failed = 0;
        let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
        for (r, expr) in &exit.registers {
            match compile_expression(expr, &config, CodeAddress(0x1_4000_1000)) {
                Ok(c) => {
                    compiled += 1;
                    println!("  {r:?}: compiled {} bytes", c.bytes.len());
                }
                Err(e) => {
                    failed += 1;
                    let key = format!("{e}").chars().take(70).collect::<String>();
                    *reasons.entry(key).or_default() += 1;
                }
            }
        }
        println!("  compiled={compiled} failed={failed}");
        for (k, v) in &reasons {
            println!("    {v}x {k}");
        }
    }
}
