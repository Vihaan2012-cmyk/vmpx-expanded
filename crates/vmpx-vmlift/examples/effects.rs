use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, ExprKind, VmDiscoveryLimits, VmEntrySource,
    VmProfileExplorerLimits, discover_vm_entries, recover_entry_paths, symbolic_pointer_offset,
};

fn describe(e: &Expr) -> String {
    if let Some(off) = symbolic_pointer_offset(e, "vsp_initial") {
        return format!("vsp{off:+#x}");
    }
    let deps = e.intrinsic_dependencies();
    let fresh = deps.iter().find(|d| d.starts_with("fresh_allocation"));
    if let Some(f) = fresh {
        let off = symbolic_pointer_offset(e, f).unwrap_or(0);
        return format!("fresh{off:+#x}");
    }
    match &e.kind {
        ExprKind::Register { name } => format!("reg {name}"),
        ExprKind::Const { value } => format!("const {value:#x}"),
        _ => format!("other deps={deps:?}"),
    }
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let bytes = fs::read(&args[1]).expect("read");
    let pe = PeImage::parse(&bytes).expect("parse");
    let discovery = discover_vm_entries(&pe,
        VmDiscoveryLimits { scan_executable_stubs: false, ..Default::default() }).expect("d");
    let domains = BTreeMap::from([(
        "native_rdx_initial".to_owned(),
        (0..=3).map(|r| Expr::constant(BitWidth::W64, r)).collect(),
    )]);
    let limits = EntryPathRecoveryLimits {
        chain: VmProfileExplorerLimits {
            max_segments: 32, max_instructions_per_segment: 2048, max_total_instructions: 8192,
            ..Default::default() },
        ..Default::default()
    };
    for entry in discovery.entries.iter().filter(|e| e.source == VmEntrySource::PeEntryPoint) {
        let Ok(family) = recover_entry_paths(&pe, entry, &domains, limits) else { continue };
        for path in &family.paths {
            let reason = path.assignments.get("native_rdx_initial").copied();
            if reason != Some(1) { continue }
            println!("=== reason {reason:?} stop {:?}", path.chain.stop_reason);
            let mut reports: Vec<&vmpx_vm::SymbolicVmExecutionReport> = vec![&family.entry_profile.symbolic];
            for s in &path.chain.segments {
                reports.extend(s.symbolic.iter());
                reports.extend(s.boundary_symbolic.iter());
                reports.extend(s.transition_pop.iter());
            }
            for (i, r) in reports.iter().enumerate() {
                if r.external_writes.is_empty() { continue }
                println!(" report {i}: {} writes", r.external_writes.len());
                for w in &r.external_writes {
                    println!("    {} <- {}", describe(&w.address), describe(&w.value));
                }
            }
            println!("--- native exit regs ---");
            if let Some(exit) = &path.native_exit {
                for (reg, v) in &exit.registers {
                    let d = describe(v);
                    if d.starts_with("const") || d.starts_with("reg") { println!("   {reg:?} = {d}"); }
                }
                println!("   ret = {}", describe(&exit.return_address));
                println!("   rflags = {}", describe(&exit.rflags));
            }
        }
    }
}
