// Emits one replacement, the input values it will be called with, and the exact
// register values the symbolic exit state predicts. A standalone harness executes
// the bytes and compares; the workspace forbids unsafe code, so execution lives
// outside it.
use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_recompile::{CodeAddress, PhysicalRegister};
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, VmDiscoveryLimits, VmEntrySource,
    VmProfileExplorerLimits, X64NativeRegister, analyze_x64_native_frame, discover_vm_entries,
    recover_entry_paths,
};
use vmpx_vmlift::{ReplacementConfig, build_replacement};

/// The stack root the harness maps and passes. Fixing it means every
/// stack-relative result is an exact constant, so no expectation needs rebasing.
const STACK_ROOT: u64 = 0x2_0008_0000;

fn physical_for(r: &X64NativeRegister) -> PhysicalRegister {
    use PhysicalRegister as P;
    use X64NativeRegister as R;
    match r {
        R::Rax => P::Rax,
        R::Rbx => P::Rbx,
        R::Rcx => P::Rcx,
        R::Rdx => P::Rdx,
        R::Rbp => P::Rbp,
        R::Rsi => P::Rsi,
        R::Rdi => P::Rdi,
        R::R8 => P::R8,
        R::R9 => P::R9,
        R::R10 => P::R10,
        R::R11 => P::R11,
        R::R12 => P::R12,
        R::R13 => P::R13,
        R::R14 => P::R14,
        R::R15 => P::R15,
    }
}

/// Index `i` names the architectural register the harness must place in
/// `in_vals[i]`. The emitter sorts exit registers lexicographically by name, so
/// the harness uses the same order and indexes expectations by it.
const ORDER: [(&str, X64NativeRegister); 15] = [
    ("native_r10_initial", X64NativeRegister::R10),
    ("native_r11_initial", X64NativeRegister::R11),
    ("native_r12_initial", X64NativeRegister::R12),
    ("native_r13_initial", X64NativeRegister::R13),
    ("native_r14_initial", X64NativeRegister::R14),
    ("native_r15_initial", X64NativeRegister::R15),
    ("native_r8_initial", X64NativeRegister::R8),
    ("native_r9_initial", X64NativeRegister::R9),
    ("native_rax_initial", X64NativeRegister::Rax),
    ("vsp_initial", X64NativeRegister::Rbp),
    ("native_rbx_initial", X64NativeRegister::Rbx),
    ("native_rcx_initial", X64NativeRegister::Rcx),
    ("native_rdi_initial", X64NativeRegister::Rdi),
    ("native_rdx_initial", X64NativeRegister::Rdx),
    ("native_rsi_initial", X64NativeRegister::Rsi),
];

/// Index into the trampoline's architectural register order.
fn register_index(name: &str) -> usize {
    match name {
        "native_rax_initial" => 0,
        "native_rbx_initial" => 1,
        "native_rcx_initial" => 2,
        "native_rdx_initial" => 3,
        "native_rsi_initial" => 4,
        "native_rdi_initial" => 5,
        "native_r8_initial" => 6,
        "native_r9_initial" => 7,
        "native_r10_initial" => 8,
        "native_r11_initial" => 9,
        "native_r12_initial" => 10,
        "native_r13_initial" => 11,
        "native_r14_initial" => 12,
        "native_r15_initial" => 13,
        "vsp_initial" => 14,
        _ => 0,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let out_dir = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/vmpxexec".to_owned());
    fs::create_dir_all(&out_dir).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let d = discover_vm_entries(
        &pe,
        VmDiscoveryLimits {
            scan_executable_stubs: false,
            ..Default::default()
        },
    )
    .unwrap();
    let entry = d
        .entries
        .iter()
        .find(|e| e.source == VmEntrySource::PeEntryPoint)
        .unwrap();
    let frame = analyze_x64_native_frame(
        &pe,
        entry.rva,
        vmpx_disasm::DirectJumpTraceLimits::default(),
    )
    .unwrap();
    let cursor = frame.cursor_architectural_register.unwrap();
    let domains = BTreeMap::from([(
        "native_rdx_initial".to_owned(),
        vec![Expr::constant(BitWidth::W64, 0)],
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
    let exit = family.paths[0].native_exit.as_ref().unwrap();

    let mut inputs: Vec<(String, PhysicalRegister)> = X64NativeRegister::ALL
        .iter()
        .map(|r| (r.symbolic_name().to_owned(), physical_for(r)))
        .collect();
    let mut symbol_rewrites: BTreeMap<String, Expr> = BTreeMap::new();
    let cursor_name = cursor.symbolic_name();
    let slot = frame
        .slots
        .iter()
        .find(|slot| {
            matches!(&slot.source,
            vmpx_vm::NativeFrameSource::Register { register }
                if register.symbolic_name() == cursor_name)
        })
        .expect("cursor save slot");
    let address = Expr::binary(
        vmpx_vm::BinaryOp::Add,
        Expr::register(BitWidth::W64, "vsp_initial"),
        Expr::constant(BitWidth::W64, u64::from(slot.frame_offset)),
    )
    .unwrap();
    symbol_rewrites.insert(cursor_name.to_owned(), Expr::load(BitWidth::W64, address));
    inputs.retain(|(name, _)| !symbol_rewrites.contains_key(name));
    inputs.push(("vsp_initial".to_owned(), physical_for(&cursor)));
    inputs.sort();
    inputs.dedup();

    let va =
        pe.optional.image_base.0 + u64::from(vmpx_rebuild::next_section_rva(&bytes).unwrap().0);
    let cfg = ReplacementConfig {
        inputs: inputs.clone(),
        max_expression_nodes: 200_000,
        max_spill_slots: 64,
        import_calls: BTreeMap::new(),
        continuation: None,
        effects: Vec::new(),
        effect_calls: Vec::new(),
        rflags: None,
        symbol_rewrites,
    };
    let r = build_replacement(&exit.registers, &cfg, CodeAddress(va)).expect("replacement");
    fs::write(format!("{out_dir}/code.bin"), &r.compiled.bytes).unwrap();
    println!("emitted {} bytes", r.compiled.bytes.len());

    // Distinct input values; the stack root is fixed and mapped by the harness.
    let mut subst: BTreeMap<String, Expr> = BTreeMap::new();
    let mut values: Vec<u64> = Vec::new();
    // Values follow architectural order so a reader can predict which register
    // holds which number: index 0 is RAX, and so on.
    let mut next = 0x1000_u64;
    let mut by_register: BTreeMap<usize, u64> = BTreeMap::new();
    for index in 0..ORDER.len() {
        let (name, _) = ORDER[index];
        let value = if name == "vsp_initial" {
            STACK_ROOT
        } else {
            next
        };
        by_register.insert(register_index(name), value);
        subst.insert(name.to_owned(), Expr::constant(BitWidth::W64, value));
        if name != "vsp_initial" {
            next = next.wrapping_add(0x1111_1111);
        }
    }
    for architectural in 0..ORDER.len() {
        values.push(by_register[&architectural]);
    }
    // The saved cursor register lives in the frame; the harness must place the
    // value the stub saved there, which is the register's entry value.
    let mut spec = String::new();
    // The harness needs both the value and which register it belongs to, because
    // the emitter's order is by name and the trampoline loads by register.
    for index in 0..ORDER.len() {
        spec.push_str(&format!("in {index} {index} {}\n", values[index]));
    }
    for slot in &frame.slots {
        let value = match &slot.source {
            vmpx_vm::NativeFrameSource::Register { register } => {
                subst.get(register.symbolic_name()).cloned()
            }
            vmpx_vm::NativeFrameSource::Constant { value }
            | vmpx_vm::NativeFrameSource::EntryImmediate { value, .. } => {
                Some(Expr::constant(BitWidth::W64, *value))
            }
            _ => None,
        };
        if let Some(vmpx_vm::ExprKind::Const { value }) = value.map(|e| e.simplify().kind) {
            spec.push_str(&format!("slot {} {}\n", slot.frame_offset, value));
        }
    }

    let mut known = 0;
    for (i, (name, reg)) in ORDER.iter().enumerate() {
        let value = exit.registers.get(reg).and_then(|e| {
            let rewritten = e.substitute(&subst).ok()?.simplify();
            match rewritten.kind {
                vmpx_vm::ExprKind::Const { value } => Some(value),
                _ => None,
            }
        });
        if let Some(v) = value {
            // Index by architectural register, matching the trampoline's outputs.
            let _ = i;
            spec.push_str(&format!("expect {} {name} {v}\n", register_index(name)));
            known += 1;
        } else {
            println!("  {name}: not a constant under these inputs");
        }
    }
    println!("{known} of {} exit registers are constant", ORDER.len());
    fs::write(format!("{out_dir}/spec.txt"), spec).unwrap();
}
