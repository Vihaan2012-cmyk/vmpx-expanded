//! End-to-end Layer-1 checks: map a real x64 DLL, invoke exports under the
//! interpreter, and verify section permissions and import-stub resolution.

use vmpx_emulate::{Access, Memory, Stop, load};
use vmpx_pe::PeImage;

const UNITOPS: &[u8] = include_bytes!("E:/VMPX/testcorpus/corpus/unitops.dll");
const COREDLL: &[u8] = include_bytes!("E:/VMPX/testcorpus/originals/coredll.dll");

const MAX_STEPS: u64 = 10_000;

/// Load `data`, invoke export `name` with `args`, run to the synthetic return
/// and return RAX.
fn call_export(data: &[u8], name: &str, args: &[u64]) -> u64 {
    let pe = PeImage::parse(data).expect("parse PE");
    let mut img = load(&pe, &vmpx_emulate::LoadOptions::default()).expect("load image");
    let rip = img.export(name).unwrap_or_else(|| panic!("export {name} missing"));
    let sentinel = img.return_sentinel();
    let mut emu = img.prepare(rip, args).expect("prepare");
    match emu.run_until(sentinel, MAX_STEPS) {
        Stop::Reached => emu.cpu.regs[0],
        other => panic!("{name} did not return cleanly: {other:?} (trace {})", emu.trace.len()),
    }
}

#[test]
fn op_add32_sub32_xor32_end_to_end() {
    assert_eq!(call_export(UNITOPS, "op_add32", &[3, 4]) & 0xffff_ffff, 7);
    assert_eq!(call_export(UNITOPS, "op_sub32", &[10, 3]) & 0xffff_ffff, 7);
    assert_eq!(call_export(UNITOPS, "op_xor32", &[0xF0, 0x0F]) & 0xffff_ffff, 0xFF);
}

#[test]
fn section_permissions_are_enforced() {
    let pe = PeImage::parse(UNITOPS).expect("parse PE");
    let mut img = load(&pe, &vmpx_emulate::LoadOptions::default()).expect("load");
    let base = img.image_base;
    let mem: &mut Memory = &mut img.memory;

    // .text (RVA 0x1000) is executable + readable, not writable.
    let text = base + 0x1000;
    assert!(mem.fetch(text, &mut [0u8; 1]).is_ok(), "text must be executable");
    assert_eq!(mem.write(text, &[0]).unwrap_err().access, Access::Write);

    // .rdata (RVA 0x2000) is readable, not executable or writable.
    let rdata = base + 0x2000;
    assert!(mem.read(rdata, &mut [0u8; 1]).is_ok(), "rdata must be readable");
    assert_eq!(mem.fetch(rdata, &mut [0u8; 1]).unwrap_err().access, Access::Exec);
    assert_eq!(mem.write(rdata, &[0]).unwrap_err().access, Access::Write);
}

#[test]
fn iat_slots_point_at_resolvable_stubs() {
    let pe = PeImage::parse(COREDLL).expect("parse coredll");
    let img = load(&pe, &vmpx_emulate::LoadOptions::default()).expect("load coredll");
    let report = pe.report();

    assert!(!img.import_stubs.is_empty(), "coredll imports KERNEL32");
    // Every IAT slot holds a stub address that resolves to its (dll, function).
    let mut checked = 0;
    for module in &report.imports {
        for sym in &module.entries {
            let iat_va = img.image_base + u64::from(sym.iat_rva.0);
            let stub = img.memory.read_uint(iat_va, 8).expect("read IAT slot");
            let (dll, _func) = img.resolve_stub(stub).expect("stub resolves");
            assert_eq!(dll, &module.name);
            checked += 1;
        }
    }
    assert!(checked > 0);
}

/// Informational: how far the realistic coredll image runs from its entry
/// before the Layer-1 environment is exhausted (first import stub or unmodeled
/// instruction). Not an assertion of unpack success.
#[test]
fn coredll_entry_run_is_reported() {
    let pe = PeImage::parse(COREDLL).expect("parse coredll");
    let mut img = load(&pe, &vmpx_emulate::LoadOptions::default()).expect("load coredll");
    let Some(entry) = img.entry else {
        eprintln!("coredll: no entry point");
        return;
    };
    let mut emu = img.prepare(entry, &[0, 1, 0]).expect("prepare");
    let stop = emu.run(MAX_STEPS);
    let last = emu.trace.last().map(|t| (t.rip, t.mnemonic));
    eprintln!(
        "coredll entry {entry:#x}: ran {} instrs, stopped {stop:?}, last {last:?}",
        emu.trace.len()
    );
}
