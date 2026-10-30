// Corpus-backed classifier validation. Ignored by default (the samples live
// outside the repo); run with `cargo test -p vmpx-detect -- --ignored`.
//
// Empirical discriminator (same coredll.dll, packing off):
//   - virt:     all probed VM entry stubs decode into coherent handler streams.
//   - ultra:    stubs recover a bootstrap dispatch but their mutated handlers
//               refuse to decode (0 clean, N mutated).
//   - mutation: its entry-stub swarm is the SAME shared VMProtect runtime VM as
//               the virt build (near-identical decode), so it classifies as
//               Virtualization. Pure native mutation of non-virtualized user
//               code is not observable at the runtime-VM entry layer.
use std::fs;

use vmpx_detect::{ProtectionMode, detect_vmprotect};
use vmpx_pe::PeImage;

const VIRT: &str = r"E:\VMPX\testcorpus\protected\coredll.virt.dll";
const MUTATION: &str = r"E:\VMPX\testcorpus\protected\coredll.mutation.dll";
const ULTRA: &str = r"E:\VMPX\testcorpus\protected\coredll.ultra.dll";

#[test]
#[ignore = "requires the external VMProtect corpus"]
fn virtualization_sample_decodes_clean_vm_streams() {
    let bytes = fs::read(VIRT).expect("virt sample must exist");
    let pe = PeImage::parse(&bytes).expect("virt sample must parse");
    let mode = detect_vmprotect(&pe).protection_mode;
    assert_eq!(mode.mode, ProtectionMode::Virtualization);
    assert!(mode.clean_vm_entries > 0);
    assert_eq!(mode.mutated_vm_entries, 0);
}

#[test]
#[ignore = "requires the external VMProtect corpus"]
fn ultra_sample_recovers_dispatch_but_handlers_are_mutated() {
    let bytes = fs::read(ULTRA).expect("ultra sample must exist");
    let pe = PeImage::parse(&bytes).expect("ultra sample must parse");
    let mode = detect_vmprotect(&pe).protection_mode;
    assert_eq!(mode.mode, ProtectionMode::Ultra);
    assert!(mode.mutated_vm_entries > 0);
    assert_eq!(mode.clean_vm_entries, 0);
}

#[test]
#[ignore = "requires the external VMProtect corpus"]
fn mutation_sample_shares_the_clean_runtime_vm() {
    // Honest limitation: the mutation build carries the identical clean runtime
    // VM, so it is indistinguishable from pure virtualization at this layer.
    let bytes = fs::read(MUTATION).expect("mutation sample must exist");
    let pe = PeImage::parse(&bytes).expect("mutation sample must parse");
    let mode = detect_vmprotect(&pe).protection_mode;
    assert_eq!(mode.mode, ProtectionMode::Virtualization);
    assert!(mode.clean_vm_entries > 0);
    assert_eq!(mode.mutated_vm_entries, 0);
}
