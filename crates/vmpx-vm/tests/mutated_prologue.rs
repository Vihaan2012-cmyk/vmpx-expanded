//! A VM entry whose `push seed; call bootstrap` skeleton is buried under
//! mutation junk must still be recognized. Before the prologue walk, discovery
//! required the seed push to be the first instruction and the call the second,
//! so Ultra (mutate-then-virtualize) entries -- which scatter junk across the
//! prologue -- scored as if they had no seed and no bootstrap at all.

use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{VmDiscoveryLimits, analyze_vm_entry_at};

/// Minimal mapped PE32+ with one executable `.text` at RVA 0x1000 (file 0x200)
/// holding `code`.
fn pe_with_text(code: &[u8]) -> Vec<u8> {
    const BASE: u64 = 0x1_4000_0000;
    let mut d = vec![0u8; 0x400];
    d[0..2].copy_from_slice(&0x5A4Du16.to_le_bytes());
    d[0x3C..0x40].copy_from_slice(&0x80u32.to_le_bytes());
    d[0x80..0x84].copy_from_slice(&0x0000_4550u32.to_le_bytes());
    let coff = 0x84;
    d[coff..coff + 2].copy_from_slice(&0x8664u16.to_le_bytes());
    d[coff + 2..coff + 4].copy_from_slice(&1u16.to_le_bytes());
    d[coff + 16..coff + 18].copy_from_slice(&0xF0u16.to_le_bytes());
    d[coff + 18..coff + 20].copy_from_slice(&0x2022u16.to_le_bytes());
    let o = coff + 20;
    d[o..o + 2].copy_from_slice(&0x020Bu16.to_le_bytes());
    d[o + 16..o + 20].copy_from_slice(&0x1000u32.to_le_bytes()); // entry rva
    d[o + 24..o + 32].copy_from_slice(&BASE.to_le_bytes());
    d[o + 32..o + 36].copy_from_slice(&0x1000u32.to_le_bytes()); // section align
    d[o + 36..o + 40].copy_from_slice(&0x200u32.to_le_bytes()); // file align
    d[o + 56..o + 60].copy_from_slice(&0x2000u32.to_le_bytes()); // size of image
    d[o + 60..o + 64].copy_from_slice(&0x200u32.to_le_bytes()); // size of headers
    d[o + 68..o + 70].copy_from_slice(&3u16.to_le_bytes());
    d[o + 72..o + 80].copy_from_slice(&0x10_0000u64.to_le_bytes());
    d[o + 80..o + 88].copy_from_slice(&0x1000u64.to_le_bytes());
    d[o + 88..o + 96].copy_from_slice(&0x10_0000u64.to_le_bytes());
    d[o + 96..o + 104].copy_from_slice(&0x1000u64.to_le_bytes());
    d[o + 108..o + 112].copy_from_slice(&16u32.to_le_bytes());
    let s = o + 0xF0;
    d[s..s + 5].copy_from_slice(b".text");
    d[s + 8..s + 12].copy_from_slice(&0x180u32.to_le_bytes()); // virtual size
    d[s + 12..s + 16].copy_from_slice(&0x1000u32.to_le_bytes()); // rva
    d[s + 16..s + 20].copy_from_slice(&0x200u32.to_le_bytes()); // raw size
    d[s + 20..s + 24].copy_from_slice(&0x200u32.to_le_bytes()); // raw ptr
    d[s + 36..s + 40].copy_from_slice(&0x6000_0020u32.to_le_bytes());
    d[0x200..0x200 + code.len()].copy_from_slice(code);
    d
}

#[test]
fn a_seed_and_bootstrap_are_found_through_prologue_mutation() {
    // 0x1000: xor eax,eax ; nop ; push 0x1234 ; mov rax,rbx ; call 0x1010 ; ret
    // The seed push is the third instruction and the call the fifth -- neither
    // in the fixed position the old gate demanded.
    let code = [
        0x31, 0xC0, // xor eax,eax
        0x90, // nop
        0x68, 0x34, 0x12, 0x00, 0x00, // push 0x1234
        0x48, 0x89, 0xD8, // mov rax,rbx
        0xE8, 0x00, 0x00, 0x00, 0x00, // call 0x1010 (rel32 = 0)
        0xC3, // 0x1010: ret (bootstrap)
    ];
    let bytes = pe_with_text(&code);
    let pe = PeImage::parse(&bytes).expect("fixture parses");

    let candidate = analyze_vm_entry_at(&pe, Rva(0x1000), VmDiscoveryLimits::default())
        .expect("entry decodes to a candidate");
    assert_eq!(
        candidate.seed,
        Some(0x1234),
        "the immediate seed is recovered past the junk before it"
    );
    assert_eq!(
        candidate.call_target,
        Some(Rva(0x1010)),
        "the bootstrap call is found past the junk after the seed"
    );
}
