//! Static histogram of vector (xmm/ymm/zmm/mm) instruction mnemonics in a mapped PE image,
//! decoded per `.pdata` function when readable, else as one linear sweep of `.text`.
//! usage: sse_histogram <mapped.bin>   (image laid out at section VAs, as `detonate --dump` writes)
use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, OpKind};
use std::collections::BTreeMap;

fn u32at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn is_vec(i: &Instruction) -> bool {
    matches!(i.mnemonic(), Mnemonic::Ldmxcsr | Mnemonic::Stmxcsr)
        || (0..i.op_count()).any(|k| {
            i.op_kind(k) == OpKind::Register && {
                let r = i.op_register(k);
                r.is_xmm() || r.is_ymm() || r.is_zmm() || r.is_mm()
            }
        })
}

fn main() {
    let img = std::fs::read(std::env::args().nth(1).expect("usage: sse_histogram <mapped.bin>")).unwrap();
    let pe = u32at(&img, 0x3c) as usize;
    let nsec = u16::from_le_bytes(img[pe + 6..pe + 8].try_into().unwrap()) as usize;
    let opt = u16::from_le_bytes(img[pe + 20..pe + 22].try_into().unwrap()) as usize;
    let (mut text, mut pdata) = ((0, 0), (0, 0));
    for i in 0..nsec {
        let o = pe + 24 + opt + i * 40;
        let (va, vs) = (u32at(&img, o + 12) as usize, u32at(&img, o + 8) as usize);
        match &img[o..o + 8] {
            b".text\0\0\0" => text = (va, vs),
            b".pdata\0\0" => pdata = (va, vs),
            _ => {}
        }
    }
    let (mut hist, mut total, mut vec_total, mut funcs) = (BTreeMap::<String, u64>::new(), 0u64, 0u64, 0u64);
    // `.pdata` is encrypted in a VMP-packed pre-init image, so function bounds are unavailable
    // there; fall back to a linear sweep of `.text` (invalid bytes are skipped, counts approximate).
    let mut per_func_vec = 0u64;
    let ranges: Vec<(usize, usize)> = if pdata.1 > 0 && (0..pdata.1 / 12).all(|k| {
        let e = pdata.0 + k * 12;
        let (b, en) = (u32at(&img, e) as usize, u32at(&img, e + 4) as usize);
        b >= text.0 && en <= text.0 + text.1 && en > b
    }) {
        (0..pdata.1 / 12).map(|k| (u32at(&img, pdata.0 + k * 12) as usize, u32at(&img, pdata.0 + k * 12 + 4) as usize)).collect()
    } else {
        vec![(text.0, text.0 + text.1)]
    };
    for (b, en) in ranges {
        funcs += 1;
        let mut d = Decoder::with_ip(64, &img[b..en], b as u64, DecoderOptions::NONE);
        let mut hit = false;
        while d.can_decode() {
            let i = d.decode();
            total += 1;
            if !i.is_invalid() && is_vec(&i) {
                vec_total += 1;
                hit = true;
                *hist.entry(format!("{:?}", i.mnemonic())).or_default() += 1;
            }
        }
        per_func_vec += u64::from(hit);
    }
    println!("functions {funcs} (with vector insns: {per_func_vec}), insns {total}, vector insns {vec_total}");
    let mut v: Vec<_> = hist.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (m, c) in v {
        println!("{c:>8} {m}");
    }
}
