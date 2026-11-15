//! Locate, from read-only guest state dumps, where the depacker keeps dicPos /
//! reps / state, and prove the r15 prob block is the stock layout to the bit.
//!
//! Replays the native decoder to each dump's input offset, then triangulates:
//! a storage slot (register, stack word, or r15-relative word) that tracks the
//! replay's known value across all three dumps is that field.
//!
//! usage: lzma_locate <stream input.bin> <dump dir>
use std::collections::BTreeMap;
use vmpx_emulate::decode_lzma1_snapshot_at_input;

fn parse_meta(path: &str) -> Option<BTreeMap<String, String>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut m = BTreeMap::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            m.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    Some(m)
}
fn hex(s: &str) -> u64 {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0)
}
fn u32le(b: &[u8], i: usize) -> Option<u32> {
    b.get(i..i + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}
fn u64le(b: &[u8], i: usize) -> Option<u64> {
    b.get(i..i + 8).map(|s| u64::from_le_bytes(s.try_into().unwrap()))
}

struct Dump {
    offset: u64,
    r15: u64,
    probs_reg: u64,
    regs: Vec<(String, u64)>,
    r15_blob: Vec<u8>,
    stack_blob: Vec<u8>,
    // replay-derived ground truth at this offset:
    out_pos: u64,
    reps: [u32; 4],
    state: u32,
    my_probs: Vec<u16>,
    image_base: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read(&args[1]).expect("stream");
    let dir = &args[2];
    let cap = 300usize << 20;

    // The stream may begin at byte 0 (fold stream.bin) or byte 1 (old capture).
    // Pick the start offset whose native replay best matches the guest r15 block
    // at the first available dump.
    let first_meta = (0..3)
        .find_map(|i| parse_meta(&format!("{dir}/state{i}_meta.txt")).map(|m| (i, m)))
        .expect("no dump meta found");
    let (fi, fm) = first_meta;
    let foff = hex(&fm["offset"]);
    let fr15 = std::fs::read(format!("{dir}/state{fi}_r15.bin")).unwrap_or_default();
    let start = [0usize, 1]
        .into_iter()
        .min_by_key(|&s| {
            match decode_lzma1_snapshot_at_input(&raw[s..], cap, 3, 0, 2, foff as usize) {
                Ok((_, _, Some(st))) => (0..st.probs.len())
                    .filter(|&j| {
                        u32le(&fr15, j * 2).map(|v| (v & 0xffff) as u16) != Some(st.probs[j])
                    })
                    .count(),
                _ => usize::MAX,
            }
        })
        .unwrap_or(1);
    println!("stream start offset = {start} (best r15 match)");
    let input = &raw[start..];

    let mut dumps = Vec::new();
    for i in 0..3 {
        let meta = match parse_meta(&format!("{dir}/state{i}_meta.txt")) {
            Some(m) => m,
            None => {
                println!("dump {i}: missing, skipping");
                continue;
            }
        };
        let offset = hex(&meta["offset"]);
        let r15 = hex(&meta["r15"]);
        let probs_reg = hex(&meta["probs"]);
        let regs: Vec<(String, u64)> = meta
            .get("regs")
            .map(|r| {
                r.split_whitespace()
                    .filter_map(|t| t.split_once('=').map(|(n, v)| (n.to_string(), hex(v))))
                    .collect()
            })
            .unwrap_or_default();
        let r15_blob = std::fs::read(format!("{dir}/state{i}_r15.bin")).unwrap_or_default();
        let stack_blob = std::fs::read(format!("{dir}/state{i}_stack.bin")).unwrap_or_default();

        let (_out, _c, snap) =
            decode_lzma1_snapshot_at_input(input, cap, 3, 0, 2, offset as usize)
                .expect("replay");
        let st = snap.expect("offset beyond stream");
        let image_base = u64le(&stack_blob, 0x38).unwrap_or(0);
        dumps.push(Dump {
            offset,
            r15,
            probs_reg,
            regs,
            r15_blob,
            stack_blob,
            out_pos: st.out_pos as u64,
            reps: st.reps,
            state: st.state,
            my_probs: st.probs,
            image_base,
        });
    }
    if dumps.is_empty() {
        println!("no dumps found in {dir}");
        std::process::exit(1);
    }

    // --- Layout proof: r15 block vs my stock-layout prob block --------------
    println!("=== prob layout (r15 block vs native stock-layout export) ===");
    for (i, d) in dumps.iter().enumerate() {
        let n = d.my_probs.len();
        let mut diffs = 0usize;
        let mut first = Vec::new();
        for j in 0..n {
            let g = u32le(&d.r15_blob, j * 2).map(|v| (v & 0xffff) as u16);
            if g != Some(d.my_probs[j]) {
                diffs += 1;
                if first.len() < 6 {
                    first.push(j);
                }
            }
        }
        let lit_ctx = d.probs_reg.wrapping_sub(d.r15);
        let lit_ok = lit_ctx >= 0xE6C && (lit_ctx - 0xE6C) % 0x300 == 0;
        println!(
            "dump {i} off={:#x}: {}/{} u16 differ (expect <~10, in-flight literal) | probs_reg-r15={:#x} literal-context {}",
            d.offset,
            diffs,
            n,
            lit_ctx,
            if lit_ok { "ALIGNED (stock)" } else { "UNALIGNED" }
        );
        if diffs > 0 && diffs <= 20 {
            print!("   differing prob indices:");
            for j in &first {
                print!(" {j}");
            }
            println!();
        }
    }

    // --- Field location by cross-dump triangulation ------------------------
    // A slot holding value v_i in dump i for all i, where v_i is the replay's
    // known field value, is that field. dicPos may be stored relative (out_pos)
    // or absolute (image_base + out_pos); try both.
    println!("\n=== field location (slots tracking replay values across all dumps) ===");
    locate_u("dicPos(rel u32)", &dumps, |d| vec![d.out_pos], 4, 64);
    locate_u("dicPos(abs u64)", &dumps, |d| vec![d.image_base + d.out_pos], 8, 64);
    locate_u("rep0", &dumps, |d| vec![d.reps[0] as u64], 4, 0);
    locate_u("rep1", &dumps, |d| vec![d.reps[1] as u64], 4, 0);
    locate_u("rep2", &dumps, |d| vec![d.reps[2] as u64], 4, 0);
    locate_u("rep3", &dumps, |d| vec![d.reps[3] as u64], 4, 0);
    locate_u("state", &dumps, |d| vec![d.state as u64], 4, 0);

    println!("\nreplay ground truth per dump:");
    for (i, d) in dumps.iter().enumerate() {
        println!(
            "  dump {i}: out_pos={:#x} reps=[{:#x},{:#x},{:#x},{:#x}] state={} image_base={:#x}",
            d.out_pos, d.reps[0], d.reps[1], d.reps[2], d.reps[3], d.state, d.image_base
        );
    }
}

/// Find storage slots whose value matches `target(d)` (within `tol`) in every
/// dump. Searches registers, the stack blob and the r15 blob.
fn locate_u(name: &str, dumps: &[Dump], target: impl Fn(&Dump) -> Vec<u64>, width: usize, tol: u64) {
    let close = |a: u64, b: u64| a.max(b) - a.min(b) <= tol;
    let read = |b: &[u8], i: usize| -> Option<u64> {
        if width == 4 {
            u32le(b, i).map(u64::from)
        } else {
            u64le(b, i)
        }
    };
    let mut hits = Vec::new();

    // Registers (compare full u64 and low u32).
    if let Some(d0) = dumps.first() {
        for (ri, (rn, _)) in d0.regs.iter().enumerate() {
            let ok = dumps.iter().all(|d| {
                let v = d.regs.get(ri).map(|x| x.1).unwrap_or(u64::MAX);
                target(d).iter().any(|t| close(v, *t) || close(v & 0xffff_ffff, *t))
            });
            if ok {
                hits.push(format!("reg:{rn}"));
            }
        }
    }
    // Stack and r15 blobs: scan every byte offset.
    for (label, pick) in [("stack", 0u8), ("r15", 1u8)] {
        let blob0: &[u8] = if pick == 0 { &dumps[0].stack_blob } else { &dumps[0].r15_blob };
        for off in 0..blob0.len().saturating_sub(width) {
            let ok = dumps.iter().all(|d| {
                let b: &[u8] = if pick == 0 { &d.stack_blob } else { &d.r15_blob };
                match read(b, off) {
                    Some(v) => target(d).iter().any(|t| close(v, *t)),
                    None => false,
                }
            });
            if ok {
                hits.push(format!("{label}+{off:#x}"));
            }
        }
    }
    if hits.is_empty() {
        println!("{name:16}: (not found)");
    } else {
        println!("{name:16}: {}", hits.join("  "));
    }
}
