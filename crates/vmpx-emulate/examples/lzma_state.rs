//! Prove the stock-layout state export/resume is byte-exact: decode a stream to
//! EOS, snapshot full decoder state at a mid-stream symbol, serialize it to the
//! stock 7-zip prob layout, rebuild a decoder from that block, finish the decode,
//! and confirm the resumed output and input consumption match the reference.
//! usage: lzma_state <input.bin> [snap_at ...]
use vmpx_emulate::{decode_lzma1, decode_lzma1_snapshot, resume_lzma1};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read(&args[1]).expect("input");
    // The capture starts one byte before the range-coder init (byte 0 ignored).
    let input = &raw[1..];
    let cap = 300usize << 20;

    let (reference, ref_consumed) = decode_lzma1(input, cap, 3, 0, 2).expect("reference decode");
    println!(
        "reference: {} bytes decompressed, {} input bytes consumed to EOS",
        reference.len(),
        ref_consumed
    );

    let snaps: Vec<u64> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse().unwrap()).collect()
    } else {
        vec![1, 1000, 500_000, 5_000_000]
    };

    let mut all_ok = true;
    for snap_at in snaps {
        let (out, consumed, snap) =
            decode_lzma1_snapshot(input, cap, 3, 0, 2, snap_at).expect("snapshot decode");
        assert_eq!(out, reference, "snapshot decode diverged from reference");
        assert_eq!(consumed, ref_consumed);
        let Some(st) = snap else {
            println!("snap_at={snap_at}: stream shorter than this symbol, skipped");
            continue;
        };
        // Resume from the snapshot: feed input from the recorded byte offset, and
        // seed the output with exactly the prefix decoded before the snapshot.
        let mut resumed = reference[..st.out_pos].to_vec();
        let r_consumed = resume_lzma1(&st, &input[st.input_pos..], cap, &mut resumed)
            .expect("resume decode");
        let ok = resumed == reference && st.input_pos + r_consumed == ref_consumed;
        all_ok &= ok;
        println!(
            "snap_at={snap_at:>8}: out_pos={:>9} input_pos={:>9} probs={} -> resume {} ({}+{}={} vs {})",
            st.out_pos,
            st.input_pos,
            st.probs.len(),
            if ok { "MATCH" } else { "MISMATCH" },
            st.input_pos,
            r_consumed,
            st.input_pos + r_consumed,
            ref_consumed
        );
    }
    println!(
        "\n{}",
        if all_ok {
            "ALL MATCH -- stock-layout export/resume is byte-exact; the guest can be handed this blob"
        } else {
            "FAIL -- export/resume diverged"
        }
    );
    std::process::exit(if all_ok { 0 } else { 1 });
}
