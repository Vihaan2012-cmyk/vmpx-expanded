//! Verify the custom LZMA1 decoder against the captured reference.
//! usage: lzma_try <input.bin> <output.bin>
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = std::fs::read(&args[1]).expect("input");
    let output = std::fs::read(&args[2]).expect("output");
    let needle = &output[..2048.min(output.len())];

    for off in [1usize, 0, 2] {
        let t0 = std::time::Instant::now();
        match vmpx_emulate::decode_lzma1(&input[off..], 300 << 20, 3, 0, 2) {
            Ok((dec, consumed)) => {
                let hit = dec.windows(needle.len()).position(|w| w == needle);
                println!(
                    "off={off}: decoded {} bytes (consumed {} input) in {:?}, needle_at={:?}",
                    dec.len(),
                    consumed,
                    t0.elapsed(),
                    hit
                );
                if hit.is_some() {
                    println!("  MATCH -- custom decoder reproduces the emulated output");
                    return;
                }
            }
            Err(e) => println!("off={off}: error: {e}"),
        }
    }
    println!("no offset reproduced the reference output");
}
