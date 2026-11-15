//! Validate the native decoder against a real sample's actual LZMA sections:
//! locate the packer table, then decode each section's stream with decode_lzma1
//! and check it reaches EOS at exactly the packer's recorded decompressed size.
//! usage: lzma_systems <packed.xpl>
use vmpx_emulate::decode_lzma1;
use vmpx_pe::PeImage;
use vmpx_unpack::{UnpackLimits, locate_packer_table};

fn main() {
    let path = std::env::args().nth(1).expect("usage: lzma_systems <packed.xpl>");
    let data = std::fs::read(&path).expect("read");
    let pe = PeImage::parse(&data).expect("parse PE");
    let (_sections, table, warnings) =
        locate_packer_table(&pe, UnpackLimits::default()).expect("locate packer table");
    for w in &warnings {
        println!("warn: {w}");
    }
    let props = table.lzma_properties[0] as u32;
    let lc = props % 9;
    let lp = (props / 9) % 5;
    let pb = props / 45;
    println!(
        "packer table: {} section(s), lc={lc} lp={lp} pb={pb}",
        table.entries.len()
    );

    let (mut ok, mut bad) = (0u32, 0u32);
    let t0 = std::time::Instant::now();
    let mut total_out = 0usize;
    for (i, e) in table.entries.iter().enumerate() {
        let off = match pe.rva_to_file_offset(e.source_rva) {
            Ok(o) => o.0 as usize,
            Err(_) => {
                println!("section {i}: source rva {:#x} not file-backed", e.source_rva.0);
                bad += 1;
                continue;
            }
        };
        let want = e.decompressed_size as usize;
        match decode_lzma1(&data[off..], want, lc, lp, pb) {
            Ok((out, consumed)) => {
                total_out += out.len();
                let exact = out.len() == want;
                if exact {
                    ok += 1;
                } else {
                    bad += 1;
                }
                println!(
                    "section {i:2}: dst={:#010x} decoded {} / want {} ({}) consumed {} input",
                    e.destination_rva.0,
                    out.len(),
                    want,
                    if exact { "MATCH" } else { "SHORT/LONG" },
                    consumed
                );
            }
            Err(err) => {
                println!("section {i:2}: decode error: {err}");
                bad += 1;
            }
        }
    }
    println!(
        "\n{ok} ok / {bad} bad; {total_out} bytes decoded in {:?}",
        t0.elapsed()
    );
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
