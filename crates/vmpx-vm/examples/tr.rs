use std::fs;
use vmpx_disasm::{DirectJumpTraceLimits, trace_direct_jump_fragments};
use vmpx_pe::{PeImage, Rva};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let rva: u32 = u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap();
    let trace =
        trace_direct_jump_fragments(&pe, Rva(rva), DirectJumpTraceLimits::default()).unwrap();
    println!(
        "entry={:?} stop.kind={:?} stop.at={:?}",
        trace.entry, trace.stop.kind, trace.stop.at
    );
    for d in trace.instructions() {
        println!("  {:?} {}", d.rva, d.instruction);
    }
}
