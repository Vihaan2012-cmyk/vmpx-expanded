use std::fs;
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    EntryProfileRecoveryLimits, VmDiscoveryLimits, analyze_vm_entry_at, analyze_x64_native_frame,
    recover_bootstrap_dispatch,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let t: u32 = u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap();
    let limits = EntryProfileRecoveryLimits::default();
    let c = analyze_vm_entry_at(&pe, Rva(t), VmDiscoveryLimits::default()).unwrap();
    let (_, b) = recover_bootstrap_dispatch(&pe, &c, limits.trace).unwrap();
    let b = b.unwrap();
    let f = analyze_x64_native_frame(&pe, c.rva, limits.trace).unwrap();
    println!(
        "vip={:?} key={:?} handler={:?} dir={:?}",
        b.roles.vip, b.roles.rolling_key, b.roles.handler, b.bytecode_direction
    );
    println!(
        "frame cursor(vsp)={:?} slots={}",
        f.cursor_architectural_register,
        f.slots.len()
    );
    println!(
        "initial vip={:?} key={:#x} handler={:?}",
        b.initial_state.vip, b.initial_state.rolling_key, b.initial_state.handler
    );
}
