use std::{env, fs, process::ExitCode};

use vmpx_pe::{PeImage, Rva};

fn main() -> ExitCode {
    let arguments = env::args().collect::<Vec<_>>();
    let [_, path, rva] = arguments.as_slice() else {
        eprintln!("usage: cargo run -p vmpx-pe --example unwind -- <pe> <rva-hex>");
        return ExitCode::FAILURE;
    };
    let rva = rva
        .strip_prefix("0x")
        .or_else(|| rva.strip_prefix("0X"))
        .unwrap_or(rva);
    let Ok(rva) = u32::from_str_radix(rva, 16) else {
        eprintln!("invalid hexadecimal RVA");
        return ExitCode::FAILURE;
    };
    let Ok(bytes) = fs::read(path) else {
        eprintln!("failed to read input");
        return ExitCode::FAILURE;
    };
    let Ok(pe) = PeImage::parse(&bytes) else {
        eprintln!("input is not a supported PE");
        return ExitCode::FAILURE;
    };
    let target = Rva(rva);
    let mut matches = 0_usize;
    for function in pe
        .report()
        .exception_functions
        .into_iter()
        .filter(|function| function.begin <= target && target < function.end)
    {
        matches += 1;
        println!("runtime_function={function:#?}");
        match pe.parse_unwind_info(function.unwind_info) {
            Ok(unwind) => println!("unwind={unwind:#?}"),
            Err(error) => println!("unwind_error={error}"),
        }
    }
    eprintln!("matching_runtime_functions={matches}");
    ExitCode::SUCCESS
}
