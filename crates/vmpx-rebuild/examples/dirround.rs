// Round-trips the directory rebuilds on an arbitrary PE.
//
// The rebuilds were designed against one VMProtect sample, which is exactly the
// situation where a shape gets baked in by accident. This runs each of them
// over any image and checks the result still parses and still validates for the
// loader, so "works on the sample" and "works on a PE" stay separate claims.
//
// Usage: dirround IMAGE [SECTION_TO_REMOVE ...]

use vmpx_pe::PeImage;
use vmpx_rebuild::directory_relocate::{
    rebuild_base_relocations, rebuild_exception_directory, rebuild_export_directory,
    rebuild_import_directory, rebuild_tls_directory, scan_iat_references,
};
use vmpx_rebuild::validate_for_loader;

fn check(label: &str, bytes: &[u8], note: String) {
    match PeImage::parse(bytes) {
        Ok(_) => match validate_for_loader(bytes) {
            Ok(_) => println!("  {label:<10} ok     {note}"),
            Err(error) => println!("  {label:<10} LOADER {error}"),
        },
        Err(error) => println!("  {label:<10} PARSE  {error}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: dirround IMAGE [SECTION ...]");
        std::process::exit(2);
    };
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("{path}: {error}");
            std::process::exit(2);
        }
    };
    let removed_owned: Vec<String> = args.iter().skip(2).cloned().collect();
    let removed: Vec<&str> = removed_owned.iter().map(String::as_str).collect();

    let pe = match PeImage::parse(&bytes) {
        Ok(pe) => pe,
        Err(error) => {
            println!("{path}: not a PE ({error})");
            return;
        }
    };
    let present = |index: usize| {
        pe.directory(index)
            .is_some_and(|directory| directory.address != 0 && directory.size != 0)
    };
    println!(
        "{path}: {} sections, export {} import {} exception {} tls {} reloc {}",
        pe.sections.len(),
        present(0),
        present(1),
        present(3),
        present(9),
        present(5)
    );

    if present(0) {
        match rebuild_export_directory(&bytes, *b".edata2\0") {
            Ok(moved) => check("export", &moved.bytes, format!("-> {}", moved.section.rva)),
            Err(error) => println!("  {:<10} refused {error}", "export"),
        }
    }
    if present(3) {
        match rebuild_exception_directory(&bytes, *b".pdata2\0", &removed) {
            Ok((moved, summary)) => check(
                "exception",
                &moved.bytes,
                format!("{} kept, {} dropped", summary.kept, summary.dropped),
            ),
            Err(error) => println!("  {:<10} refused {error}", "exception"),
        }
    }
    if present(1) {
        match scan_iat_references(&bytes, &removed) {
            Ok(scan) => println!(
                "  {:<10} {} surviving reference(s), {} function(s) decoded, {} byte(s) unverified",
                "iat-scan",
                scan.references.len(),
                scan.decoded_functions,
                scan.unverified_bytes
            ),
            Err(error) => println!("  {:<10} refused {error}", "iat-scan"),
        }
        match rebuild_import_directory(&bytes, *b".idata2\0") {
            Ok((moved, summary)) => check(
                "import",
                &moved.bytes,
                format!("{} modules, {} symbols", summary.modules, summary.symbols),
            ),
            Err(error) => println!("  {:<10} refused {error}", "import"),
        }
    }
    if present(9) {
        match rebuild_tls_directory(&bytes, *b".tls2\0\0\0", &removed) {
            Ok((moved, summary)) => {
                let fixups = summary.needs_relocation.clone();
                check(
                    "tls",
                    &moved.bytes,
                    format!(
                        "{} callback(s) kept, {} dropped",
                        summary.kept_callbacks,
                        summary.dropped_callbacks.len()
                    ),
                );
                match rebuild_base_relocations(&moved.bytes, *b".reloc2\0", &removed, &fixups) {
                    Ok((relocated, counts)) => check(
                        "reloc",
                        &relocated.bytes,
                        format!(
                            "{} kept, {} dropped, {} added",
                            counts.kept, counts.dropped, counts.added
                        ),
                    ),
                    Err(error) => println!("  {:<10} refused {error}", "reloc"),
                }
            }
            Err(error) => println!("  {:<10} refused {error}", "tls"),
        }
    } else if present(5) {
        match rebuild_base_relocations(&bytes, *b".reloc2\0", &removed, &[]) {
            Ok((moved, counts)) => check(
                "reloc",
                &moved.bytes,
                format!("{} kept, {} dropped", counts.kept, counts.dropped),
            ),
            Err(error) => println!("  {:<10} refused {error}", "reloc"),
        }
    }
}
