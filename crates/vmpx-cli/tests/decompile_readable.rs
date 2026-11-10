use std::{path::Path, process::Command};

/// The `--readable`, `--dialect` and `--fold` flags must parse and be documented.
/// This runs everywhere; it needs no sample image.
#[test]
fn decompile_help_lists_readable_flags() {
    let out = Command::new(env!("CARGO_BIN_EXE_vmpx"))
        .args(["decompile", "--help"])
        .output()
        .expect("run vmpx decompile --help");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("--readable"), "missing --readable:\n{text}");
    assert!(text.contains("--dialect"), "missing --dialect:\n{text}");
    assert!(text.contains("--fold"), "missing --fold:\n{text}");
}

/// End-to-end golden for `--dialect rust --fold proven` on the integrity guard at
/// 0xc4a0add. `proven` must net the whole rdtsc-timing + .vmp2-checksum chain down
/// to its proven constant, leaving only the frame store and the INT3 probe; the
/// context argument is recognised, modeled as a packed `Ctx` struct, and its
/// `ctx + off` accesses render as struct fields.
///
/// Gated on the sample image: CI without `xpl_mapped.bin` skips rather than fails.
/// Point `VMPX_GOLDEN_IMAGE` at the mapped post-init image to run it.
#[test]
fn proven_fold_collapses_c4a0add_guard() {
    let image =
        std::env::var("VMPX_GOLDEN_IMAGE").unwrap_or_else(|_| r"E:\VMPX\xpl_mapped.bin".to_owned());
    if !Path::new(&image).exists() {
        eprintln!("golden image {image} absent; skipping proven-fold golden");
        return;
    }

    let out = Command::new(env!("CARGO_BIN_EXE_vmpx"))
        .args([
            "decompile",
            &image,
            "--rva",
            "0xc4a0add",
            "--dialect",
            "rust",
            "--fold",
            "proven",
        ])
        .env("VMPX_DECOMPILE_SERVE_RO", "1")
        .env("VMPX_TRUST_MAPPED_IMAGE", "1")
        .output()
        .expect("run vmpx decompile");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);

    // The exact collapsed body: timing + checksum net to a proven constant, over
    // the recognised context pointer `ctx`, modeled as a packed `Ctx` struct so
    // `ctx + off` accesses read as fields. The repeated field load is factored into
    // `v98` (per-statement CSE) and the scaled indicator reads as a conditional,
    // not `(cond as u64) * 2`.
    assert!(
        text.contains("#[repr(C, packed)]\nstruct Ctx {"),
        "ctx struct not synthesized:\n{text}"
    );
    assert!(
        text.contains("const _: () = assert!(core::mem::offset_of!(Ctx, f98) == 0x98);"),
        "field offset not pinned by a compile-time assertion:\n{text}"
    );
    assert!(
        text.contains("(*ctx).fa0 = a0;"),
        "store not rendered as a struct field:\n{text}"
    );
    assert!(
        text.contains("let v98 = (*ctx).f98;"),
        "repeated field load not factored into a struct-field local:\n{text}"
    );
    assert!(
        text.contains("return (if read_u8(v98) == INT3 { 2 } else { 0 }).wrapping_add(v98);"),
        "proven body changed:\n{text}"
    );
    // The context argument is a struct pointer, not a bare integer address.
    assert!(
        text.contains("ctx: *mut Ctx) -> u64"),
        "ctx not modeled as a struct pointer in the signature:\n{text}"
    );
    // None of the raw timing/checksum machinery may survive a proven fold.
    for leaked in ["rdtsc", "table_checksum", "memory_snapshot_epoch", "udiv32"] {
        assert!(
            !text.contains(leaked),
            "{leaked} leaked into proven fold:\n{text}"
        );
    }
}
