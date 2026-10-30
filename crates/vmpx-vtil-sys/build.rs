//! Links the native VTIL bridge when the `vtil` feature is enabled.
//!
//! `VMPX_VTIL_LIB_DIRS` is a platform path list (`;` on Windows, `:` elsewhere)
//! of directories holding the static libraries produced by building
//! `native/vtil-bridge` with `CMake` (use a Release build on MSVC).

use std::env;

const LIBS: &[&str] = &[
    "vmpx-vtil-bridge",
    "VTIL-Compiler",
    "VTIL-SymEx",
    "VTIL-Architecture",
    "VTIL-Common",
    "keystone",
    "capstone",
];

fn main() {
    println!("cargo:rerun-if-env-changed=VMPX_VTIL_LIB_DIRS");
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_VTIL").is_none() {
        return;
    }
    let dirs = env::var_os("VMPX_VTIL_LIB_DIRS").unwrap_or_else(|| {
        panic!(
            "feature `vtil` requires VMPX_VTIL_LIB_DIRS: build native/vtil-bridge with CMake \
             and list the directories containing its static libraries"
        )
    });
    for dir in env::split_paths(&dirs) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in LIBS {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        // keystone pulls in SHGetKnownFolderPath and COM GUIDs.
        println!("cargo:rustc-link-lib=dylib=shell32");
        println!("cargo:rustc-link-lib=dylib=ole32");
        // VTIL's static initialisers overflow the default 1 MiB main stack
        // (its CMake adds the same flag). Link args do not propagate to
        // dependents, so any binary linking this crate must pass it too.
        println!("cargo:rustc-link-arg-tests=/STACK:67108864");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
