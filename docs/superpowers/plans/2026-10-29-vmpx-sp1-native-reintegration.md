# SP1 — Native Reintegration + Image Emission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the replacement machine code the `clean` pipeline already compiles (and currently discards) into a loadable PE with the entry/TLS callbacks redirected to it, written in sound mode only when proven and otherwise behind an explicit `--experimental` path.

**Architecture:** Two of SP1's four designed components already exist and are tested in `vmpx-rebuild` — `append_section` (file-backed section append with full header/alignment/reparse validation) and `set_entry_point` (entry redirect). SP1 adds: a naked domain-dispatcher encoder in `vmpx-recompile` (hand-encoded with iced `code_asm`, because `compile()` emits a Win64 frame and a dispatcher must preserve all incoming registers), a TLS-callback redirect in `vmpx-rebuild` mirroring `set_entry_point`, a thin `reintegrate` orchestration seam in `vmpx-rebuild` that chains append → redirect → loader-validate, and the `clean.rs`/`main.rs` wiring that assembles the section payload (bodies then dispatchers) and gates sound vs experimental output.

**Tech Stack:** Rust (edition 2024), `iced-x86` (`code_asm` feature, already a `vmpx-recompile` dependency), `vmpx-pe`, `vmpx-rebuild`, `vmpx-recompile`, `vmpx-vmlift`, `clap`.

**Spec:** `docs/superpowers/specs/2026-10-29-vmpx-sp1-native-reintegration-design.md`

## Global Constraints

- Rust edition 2024, `rust-version = "1.88"`.
- `unsafe_code = "forbid"` workspace-wide — no `unsafe` in any task.
- No new external crate dependencies. The dispatcher uses `iced-x86` which is already a `vmpx-recompile` dependency; everything else uses existing workspace crates.
- Lean comments: a short line only where the "why" is non-obvious. Do not mimic the repo's verbose comment style. Do not edit existing comments.
- New executable section name: `.vmpx0` (8-byte field `*b".vmpx0\0\0"`), characteristics `IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_EXECUTE` = `0x6000_0020`.
- Behavior-changing modes are opt-in and labeled: sound `output` is written only when `blockers` is empty; `--experimental <path>` writes regardless, with a stderr warning and `experimental: true` plus the blockers list in the report, and never changes sound-`output` semantics.
- Static-only: nothing executes the target.

---

## File structure

- `crates/vmpx-recompile/src/dispatch.rs` — **new.** Naked domain-dispatcher encoder. One responsibility: given a domain register, `(value, body_va)` arms, a fallback VA, and the dispatcher's own VA, emit position-correct `cmp`/`je`/`jmp` bytes.
- `crates/vmpx-recompile/src/lib.rs` — **modify.** Export the dispatcher.
- `crates/vmpx-rebuild/src/reintegrate.rs` — **new.** TLS-callback redirect + the `reintegrate` orchestration seam (append → entry/TLS redirect → loader-validate). One responsibility: compose a prebuilt code payload into an image and point entries at it. Takes bytes, not codegen types, so `vmpx-rebuild` gains no dependency on `vmpx-recompile`.
- `crates/vmpx-rebuild/src/lib.rs` — **modify.** Declare the module and export its items.
- `crates/vmpx-cli/src/clean.rs` — **modify.** Stop discarding `emitted_section`; run the recompilation stage against the conventional image; assemble the payload (bodies then dispatchers); call `reintegrate`; populate the `reintegration` stage report; return the reintegrated image for the experimental path.
- `crates/vmpx-cli/src/main.rs` — **modify.** Add `--experimental <path>` to `Command::Clean`, with alias guards, the warning, and the write.

---

### Task 1: Naked domain-dispatcher encoder

Emits a register-preserving trampoline: for each recovered input value, compare the domain register and branch to that value's body; otherwise fall through to the preserved original entry. Hand-encoded because `compile()` would wrap it in a Win64 frame that clobbers the incoming register state the bodies expect.

**Files:**
- Create: `crates/vmpx-recompile/src/dispatch.rs`
- Modify: `crates/vmpx-recompile/src/lib.rs`
- Test: in `crates/vmpx-recompile/src/dispatch.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `PhysicalRegister` (from this crate's `ir`), `iced-x86` `code_asm`.
- Produces:
  ```rust
  pub struct DispatchArm { pub value: u64, pub body: u64 }   // body = absolute VA
  pub fn emit_domain_dispatcher(
      domain: PhysicalRegister,
      arms: &[DispatchArm],
      fallback: u64,          // absolute VA of the preserved original entry
      dispatcher_va: u64,     // absolute VA where these bytes will be placed
  ) -> Result<Vec<u8>, DispatchError>;

  #[derive(Debug, thiserror::Error)]
  pub enum DispatchError {
      #[error("domain value {0} does not fit in a 32-bit compare immediate")]
      ValueTooWide(u64),
      #[error("iced assembly failed: {0}")]
      Assemble(String),
  }
  ```

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhysicalRegister;
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};

    fn mnemonics(bytes: &[u8], ip: u64) -> Vec<Mnemonic> {
        let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
        decoder.iter().map(|insn| insn.mnemonic()).collect()
    }

    #[test]
    fn single_arm_compares_and_branches_then_falls_through() {
        let bytes = emit_domain_dispatcher(
            PhysicalRegister::Rdx,
            &[DispatchArm { value: 1, body: 0x1_4000_3000 }],
            0x1_4000_2000,
            0x1_4000_4000,
        )
        .expect("encode dispatcher");
        let seq = mnemonics(&bytes, 0x1_4000_4000);
        assert_eq!(seq.first().copied(), Some(Mnemonic::Cmp));
        assert!(seq.contains(&Mnemonic::Je));
        assert_eq!(seq.last().copied(), Some(Mnemonic::Jmp)); // fallback
    }

    #[test]
    fn rejects_oversized_domain_value() {
        let err = emit_domain_dispatcher(
            PhysicalRegister::Rdx,
            &[DispatchArm { value: u64::from(u32::MAX) + 1, body: 0x1000 }],
            0x2000,
            0x4000,
        )
        .expect_err("value must fit in imm32");
        assert!(matches!(err, DispatchError::ValueTooWide(_)));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vmpx-recompile dispatch`
Expected: FAIL — `emit_domain_dispatcher` / `dispatch` module not found.

- [ ] **Step 3: Write minimal implementation**

```rust
use crate::PhysicalRegister;
use iced_x86::code_asm::{
    CodeAssembler,
    registers::gpr64::{
        r8, r9, r10, r11, r12, r13, r14, r15, rax, rbp, rbx, rcx, rdi, rdx, rsi,
    },
};
use iced_x86::code_asm::AsmRegister64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DispatchArm { pub value: u64, pub body: u64 }

#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    #[error("domain value {0} does not fit in a 32-bit compare immediate")]
    ValueTooWide(u64),
    #[error("iced assembly failed: {0}")]
    Assemble(String),
}

fn reg(register: PhysicalRegister) -> AsmRegister64 {
    match register {
        PhysicalRegister::Rax => rax, PhysicalRegister::Rcx => rcx,
        PhysicalRegister::Rdx => rdx, PhysicalRegister::Rbx => rbx,
        PhysicalRegister::Rbp => rbp, PhysicalRegister::Rsi => rsi,
        PhysicalRegister::Rdi => rdi, PhysicalRegister::R8 => r8,
        PhysicalRegister::R9 => r9, PhysicalRegister::R10 => r10,
        PhysicalRegister::R11 => r11, PhysicalRegister::R12 => r12,
        PhysicalRegister::R13 => r13, PhysicalRegister::R14 => r14,
        PhysicalRegister::R15 => r15,
    }
}

/// Register-preserving trampoline placed at `dispatcher_va`. `cmp` touches only
/// flags, which a VM entry does not consume as input.
pub fn emit_domain_dispatcher(
    domain: PhysicalRegister,
    arms: &[DispatchArm],
    fallback: u64,
    dispatcher_va: u64,
) -> Result<Vec<u8>, DispatchError> {
    let mut asm = CodeAssembler::new(64).map_err(|e| DispatchError::Assemble(e.to_string()))?;
    let domain = reg(domain);
    for arm in arms {
        let imm = i32::try_from(arm.value).map_err(|_| DispatchError::ValueTooWide(arm.value))?;
        asm.cmp(domain, imm).map_err(|e| DispatchError::Assemble(e.to_string()))?;
        asm.je(arm.body).map_err(|e| DispatchError::Assemble(e.to_string()))?;
    }
    asm.jmp(fallback).map_err(|e| DispatchError::Assemble(e.to_string()))?;
    asm.assemble(dispatcher_va).map_err(|e| DispatchError::Assemble(e.to_string()))
}
```

Then in `crates/vmpx-recompile/src/lib.rs` add `mod dispatch;` and `pub use dispatch::{DispatchArm, DispatchError, emit_domain_dispatcher};`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vmpx-recompile dispatch`
Expected: PASS. If an iced `code_asm` branch method signature differs (e.g. `je`/`jmp` target type), adjust to the signature iced exposes for absolute-target branches; the decode assertions confirm correctness.

- [ ] **Step 5: Commit**

```bash
git add crates/vmpx-recompile/src/dispatch.rs crates/vmpx-recompile/src/lib.rs
git commit -m "feat(recompile): add naked domain-dispatcher encoder"
```

---

### Task 2: TLS-callback redirect in vmpx-rebuild

Mirrors `set_entry_point`, but rewrites one entry of the TLS callback array instead of `AddressOfEntryPoint`. Needed because a loader-reachable TLS callback is a virtualized routine whose replacement must be pointed to.

**Files:**
- Create: `crates/vmpx-rebuild/src/reintegrate.rs`
- Modify: `crates/vmpx-rebuild/src/lib.rs`
- Test: in `crates/vmpx-rebuild/src/reintegrate.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `vmpx_pe::{PeImage, Rva}`, and `vmpx-rebuild`'s private `write_bytes`/`add_offset` helpers (make them `pub(crate)` in this task).
- Produces:
  ```rust
  pub fn set_tls_callback(input: &[u8], index: usize, target: Rva) -> Result<Vec<u8>, RebuildError>;
  ```
  Adds `RebuildError` variants: `NoTlsDirectory`, `TlsCallbackIndex { index: usize, count: usize }`, `TlsCallbackNotExecutable { rva: Rva }`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_pe::PeImage;

    // Builds a PE32+ with one R+X section and a TLS directory whose callback
    // array holds one entry, returning (bytes, callback_rva, image_base).
    fn pe_with_one_tls_callback() -> (Vec<u8>, Rva, u64) {
        // Reuse the shape of vmpx-rebuild's existing `minimal_pe` test helper,
        // extended with a populated IMAGE_DIRECTORY_ENTRY_TLS. Construct inline
        // here (the lib.rs helper is test-private to its module).
        unimplemented!("construct a minimal PE32+ with a single TLS callback")
    }

    #[test]
    fn redirects_a_tls_callback_and_survives_reparse() {
        let (input, _callback_rva, image_base) = pe_with_one_tls_callback();
        let target = Rva(0x2000); // inside the R+X section
        let output = set_tls_callback(&input, 0, target).expect("redirect tls callback");
        let pe = PeImage::parse(&output).expect("reparse");
        let tls = pe.report().tls.expect("tls present");
        assert_eq!(tls.callbacks[0], image_base + u64::from(target.0));
    }

    #[test]
    fn rejects_out_of_range_index() {
        let (input, _, _) = pe_with_one_tls_callback();
        assert!(matches!(
            set_tls_callback(&input, 9, Rva(0x2000)),
            Err(RebuildError::TlsCallbackIndex { index: 9, count: 1 })
        ));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vmpx-rebuild reintegrate::tests::redirects_a_tls_callback`
Expected: FAIL — `set_tls_callback` undefined / test helper `unimplemented!`.

- [ ] **Step 3: Write minimal implementation**

First implement the test helper `pe_with_one_tls_callback` by adapting the `minimal_pe` construction already in `crates/vmpx-rebuild/src/lib.rs` tests: set data-directory count to cover index 9 (TLS), point `IMAGE_DIRECTORY_ENTRY_TLS` at an `IMAGE_TLS_DIRECTORY64` in the section's file data, set its `AddressOfCallBacks` to a VA (`image_base + callback_rva`), and write one 8-byte callback VA (then a null terminator) at `callback_rva`.

Then the function:

```rust
use vmpx_pe::{IMAGE_DIRECTORY_ENTRY_TLS, PeImage, Rva, Va};
use crate::{RebuildError, add_offset, write_bytes};

const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;

/// Rewrites one TLS callback array entry to an executable, file-backed RVA and
/// reparses before returning.
pub fn set_tls_callback(input: &[u8], index: usize, target: Rva) -> Result<Vec<u8>, RebuildError> {
    let pe = PeImage::parse(input).map_err(|source| RebuildError::Pe { stage: "input", source })?;
    let executable = pe.sections.iter().any(|s| {
        s.characteristics & IMAGE_SCN_MEM_EXECUTE != 0 && s.contains_mapped_rva(target)
    });
    if !executable {
        return Err(RebuildError::TlsCallbackNotExecutable { rva: target });
    }
    let tls = pe.report().tls.ok_or(RebuildError::NoTlsDirectory)?;
    let count = tls.callbacks.len();
    if index >= count {
        return Err(RebuildError::TlsCallbackIndex { index, count });
    }
    let directory = pe.directory(IMAGE_DIRECTORY_ENTRY_TLS).ok_or(RebuildError::NoTlsDirectory)?;
    // AddressOfCallBacks is a VA at offset 24 of IMAGE_TLS_DIRECTORY64.
    let tls_file = pe.rva_to_file_offset(Rva(directory.address))
        .map_err(|e| RebuildError::OutputInvariant(e.to_string()))?;
    let array_va_offset = add_offset(tls_file, 24, "AddressOfCallBacks offset")?;
    let array_va = read_u64(input, array_va_offset)?;
    let array_rva = pe.checked_rva_of(Va(array_va))
        .ok_or_else(|| RebuildError::OutputInvariant("TLS callback array VA below image base".to_owned()))?;
    let slot_rva = Rva(array_rva.0 + u32::try_from(index * 8).map_err(|_| RebuildError::Overflow { what: "tls slot" })?);
    let slot_file = pe.rva_to_file_offset(slot_rva)
        .map_err(|e| RebuildError::OutputInvariant(e.to_string()))?;
    let new_va = pe.optional.image_base.0 + u64::from(target.0);

    let mut output = input.to_vec();
    write_bytes(&mut output, slot_file, &new_va.to_le_bytes(), "tls callback slot")?;
    let reparsed = PeImage::parse(&output).map_err(|source| RebuildError::Pe { stage: "output", source })?;
    let got = reparsed.report().tls.ok_or(RebuildError::NoTlsDirectory)?;
    if got.callbacks.get(index).copied() != Some(new_va) {
        return Err(RebuildError::OutputInvariant("TLS callback did not survive reparse".to_owned()));
    }
    Ok(output)
}

fn read_u64(data: &[u8], offset: vmpx_pe::FileOffset) -> Result<u64, RebuildError> {
    let start = usize::try_from(offset.0).map_err(|_| RebuildError::Overflow { what: "tls read" })?;
    let bytes = data.get(start..start + 8).ok_or(RebuildError::Overflow { what: "tls read" })?;
    Ok(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
}
```

Add the three new variants to `RebuildError` in `lib.rs`, make `write_bytes` and `add_offset` `pub(crate)`, add `mod reintegrate;` and `pub use reintegrate::set_tls_callback;`. Confirm the exact field names on the parsed TLS report (`tls.callbacks`, element type `u64` VA) against `vmpx_pe`; adjust the assertion if the model stores RVAs instead of VAs.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vmpx-rebuild reintegrate`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/vmpx-rebuild/src/reintegrate.rs crates/vmpx-rebuild/src/lib.rs
git commit -m "feat(rebuild): add TLS callback redirect"
```

---

### Task 3: Reintegration orchestration seam

A thin, byte-level seam that chains the existing `append_section` with the redirects and a loader-validation read. It takes a prebuilt payload and absolute redirect RVAs so `vmpx-rebuild` stays free of any codegen dependency.

**Files:**
- Modify: `crates/vmpx-rebuild/src/reintegrate.rs`
- Modify: `crates/vmpx-rebuild/src/lib.rs` (exports)
- Test: in `crates/vmpx-rebuild/src/reintegrate.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `append_section`, `AppendSection`, `AppendedSection`, `set_entry_point`, `set_tls_callback`, `validate_for_loader`, `next_section_rva`.
- Produces:
  ```rust
  pub struct ReintegrationRequest<'a> {
      pub section_name: [u8; 8],
      pub payload: &'a [u8],
      pub entry_redirect: Option<Rva>,       // absolute RVA within the new section
      pub tls_redirects: &'a [(usize, Rva)], // (callback index, absolute RVA)
  }
  pub struct ReintegrationReport {
      pub section: AppendedSection,
      pub entry_redirected: bool,
      pub tls_redirected: usize,
      pub loader_valid: bool,
      pub diagnostics: Vec<String>,
  }
  pub fn reintegrate(conventional: &[u8], request: ReintegrationRequest<'_>)
      -> Result<(Vec<u8>, ReintegrationReport), RebuildError>;
  ```

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn reintegrates_payload_and_redirects_entry() {
    // minimal conventional PE32+ with one R+X section and header slack for one
    // more section header (reuse Task 2's builder without the TLS directory).
    let input = minimal_conventional_pe();
    let planned = crate::next_section_rva(&input).expect("planned rva");
    let payload = vec![0x90u8, 0x90, 0xC3]; // nop; nop; ret — stand-in body
    let (bytes, report) = reintegrate(&input, ReintegrationRequest {
        section_name: *b".vmpx0\0\0",
        payload: &payload,
        entry_redirect: Some(planned),
        tls_redirects: &[],
    }).expect("reintegrate");
    let pe = vmpx_pe::PeImage::parse(&bytes).expect("reparse");
    assert_eq!(pe.optional.address_of_entry_point, planned);
    assert_eq!(report.section.rva, planned);
    assert!(report.entry_redirected);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vmpx-rebuild reintegrate::tests::reintegrates_payload`
Expected: FAIL — `reintegrate` undefined.

- [ ] **Step 3: Write minimal implementation**

```rust
use crate::{AppendSection, AppendedSection, append_section, set_entry_point, validate_for_loader};

const REINTEGRATION_CHARACTERISTICS: u32 = 0x6000_0020; // CODE | READ | EXECUTE

pub fn reintegrate(
    conventional: &[u8],
    request: ReintegrationRequest<'_>,
) -> Result<(Vec<u8>, ReintegrationReport), RebuildError> {
    let rebuilt = append_section(conventional, AppendSection {
        name: request.section_name,
        data: request.payload,
        characteristics: REINTEGRATION_CHARACTERISTICS,
    })?;
    let mut bytes = rebuilt.bytes;
    let mut entry_redirected = false;
    if let Some(entry) = request.entry_redirect {
        bytes = set_entry_point(&bytes, entry)?;
        entry_redirected = true;
    }
    let mut tls_redirected = 0;
    for (index, target) in request.tls_redirects {
        bytes = set_tls_callback(&bytes, *index, *target)?;
        tls_redirected += 1;
    }
    let mut diagnostics = Vec::new();
    let loader_valid = match validate_for_loader(&bytes) {
        Ok(_) => true,
        Err(error) => { diagnostics.push(error.to_string()); false }
    };
    Ok((bytes, ReintegrationReport {
        section: rebuilt.section,
        entry_redirected,
        tls_redirected,
        loader_valid,
        diagnostics,
    }))
}
```

Export `ReintegrationRequest`, `ReintegrationReport`, `reintegrate` from `lib.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vmpx-rebuild reintegrate`
Expected: PASS (all reintegrate tests).

- [ ] **Step 5: Commit**

```bash
git add crates/vmpx-rebuild/src/reintegrate.rs crates/vmpx-rebuild/src/lib.rs
git commit -m "feat(rebuild): add reintegration orchestration seam"
```

---

### Task 4: Wire reintegration into the clean pipeline

Stop discarding `emitted_section`; run emission against the conventional image so body RVAs match where `append_section` will place them; build one dispatcher per multi-path entry; assemble the payload (all bodies, then all dispatchers); call `reintegrate`; record the result and return the reintegrated image.

**Files:**
- Modify: `crates/vmpx-cli/src/clean.rs`
- Test: `crates/vmpx-cli/tests/clean_command.rs` (extend; gated the same way existing fixture tests are) and a focused unit added to `clean.rs`.

**Interfaces:**
- Consumes: `vmpx_recompile::{DispatchArm, emit_domain_dispatcher}`, `vmpx_rebuild::{ReintegrationRequest, next_section_rva, reintegrate}`, the existing `build_replacement` emission, `physical_for`.
- Produces: `analyze_clean_with_image` returns `Result<(CleanReport, Option<Vec<u8>>, Option<Vec<u8>>)>` — adding a third element: the reintegrated experimental image when the machinery completed (conventional image stays the second element). The `reintegration` `CleanStageReport` gains metrics `section_rva`, `section_size`, `entries_redirected`, `dispatchers`, and `complete == true` when `reintegrate` succeeded.

- [ ] **Step 1: Write the failing test**

Add a unit test in `clean.rs` that exercises only the payload-layout helper (the part that is testable without a real VM sample). Introduce a pure helper and test it:

```rust
// Returns (payload, entry_dispatcher_rva, per-entry dispatcher RVAs). Bodies are
// placed first at `base`, dispatchers after, so body VAs stay as compiled.
fn layout_reintegration_payload(
    base: Rva, image_base: u64,
    bodies: &[(u64 /*domain value*/, Vec<u8>)],
    domain_reg: vmpx_recompile::PhysicalRegister,
    original_entry_rva: Rva,
) -> Result<(Vec<u8>, Rva), String>;

#[cfg(test)]
mod reintegration_tests {
    use super::*;
    use vmpx_pe::Rva;
    use vmpx_recompile::PhysicalRegister;

    #[test]
    fn dispatcher_follows_bodies_and_targets_each_body() {
        let base = Rva(0x2000);
        let image_base = 0x1_4000_0000;
        let bodies = vec![(0u64, vec![0x90, 0xC3]), (1u64, vec![0x90, 0x90, 0xC3])];
        let (payload, dispatcher_rva) = layout_reintegration_payload(
            base, image_base, &bodies, PhysicalRegister::Rdx, Rva(0x1000),
        ).expect("layout");
        // bodies total 5 bytes, so the dispatcher starts at base + 5.
        assert_eq!(dispatcher_rva, Rva(0x2005));
        assert_eq!(&payload[..5], &[0x90, 0xC3, 0x90, 0x90, 0xC3]);
        assert!(payload.len() > 5); // dispatcher bytes appended
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vmpx-cli reintegration_tests`
Expected: FAIL — `layout_reintegration_payload` undefined.

- [ ] **Step 3: Write minimal implementation**

Add the helper, then use it in the recompilation stage. Helper:

```rust
fn layout_reintegration_payload(
    base: Rva,
    image_base: u64,
    bodies: &[(u64, Vec<u8>)],
    domain_reg: vmpx_recompile::PhysicalRegister,
    original_entry_rva: Rva,
) -> Result<(Vec<u8>, Rva), String> {
    let mut payload = Vec::new();
    let mut arms = Vec::new();
    let mut offset = 0u32;
    for (value, bytes) in bodies {
        let body_rva = base.0.checked_add(offset).ok_or("body rva overflow")?;
        arms.push(vmpx_recompile::DispatchArm { value: *value, body: image_base + u64::from(body_rva) });
        payload.extend_from_slice(bytes);
        offset = offset.checked_add(u32::try_from(bytes.len()).map_err(|_| "body too large")?)
            .ok_or("offset overflow")?;
    }
    let dispatcher_rva = Rva(base.0.checked_add(offset).ok_or("dispatcher rva overflow")?);
    let dispatcher = vmpx_recompile::emit_domain_dispatcher(
        domain_reg, &arms,
        image_base + u64::from(original_entry_rva.0),
        image_base + u64::from(dispatcher_rva.0),
    ).map_err(|e| e.to_string())?;
    payload.extend_from_slice(&dispatcher);
    Ok((payload, dispatcher_rva))
}
```

Then in the recompilation stage of `analyze_clean_with_image`:
- compute the conventional image first (move the existing `relayout_conventional` call above the recompilation stage, or compute `planned_rva = next_section_rva(&conventional_bytes)` and compile bodies at `image_base + planned_rva`);
- group emitted bodies by entry and by their domain value (from `assignments`);
- for a single-value entry, set `entry_redirect`/tls target to the sole body RVA (no dispatcher); for a multi-value entry, call `layout_reintegration_payload` and redirect to `dispatcher_rva`;
- concatenate all entries' payloads, call `reintegrate` with the header-entry `entry_redirect` and any `tls_redirects`;
- set `reintegration.complete = true` and its metrics on success; push the reintegrated bytes out as the new third return element;
- leave `blockers`, `ready_to_write`, and sound `output` behavior unchanged.

Update the one caller of `analyze_clean_with_image` (`Command::Clean` in `main.rs`) and the `analyze_clean` wrapper to the new 3-tuple.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vmpx-cli` and `cargo build -p vmpx-cli`
Expected: PASS; crate builds. The full-pipeline path is verified out-of-band in Task 5 (no CI VM fixture).

- [ ] **Step 5: Commit**

```bash
git add crates/vmpx-cli/src/clean.rs crates/vmpx-cli/src/main.rs
git commit -m "feat(cli): assemble payload and reintegrate emitted replacements"
```

---

### Task 5: `--experimental` output path

Expose the reintegrated image: sound `output` is unchanged, a new `--experimental <path>` writes the reintegrated bytes regardless of blockers, with a warning and report flag.

**Files:**
- Modify: `crates/vmpx-cli/src/main.rs`
- Test: `crates/vmpx-cli/tests/clean_command.rs`

**Interfaces:**
- Consumes: the 3-tuple from `analyze_clean_with_image` (Task 4).
- Produces: `Command::Clean` gains `experimental: Option<PathBuf>`; `CleanReport` gains `pub experimental: bool`.

- [ ] **Step 1: Write the failing test**

```rust
// In clean_command.rs — exercises arg plumbing and the no-op/clean-input path,
// which needs no VM fixture.
#[test]
fn experimental_flag_is_rejected_when_it_aliases_output() {
    let dir = unique_dir("exp-alias");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("in.bin");
    std::fs::write(&input, minimal_conventional_pe_bytes()).unwrap();
    let out = dir.join("out.bin");
    let output = Command::new(env!("CARGO_BIN_EXE_vmpx"))
        .args(["clean", input.to_str().unwrap(), out.to_str().unwrap(),
               "--experimental", out.to_str().unwrap()])
        .output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("aliases"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vmpx-cli experimental_flag_is_rejected`
Expected: FAIL — unknown `--experimental` argument.

- [ ] **Step 3: Write minimal implementation**

In `main.rs`, add `#[arg(long)] experimental: Option<PathBuf>` to `Command::Clean`; add alias guards mirroring the existing `reconstructed`/`report` checks (reject when `experimental` equals input/output/report/reconstructed or canonicalizes to the input). After `analyze_clean_with_image` returns `(clean, conventional, reintegrated)`:

```rust
if let Some(path) = &experimental {
    match &reintegrated {
        Some(bytes) => {
            fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
            eprintln!("[experimental] WARNING: unverified devirtualized output written to {}; {} blocker(s) remain", path.display(), clean.blockers.len());
            println!("[experimental] {}", path.display());
        }
        None => eprintln!("[experimental] no reintegrated image was produced; nothing written to {}", path.display()),
    }
}
```

Set `clean.experimental = experimental.is_some();` before serializing the report. Leave the sound `output` write path exactly as is.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vmpx-cli` ; `cargo build --release`
Expected: PASS; release build succeeds.

- [ ] **Step 5: Commit**

```bash
git add crates/vmpx-cli/src/main.rs crates/vmpx-cli/tests/clean_command.rs
git commit -m "feat(cli): add --experimental reintegrated output path"
```

---

## Out-of-band verification (manual, not CI)

After Task 5, on the author's Windows machine with a local VMProtect sample (never committed):

1. `cargo run -p vmpx-cli --release -- clean sample.dll out.dll --experimental sample.devirt.dll --report report.json`
2. Confirm `report.json` shows `reintegration.complete == true`, a `.vmpx0` section, and the entry/TLS redirect metrics.
3. Confirm `sample.devirt.dll` reparses (`clean`/`info` on it) and `validate_for_loader` passed.
4. Load/run it and compare observable behavior against the protected original.

---

## Self-review

**Spec coverage.** SP1 spec components map to tasks: section append + entry redirect → already in `vmpx-rebuild` (used by Task 3); domain dispatcher → Task 1; TLS redirect → Task 2; `reintegrate` orchestrator → Task 3; layout contract (bodies first, dispatchers after) → Task 4 `layout_reintegration_payload`; sound vs `--experimental` → Tasks 4–5; fixture-first testing → unit tests in Tasks 1–3, plumbing test in Task 4, arg test in Task 5, with the real-sample path explicitly out-of-band. Integrity/anti-tamper and YAGNI boundaries (no OEP/imports/stripping/optimization) are respected — none of those are touched.

**Placeholder scan.** The only intentional `unimplemented!` is a test-fixture builder the implementer completes in Task 2 Step 3 (instructions given inline). No TODO/TBD in shipped code steps.

**Type consistency.** `DispatchArm { value, body }`, `emit_domain_dispatcher(domain, arms, fallback, dispatcher_va)`, `reintegrate(conventional, ReintegrationRequest)`, and the 3-tuple return of `analyze_clean_with_image` are used consistently across Tasks 1→3→4→5. `PhysicalRegister` is the `vmpx-recompile` enum throughout.

**Risk the executor must confirm at Step 4 of Tasks 1–2:** exact iced `code_asm` branch-target signature, and whether the parsed TLS report stores callback VAs or RVAs. Both are caught by the decode/reparse assertions in those tasks.
