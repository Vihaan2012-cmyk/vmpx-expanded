use crate::{
    AppendSection, AppendedSection, RebuildError, add_offset, append_section, set_entry_point,
    validate_for_loader, write_bytes,
};
use vmpx_pe::{FileOffset, IMAGE_DIRECTORY_ENTRY_TLS, PeImage, PeKind, Rva, Va};

const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const REINTEGRATION_CHARACTERISTICS: u32 = 0x6000_0020; // CODE | READ | EXECUTE

/// Inputs for one reintegration pass: a file-backed payload section plus any
/// redirects to point at it. `vmpx-rebuild` stays free of codegen crates by
/// taking already-rendered bytes and absolute RVAs rather than building them.
pub struct ReintegrationRequest<'a> {
    pub section_name: [u8; 8],
    pub payload: &'a [u8],
    pub entry_redirect: Option<Rva>,
    pub tls_redirects: &'a [(usize, Rva)],
}

/// Outcome of a reintegration pass: what was appended, which redirects landed,
/// and whether the result still satisfies the OS loader.
pub struct ReintegrationReport {
    pub section: AppendedSection,
    pub entry_redirected: bool,
    pub tls_redirected: usize,
    pub loader_valid: bool,
    pub diagnostics: Vec<String>,
}

/// Appends `request.payload` as a new section, then applies the requested
/// entry-point and TLS-callback redirects, then records a loader-validation
/// pass/fail without failing the call on an invalid result.
///
/// # Errors
///
/// Propagates any error from `append_section`, `set_entry_point`, or
/// `set_tls_callback`.
pub fn reintegrate(
    conventional: &[u8],
    request: ReintegrationRequest<'_>,
) -> Result<(Vec<u8>, ReintegrationReport), RebuildError> {
    let rebuilt = append_section(
        conventional,
        AppendSection {
            name: request.section_name,
            data: request.payload,
            characteristics: REINTEGRATION_CHARACTERISTICS,
        },
    )?;
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
        Err(error) => {
            diagnostics.push(error.to_string());
            false
        }
    };
    Ok((
        bytes,
        ReintegrationReport {
            section: rebuilt.section,
            entry_redirected,
            tls_redirected,
            loader_valid,
            diagnostics,
        },
    ))
}

/// Rewrites one TLS callback array entry to an executable, file-backed RVA
/// and reparses the resulting image before returning it.
///
/// # Errors
///
/// Rejects malformed or non-PE32+ images, a missing TLS directory, an
/// out-of-range callback index, a non-executable or non-file-backed target,
/// and any value that does not survive output reparse.
pub fn set_tls_callback(input: &[u8], index: usize, target: Rva) -> Result<Vec<u8>, RebuildError> {
    let pe = PeImage::parse(input).map_err(|source| RebuildError::Pe {
        stage: "input",
        source,
    })?;
    // The TLS directory layout and 8-byte callback VAs below are PE32+ only.
    if pe.optional.kind != PeKind::Pe32Plus {
        return Err(RebuildError::UnsupportedKind(pe.optional.kind));
    }
    let executable = pe.sections.iter().any(|section| {
        section.characteristics & IMAGE_SCN_MEM_EXECUTE != 0 && section.contains_mapped_rva(target)
    });
    if !executable {
        return Err(RebuildError::TlsCallbackNotExecutable { rva: target });
    }
    pe.rva_to_file_offset(target)
        .map_err(|_| RebuildError::TlsCallbackNotFileBacked { rva: target })?;

    let tls = pe.report().tls.ok_or(RebuildError::NoTlsDirectory)?;
    let count = tls.callbacks.len();
    if index >= count {
        return Err(RebuildError::TlsCallbackIndex { index, count });
    }

    let directory = pe
        .directory(IMAGE_DIRECTORY_ENTRY_TLS)
        .ok_or(RebuildError::NoTlsDirectory)?;
    let tls_offset = pe
        .rva_to_file_offset(Rva(directory.address))
        .map_err(|source| RebuildError::Pe {
            stage: "TLS directory",
            source,
        })?;
    // AddressOfCallBacks is a VA at offset 24 of IMAGE_TLS_DIRECTORY64.
    let array_va_offset = add_offset(tls_offset, 24, "AddressOfCallBacks offset")?;
    let callback_array_va = Va(read_u64(input, array_va_offset)?);
    let array_rva = pe
        .optional
        .image_base
        .checked_rva_of(callback_array_va)
        .ok_or_else(|| {
            RebuildError::OutputInvariant("TLS callback array VA is below image base".to_owned())
        })?;
    let slot_delta = u32::try_from(index.checked_mul(8).ok_or(RebuildError::Overflow {
        what: "tls slot index",
    })?)
    .map_err(|_| RebuildError::Overflow {
        what: "tls slot index",
    })?;
    let slot_rva = array_rva.checked_add(slot_delta).ok_or(RebuildError::Overflow {
        what: "tls slot rva",
    })?;
    let slot_offset = pe.rva_to_file_offset(slot_rva).map_err(|source| RebuildError::Pe {
        stage: "TLS callback slot",
        source,
    })?;
    let new_va = pe
        .optional
        .image_base
        .checked_add_rva(target)
        .ok_or(RebuildError::Overflow {
            what: "tls callback VA",
        })?;

    let mut output = input.to_vec();
    write_bytes(&mut output, slot_offset, &new_va.0.to_le_bytes(), "tls callback slot")?;

    let reparsed = PeImage::parse(&output).map_err(|source| RebuildError::Pe {
        stage: "output",
        source,
    })?;
    let got = reparsed.report().tls.ok_or(RebuildError::NoTlsDirectory)?;
    if got.callbacks.get(index).copied() != Some(new_va) {
        return Err(RebuildError::OutputInvariant(
            "TLS callback did not survive reparse".to_owned(),
        ));
    }
    Ok(output)
}

fn read_u64(data: &[u8], offset: FileOffset) -> Result<u64, RebuildError> {
    let start = usize::try_from(offset.0).map_err(|_| RebuildError::Overflow { what: "tls read" })?;
    let end = start.checked_add(8).ok_or(RebuildError::Overflow { what: "tls read" })?;
    let bytes = data.get(start..end).ok_or(RebuildError::Overflow { what: "tls read" })?;
    Ok(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_pe::PeImage;

    const NT_OFFSET: usize = 0x80;
    const OPTIONAL_OFFSET: usize = NT_OFFSET + 4 + 20;
    const SECTION_TABLE_OFFSET: usize = OPTIONAL_OFFSET + 0xF0;
    const IMAGE_BASE: u64 = 0x1_4000_0000;
    const FIRST_RAW: usize = 0x200;
    const SECTION_RVA: u32 = 0x1000;
    const SECTION_SPAN: u32 = 0x2000; // covers target RVA 0x2000 used by the redirect test

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    /// A PE32+ with one R+X `.text` section (RVA 0x1000, span 0x2000) and
    /// header slack for one more 40-byte section header. No TLS directory.
    fn minimal_conventional_pe() -> Vec<u8> {
        let mut bytes = vec![0_u8; FIRST_RAW + usize::try_from(SECTION_SPAN).expect("small span")];
        put_u16(&mut bytes, 0, 0x5A4D);
        put_u32(&mut bytes, 0x3C, u32::try_from(NT_OFFSET).expect("small NT offset"));
        put_u32(&mut bytes, NT_OFFSET, 0x0000_4550);
        let coff = NT_OFFSET + 4;
        put_u16(&mut bytes, coff, 0x8664);
        put_u16(&mut bytes, coff + 2, 1);
        put_u16(&mut bytes, coff + 16, 0xF0);
        put_u16(&mut bytes, coff + 18, 0x2022);
        put_u16(&mut bytes, OPTIONAL_OFFSET, 0x020B);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 16, SECTION_RVA);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 20, SECTION_RVA);
        put_u64(&mut bytes, OPTIONAL_OFFSET + 24, IMAGE_BASE);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 32, 0x1000);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 36, 0x200);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 56, SECTION_RVA + SECTION_SPAN);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 60, u32::try_from(FIRST_RAW).expect("small header size"));
        put_u16(&mut bytes, OPTIONAL_OFFSET + 68, 3);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 108, 1);
        let section = SECTION_TABLE_OFFSET;
        bytes[section..section + 8].copy_from_slice(b".text\0\0\0");
        put_u32(&mut bytes, section + 8, SECTION_SPAN);
        put_u32(&mut bytes, section + 12, SECTION_RVA);
        put_u32(&mut bytes, section + 16, SECTION_SPAN);
        put_u32(&mut bytes, section + 20, u32::try_from(FIRST_RAW).expect("small raw offset"));
        put_u32(&mut bytes, section + 36, 0x6000_0020);
        bytes[FIRST_RAW] = 0xC3;
        bytes
    }

    const TLS_DIR_RVA: u32 = SECTION_RVA + 0x100;
    const CALLBACK_ARRAY_RVA: u32 = SECTION_RVA + 0x200;
    const CALLBACK_RVA: Rva = Rva(SECTION_RVA + 0x50);

    /// Builds on [`minimal_conventional_pe`] with a populated
    /// `IMAGE_DIRECTORY_ENTRY_TLS` whose callback array holds one callback VA.
    /// Returns `(bytes, callback_rva, image_base)`.
    fn pe_with_one_tls_callback() -> (Vec<u8>, Rva, u64) {
        let mut bytes = minimal_conventional_pe();

        put_u32(&mut bytes, OPTIONAL_OFFSET + 108, 10); // enough directories to reach TLS (index 9)
        let tls_directory_entry = OPTIONAL_OFFSET + 112 + IMAGE_DIRECTORY_ENTRY_TLS * 8;
        put_u32(&mut bytes, tls_directory_entry, TLS_DIR_RVA);
        put_u32(&mut bytes, tls_directory_entry + 4, 40);

        let tls_dir_offset = FIRST_RAW + usize::try_from(TLS_DIR_RVA - SECTION_RVA).expect("small delta");
        let callback_array_va = IMAGE_BASE + u64::from(CALLBACK_ARRAY_RVA);
        put_u64(&mut bytes, tls_dir_offset + 24, callback_array_va);

        let callback_array_offset =
            FIRST_RAW + usize::try_from(CALLBACK_ARRAY_RVA - SECTION_RVA).expect("small delta");
        put_u64(&mut bytes, callback_array_offset, IMAGE_BASE + u64::from(CALLBACK_RVA.0));
        put_u64(&mut bytes, callback_array_offset + 8, 0); // terminator

        (bytes, CALLBACK_RVA, IMAGE_BASE)
    }

    #[test]
    fn redirects_a_tls_callback_and_survives_reparse() {
        let (input, _callback_rva, image_base) = pe_with_one_tls_callback();
        let target = Rva(0x2000); // inside the R+X section
        let output = set_tls_callback(&input, 0, target).expect("redirect tls callback");
        let pe = PeImage::parse(&output).expect("reparse");
        let tls = pe.report().tls.expect("tls present");
        assert_eq!(tls.callbacks[0], Va(image_base + u64::from(target.0)));
    }

    #[test]
    fn rejects_a_target_that_is_executable_but_not_file_backed() {
        let (mut input, _, _) = pe_with_one_tls_callback();
        // Shrink raw data so RVA 0x2000 stays mapped but has no file bytes.
        put_u32(&mut input, SECTION_TABLE_OFFSET + 16, 0x1000);
        assert!(matches!(
            set_tls_callback(&input, 0, Rva(0x2000)),
            Err(RebuildError::TlsCallbackNotFileBacked { rva: Rva(0x2000) })
        ));
    }

    #[test]
    fn rejects_a_non_pe32_plus_image() {
        let (mut input, _, _) = pe_with_one_tls_callback();
        put_u16(&mut input, OPTIONAL_OFFSET, 0x010B);
        assert!(matches!(
            set_tls_callback(&input, 0, Rva(0x2000)),
            Err(RebuildError::UnsupportedKind(PeKind::Pe32))
        ));
    }

    #[test]
    fn rejects_out_of_range_index() {
        let (input, _, _) = pe_with_one_tls_callback();
        assert!(matches!(
            set_tls_callback(&input, 9, Rva(0x2000)),
            Err(RebuildError::TlsCallbackIndex { index: 9, count: 1 })
        ));
    }

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
        let pe = PeImage::parse(&bytes).expect("reparse");
        assert_eq!(pe.optional.address_of_entry_point, planned);
        assert_eq!(report.section.rva, planned);
        assert!(report.entry_redirected);
    }
}
