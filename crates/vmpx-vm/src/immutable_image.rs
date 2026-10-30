//! Memory source that exposes only bytes the loader maps read-only.

use vmpx_pe::{PeImage, Va};

use crate::{NativeCallImportResolution, SymbolicMemorySource};

const SCN_MEM_READ: u32 = 0x4000_0000;
const SCN_MEM_WRITE: u32 = 0x8000_0000;

/// Image bytes in headers and non-writable readable sections. Bytes in
/// writable sections can change before the read, so they are never exposed.
#[derive(Clone, Copy, Debug)]
pub struct ReadOnlyImage<'a, 'data>(pub &'a PeImage<'data>);

impl SymbolicMemorySource for ReadOnlyImage<'_, '_> {
    fn read_byte(&self, address: Va) -> Option<u8> {
        let pe = self.0;
        let rva = pe.optional.image_base.checked_rva_of(address)?;
        if rva.0 >= pe.optional.size_of_headers {
            let section = pe
                .sections
                .iter()
                .find(|section| section.contains_mapped_rva(rva))?;
            if section.characteristics & SCN_MEM_READ == 0
                || section.characteristics & SCN_MEM_WRITE != 0
            {
                return None;
            }
        }
        SymbolicMemorySource::read_byte(pe, address)
    }

    fn resolve_native_call_import(&self, target: u64) -> Option<NativeCallImportResolution> {
        SymbolicMemorySource::resolve_native_call_import(self.0, target)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const IMAGE_BASE: u64 = 0x1_8000_0000;
    pub(crate) const RDATA_VA: u64 = IMAGE_BASE + 0x1000;
    pub(crate) const DATA_VA: u64 = IMAGE_BASE + 0x2000;

    pub(crate) fn two_section_pe64() -> Vec<u8> {
        let mut data = vec![0_u8; 0x800];
        data[0..2].copy_from_slice(&0x5A4D_u16.to_le_bytes());
        data[0x3C..0x40].copy_from_slice(&0x80_u32.to_le_bytes());
        data[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
        let coff = 0x84;
        data[coff..coff + 2].copy_from_slice(&0x8664_u16.to_le_bytes());
        data[coff + 2..coff + 4].copy_from_slice(&2_u16.to_le_bytes());
        data[coff + 16..coff + 18].copy_from_slice(&0xF0_u16.to_le_bytes());
        data[coff + 18..coff + 20].copy_from_slice(&0x2022_u16.to_le_bytes());
        let optional = coff + 20;
        data[optional..optional + 2].copy_from_slice(&0x20B_u16.to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&IMAGE_BASE.to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 56..optional + 60].copy_from_slice(&0x3000_u32.to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&3_u16.to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&16_u32.to_le_bytes());
        for (index, (name, rva, raw, flags)) in [
            (&b".rdata"[..], 0x1000_u32, 0x400_u32, 0x4000_0040_u32),
            (&b".data"[..], 0x2000, 0x600, 0xC000_0040),
        ]
        .into_iter()
        .enumerate()
        {
            let section = optional + 0xF0 + index * 40;
            data[section..section + name.len()].copy_from_slice(name);
            data[section + 8..section + 12].copy_from_slice(&0x200_u32.to_le_bytes());
            data[section + 12..section + 16].copy_from_slice(&rva.to_le_bytes());
            data[section + 16..section + 20].copy_from_slice(&0x200_u32.to_le_bytes());
            data[section + 20..section + 24].copy_from_slice(&raw.to_le_bytes());
            data[section + 36..section + 40].copy_from_slice(&flags.to_le_bytes());
        }
        data[0x400..0x404].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
        data[0x600..0x604].copy_from_slice(&0x1234_5678_u32.to_le_bytes());
        data
    }

    #[test]
    fn exposes_read_only_sections_and_hides_writable_ones() {
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let image = ReadOnlyImage(&pe);
        assert_eq!(image.read_byte(Va(RDATA_VA)), Some(0xEF));
        assert_eq!(image.read_byte(Va(RDATA_VA + 3)), Some(0xDE));
        assert_eq!(image.read_byte(Va(DATA_VA)), None);
        assert_eq!(
            SymbolicMemorySource::read_byte(&pe, Va(DATA_VA)),
            Some(0x78)
        );
        assert_eq!(image.read_byte(Va(IMAGE_BASE + 0x2800)), None);
        assert_eq!(image.read_byte(Va(0x10)), None);
    }
}
