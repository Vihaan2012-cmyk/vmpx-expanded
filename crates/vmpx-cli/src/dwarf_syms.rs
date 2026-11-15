//! DWARF debug-symbol emission for the reconstructed clean PE.
//!
//! Names every function the `auto` pipeline rebuilds (devirtualized guards,
//! relocated mutated natives, the OEP) with a `DW_TAG_subprogram` carrying its
//! address range in the clean image, and embeds the `.debug_*` sections INTO
//! the PE -- so a disassembler loads real names instead of `sub_<addr>`, and so
//! the downstream cv2pdb (DWARF->PDB) path, which reads DWARF out of the PE's
//! own sections, keeps working. The constructs are deliberately vanilla
//! MinGW/GCC (compile_unit + base type + subprogram), nothing cv2pdb chokes on.
//!
//! PE section names are 8 bytes, but `.debug_info` is longer. We use the same
//! COFF long-name mechanism MinGW's linker does: the section `Name` holds
//! `/<decimal>` into a COFF string table appended at `PointerToSymbolTable`.

use std::collections::BTreeMap;

use gimli::write::{Address, AttributeValue, DwarfUnit, EndianVec, Sections};
use gimli::{Encoding, Format, LittleEndian};
use vmpx_rebuild::{AppendSection, append_section};

/// INITIALIZED_DATA | MEM_DISCARDABLE | MEM_READ -- how MinGW marks `.debug_*`.
const DEBUG_CHARS: u32 = 0x4200_0040;

/// One reconstructed function to describe.
pub struct Subprogram {
    pub name: String,
    /// Virtual address of the function entry in the clean image (base + rva).
    pub low_pc: u64,
    /// Virtual address one past the function's last byte, when known.
    pub high_pc: Option<u64>,
}

/// Build the `.debug_*` section bytes for `subs`. Returns `(section-name, bytes)`
/// pairs in a stable order; empty sections are omitted.
fn build_sections(subs: &[Subprogram]) -> Result<Vec<(&'static str, Vec<u8>)>, String> {
    let encoding = Encoding {
        format: Format::Dwarf32,
        version: 4,
        address_size: 8,
    };
    let mut dwarf = DwarfUnit::new(encoding);
    let root = dwarf.unit.root();
    {
        let r = dwarf.unit.get_mut(root);
        r.set(
            gimli::DW_AT_producer,
            AttributeValue::String(b"vmpx-expanded".to_vec()),
        );
        r.set(
            gimli::DW_AT_name,
            AttributeValue::String(b"clean.c".to_vec()),
        );
        r.set(
            gimli::DW_AT_comp_dir,
            AttributeValue::String(b".".to_vec()),
        );
        r.set(
            gimli::DW_AT_language,
            AttributeValue::Language(gimli::DW_LANG_C),
        );
    }
    // A single base type so each subprogram can carry a return type -- plain
    // `unsigned long` (8 bytes), the kind GCC emits.
    let u64_ty = dwarf.unit.add(root, gimli::DW_TAG_base_type);
    {
        let b = dwarf.unit.get_mut(u64_ty);
        b.set(
            gimli::DW_AT_name,
            AttributeValue::String(b"long unsigned int".to_vec()),
        );
        b.set(gimli::DW_AT_byte_size, AttributeValue::Udata(8));
        b.set(
            gimli::DW_AT_encoding,
            AttributeValue::Encoding(gimli::DW_ATE_unsigned),
        );
    }
    for sub in subs {
        let die = dwarf.unit.add(root, gimli::DW_TAG_subprogram);
        let d = dwarf.unit.get_mut(die);
        d.set(
            gimli::DW_AT_name,
            AttributeValue::String(sub.name.clone().into_bytes()),
        );
        d.set(gimli::DW_AT_external, AttributeValue::Flag(true));
        d.set(gimli::DW_AT_prototyped, AttributeValue::Flag(true));
        d.set(gimli::DW_AT_type, AttributeValue::UnitRef(u64_ty));
        d.set(
            gimli::DW_AT_low_pc,
            AttributeValue::Address(Address::Constant(sub.low_pc)),
        );
        if let Some(high) = sub.high_pc {
            if high > sub.low_pc {
                // DWARF4 offset form: high_pc is the size, relative to low_pc.
                d.set(gimli::DW_AT_high_pc, AttributeValue::Udata(high - sub.low_pc));
            }
        }
    }

    let mut sections = Sections::new(EndianVec::new(LittleEndian));
    dwarf
        .write(&mut sections)
        .map_err(|e| format!("dwarf write: {e}"))?;
    let mut out: Vec<(&'static str, Vec<u8>)> = Vec::new();
    sections
        .for_each(|id, data| {
            let bytes = data.slice().to_vec();
            if !bytes.is_empty() {
                out.push((id.name(), bytes));
            }
            Ok::<(), gimli::write::Error>(())
        })
        .map_err(|e| format!("dwarf collect: {e}"))?;
    Ok(out)
}

/// Patch the 8-byte `Name` of the section whose `VirtualAddress == rva` to the
/// COFF long-name reference `/<string_table_offset>`.
fn set_long_name(image: &mut [u8], rva: u32, strtab_offset: u32) -> Result<(), String> {
    let e_lfanew = u32::from_le_bytes(
        image
            .get(0x3c..0x40)
            .ok_or("no e_lfanew")?
            .try_into()
            .unwrap(),
    ) as usize;
    let coff = e_lfanew + 4;
    let num_sections = u16::from_le_bytes(image[coff + 2..coff + 4].try_into().unwrap()) as usize;
    let size_opt = u16::from_le_bytes(image[coff + 16..coff + 18].try_into().unwrap()) as usize;
    let table = coff + 20 + size_opt;
    for i in 0..num_sections {
        let hdr = table + i * 40;
        let va = u32::from_le_bytes(image[hdr + 12..hdr + 16].try_into().unwrap());
        if va == rva {
            let mut name = [0u8; 8];
            let s = format!("/{strtab_offset}");
            let b = s.as_bytes();
            if b.len() > 8 {
                return Err(format!("strtab offset {strtab_offset} name too long"));
            }
            name[..b.len()].copy_from_slice(b);
            image[hdr..hdr + 8].copy_from_slice(&name);
            return Ok(());
        }
    }
    Err(format!("no section header at rva {rva:#x}"))
}

/// Set the COFF `PointerToSymbolTable`/`NumberOfSymbols` (symbol table empty;
/// the field just locates the trailing string table).
fn set_symtab_pointer(image: &mut [u8], ptr: u32) {
    let e_lfanew =
        u32::from_le_bytes(image[0x3c..0x40].try_into().unwrap()) as usize;
    let coff = e_lfanew + 4;
    image[coff + 8..coff + 12].copy_from_slice(&ptr.to_le_bytes());
    image[coff + 12..coff + 16].copy_from_slice(&0u32.to_le_bytes());
}

/// Make room in the PE header region for `extra` more section headers, by
/// shifting all section raw data down one or more `FileAlignment` units and
/// bumping each `PointerToRawData` and `SizeOfHeaders`. A no-op when there is
/// already room. Section RVAs (and thus every RVA-based directory) are
/// untouched; only file offsets move.
fn ensure_header_room(image: &[u8], extra: usize) -> Result<Vec<u8>, String> {
    let e_lfanew = u32::from_le_bytes(image[0x3c..0x40].try_into().unwrap()) as usize;
    let coff = e_lfanew + 4;
    let num = u16::from_le_bytes(image[coff + 2..coff + 4].try_into().unwrap()) as usize;
    let size_opt = u16::from_le_bytes(image[coff + 16..coff + 18].try_into().unwrap()) as usize;
    let opt = coff + 20;
    let sect_table = opt + size_opt;
    let file_align = u32::from_le_bytes(image[opt + 36..opt + 40].try_into().unwrap());
    let size_of_headers = u32::from_le_bytes(image[opt + 60..opt + 64].try_into().unwrap());
    if file_align == 0 {
        return Err("zero FileAlignment".into());
    }
    let mut first_raw = u32::MAX;
    for i in 0..num {
        let h = sect_table + i * 40;
        let raw_size = u32::from_le_bytes(image[h + 16..h + 20].try_into().unwrap());
        let praw = u32::from_le_bytes(image[h + 20..h + 24].try_into().unwrap());
        if raw_size != 0 && praw != 0 {
            first_raw = first_raw.min(praw);
        }
    }
    if first_raw == u32::MAX {
        first_raw = size_of_headers;
    }
    let capacity = first_raw.min(size_of_headers);
    let required = u32::try_from(sect_table + (num + extra) * 40).map_err(|_| "header offset")?;
    if required <= capacity {
        return Ok(image.to_vec());
    }
    let shift = required.saturating_sub(capacity).div_ceil(file_align) * file_align;
    // SizeOfHeaders (grown) must still fit before the first section's RVA.
    let sect_align = u32::from_le_bytes(image[opt + 32..opt + 36].try_into().unwrap());
    let first_rva = {
        let h = sect_table; // first section header; VirtualAddress at +12
        u32::from_le_bytes(image[h + 12..h + 16].try_into().unwrap())
    };
    if sect_align != 0 && (size_of_headers + shift) > first_rva {
        return Err(format!(
            "header expansion {shift:#x} would collide with first section rva {first_rva:#x}"
        ));
    }
    let cut = first_raw as usize;
    let mut out = Vec::with_capacity(image.len() + shift as usize);
    out.extend_from_slice(&image[..cut]);
    out.extend(std::iter::repeat_n(0u8, shift as usize));
    out.extend_from_slice(&image[cut..]);
    for i in 0..num {
        let h = sect_table + i * 40;
        let raw_size = u32::from_le_bytes(out[h + 16..h + 20].try_into().unwrap());
        let praw = u32::from_le_bytes(out[h + 20..h + 24].try_into().unwrap());
        if raw_size != 0 && praw >= first_raw {
            out[h + 20..h + 24].copy_from_slice(&(praw + shift).to_le_bytes());
        }
    }
    out[opt + 60..opt + 64].copy_from_slice(&(size_of_headers + shift).to_le_bytes());
    Ok(out)
}

/// Embed DWARF describing `subs` into `image`, returning the new PE bytes.
pub fn embed_dwarf(image: Vec<u8>, subs: &[Subprogram]) -> Result<Vec<u8>, String> {
    let sections = build_sections(subs)?;
    if sections.is_empty() {
        return Ok(image);
    }
    let mut image = ensure_header_room(&image, sections.len() + 1)?;
    // Append each `.debug_*` as a discardable data section under a temp 8-byte
    // name, remembering its rva + full name.
    let mut appended: Vec<(u32, &'static str)> = Vec::new();
    for (idx, (name, data)) in sections.iter().enumerate() {
        let mut temp = *b"/dbg0000";
        temp[7] = b'0' + u8::try_from(idx).unwrap_or(0);
        let rebuilt = append_section(
            &image,
            AppendSection {
                name: temp,
                data,
                characteristics: DEBUG_CHARS,
            },
        )
        .map_err(|e| format!("append {name}: {e}"))?;
        let rva = rebuilt.section.rva.0;
        image = rebuilt.bytes;
        appended.push((rva, name));
    }
    // Build the COFF string table: [u32 total size][name\0]... first string at
    // offset 4. Record each name's offset, then rewrite the headers.
    let strtab_pos = u32::try_from(image.len()).map_err(|_| "image too large")?;
    let mut strtab: Vec<u8> = vec![0, 0, 0, 0]; // size filled in below
    let mut offsets: BTreeMap<&str, u32> = BTreeMap::new();
    for (_, name) in &appended {
        if offsets.contains_key(*name) {
            continue;
        }
        offsets.insert(name, u32::try_from(strtab.len()).unwrap());
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
    }
    let total = u32::try_from(strtab.len()).map_err(|_| "strtab too large")?;
    strtab[0..4].copy_from_slice(&total.to_le_bytes());
    for (rva, name) in &appended {
        set_long_name(&mut image, *rva, offsets[*name])?;
    }
    image.extend_from_slice(&strtab);
    set_symtab_pointer(&mut image, strtab_pos);
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dwarf_round_trips_named_subprograms() {
        let subs = vec![
            Subprogram {
                name: "vmp_guard_c4a0add".into(),
                low_pc: 0x1_4000_1000,
                high_pc: Some(0x1_4000_1050),
            },
            Subprogram {
                name: "OEP_DllEntryPoint".into(),
                low_pc: 0x1_4000_1200,
                high_pc: None,
            },
        ];
        let secs = build_sections(&subs).unwrap();
        let map: BTreeMap<&str, Vec<u8>> = secs.iter().map(|(n, b)| (*n, b.clone())).collect();
        assert!(map.contains_key(".debug_info"), "has .debug_info");
        assert!(map.contains_key(".debug_abbrev"), "has .debug_abbrev");

        // Parse it back with gimli read and confirm both names + an address.
        let endian = gimli::RunTimeEndian::Little;
        let load = |id: gimli::SectionId| -> Result<gimli::EndianSlice<'_, gimli::RunTimeEndian>, gimli::Error> {
            let data = map.get(id.name()).map(|v| v.as_slice()).unwrap_or(&[][..]);
            Ok(gimli::EndianSlice::new(data, endian))
        };
        let dwarf = gimli::Dwarf::load(load).unwrap();
        let mut names = Vec::new();
        let mut saw_addr = false;
        let mut iter = dwarf.units();
        let header = iter.next().unwrap().unwrap();
        let unit = dwarf.unit(header).unwrap();
        let mut entries = unit.entries();
        while let Some(entry) = entries.next_dfs().unwrap() {
            if entry.tag() != gimli::DW_TAG_subprogram {
                continue;
            }
            if let Some(gimli::AttributeValue::String(s)) =
                entry.attr_value(gimli::DW_AT_name)
            {
                names.push(String::from_utf8_lossy(s.slice()).into_owned());
            }
            if let Some(gimli::AttributeValue::Addr(a)) =
                entry.attr_value(gimli::DW_AT_low_pc)
            {
                saw_addr |= a >= 0x1_4000_1000;
            }
        }
        assert!(names.iter().any(|n| n == "vmp_guard_c4a0add"), "{names:?}");
        assert!(names.iter().any(|n| n == "OEP_DllEntryPoint"), "{names:?}");
        assert!(saw_addr, "a subprogram carried its low_pc address");
    }
}
