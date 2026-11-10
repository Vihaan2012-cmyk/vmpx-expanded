//! Import recovery for `vmpx auto`: name the application's encrypted `.idata`
//! slots from a post-init snapshot and bind them in place with a real import
//! directory (the loader fills `FirstThunk`, so the encrypted bytes are
//! overwritten).

use std::collections::BTreeMap;

use vmpx_emulate::snapshot;
use vmpx_pe::PeImage;
use vmpx_rebuild::{AppendSection, append_section, serialize_import_table};
use vmpx_rebuild::{ImportModuleSpec, ImportTarget};

use crate::logkit::Log;

const STUB_REGION: u64 = 0x0000_0020_0000_0000;
const IMPORT_DIRECTORY: usize = 1;
const IAT_DIRECTORY: usize = 12;
const DESCRIPTOR_SIZE: usize = 20;
const FIRST_THUNK_FIELD: usize = 16;

/// Bind imports named from a post-init snapshot: read each `.idata` slot's
/// sentinel and name it by the loader's flattened import order.
pub fn bind(
    log: &Log,
    intact: &PeImage<'_>,
    intact_bytes: &[u8],
    stripped: &[u8],
    snapshot_path: &std::path::Path,
) -> Result<Vec<u8>, String> {
    let base = intact.optional.image_base.0;
    let (_cpu, mem) = snapshot::load(snapshot_path).map_err(|e| format!("snapshot: {e}"))?;
    let names = flattened_import_names(intact);
    let Some(idata) = intact.sections.iter().find(|s| s.name.as_str() == ".idata") else {
        return Err("no .idata section".to_owned());
    };
    let start = idata.virtual_address.0;
    let size = idata.virtual_size.max(idata.size_of_raw_data);
    let mut slots: Vec<(u32, String, String)> = Vec::new();
    for offset in (0..size).step_by(8) {
        let rva = start + offset;
        if let Ok(value) = mem.read_uint(base + u64::from(rva), 8)
            && let Some(index) = value
                .checked_sub(STUB_REGION)
                .filter(|d| d % 8 == 0)
                .map(|d| (d / 8) as usize)
                .filter(|i| *i < names.len())
        {
            let (module, func) = &names[index];
            slots.push((rva, module.clone(), func.clone()));
        }
    }
    bind_slots(log, intact, intact_bytes, stripped, slots)
}

/// The import entry a recovered symbol denotes. `#N` is the sentinel both
/// recovery paths (`flattened_import_names`, `detonate_imports`) use for a
/// name-less by-ordinal import, so it becomes a real ordinal entry the loader
/// resolves by ordinal; a literal `#N` NAME would never resolve to an export.
/// Anything else, including a `#` not followed by a `u16`, stays a by-name
/// import.
fn import_target(name: &str) -> ImportTarget {
    match name.strip_prefix('#').and_then(|n| n.parse::<u16>().ok()) {
        Some(ordinal) => ImportTarget::Ordinal(ordinal),
        None => ImportTarget::Name {
            name: name.to_owned(),
            hint: 0,
        },
    }
}

/// The loader's flattened import order, which assigns each symbol its sentinel.
fn flattened_import_names(intact: &PeImage<'_>) -> Vec<(String, String)> {
    let mut names = Vec::new();
    for module in &intact.report().imports {
        for sym in &module.entries {
            names.push((
                module.name.clone(),
                sym.name
                    .clone()
                    .unwrap_or_else(|| format!("#{}", sym.ordinal.unwrap_or(0))),
            ));
        }
    }
    names
}

/// Bind a set of `(slot_rva, module, symbol)` triples -- from a snapshot or a
/// live detonation -- into a real import directory whose `FirstThunk`s point at
/// the existing `.idata` runs, so the loader fills the slots the code reads.
pub fn bind_slots(
    log: &Log,
    intact: &PeImage<'_>,
    intact_bytes: &[u8],
    stripped: &[u8],
    slots: Vec<(u32, String, String)>,
) -> Result<Vec<u8>, String> {
    let mut by_module: BTreeMap<String, Vec<(u32, String)>> = BTreeMap::new();
    for (rva, module, func) in slots {
        by_module.entry(module).or_default().push((rva, func));
    }
    let total: usize = by_module.values().map(Vec::len).sum();
    if total == 0 {
        return Err("no import slot resolved".to_owned());
    }

    // One descriptor per contiguous run of slots.
    let mut runs: Vec<(String, u32, Vec<String>)> = Vec::new();
    for (module, slots) in &by_module {
        let mut sorted = slots.clone();
        sorted.sort_by_key(|(rva, _)| *rva);
        for (rva, func) in sorted {
            match runs.last_mut() {
                Some((m, start, funcs))
                    if *m == *module
                        && *start + 8 * u32::try_from(funcs.len()).unwrap_or(u32::MAX) == rva =>
                {
                    funcs.push(func);
                }
                _ => runs.push((module.clone(), rva, vec![func])),
            }
        }
    }

    let specs: Vec<ImportModuleSpec> = runs
        .iter()
        .map(|(module, _, funcs)| ImportModuleSpec {
            name: module.clone(),
            symbols: funcs.iter().map(|name| import_target(name)).collect(),
        })
        .collect();
    let section_rva = vmpx_rebuild::next_section_rva(stripped).map_err(|e| e.to_string())?;
    let table = serialize_import_table(intact.optional.kind, section_rva, &specs)
        .map_err(|e| format!("serialize: {e}"))?;

    // Point each descriptor's FirstThunk at its existing .idata run.
    let mut payload = table.bytes.clone();
    for (index, (_, start, _)) in runs.iter().enumerate() {
        let at = index * DESCRIPTOR_SIZE + FIRST_THUNK_FIELD;
        payload
            .get_mut(at..at + 4)
            .ok_or("descriptor past payload")?
            .copy_from_slice(&start.to_le_bytes());
    }
    let rebuilt = append_section(
        stripped,
        AppendSection {
            name: *b".vmpximp",
            data: &payload,
            characteristics: 0xC000_0040,
        },
    )
    .map_err(|e| e.to_string())?;
    let mut image = rebuilt.bytes;

    // Install directory 1 and point the IAT directory at the .idata span.
    let pe_off = u32::from_le_bytes(image[0x3c..0x40].try_into().expect("4")) as usize;
    let dirs = pe_off + 24 + 112;
    let put = |image: &mut Vec<u8>, index: usize, rva: u32, sz: u32| {
        let at = dirs + index * 8;
        image[at..at + 4].copy_from_slice(&rva.to_le_bytes());
        image[at + 4..at + 8].copy_from_slice(&sz.to_le_bytes());
    };
    put(
        &mut image,
        IMPORT_DIRECTORY,
        table.import_directory.rva.0,
        table.import_directory.size,
    );
    let low = runs.iter().map(|(_, s, _)| *s).min().unwrap_or(0);
    let high = runs
        .iter()
        .map(|(_, s, f)| s + 8 * u32::try_from(f.len()).unwrap_or(0))
        .max()
        .unwrap_or(0);
    put(&mut image, IAT_DIRECTORY, low, high.saturating_sub(low));

    let _ = intact_bytes;
    log.step(
        "imports bound",
        &format!(
            "{total} symbol(s) in {} descriptor(s) over {} module(s)",
            runs.len(),
            by_module.len()
        ),
    );
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::import_target;
    use vmpx_rebuild::ImportTarget;

    #[test]
    fn ordinal_sentinel_becomes_a_by_ordinal_import() {
        assert_eq!(import_target("#123"), ImportTarget::Ordinal(123));
        assert_eq!(import_target("#0"), ImportTarget::Ordinal(0));
        assert_eq!(import_target("#65535"), ImportTarget::Ordinal(65535));
    }

    #[test]
    fn real_names_stay_by_name() {
        assert_eq!(
            import_target("MessageBoxW"),
            ImportTarget::Name {
                name: "MessageBoxW".to_owned(),
                hint: 0
            }
        );
    }

    #[test]
    fn non_numeric_or_overflowing_hash_stays_a_name() {
        // A `#` not followed by a u16 is not an ordinal sentinel.
        assert_eq!(
            import_target("#notanumber"),
            ImportTarget::Name {
                name: "#notanumber".to_owned(),
                hint: 0
            }
        );
        assert_eq!(
            import_target("#99999"),
            ImportTarget::Name {
                name: "#99999".to_owned(),
                hint: 0
            }
        );
    }
}
