//! Memory source that exposes only bytes the loader maps read-only.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

use vmpx_pe::{ImageBase, ImportModule, PeImage, Va};

use crate::{
    NativeCallImportDescriptor, NativeCallImportMatchKind, NativeCallImportResolution,
    SymbolicMemorySource,
};

/// Concrete runtime memory observed on ONE detonation: a plain, crate-owned map
/// from virtual address to captured bytes. This is NOT a proof — it reflects a
/// single run/environment. The CLI builds it from a detonation's observed
/// globals; vmpx-vm never produces it and never trusts it on the sound path.
///
/// Kept deliberately free of any `vmpx-emulate` type (the crate owns this plain
/// structure; the CLI converts `ObservedFacts.globals` into it).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObservedMemory {
    slots: BTreeMap<u64, Vec<u8>>,
}

impl ObservedMemory {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds the seed from `(va, bytes)` slots, e.g. a detonation's observed
    /// globals.
    pub fn from_slots(slots: impl IntoIterator<Item = (u64, Vec<u8>)>) -> Self {
        Self {
            slots: slots.into_iter().collect(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Observed byte at `va`, if any recorded slot covers it.
    #[must_use]
    pub fn read_byte(&self, va: u64) -> Option<u8> {
        let (&base, bytes) = self.slots.range(..=va).next_back()?;
        let offset = usize::try_from(va.checked_sub(base)?).ok()?;
        bytes.get(offset).copied()
    }
}

/// A dispatch step of more than this many bytecode bytes is a real VM branch
/// (straight-line handlers advance the VIP by a few bytes at a time).
pub const FAR_BRANCH_DELTA: u64 = 0x100;

/// Concrete VM branch targets observed on ONE detonation: a plain, crate-owned
/// map from a branch SITE identity (the VM branch's VIP / bytecode address) to
/// the concrete target VIP(s) the branch reached on that run. This is NOT a
/// proof — it reflects a single run/input. The CLI builds it from a detonation's
/// recorded branch targets; vmpx-vm never produces it and never trusts it on the
/// sound path. A seeded target is followed only when it independently validates
/// as real in-image bytecode (see cfg.rs), so observation can seed a resolution
/// but can never turn a data address into code.
///
/// Kept deliberately free of any `vmpx-emulate` type (the crate owns this plain
/// structure, mirroring [`ObservedMemory`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObservedBranchTargets {
    sites: BTreeMap<u64, Vec<u64>>,
}

impl ObservedBranchTargets {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds the seed from `(branch_site_vip, target_vips)` pairs, e.g. a
    /// detonation's recorded VM branch targets correlated to their branch site.
    pub fn from_sites(sites: impl IntoIterator<Item = (u64, Vec<u64>)>) -> Self {
        Self {
            sites: sites.into_iter().collect(),
        }
    }

    /// Builds the seed from raw dynamic VIP->VIP dispatch edges `(from, to)`
    /// (e.g. the emulate observer's `ObservedVmEdge`s). Only FAR edges —
    /// `|to - from| > FAR_BRANCH_DELTA` — are real VM branches; the rest are
    /// straight-line dispatch steps and are dropped. Targets are de-duplicated
    /// and sorted per site. Evidence only: the edge set is a lower bound (only
    /// branches actually executed on one run).
    pub fn from_vip_edges(edges: impl IntoIterator<Item = (u64, u64)>) -> Self {
        let mut sites: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for (from, to) in edges {
            if from.abs_diff(to) > FAR_BRANCH_DELTA {
                sites.entry(from).or_default().push(to);
            }
        }
        for targets in sites.values_mut() {
            targets.sort_unstable();
            targets.dedup();
        }
        Self { sites }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Branch site VIPs with at least one recorded target, ascending.
    pub fn sites(&self) -> impl Iterator<Item = u64> + '_ {
        self.sites.keys().copied()
    }

    /// Number of distinct branch sites with at least one observed target.
    #[must_use]
    pub fn site_count(&self) -> usize {
        self.sites.len()
    }

    /// Observed concrete target VIP(s) for the branch at `site`, or an empty
    /// slice when none were recorded.
    #[must_use]
    pub fn targets_for(&self, site: u64) -> &[u64] {
        self.sites.get(&site).map_or(&[], Vec::as_slice)
    }
}

/// Read-only image bytes overlaid with observed runtime memory.
///
/// `read_byte` returns only sound static-image bytes (identical to
/// [`ReadOnlyImage`]). `observed_byte` exposes the observed seed: where a const
/// slot is zero/absent in the static image, the symbolic executor may consult
/// it to resolve a provably-null read. Consulting a recorded slot flips
/// `seeded`, so the caller can label the recovery with the observed assumption.
/// EXPERIMENTAL — never the sound path.
pub struct ObservedImage<'a, 'data> {
    image: ReadOnlyImage<'a, 'data>,
    observed: &'a ObservedMemory,
    seeded: &'a Cell<bool>,
}

impl<'a, 'data> ObservedImage<'a, 'data> {
    #[must_use]
    pub fn new(
        image: ReadOnlyImage<'a, 'data>,
        observed: &'a ObservedMemory,
        seeded: &'a Cell<bool>,
    ) -> Self {
        Self {
            image,
            observed,
            seeded,
        }
    }
}

impl SymbolicMemorySource for ObservedImage<'_, '_> {
    fn read_byte(&self, address: Va) -> Option<u8> {
        self.image.read_byte(address)
    }

    fn read_only_byte(&self, address: Va) -> Option<u8> {
        self.image.read_only_byte(address)
    }

    fn is_read_only_address(&self, address: Va) -> bool {
        self.image.is_read_only_address(address)
    }

    fn protection_stability_proven(&self) -> bool {
        self.image.protection_stability_proven()
    }

    fn observed_byte(&self, address: Va) -> Option<u8> {
        let byte = self.observed.read_byte(address.get())?;
        self.seeded.set(true);
        Some(byte)
    }

    fn resolve_native_call_import(&self, target: u64) -> Option<NativeCallImportResolution> {
        self.image.resolve_native_call_import(target)
    }
}

const SCN_MEM_READ: u32 = 0x4000_0000;
const SCN_MEM_WRITE: u32 = 0x8000_0000;

static VOLATILE_PAGES: std::sync::RwLock<BTreeSet<u64>> =
    std::sync::RwLock::new(BTreeSet::new());
/// Disjoint inclusive byte ranges keyed by start.
static VOLATILE_RANGES: std::sync::RwLock<BTreeMap<u64, u64>> =
    std::sync::RwLock::new(BTreeMap::new());

/// DYNAMIC-EVIDENCE VETO: declares 4 KiB pages (virtual address >> 12) that a
/// run showed to change after load. No byte of such a page is served as a
/// constant from the file. It can only REMOVE served bytes; it is never
/// evidence that a page is stable.
pub fn set_volatile_image_pages(pages: impl IntoIterator<Item = u64>) {
    *VOLATILE_PAGES
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = pages.into_iter().collect();
}

/// DYNAMIC-EVIDENCE VETO at byte granularity: inclusive `(start, end)` virtual
/// address ranges whose bytes differed between the file image and a post-init
/// run. A volatile byte reads as a fresh unknown on every path (suppressed or
/// not) instead of as the file's value. Veto only, never a value source.
pub fn set_volatile_image_ranges(ranges: impl IntoIterator<Item = (u64, u64)>) {
    *VOLATILE_RANGES
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) =
        ranges.into_iter().filter(|(s, e)| s <= e).collect();
}

fn is_volatile_page(address: Va) -> bool {
    let address = address.get();
    if VOLATILE_PAGES
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .contains(&(address >> 12))
    {
        return true;
    }
    VOLATILE_RANGES
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .range(..=address)
        .next_back()
        .is_some_and(|(_, &end)| address <= end)
}

const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const SCN_CNT_CODE: u32 = 0x0000_0020;

/// An import whose name could change page protection, write another mapping's
/// pages, or resolve such an API dynamically. Conservative: ordinals, any
/// `ntdll` import and unnamed symbols count.
fn import_may_reprotect(module: &str, name: Option<&str>) -> bool {
    let Some(name) = name else { return true };
    if module.to_ascii_lowercase().starts_with("ntdll") {
        return true;
    }
    let native_api = |prefix: &str| {
        name.strip_prefix(prefix)
            .and_then(|rest| rest.chars().next())
            .is_some_and(char::is_uppercase)
    };
    if name.starts_with("Ldr") || native_api("Nt") || native_api("Zw") {
        return true;
    }
    let name = name.to_ascii_lowercase();
    [
        "protect",
        "writeprocessmemory",
        "writevirtualmemory",
        "mapviewof",
        "unmapviewof",
        "getprocaddress",
        "loadlibrary",
        "getmodulehandle",
        "setprocessmitigation",
        "setinformationvirtualmemory",
    ]
    .iter()
    .any(|needle| name.contains(needle))
}

/// Raw OS access that bypasses the import table: `syscall`, `sysenter`,
/// `int 2e`, or a PEB/TEB fetch (`gs:[60h]`/`gs:[30h]`, `fs:[30h]`) from which
/// a loader-style export walk could find a protection API.
fn contains_raw_os_access(bytes: &[u8]) -> bool {
    if bytes
        .windows(2)
        .any(|pair| matches!(pair, [0x0F, 0x05] | [0x0F, 0x34] | [0xCD, 0x2E]))
    {
        return true;
    }
    bytes.windows(4).enumerate().any(|(index, dword)| {
        matches!(dword, [0x60 | 0x30, 0, 0, 0])
            && bytes[index.saturating_sub(6)..index]
                .iter()
                .any(|byte| matches!(byte, 0x64 | 0x65))
    })
}

static REPROTECT_CACHE: std::sync::Mutex<Vec<(usize, usize, bool)>> =
    std::sync::Mutex::new(Vec::new());

/// PROOF that nothing in the image can change page protection, so every
/// non-writable page keeps the loader's protection for the whole run: no import
/// that could reprotect or resolve such an API, no section both writable and
/// executable (self-modifying code could mint a raw system call), and no
/// `syscall`/`sysenter`/`int 2e` byte pair or PEB/TEB fetch in any executable section. Over-
/// approximate (false positives only withhold the proof). Says nothing about
/// code outside this image.
pub(crate) fn image_cannot_reprotect(pe: &PeImage<'_>) -> bool {
    let data = pe.data();
    let key = (data.as_ptr() as usize, data.len());
    if let Some(&(_, _, cached)) = REPROTECT_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .find(|entry| (entry.0, entry.1) == key)
    {
        return cached;
    }
    let proven = !pe.report().imports.iter().any(|module| {
        module
            .entries
            .iter()
            .any(|symbol| import_may_reprotect(&module.name, symbol.name.as_deref()))
    }) && pe.sections.iter().all(|section| {
        let executable = section.characteristics & (SCN_MEM_EXECUTE | SCN_CNT_CODE) != 0;
        if !executable {
            return true;
        }
        if section.characteristics & SCN_MEM_WRITE != 0 {
            return false;
        }
        let start = section.pointer_to_raw_data.0 as usize;
        let end = start.saturating_add(section.size_of_raw_data as usize);
        data.get(start..end.min(data.len()))
            .is_some_and(|raw| !contains_raw_os_access(raw))
    });
    REPROTECT_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push((key.0, key.1, proven));
    proven
}

/// Image bytes in headers and non-writable readable sections. Bytes in
/// writable sections can change before the read, so they are never exposed.
#[derive(Clone, Copy, Debug)]
pub struct ReadOnlyImage<'a, 'data>(pub &'a PeImage<'data>);

impl SymbolicMemorySource for ReadOnlyImage<'_, '_> {
    fn read_only_byte(&self, address: Va) -> Option<u8> {
        self.read_byte(address)
    }

    fn is_read_only_address(&self, address: Va) -> bool {
        let pe = self.0;
        let Some(rva) = pe.optional.image_base.checked_rva_of(address) else {
            return false;
        };
        if rva.0 < pe.optional.size_of_headers {
            return true;
        }
        pe.sections.iter().find(|section| section.contains_mapped_rva(rva)).is_some_and(|section| {
            section.characteristics & SCN_MEM_READ != 0 && section.characteristics & SCN_MEM_WRITE == 0
        })
    }

    fn protection_stability_proven(&self) -> bool {
        image_cannot_reprotect(self.0)
    }

    fn read_byte(&self, address: Va) -> Option<u8> {
        if is_volatile_page(address) {
            return None;
        }
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
        let direct = SymbolicMemorySource::resolve_native_call_import(self.0, target);
        if !matches!(direct, Some(NativeCallImportResolution::NoMatch { .. })) {
            return direct;
        }
        let imports = self.0.report().imports;
        let thunk = resolve_import_thunk(
            &imports,
            self.0.optional.image_base,
            |address| SymbolicMemorySource::read_byte(self, address),
            target,
        );
        thunk.or(direct)
    }
}

/// Resolves a call target that is the VA of a `jmp qword ptr [rip+disp32]`
/// stub whose slot is exactly an import's IAT entry. The caller supplies only
/// bytes that cannot change after load, so the stub provably forwards to that
/// import. Anything else returns `None`.
pub(crate) fn resolve_import_thunk(
    imports: &[ImportModule],
    image_base: ImageBase,
    read_byte: impl Fn(Va) -> Option<u8>,
    target: u64,
) -> Option<NativeCallImportResolution> {
    let mut stub = [0_u8; 6];
    for (offset, byte) in stub.iter_mut().enumerate() {
        *byte = read_byte(Va(target.checked_add(offset as u64)?))?;
    }
    if stub[..2] != [0xFF, 0x25] {
        return None;
    }
    let displacement = i32::from_le_bytes(stub[2..].try_into().ok()?);
    let slot = target
        .checked_add(6)?
        .checked_add_signed(i64::from(displacement))?;
    let slot_rva = image_base.checked_rva_of(Va(slot))?;
    let mut matches = imports
        .iter()
        .flat_map(|module| module.entries.iter().map(move |symbol| (module, symbol)))
        .filter(|(_, symbol)| symbol.iat_rva == slot_rva)
        .map(|(module, symbol)| NativeCallImportDescriptor {
            module: module.name.clone(),
            iat_rva: symbol.iat_rva,
            lookup_value: symbol.lookup_value,
            name: symbol.name.clone(),
            ordinal: symbol.ordinal,
            match_kinds: vec![NativeCallImportMatchKind::ImportThunk],
        })
        .collect::<Vec<_>>();
    match matches.len() {
        0 => None,
        1 => Some(NativeCallImportResolution::Unique {
            target,
            import: matches.remove(0),
        }),
        _ => Some(NativeCallImportResolution::Ambiguous {
            target,
            imports: matches,
        }),
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
    fn reprotect_capability_is_detected_conservatively() {
        assert!(import_may_reprotect("kernel32.dll", Some("VirtualProtect")));
        assert!(import_may_reprotect("KERNEL32.dll", Some("GetProcAddress")));
        assert!(import_may_reprotect("kernel32.dll", Some("WriteProcessMemory")));
        assert!(import_may_reprotect("ntdll.dll", Some("RtlFoo")));
        assert!(import_may_reprotect("kernel32.dll", Some("LdrGetDllHandle")));
        assert!(import_may_reprotect("x.dll", Some("NtQueryInformationProcess")));
        assert!(!import_may_reprotect("kernel32.dll", Some("Sleep")));
        assert!(!import_may_reprotect("kernel32.dll", Some("GetCurrentThread")));
        assert!(import_may_reprotect("kernel32.dll", None));
        assert!(!import_may_reprotect("kernel32.dll", Some("GetTickCount")));
        assert!(contains_raw_os_access(&[0x90, 0x0F, 0x05, 0xC3]));
        assert!(contains_raw_os_access(&[0xCD, 0x2E]));
        assert!(!contains_raw_os_access(&[0x90, 0x0F, 0x1F, 0x05]));
        assert!(contains_raw_os_access(&[0x65, 0x48, 0x8B, 0x04, 0x25, 0x60, 0, 0, 0]));
        assert!(!contains_raw_os_access(&[0x48, 0x8B, 0x04, 0x25, 0x60, 0, 0, 0]));
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        assert!(image_cannot_reprotect(&pe), "no imports, no executable section");
        assert!(ReadOnlyImage(&pe).protection_stability_proven());
    }

    #[test]
    fn volatile_bytes_are_never_served() {
        // The veto is process-global: probe addresses no other test reads.
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let source = ReadOnlyImage(&pe);
        let probe = RDATA_VA + 0x100;
        assert_eq!(source.read_byte(Va(probe)), Some(0));
        assert_eq!(source.read_only_byte(Va(probe)), Some(0));
        set_volatile_image_ranges([(probe + 1, probe + 2)]);
        assert_eq!(source.read_byte(Va(probe)), Some(0));
        assert_eq!(source.read_byte(Va(probe + 1)), None);
        assert_eq!(source.read_only_byte(Va(probe + 2)), None);
        assert_eq!(source.read_byte(Va(probe + 3)), Some(0));
        set_volatile_image_ranges([]);
        assert_eq!(source.read_byte(Va(probe + 1)), Some(0));
        set_volatile_image_pages([0x5000_0000 >> 12]);
        assert!(is_volatile_page(Va(0x5000_0123)));
        assert!(!is_volatile_page(Va(probe)));
        set_volatile_image_pages([]);
    }

    #[test]
    fn observed_memory_reads_bytes_within_recorded_slots_only() {
        let observed = ObservedMemory::from_slots([(0x1000_u64, vec![0xAA, 0xBB, 0xCC])]);
        assert!(!observed.is_empty());
        assert_eq!(observed.read_byte(0x1000), Some(0xAA));
        assert_eq!(observed.read_byte(0x1002), Some(0xCC));
        assert_eq!(observed.read_byte(0x1003), None); // past the slot
        assert_eq!(observed.read_byte(0x0FFF), None); // before the slot
        assert!(ObservedMemory::new().read_byte(0x1000).is_none());
    }

    #[test]
    fn vip_edges_keep_only_far_branches_deduped_per_site() {
        let observed = ObservedBranchTargets::from_vip_edges([
            (0x1000_u64, 0x1004_u64), // straight-line step: dropped
            (0x1004, 0x2000),         // far branch
            (0x1004, 0x2000),         // duplicate
            (0x1004, 0x0800),         // far, backwards
            (0x3000, 0x3100),         // exactly at the threshold: not far
        ]);
        assert_eq!(observed.site_count(), 1);
        assert_eq!(observed.targets_for(0x1004), &[0x0800, 0x2000]);
        assert!(observed.targets_for(0x1000).is_empty());
        assert!(observed.targets_for(0x3000).is_empty());
    }

    #[test]
    fn observed_branch_targets_returns_recorded_sites_only() {
        let observed =
            ObservedBranchTargets::from_sites([(0x1000_u64, vec![0x2000, 0x3000]), (0x1800, vec![])]);
        assert!(!observed.is_empty());
        assert_eq!(observed.targets_for(0x1000), &[0x2000, 0x3000]);
        assert_eq!(observed.targets_for(0x1800), &[] as &[u64]);
        assert_eq!(observed.targets_for(0x9999), &[] as &[u64]); // unrecorded site
        assert!(ObservedBranchTargets::new().is_empty());
        assert_eq!(ObservedBranchTargets::new().targets_for(0x1000), &[] as &[u64]);
    }

    #[test]
    fn observed_image_overlays_the_seed_and_flags_consultation() {
        let bytes = two_section_pe64();
        let pe = PeImage::parse(&bytes).expect("synthetic PE");
        let observed = ObservedMemory::from_slots([(DATA_VA, vec![0x99])]);
        let seeded = Cell::new(false);
        let image = ObservedImage::new(ReadOnlyImage(&pe), &observed, &seeded);
        // read_byte stays the sound read-only view (writable .data stays hidden).
        assert_eq!(image.read_byte(Va(DATA_VA)), None);
        assert!(!seeded.get());
        // observed_byte exposes the seed and flips the label flag.
        assert_eq!(image.observed_byte(Va(DATA_VA)), Some(0x99));
        assert!(seeded.get());
        // An unrecorded address yields no seed.
        assert_eq!(image.observed_byte(Va(DATA_VA + 1)), None);
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

    fn module(slot_rva: u32, name: &str) -> vmpx_pe::ImportModule {
        vmpx_pe::ImportModule {
            name: "kernel32.dll".to_owned(),
            descriptor_rva: vmpx_pe::Rva(0),
            original_first_thunk: vmpx_pe::Rva(0),
            first_thunk: vmpx_pe::Rva(slot_rva),
            entries: vec![vmpx_pe::ImportSymbol {
                iat_rva: vmpx_pe::Rva(slot_rva),
                lookup_value: 0x9000,
                name: Some(name.to_owned()),
                hint: None,
                ordinal: None,
            }],
        }
    }

    fn stub_reader(stub_va: u64, bytes: &[u8]) -> impl Fn(Va) -> Option<u8> + '_ {
        move |address| {
            let offset = address.0.checked_sub(stub_va)?;
            bytes.get(usize::try_from(offset).ok()?).copied()
        }
    }

    #[test]
    fn jmp_stub_to_iat_slot_resolves_to_that_import() {
        let stub_va = IMAGE_BASE + 0x1100;
        let slot_rva = 0x2040_u32;
        let disp = (i64::from(slot_rva) - 0x1100 - 6) as i32;
        let mut stub = vec![0xFF, 0x25];
        stub.extend_from_slice(&disp.to_le_bytes());
        let imports = [module(slot_rva, "CreateFileW")];
        let read = stub_reader(stub_va, &stub);
        let Some(NativeCallImportResolution::Unique { import, target }) =
            resolve_import_thunk(&imports, ImageBase(IMAGE_BASE), read, stub_va)
        else {
            panic!("stub into an IAT slot must resolve");
        };
        assert_eq!(target, stub_va);
        assert_eq!(import.name.as_deref(), Some("CreateFileW"));
        assert_eq!(
            import.match_kinds,
            vec![NativeCallImportMatchKind::ImportThunk]
        );
    }

    #[test]
    fn stub_not_into_an_iat_slot_or_not_a_jmp_stays_unresolved() {
        let stub_va = IMAGE_BASE + 0x1100;
        let disp = (0x2040_i64 - 0x1100 - 6) as i32;
        let mut stub = vec![0xFF, 0x25];
        stub.extend_from_slice(&disp.to_le_bytes());
        let other_slot = [module(0x2048, "Other")];
        assert!(
            resolve_import_thunk(
                &other_slot,
                ImageBase(IMAGE_BASE),
                stub_reader(stub_va, &stub),
                stub_va
            )
            .is_none()
        );
        let slot = [module(0x2040, "CreateFileW")];
        let mut call = stub.clone();
        call[1] = 0x15;
        assert!(
            resolve_import_thunk(
                &slot,
                ImageBase(IMAGE_BASE),
                stub_reader(stub_va, &call),
                stub_va
            )
            .is_none()
        );
        assert!(resolve_import_thunk(&slot, ImageBase(IMAGE_BASE), |_| None, stub_va).is_none());
    }
}
