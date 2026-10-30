//! Static recovery of VMProtect-packed sections.
//!
//! # Real VMProtect 3.11 packed layout (reverse-engineered from genuine output)
//!
//! When "Pack the output file" is enabled, VMProtect strips the raw bytes of
//! every initialized section of the pre-pack image (`PointerToRawData == 0`,
//! `SizeOfRawData == 0`, `VirtualSize != 0`) and stores each section's bytes as
//! an independent **raw LZMA1 stream** inside the protector's own raw-data
//! section. The loader stub that places them is itself virtualized, so there is
//! **no plaintext descriptor table** (the earlier `PACKER_INFO` model — 8-byte
//! source/destination pairs, plaintext or rotating-key XOR — does not occur in
//! real output and has been removed).
//!
//! Each stream is a header-less LZMA1 stream with the standard VMProtect
//! properties `lc=3, lp=0, pb=2` and an end-of-stream marker. A stream's
//! decompressed length equals the destination section's `VirtualSize` rounded
//! up to `FileAlignment` (the original on-disk section size). Streams are not
//! stored in section order and carry no self-identifying metadata, so they are
//! located structurally: every zero-initialised offset in the protector's
//! raw-data sections is probed, and an offset is accepted as a stream for a
//! destination only when it decodes to *exactly* that destination's rounded
//! size and is immediately followed by a clean EOS marker (validated with three
//! independent `lzma-rs` decodes — see [`section_stream_matches`]).
//!
//! # Soundness
//!
//! A stream is bound to a destination only when its decompressed size matches
//! exactly. When several destinations share one rounded size, the located
//! streams for that size must number *exactly* as many as those destinations
//! (otherwise the file is rejected as ambiguous); they are then paired in
//! ascending file-offset / ascending-RVA order, which is the sole residual
//! assumption and is surfaced as a report warning. The rebuilt image must
//! reparse in memory, and (via `--conventional`) pass loader validation.
//! Any count/size/range/decode mismatch fails closed with a precise error.

#![allow(clippy::doc_markdown)]

use std::{
    collections::HashMap,
    io::{self, BufRead, Read, Write},
};

use serde::Serialize;
use thiserror::Error;
use vmpx_pe::{FileOffset, PeError, PeImage, Rva, Section};

const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
/// Standard VMProtect packer LZMA properties: `lc=3, lp=0, pb=2`, 64 MiB dict.
const VMPROTECT_LZMA_PROPERTIES: [u8; 5] = [0x5D, 0x00, 0x00, 0x00, 0x04];
const MAX_PACKED_SECTIONS: usize = 256;
/// Upper bound on located streams per distinct size before declaring ambiguity.
const MAX_STREAMS_PER_SIZE: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct UnpackLimits {
    pub max_image_size: usize,
    pub max_packed_sections: usize,
}

impl Default for UnpackLimits {
    fn default() -> Self {
        Self {
            max_image_size: 2 * 1024 * 1024 * 1024,
            max_packed_sections: MAX_PACKED_SECTIONS,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PackerLayoutKind {
    /// Real VMProtect 3.x layout: one header-less raw LZMA1 stream per section.
    RawLzma1Sections,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PackerEntry {
    pub source_rva: Rva,
    pub destination_rva: Rva,
    pub decompressed_size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PackerTable {
    pub kind: PackerLayoutKind,
    pub file_offset: FileOffset,
    pub lzma_properties: [u8; 5],
    pub entries: Vec<PackerEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct UnpackedBlock {
    pub section_index: usize,
    pub section_name: String,
    pub source_rva: Rva,
    pub destination_rva: Rva,
    pub section_virtual_size: u32,
    pub destination_capacity: u32,
    pub decompressed_size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct UnpackReport {
    pub schema_version: u32,
    pub input_size: u64,
    pub output_size: u64,
    pub packed_sections: usize,
    pub packer_table: PackerTable,
    pub blocks: Vec<UnpackedBlock>,
    pub in_memory_reparsed: bool,
    pub disk_reparsed: bool,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct UnpackResult {
    pub image: Vec<u8>,
    pub report: UnpackReport,
}

#[derive(Debug, Error)]
pub enum UnpackError {
    #[error("[pe] {0}")]
    Pe(#[from] PeError),

    #[error("[locate] {0}")]
    Locate(String),

    #[error("[decompress] block {block} ({section}): {message}")]
    Decompress {
        block: usize,
        section: String,
        message: String,
    },

    #[error("[reconstruct] {0}")]
    Reconstruct(String),
}

fn round_up(value: u32, align: u32) -> Option<u32> {
    if align <= 1 {
        return Some(value);
    }
    let blocks = value.checked_add(align - 1)? / align;
    blocks.checked_mul(align)
}

fn lzma_source_offset(pe: &PeImage<'_>, rva: Rva) -> Option<usize> {
    let raw = usize::try_from(pe.rva_to_file_offset(rva).ok()?.0).ok()?;
    (pe.data().get(raw) == Some(&0)).then_some(raw)
}

struct BoundedWriter<'a> {
    destination: &'a mut [u8],
    written: usize,
}

impl Write for BoundedWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let remaining = self.destination.len().saturating_sub(self.written);
        if buffer.len() > remaining {
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                "decompressed block exceeds destination section",
            ));
        }
        let end = self.written + buffer.len();
        self.destination[self.written..end].copy_from_slice(buffer);
        self.written = end;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A capped sink that counts bytes and refuses to exceed `cap`.
struct CountingWriter {
    written: usize,
    cap: usize,
}

impl Write for CountingWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.written + buffer.len() > self.cap {
            return Err(io::Error::new(io::ErrorKind::FileTooLarge, "cap exceeded"));
        }
        self.written += buffer.len();
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Presents a synthesized 13-byte LZMA header (`properties` + little-endian
/// `declared_size`) followed by `payload`, without copying the payload.
struct HeaderPayloadReader<'a> {
    header: [u8; 13],
    payload: &'a [u8],
    position: usize,
}

impl<'a> HeaderPayloadReader<'a> {
    fn new(properties: [u8; 5], declared_size: u64, payload: &'a [u8]) -> Self {
        let mut header = [0_u8; 13];
        header[..5].copy_from_slice(&properties);
        header[5..].copy_from_slice(&declared_size.to_le_bytes());
        Self {
            header,
            payload,
            position: 0,
        }
    }
}

impl Read for HeaderPayloadReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let count = available.len().min(buffer.len());
        buffer[..count].copy_from_slice(&available[..count]);
        self.consume(count);
        Ok(count)
    }
}

impl BufRead for HeaderPayloadReader<'_> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.position < self.header.len() {
            return Ok(&self.header[self.position..]);
        }
        let payload_position = self.position - self.header.len();
        Ok(self.payload.get(payload_position..).unwrap_or_default())
    }

    fn consume(&mut self, amount: usize) {
        let total = self.header.len().saturating_add(self.payload.len());
        self.position = self.position.saturating_add(amount).min(total);
    }
}

fn decode_into<W: Write>(
    properties: [u8; 5],
    declared_size: u64,
    payload: &[u8],
    output: &mut W,
) -> Result<(), String> {
    let mut input = HeaderPayloadReader::new(properties, declared_size, payload);
    let options = lzma_rs::decompress::Options {
        unpacked_size: lzma_rs::decompress::UnpackedSize::ReadFromHeader,
        memlimit: Some(256 * 1024 * 1024),
        allow_incomplete: false,
    };
    lzma_rs::lzma_decompress_with_options(&mut input, output, &options).map_err(|e| e.to_string())
}

/// Decode exactly `size` bytes of a header-less raw LZMA1 stream at `payload`
/// into a counting sink; succeeds only when exactly `size` bytes are produced.
fn decode_counted(properties: [u8; 5], size: usize, payload: &[u8]) -> Result<(), String> {
    let mut sink = CountingWriter {
        written: 0,
        cap: size,
    };
    decode_into(properties, size as u64, payload, &mut sink)?;
    (sink.written == size)
        .then_some(())
        .ok_or_else(|| format!("produced {:#X} bytes, expected {size:#X}", sink.written))
}

/// Structural test: does `payload` begin a valid raw LZMA1 stream that
/// decompresses to *exactly* `size` bytes and ends on a clean EOS marker?
///
/// Three independent `lzma-rs` decodes make this sound against `lzma-rs`'s
/// lenient range decoder:
/// 1. exact-size decode yields exactly `size` bytes (so `len >= size`);
/// 2. a `size + 1` decode fails (the EOS marker sits exactly at `size`, so
///    `len <= size`);
/// 3. an unknown-size decode trips over trailing file bytes *after* a valid EOS
///    (`"end-of-stream marker"`), rejecting mis-decodes that merely tolerate
///    `size` garbage bytes before faulting on an LZ distance.
fn section_stream_matches(properties: [u8; 5], payload: &[u8], size: usize) -> bool {
    if payload.first() != Some(&0) {
        return false;
    }
    if decode_counted(properties, size, payload).is_err() {
        return false;
    }
    if decode_counted(properties, size + 1, payload).is_ok() {
        return false;
    }
    let mut sink = CountingWriter {
        written: 0,
        cap: size + 64,
    };
    matches!(
        decode_into(properties, u64::MAX, payload, &mut sink),
        Err(message) if message.contains("end-of-stream marker")
    )
}

/// Virtual-only initialized sections: the decompression destinations.
fn packed_sections(pe: &PeImage<'_>, limit: usize) -> Result<Vec<Section>, UnpackError> {
    let sections: Vec<_> = pe
        .sections
        .iter()
        .filter(|section| {
            section.pointer_to_raw_data.0 == 0
                && section.size_of_raw_data == 0
                && section.virtual_size != 0
                && section.characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA == 0
        })
        .cloned()
        .collect();
    if sections.is_empty() {
        return Err(UnpackError::Locate(
            "no initialized virtual-only destination sections were found".to_owned(),
        ));
    }
    if sections.len() > limit {
        return Err(UnpackError::Locate(format!(
            "{} packed-section candidates exceed limit {limit}",
            sections.len()
        )));
    }
    Ok(sections)
}

/// File ranges of the protector's raw-data sections (where streams live).
fn raw_data_ranges(pe: &PeImage<'_>) -> Vec<(usize, usize)> {
    let len = pe.data().len();
    pe.sections
        .iter()
        .filter(|s| s.pointer_to_raw_data.0 != 0 && s.size_of_raw_data != 0)
        .filter_map(|s| {
            let start = usize::try_from(s.pointer_to_raw_data.0).ok()?;
            let end = start
                .checked_add(usize::try_from(s.size_of_raw_data).ok()?)?
                .min(len);
            (start < end).then_some((start, end))
        })
        .collect()
}

/// Capacity (bytes) available at `section` before the next section / image end.
fn destination_capacity(pe: &PeImage<'_>, section: &Section) -> Option<u32> {
    let next = pe
        .sections
        .iter()
        .filter_map(|c| (c.virtual_address.0 > section.virtual_address.0).then_some(c.virtual_address.0))
        .min()
        .unwrap_or(pe.optional.size_of_image);
    next.checked_sub(section.virtual_address.0)
}

/// Locates the per-section raw LZMA1 streams and binds each to its destination.
///
/// # Errors
///
/// Returns an error when destinations are absent, a destination's rounded size
/// does not fit its capacity, the number of located streams for any size does
/// not match the number of same-sized destinations, or limits are exceeded.
#[allow(clippy::too_many_lines)]
pub fn locate_packer_table(
    pe: &PeImage<'_>,
    limits: UnpackLimits,
) -> Result<(Vec<Section>, PackerTable, Vec<String>), UnpackError> {
    let destinations = packed_sections(pe, limits.max_packed_sections)?;
    let file_alignment = pe.optional.file_alignment;

    // Each destination's expected decompressed size = VirtualSize rounded up to
    // FileAlignment, which must fit the section's capacity.
    let mut targets = Vec::with_capacity(destinations.len());
    for section in &destinations {
        let size = round_up(section.virtual_size, file_alignment).ok_or_else(|| {
            UnpackError::Locate(format!("section {} rounded size overflows", section.name))
        })?;
        let capacity = destination_capacity(pe, section).ok_or_else(|| {
            UnpackError::Locate(format!("section {} capacity underflows", section.name))
        })?;
        if size > capacity {
            return Err(UnpackError::Locate(format!(
                "section {} rounded size {size:#X} exceeds capacity {capacity:#X}",
                section.name
            )));
        }
        if usize::try_from(size).unwrap_or(usize::MAX) > limits.max_image_size {
            return Err(UnpackError::Locate(format!(
                "section {} rounded size {size:#X} exceeds image limit",
                section.name
            )));
        }
        targets.push(size);
    }

    let ranges = raw_data_ranges(pe);
    let data = pe.data();
    let mut distinct: Vec<u32> = targets.clone();
    distinct.sort_unstable();
    distinct.dedup();

    // Structurally locate every stream of each distinct target size.
    let mut candidates: HashMap<u32, Vec<usize>> = HashMap::new();
    for &size in &distinct {
        let size_usize = usize::try_from(size).unwrap_or(usize::MAX);
        let mut offsets = Vec::new();
        'ranges: for &(start, end) in &ranges {
            for offset in start..end {
                if data[offset] != 0 {
                    continue;
                }
                if section_stream_matches(VMPROTECT_LZMA_PROPERTIES, &data[offset..], size_usize) {
                    offsets.push(offset);
                    if offsets.len() > MAX_STREAMS_PER_SIZE {
                        break 'ranges;
                    }
                }
            }
        }
        candidates.insert(size, offsets);
    }

    // Bind streams to destinations: exact per-size count match (fail closed on
    // ambiguity), paired in ascending file-offset / ascending-RVA order.
    let mut warnings = Vec::new();
    for &size in &distinct {
        let dest_count = targets.iter().filter(|&&t| t == size).count();
        let found = candidates.get(&size).map_or(0, Vec::len);
        if found != dest_count {
            return Err(UnpackError::Locate(format!(
                "{found} raw LZMA1 stream(s) of decompressed size {size:#X} found for \
                 {dest_count} destination section(s) of that size - layout is ambiguous"
            )));
        }
        if dest_count > 1 {
            let names: Vec<_> = destinations
                .iter()
                .zip(&targets)
                .filter(|(_, t)| **t == size)
                .map(|(s, _)| s.name.clone())
                .collect();
            warnings.push(format!(
                "sections {names:?} share decompressed size {size:#X}; streams assigned by \
                 ascending file offset to ascending section RVA"
            ));
        }
    }
    for offsets in candidates.values_mut() {
        offsets.sort_unstable();
    }

    // Destinations are already in section (ascending-RVA) order; assign each the
    // next unused stream of its size.
    let mut cursor: HashMap<u32, usize> = HashMap::new();
    let mut entries = Vec::with_capacity(destinations.len());
    for (section, &size) in destinations.iter().zip(&targets) {
        let index = cursor.entry(size).or_insert(0);
        let offset = candidates[&size][*index];
        *index += 1;
        let source_rva = pe
            .file_offset_to_rva(FileOffset(u64::try_from(offset).unwrap_or(u64::MAX)))
            .map_err(UnpackError::from)?;
        entries.push(PackerEntry {
            source_rva,
            destination_rva: section.virtual_address,
            decompressed_size: size,
        });
    }

    let file_offset = entries
        .iter()
        .filter_map(|e| pe.rva_to_file_offset(e.source_rva).ok())
        .min_by_key(|o| o.0)
        .unwrap_or(FileOffset(0));

    let table = PackerTable {
        kind: PackerLayoutKind::RawLzma1Sections,
        file_offset,
        lzma_properties: VMPROTECT_LZMA_PROPERTIES,
        entries,
    };
    Ok((destinations, table, warnings))
}

/// Restores packed initialized sections into an explicit RVA-layout image.
///
/// This output is an analysis artifact, not a conventional rebuilt PE.
///
/// # Errors
///
/// Returns an error for malformed PE structure, ambiguous packer metadata,
/// resource-limit violations, invalid LZMA streams, or overlapping/out-of-range
/// reconstruction data.
#[allow(clippy::too_many_lines)]
pub fn unpack_to_mapped_image(
    input: &[u8],
    limits: UnpackLimits,
) -> Result<UnpackResult, UnpackError> {
    let pe = PeImage::parse(input)?;
    let image_size = usize::try_from(pe.optional.size_of_image)
        .map_err(|_| UnpackError::Reconstruct("SizeOfImage cannot be represented".to_owned()))?;
    if image_size == 0 || image_size > limits.max_image_size {
        return Err(UnpackError::Reconstruct(format!(
            "SizeOfImage {image_size:#X} exceeds configured limit {:#X}",
            limits.max_image_size
        )));
    }
    let header_size = usize::try_from(pe.optional.size_of_headers)
        .map_err(|_| UnpackError::Reconstruct("SizeOfHeaders cannot be represented".to_owned()))?;
    if header_size > input.len() || header_size > image_size {
        return Err(UnpackError::Reconstruct(format!(
            "SizeOfHeaders {header_size:#X} exceeds input or output bounds"
        )));
    }
    let (packed, table, mut warnings) = locate_packer_table(&pe, limits)?;
    let mut output = vec![0_u8; image_size];
    output[..header_size].copy_from_slice(&input[..header_size]);

    for section in &pe.sections {
        if section.pointer_to_raw_data.0 == 0 || section.size_of_raw_data == 0 {
            continue;
        }
        let source = usize::try_from(section.pointer_to_raw_data.0).map_err(|_| {
            UnpackError::Reconstruct(format!("section {} raw offset is too large", section.name))
        })?;
        let size = usize::try_from(section.size_of_raw_data).map_err(|_| {
            UnpackError::Reconstruct(format!("section {} raw size is too large", section.name))
        })?;
        let destination = usize::try_from(section.virtual_address.0).map_err(|_| {
            UnpackError::Reconstruct(format!("section {} RVA is too large", section.name))
        })?;
        let source_end = source.checked_add(size).ok_or_else(|| {
            UnpackError::Reconstruct(format!("section {} source range overflows", section.name))
        })?;
        let destination_end = destination.checked_add(size).ok_or_else(|| {
            UnpackError::Reconstruct(format!(
                "section {} destination range overflows",
                section.name
            ))
        })?;
        let source_bytes = input.get(source..source_end).ok_or_else(|| {
            UnpackError::Reconstruct(format!(
                "section {} source range exceeds input",
                section.name
            ))
        })?;
        let destination_bytes = output
            .get_mut(destination..destination_end)
            .ok_or_else(|| {
                UnpackError::Reconstruct(format!(
                    "section {} destination range exceeds image",
                    section.name
                ))
            })?;
        destination_bytes.copy_from_slice(source_bytes);
    }

    let section_table = usize::try_from(pe.nt_headers_offset.0)
        .ok()
        .and_then(|offset| offset.checked_add(4 + 20))
        .and_then(|offset| offset.checked_add(usize::from(pe.coff.size_of_optional_header)))
        .ok_or_else(|| UnpackError::Reconstruct("section-table offset overflows".to_owned()))?;
    for section in &pe.sections {
        let section_header = section_table
            .checked_add(section.index.checked_mul(40).ok_or_else(|| {
                UnpackError::Reconstruct("section-header index overflows".to_owned())
            })?)
            .ok_or_else(|| {
                UnpackError::Reconstruct("section-header offset overflows".to_owned())
            })?;
        let raw_size_range = section_header + 16..section_header + 20;
        let raw_pointer_range = section_header + 20..section_header + 24;
        output
            .get_mut(raw_size_range)
            .ok_or_else(|| {
                UnpackError::Reconstruct("section raw-size field exceeds output".to_owned())
            })?
            .copy_from_slice(&section.virtual_size.to_le_bytes());
        output
            .get_mut(raw_pointer_range)
            .ok_or_else(|| {
                UnpackError::Reconstruct("section raw-pointer field exceeds output".to_owned())
            })?
            .copy_from_slice(&section.virtual_address.0.to_le_bytes());
    }

    let destinations_by_rva: HashMap<_, _> = packed
        .iter()
        .map(|section| (section.virtual_address.0, section))
        .collect();
    let mut blocks = Vec::with_capacity(table.entries.len());
    for (block_index, entry) in table.entries.iter().enumerate() {
        let section = destinations_by_rva
            .get(&entry.destination_rva.0)
            .ok_or_else(|| {
                UnpackError::Reconstruct(format!(
                    "block {block_index} destination {} has no section",
                    entry.destination_rva
                ))
            })?;
        let source =
            lzma_source_offset(&pe, entry.source_rva).ok_or_else(|| UnpackError::Decompress {
                block: block_index,
                section: section.name.clone(),
                message: format!("source RVA {} is not a raw LZMA stream", entry.source_rva),
            })?;
        let destination = usize::try_from(entry.destination_rva.0).map_err(|_| {
            UnpackError::Reconstruct(format!("block {block_index} destination is too large"))
        })?;
        let capacity_u32 = destination_capacity(&pe, section)
            .filter(|capacity| *capacity >= entry.decompressed_size)
            .ok_or_else(|| {
                UnpackError::Reconstruct(format!(
                    "block {block_index} decompressed size exceeds destination capacity"
                ))
            })?;
        let expected = usize::try_from(entry.decompressed_size).map_err(|_| {
            UnpackError::Reconstruct(format!("block {block_index} size is too large"))
        })?;
        let destination_end = destination.checked_add(expected).ok_or_else(|| {
            UnpackError::Reconstruct(format!("block {block_index} destination overflows"))
        })?;
        let destination_bytes = output
            .get_mut(destination..destination_end)
            .ok_or_else(|| {
                UnpackError::Reconstruct(format!("block {block_index} destination exceeds image"))
            })?;
        let mut writer = BoundedWriter {
            destination: destination_bytes,
            written: 0,
        };
        decode_into(
            table.lzma_properties,
            entry.decompressed_size.into(),
            &input[source..],
            &mut writer,
        )
        .map_err(|message| UnpackError::Decompress {
            block: block_index,
            section: section.name.clone(),
            message,
        })?;
        if writer.written != expected {
            return Err(UnpackError::Decompress {
                block: block_index,
                section: section.name.clone(),
                message: format!(
                    "decoded {:#X} bytes, expected {expected:#X}",
                    writer.written
                ),
            });
        }
        blocks.push(UnpackedBlock {
            section_index: section.index,
            section_name: section.name.clone(),
            source_rva: entry.source_rva,
            destination_rva: entry.destination_rva,
            section_virtual_size: section.virtual_size,
            destination_capacity: capacity_u32,
            decompressed_size: entry.decompressed_size,
        });
    }

    PeImage::parse(&output).map_err(|error| {
        UnpackError::Reconstruct(format!("mapped output failed in-memory reparse: {error}"))
    })?;
    warnings.sort();
    let report = UnpackReport {
        schema_version: 2,
        input_size: u64::try_from(input.len()).unwrap_or(u64::MAX),
        output_size: u64::try_from(output.len()).unwrap_or(u64::MAX),
        packed_sections: packed.len(),
        packer_table: table,
        blocks,
        in_memory_reparsed: true,
        disk_reparsed: false,
        warnings,
    };
    Ok(UnpackResult {
        image: output,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode `plain` as a header-less raw LZMA1 stream (strip the 13-byte
    /// `.lzma` header `lzma-rs` writes) — the exact shape VMProtect stores.
    fn raw_lzma1(plain: &[u8]) -> Vec<u8> {
        let mut framed = Vec::new();
        lzma_rs::lzma_compress(&mut io::Cursor::new(plain), &mut framed).expect("compress");
        framed.split_off(13)
    }

    #[test]
    fn matches_a_real_raw_lzma1_section_stream() {
        // A stream decoding to exactly `size` bytes, followed by trailing bytes,
        // is recognised; sizes off by one are not.
        let payload = vec![0xAB_u8; 0x200];
        let mut stream = raw_lzma1(&payload);
        stream.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]); // trailing PE bytes
        assert!(section_stream_matches(
            VMPROTECT_LZMA_PROPERTIES,
            &stream,
            0x200
        ));
        assert!(!section_stream_matches(
            VMPROTECT_LZMA_PROPERTIES,
            &stream,
            0x1FF
        ));
        assert!(!section_stream_matches(
            VMPROTECT_LZMA_PROPERTIES,
            &stream,
            0x201
        ));
    }

    #[test]
    fn decode_counted_round_trips_raw_lzma1() {
        let plain: Vec<u8> = (0..0x400_u32).map(|i| (i ^ (i >> 3)) as u8).collect();
        let stream = raw_lzma1(&plain);
        assert!(decode_counted(VMPROTECT_LZMA_PROPERTIES, plain.len(), &stream).is_ok());
    }

    #[test]
    fn rejects_non_stream_and_truncated_bytes() {
        // Arbitrary high-entropy bytes are not a clean EOS-terminated stream.
        let garbage: Vec<u8> = (0..0x400_u16).map(|i| (i.wrapping_mul(2654) >> 3) as u8).collect();
        assert!(!section_stream_matches(VMPROTECT_LZMA_PROPERTIES, &garbage, 0x200));

        // A valid stream with its tail removed must fail the exact-size decode.
        let payload = vec![0xCD_u8; 0x200];
        let mut stream = raw_lzma1(&payload);
        stream.truncate(stream.len() / 2);
        assert!(decode_counted(VMPROTECT_LZMA_PROPERTIES, 0x200, &stream).is_err());
    }

    #[test]
    fn round_up_matches_file_alignment() {
        assert_eq!(round_up(0x156, 0x200), Some(0x200));
        assert_eq!(round_up(0xAA, 0x200), Some(0x200));
        assert_eq!(round_up(0xE7_1BC, 0x200), Some(0xE7_200));
        assert_eq!(round_up(0x200, 0x200), Some(0x200));
        assert_eq!(round_up(0, 0x200), Some(0));
    }

    #[test]
    fn bounded_writer_refuses_overflow() {
        let mut destination = [0_u8; 3];
        let mut writer = BoundedWriter {
            destination: &mut destination,
            written: 0,
        };
        assert_eq!(writer.write(b"abc").expect("fits"), 3);
        assert!(writer.write(b"d").is_err());
        assert_eq!(&destination, b"abc");
    }
}
