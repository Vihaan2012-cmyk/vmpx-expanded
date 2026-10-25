use std::{
    collections::HashMap,
    io::{self, BufRead, Read, Write},
};

use serde::Serialize;
use thiserror::Error;
use vmpx_pe::{FileOffset, PeError, PeImage, Rva, Section};

const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
const DEFAULT_LZMA_PROPERTIES: [u8; 5] = [0x5D, 0x00, 0x00, 0x00, 0x04];
const MAX_PACKED_SECTIONS: usize = 256;
const MAX_TABLE_CANDIDATES: usize = 16;

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
    LegacyPlaintext,
    EncodedRotatingKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PackerEntry {
    pub source_rva: Rva,
    pub destination_rva: Rva,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PackerTable {
    pub kind: PackerLayoutKind,
    pub file_offset: FileOffset,
    pub destination_key: Option<u32>,
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

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    let bytes = data.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

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

fn lzma_source_offset(pe: &PeImage<'_>, rva: Rva) -> Option<usize> {
    let raw = usize::try_from(pe.rva_to_file_offset(rva).ok()?.0).ok()?;
    (pe.data().get(raw) == Some(&0)).then_some(raw)
}

fn validate_sources(pe: &PeImage<'_>, data: &[u8], table_offset: usize, count: usize) -> bool {
    for index in 0..count {
        let Some(entry_offset) = index
            .checked_mul(8)
            .and_then(|delta| table_offset.checked_add(delta))
        else {
            return false;
        };
        let Some(source) = read_u32(data, entry_offset) else {
            return false;
        };
        if lzma_source_offset(pe, Rva(source)).is_none() {
            return false;
        }
    }
    true
}

fn find_legacy_tables(pe: &PeImage<'_>, destinations: &[Section]) -> Vec<PackerTable> {
    let data = pe.data();
    let Some(table_size) = destinations.len().checked_mul(8) else {
        return Vec::new();
    };
    if table_size > data.len() {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    for table_offset in 8..=data.len() - table_size {
        let destinations_match = destinations.iter().enumerate().all(|(index, section)| {
            read_u32(data, table_offset + index * 8 + 4) == Some(section.virtual_address.0)
        });
        if !destinations_match || !validate_sources(pe, data, table_offset, destinations.len()) {
            continue;
        }
        let Some(properties_rva) = read_u32(data, table_offset - 8) else {
            continue;
        };
        let Ok(properties_offset) = pe.rva_to_file_offset(Rva(properties_rva)) else {
            continue;
        };
        let Ok(properties_offset) = usize::try_from(properties_offset.0) else {
            continue;
        };
        let Some(properties) = data.get(properties_offset..properties_offset.saturating_add(5))
        else {
            continue;
        };
        let mut lzma_properties = [0_u8; 5];
        lzma_properties.copy_from_slice(properties);
        let entries = destinations
            .iter()
            .enumerate()
            .filter_map(|(index, section)| {
                Some(PackerEntry {
                    source_rva: Rva(read_u32(data, table_offset + index * 8)?),
                    destination_rva: section.virtual_address,
                })
            })
            .collect();
        candidates.push(PackerTable {
            kind: PackerLayoutKind::LegacyPlaintext,
            file_offset: FileOffset(u64::try_from(table_offset).unwrap_or(u64::MAX)),
            destination_key: None,
            lzma_properties,
            entries,
        });
        if candidates.len() >= MAX_TABLE_CANDIDATES {
            break;
        }
    }
    candidates
}

fn decode_destinations(
    stored: &[u32],
    key: u32,
    destination_indexes: &HashMap<u32, usize>,
) -> Option<Vec<Rva>> {
    let mut seen = vec![false; destination_indexes.len()];
    let mut decoded = Vec::with_capacity(stored.len());
    for (index, encoded) in stored.iter().copied().enumerate() {
        let rotation = u32::try_from((index * 7) & 31).ok()?;
        let destination = encoded ^ key.rotate_left(rotation);
        let destination_index = *destination_indexes.get(&destination)?;
        if seen[destination_index] {
            return None;
        }
        seen[destination_index] = true;
        decoded.push(Rva(destination));
    }
    seen.iter().all(|value| *value).then_some(decoded)
}

fn find_encoded_tables(pe: &PeImage<'_>, destinations: &[Section]) -> Vec<PackerTable> {
    let data = pe.data();
    let Some(table_size) = destinations.len().checked_mul(8) else {
        return Vec::new();
    };
    if table_size > data.len() {
        return Vec::new();
    }
    let destination_indexes: HashMap<_, _> = destinations
        .iter()
        .enumerate()
        .map(|(index, section)| (section.virtual_address.0, index))
        .collect();
    let mut candidates = Vec::new();
    let mut table_offset = 0_usize;
    while table_offset <= data.len() - table_size {
        // Two cheap source checks reject almost every byte position. The full
        // source pass is done only for the rare survivor.
        let first_source = read_u32(data, table_offset).map(Rva);
        let second_source = read_u32(data, table_offset + 8).map(Rva);
        if first_source
            .and_then(|rva| lzma_source_offset(pe, rva))
            .is_none()
            || second_source
                .and_then(|rva| lzma_source_offset(pe, rva))
                .is_none()
            || !validate_sources(pe, data, table_offset, destinations.len())
        {
            table_offset += 1;
            continue;
        }
        let stored: Vec<_> = (0..destinations.len())
            .filter_map(|index| read_u32(data, table_offset + index * 8 + 4))
            .collect();
        if stored.len() != destinations.len() {
            table_offset += 1;
            continue;
        }
        for seed in destinations {
            let key = stored[0] ^ seed.virtual_address.0;
            let Some(decoded) = decode_destinations(&stored, key, &destination_indexes) else {
                continue;
            };
            let entries = decoded
                .into_iter()
                .enumerate()
                .filter_map(|(index, destination_rva)| {
                    Some(PackerEntry {
                        source_rva: Rva(read_u32(data, table_offset + index * 8)?),
                        destination_rva,
                    })
                })
                .collect();
            candidates.push(PackerTable {
                kind: PackerLayoutKind::EncodedRotatingKey,
                file_offset: FileOffset(u64::try_from(table_offset).unwrap_or(u64::MAX)),
                destination_key: Some(key),
                lzma_properties: DEFAULT_LZMA_PROPERTIES,
                entries,
            });
            if candidates.len() >= MAX_TABLE_CANDIDATES {
                return candidates;
            }
        }
        table_offset += 1;
    }
    candidates
}

/// Finds a uniquely validated legacy or rotating-key `PACKER_INFO` table.
///
/// # Errors
///
/// Returns an error when destination sections are absent, resource limits are
/// exceeded, no table validates, or multiple candidates remain ambiguous.
pub fn locate_packer_table(
    pe: &PeImage<'_>,
    limits: UnpackLimits,
) -> Result<(Vec<Section>, PackerTable), UnpackError> {
    let destinations = packed_sections(pe, limits.max_packed_sections)?;
    let legacy = find_legacy_tables(pe, &destinations);
    if legacy.len() == 1 {
        return Ok((destinations, legacy[0].clone()));
    }
    if legacy.len() > 1 {
        return Err(UnpackError::Locate(format!(
            "{} legacy PACKER_INFO candidates are ambiguous",
            legacy.len()
        )));
    }
    let encoded = find_encoded_tables(pe, &destinations);
    match encoded.as_slice() {
        [table] => Ok((destinations, table.clone())),
        [] => Err(UnpackError::Locate(
            "no validated legacy or rotating-key PACKER_INFO table was found".to_owned(),
        )),
        _ => Err(UnpackError::Locate(format!(
            "{} encoded PACKER_INFO candidates are ambiguous",
            encoded.len()
        ))),
    }
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

struct HeaderPayloadReader<'a> {
    header: [u8; 13],
    payload: &'a [u8],
    position: usize,
}

impl<'a> HeaderPayloadReader<'a> {
    fn new(properties: [u8; 5], payload: &'a [u8]) -> Self {
        let mut header = [0_u8; 13];
        header[..5].copy_from_slice(&properties);
        header[5..].copy_from_slice(&u64::MAX.to_le_bytes());
        Self {
            header,
            payload,
            position: 0,
        }
    }

    fn payload_position(&self) -> usize {
        self.position.saturating_sub(self.header.len())
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

fn decompress_attempt(
    properties: [u8; 5],
    compressed: &[u8],
    destination: &mut [u8],
) -> (Result<(), String>, usize, usize) {
    let mut input = HeaderPayloadReader::new(properties, compressed);
    let mut output = BoundedWriter {
        destination,
        written: 0,
    };
    let options = lzma_rs::decompress::Options {
        unpacked_size: lzma_rs::decompress::UnpackedSize::ReadFromHeader,
        memlimit: Some(128 * 1024 * 1024),
        allow_incomplete: false,
    };
    let result = lzma_rs::lzma_decompress_with_options(&mut input, &mut output, &options)
        .map_err(|error| error.to_string());
    (result, input.payload_position(), output.written)
}

fn decompress_block(
    properties: [u8; 5],
    compressed: &[u8],
    destination: &mut [u8],
) -> Result<usize, String> {
    let (first_result, consumed, written) = decompress_attempt(properties, compressed, destination);
    let first_error = match first_result {
        Ok(()) => return Ok(written),
        Err(error) => error,
    };
    if consumed == 0 || consumed >= compressed.len() {
        return Err(first_error);
    }

    // Embedded streams are followed by unrelated PE bytes. lzma-rs correctly
    // rejects such trailing input for a standalone stream, so retry with the
    // exact amount consumed through the end marker. This is evidence-derived,
    // not a guessed compressed size.
    destination.fill(0);
    let (retry_result, _, retry_written) =
        decompress_attempt(properties, &compressed[..consumed], destination);
    retry_result.map(|()| retry_written).map_err(|retry_error| {
        format!(
            "initial decode failed ({first_error}); exact-consumption retry failed ({retry_error})"
        )
    })
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
    let (packed, table) = locate_packer_table(&pe, limits)?;
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
    let mut warnings = Vec::new();
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
        let destination_limit = pe
            .sections
            .iter()
            .filter_map(|candidate| {
                (candidate.virtual_address.0 > entry.destination_rva.0)
                    .then_some(candidate.virtual_address.0)
            })
            .min()
            .unwrap_or(pe.optional.size_of_image);
        let capacity_u32 = destination_limit
            .checked_sub(entry.destination_rva.0)
            .filter(|capacity| *capacity >= section.virtual_size)
            .ok_or_else(|| {
                UnpackError::Reconstruct(format!(
                    "block {block_index} destination span is smaller than section virtual size"
                ))
            })?;
        let capacity = usize::try_from(capacity_u32).map_err(|_| {
            UnpackError::Reconstruct(format!("block {block_index} capacity is too large"))
        })?;
        let destination_end = destination.checked_add(capacity).ok_or_else(|| {
            UnpackError::Reconstruct(format!("block {block_index} destination overflows"))
        })?;
        let destination_bytes = output
            .get_mut(destination..destination_end)
            .ok_or_else(|| {
                UnpackError::Reconstruct(format!("block {block_index} destination exceeds image"))
            })?;
        let written = decompress_block(table.lzma_properties, &input[source..], destination_bytes)
            .map_err(|message| UnpackError::Decompress {
                block: block_index,
                section: section.name.clone(),
                message,
            })?;
        if written < usize::try_from(section.virtual_size).unwrap_or(usize::MAX) {
            warnings.push(format!(
                "block {block_index} ({}) produced {written:#X} bytes for virtual size {:#X}",
                section.name, section.virtual_size
            ));
        }
        blocks.push(UnpackedBlock {
            section_index: section.index,
            section_name: section.name.clone(),
            source_rva: entry.source_rva,
            destination_rva: entry.destination_rva,
            section_virtual_size: section.virtual_size,
            destination_capacity: capacity_u32,
            decompressed_size: u32::try_from(written).map_err(|_| {
                UnpackError::Reconstruct(format!("block {block_index} output exceeds 32 bits"))
            })?,
        });
    }

    PeImage::parse(&output).map_err(|error| {
        UnpackError::Reconstruct(format!("mapped output failed in-memory reparse: {error}"))
    })?;
    let report = UnpackReport {
        schema_version: 1,
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

    #[test]
    fn rotating_key_destinations_require_a_permutation() {
        let destinations = [0x1000_u32, 0x5000, 0x9000];
        let key = 0xA1B2_C3D4_u32;
        let stored: Vec<_> = destinations
            .iter()
            .enumerate()
            .map(|(index, destination)| {
                let rotation = u32::try_from((index * 7) & 31).unwrap_or(0);
                destination ^ key.rotate_left(rotation)
            })
            .collect();
        let indexes = destinations
            .iter()
            .enumerate()
            .map(|(index, value)| (*value, index))
            .collect();
        assert_eq!(
            decode_destinations(&stored, key, &indexes),
            Some(destinations.into_iter().map(Rva).collect())
        );
        let duplicates = [stored[0], stored[0], stored[2]];
        assert_eq!(decode_destinations(&duplicates, key, &indexes), None);
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
