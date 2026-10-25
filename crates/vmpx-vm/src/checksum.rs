use serde::Serialize;
use thiserror::Error;
use vmpx_pe::Va;

use crate::BitWidth;

/// Source used to initialize a table-driven checksum accumulator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChecksumAccumulatorSource {
    Constant { value: u32 },
    VmStack { offset: i32 },
}

/// Recovered table-index expression. The first supported recurrence is the
/// common byte-table form `(accumulator & 0xff) ^ input_byte`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChecksumTableIndex {
    AccumulatorLowByteXorInputByte,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChecksumShift {
    LogicalLeft,
    LogicalRight,
}

/// Recovered per-byte accumulator combination.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChecksumCombine {
    ShiftedAccumulatorXorTableEntry,
}

/// Exact recovered contract for a byte-at-a-time, table-driven checksum VM
/// operation. Final NOT is applied before `final_xor`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TableChecksumMetadata {
    pub checksum_width: BitWidth,
    pub accumulator_source: ChecksumAccumulatorSource,
    pub initial_xor: u32,
    pub pointer_stack_offset: i32,
    pub length_stack_offset: i32,
    pub length_width: BitWidth,
    pub pointer_stride: i8,
    pub table_address: Va,
    pub table_entries: u16,
    pub table_entry_width: BitWidth,
    pub table_index: ChecksumTableIndex,
    pub shift: ChecksumShift,
    pub shift_bits: u8,
    pub combine: ChecksumCombine,
    /// Constant XOR value applied after every table combination.
    pub iteration_xor: u32,
    pub final_not: bool,
    pub final_xor: u32,
    /// Result offset relative to VSP after `vsp_delta` has been applied.
    pub result_stack_offset: i32,
    pub vsp_delta: i32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "metadata", rename_all = "snake_case")]
pub enum VmSemanticMetadata {
    TableChecksum(TableChecksumMetadata),
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum TableChecksumMetadataError {
    #[error("table checksum width must be W32, got {0:?}")]
    ChecksumWidth(BitWidth),

    #[error("table checksum requires 256 table entries, got {0}")]
    TableEntries(u16),

    #[error("table checksum entry width must be W32, got {0:?}")]
    TableEntryWidth(BitWidth),

    #[error("table checksum shift count {0} is outside 0..32")]
    ShiftBits(u8),

    #[error("table checksum pointer stride cannot be zero")]
    ZeroPointerStride,
}

impl TableChecksumMetadata {
    /// Checks invariants required by the currently supported exact recurrence.
    ///
    /// # Errors
    ///
    /// Returns a structured error for a non-W32 accumulator/table, a table
    /// other than 256 entries, an invalid logical-shift count, or zero stride.
    pub fn validate(&self) -> Result<(), TableChecksumMetadataError> {
        if self.checksum_width != BitWidth::W32 {
            return Err(TableChecksumMetadataError::ChecksumWidth(
                self.checksum_width,
            ));
        }
        if self.table_entries != 256 {
            return Err(TableChecksumMetadataError::TableEntries(self.table_entries));
        }
        if self.table_entry_width != BitWidth::W32 {
            return Err(TableChecksumMetadataError::TableEntryWidth(
                self.table_entry_width,
            ));
        }
        if self.shift_bits >= 32 {
            return Err(TableChecksumMetadataError::ShiftBits(self.shift_bits));
        }
        if self.pointer_stride == 0 {
            return Err(TableChecksumMetadataError::ZeroPointerStride);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_metadata() -> TableChecksumMetadata {
        TableChecksumMetadata {
            checksum_width: BitWidth::W32,
            accumulator_source: ChecksumAccumulatorSource::Constant { value: 0 },
            initial_xor: u32::MAX,
            pointer_stack_offset: 0,
            length_stack_offset: 8,
            length_width: BitWidth::W64,
            pointer_stride: 1,
            table_address: Va(0x1_4000_8000),
            table_entries: 256,
            table_entry_width: BitWidth::W32,
            table_index: ChecksumTableIndex::AccumulatorLowByteXorInputByte,
            shift: ChecksumShift::LogicalRight,
            shift_bits: 8,
            combine: ChecksumCombine::ShiftedAccumulatorXorTableEntry,
            iteration_xor: 0,
            final_not: true,
            final_xor: 0,
            result_stack_offset: 0,
            vsp_delta: 12,
        }
    }

    #[test]
    fn validates_exact_supported_recurrence_shape() {
        assert_eq!(valid_metadata().validate(), Ok(()));

        let mut invalid = valid_metadata();
        invalid.table_entries = 255;
        assert_eq!(
            invalid.validate(),
            Err(TableChecksumMetadataError::TableEntries(255))
        );
    }
}
