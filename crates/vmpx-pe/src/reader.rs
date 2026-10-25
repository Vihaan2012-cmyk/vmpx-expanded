use crate::PeError;

#[derive(Clone, Copy)]
pub(crate) struct Reader<'a> {
    data: &'a [u8],
}

impl<'a> Reader<'a> {
    pub(crate) const fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    pub(crate) const fn len(self) -> usize {
        self.data.len()
    }

    pub(crate) fn bytes(self, offset: usize, size: usize, what: &str) -> Result<&'a [u8], PeError> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| PeError::parse(offset, format!("{what} range overflows")))?;
        self.data
            .get(offset..end)
            .ok_or_else(|| PeError::parse(offset, format!("{what} exceeds file bounds")))
    }

    pub(crate) fn u16(self, offset: usize, what: &str) -> Result<u16, PeError> {
        let bytes = self.bytes(offset, 2, what)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(crate) fn u32(self, offset: usize, what: &str) -> Result<u32, PeError> {
        let bytes = self.bytes(offset, 4, what)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(crate) fn u64(self, offset: usize, what: &str) -> Result<u64, PeError> {
        let bytes = self.bytes(offset, 8, what)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    pub(crate) fn c_string(
        self,
        offset: usize,
        max_len: usize,
        what: &str,
    ) -> Result<String, PeError> {
        let tail = self.bytes(offset, 1, what)?;
        let available = self.len().saturating_sub(offset).min(max_len);
        let bounded = &self.data[offset..offset + available];
        let end = bounded
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| PeError::parse(offset, format!("unterminated {what}")))?;
        let _ = tail;
        Ok(String::from_utf8_lossy(&bounded[..end]).into_owned())
    }
}
