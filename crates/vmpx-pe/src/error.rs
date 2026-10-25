use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum PeError {
    #[error("PE parse error at file offset {offset:#X}: {message}")]
    Parse { offset: u64, message: String },

    #[error("unsupported PE feature at file offset {offset:#X}: {message}")]
    Unsupported { offset: u64, message: String },

    #[error("address translation failed: {message}")]
    Translation { message: String },
}

impl PeError {
    pub(crate) fn parse(offset: usize, message: impl Into<String>) -> Self {
        Self::Parse {
            offset: u64::try_from(offset).unwrap_or(u64::MAX),
            message: message.into(),
        }
    }

    pub(crate) fn translation(message: impl Into<String>) -> Self {
        Self::Translation {
            message: message.into(),
        }
    }
}
