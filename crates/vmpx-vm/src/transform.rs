use serde::Serialize;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BitWidth {
    W8,
    W16,
    W32,
    W64,
}

impl BitWidth {
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::W8 => 8,
            Self::W16 => 16,
            Self::W32 => 32,
            Self::W64 => 64,
        }
    }

    #[must_use]
    pub const fn bytes(self) -> u32 {
        self.bits() / 8
    }

    #[must_use]
    pub const fn from_bytes(bytes: u32) -> Option<Self> {
        match bytes {
            1 => Some(Self::W8),
            2 => Some(Self::W16),
            4 => Some(Self::W32),
            8 => Some(Self::W64),
            _ => None,
        }
    }

    #[must_use]
    pub const fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            8 => Some(Self::W8),
            16 => Some(Self::W16),
            32 => Some(Self::W32),
            64 => Some(Self::W64),
            _ => None,
        }
    }

    #[must_use]
    pub const fn mask(self) -> u64 {
        match self {
            Self::W8 => u8::MAX as u64,
            Self::W16 => u16::MAX as u64,
            Self::W32 => u32::MAX as u64,
            Self::W64 => u64::MAX,
        }
    }

    #[must_use]
    pub const fn truncate(self, value: u64) -> u64 {
        value & self.mask()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformOp {
    Add,
    Sub,
    Xor,
    Not,
    Neg,
    Rol,
    Ror,
    Bswap,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct Transform {
    pub op: TransformOp,
    pub width: BitWidth,
    pub operand: u64,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum TransformError {
    #[error("BSWAP is supported only at 16, 32, or 64 bits")]
    UnsupportedBswapWidth,
}

impl Transform {
    /// Applies the width-truncated reversible transformation.
    ///
    /// # Errors
    ///
    /// `BSWAP` at eight bits is rejected rather than silently treated as a
    /// different operation.
    pub fn apply(self, value: u64) -> Result<u64, TransformError> {
        let value = self.width.truncate(value);
        let operand = self.width.truncate(self.operand);
        let bits = self.width.bits();
        let result = match self.op {
            TransformOp::Add => value.wrapping_add(operand),
            TransformOp::Sub => value.wrapping_sub(operand),
            TransformOp::Xor => value ^ operand,
            TransformOp::Not => !value,
            TransformOp::Neg => value.wrapping_neg(),
            TransformOp::Rol => {
                let count = u32::try_from(operand % u64::from(bits)).unwrap_or(0);
                rotate_left_width(value, count, self.width)
            }
            TransformOp::Ror => {
                let count = u32::try_from(operand % u64::from(bits)).unwrap_or(0);
                rotate_right_width(value, count, self.width)
            }
            TransformOp::Bswap => match self.width {
                BitWidth::W8 => return Err(TransformError::UnsupportedBswapWidth),
                BitWidth::W16 => u64::from(u16::try_from(value).unwrap_or(0).swap_bytes()),
                BitWidth::W32 => u64::from(u32::try_from(value).unwrap_or(0).swap_bytes()),
                BitWidth::W64 => value.swap_bytes(),
            },
        };
        Ok(self.width.truncate(result))
    }

    #[must_use]
    pub const fn inverse(self) -> Self {
        let op = match self.op {
            TransformOp::Add => TransformOp::Sub,
            TransformOp::Sub => TransformOp::Add,
            TransformOp::Rol => TransformOp::Ror,
            TransformOp::Ror => TransformOp::Rol,
            TransformOp::Xor => TransformOp::Xor,
            TransformOp::Not => TransformOp::Not,
            TransformOp::Neg => TransformOp::Neg,
            TransformOp::Bswap => TransformOp::Bswap,
        };
        Self { op, ..self }
    }
}

fn rotate_left_width(value: u64, count: u32, width: BitWidth) -> u64 {
    match width {
        BitWidth::W8 => u64::from(u8::try_from(value).unwrap_or(0).rotate_left(count)),
        BitWidth::W16 => u64::from(u16::try_from(value).unwrap_or(0).rotate_left(count)),
        BitWidth::W32 => u64::from(u32::try_from(value).unwrap_or(0).rotate_left(count)),
        BitWidth::W64 => value.rotate_left(count),
    }
}

fn rotate_right_width(value: u64, count: u32, width: BitWidth) -> u64 {
    match width {
        BitWidth::W8 => u64::from(u8::try_from(value).unwrap_or(0).rotate_right(count)),
        BitWidth::W16 => u64::from(u16::try_from(value).unwrap_or(0).rotate_right(count)),
        BitWidth::W32 => u64::from(u32::try_from(value).unwrap_or(0).rotate_right(count)),
        BitWidth::W64 => value.rotate_right(count),
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct TransformChain {
    pub transforms: Vec<Transform>,
}

impl TransformChain {
    /// Applies transforms from first to last.
    ///
    /// # Errors
    ///
    /// Propagates invalid width/operation combinations from a transform.
    pub fn apply(&self, mut value: u64) -> Result<u64, TransformError> {
        for transform in &self.transforms {
            value = transform.apply(value)?;
        }
        Ok(value)
    }

    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            transforms: self
                .transforms
                .iter()
                .rev()
                .map(|transform| transform.inverse())
                .collect(),
        }
    }

    pub fn simplify(&mut self) {
        let mut simplified: Vec<Transform> = Vec::with_capacity(self.transforms.len());
        for transform in self.transforms.drain(..) {
            if simplified
                .last()
                .is_some_and(|previous| previous.inverse() == transform)
            {
                simplified.pop();
            } else {
                simplified.push(transform);
            }
        }
        self.transforms = simplified;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    #[test]
    fn inverse_round_trip_for_randomized_values_and_widths() {
        let mut state = 0xA55A_1122_3344_7788;
        for width in [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64] {
            for op in [
                TransformOp::Add,
                TransformOp::Sub,
                TransformOp::Xor,
                TransformOp::Not,
                TransformOp::Neg,
                TransformOp::Rol,
                TransformOp::Ror,
            ] {
                for _ in 0..1_000 {
                    let value = pseudo_random(&mut state);
                    let transform = Transform {
                        op,
                        width,
                        operand: pseudo_random(&mut state),
                    };
                    let encoded = transform.apply(value).expect("supported transform");
                    let decoded = transform
                        .inverse()
                        .apply(encoded)
                        .expect("supported inverse");
                    assert_eq!(decoded, width.truncate(value));
                }
            }
        }
    }

    #[test]
    fn bswap_round_trip_for_supported_widths() {
        for width in [BitWidth::W16, BitWidth::W32, BitWidth::W64] {
            let transform = Transform {
                op: TransformOp::Bswap,
                width,
                operand: 0,
            };
            let value = width.truncate(0x0123_4567_89AB_CDEF);
            assert_eq!(
                transform
                    .inverse()
                    .apply(transform.apply(value).expect("bswap"))
                    .expect("inverse"),
                value
            );
        }
    }
}
