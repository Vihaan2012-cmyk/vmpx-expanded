use core::fmt;

use serde::{Deserialize, Serialize};

macro_rules! address_type {
    ($name:ident, $inner:ty) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Default,
            Eq,
            Hash,
            Ord,
            PartialEq,
            PartialOrd,
            Serialize,
            Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub $inner);

        impl $name {
            #[must_use]
            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "0x{:X}", self.0)
            }
        }
    };
}

address_type!(Va, u64);
address_type!(Rva, u32);
address_type!(FileOffset, u64);
address_type!(ImageBase, u64);

impl ImageBase {
    #[must_use]
    pub fn checked_add_rva(self, rva: Rva) -> Option<Va> {
        self.0.checked_add(u64::from(rva.0)).map(Va)
    }

    #[must_use]
    pub fn checked_rva_of(self, va: Va) -> Option<Rva> {
        let delta = va.0.checked_sub(self.0)?;
        u32::try_from(delta).ok().map(Rva)
    }
}

impl Rva {
    #[must_use]
    pub fn checked_add(self, amount: u32) -> Option<Self> {
        self.0.checked_add(amount).map(Self)
    }
}

impl Va {
    #[must_use]
    pub fn checked_add(self, amount: u64) -> Option<Self> {
        self.0.checked_add(amount).map(Self)
    }

    #[must_use]
    pub fn checked_add_signed(self, amount: i64) -> Option<Self> {
        if amount >= 0 {
            self.checked_add(amount.cast_unsigned())
        } else {
            self.0.checked_sub(amount.unsigned_abs()).map(Self)
        }
    }
}

impl FileOffset {
    #[must_use]
    pub fn checked_add(self, amount: u64) -> Option<Self> {
        self.0.checked_add(amount).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_base_conversion_is_checked() {
        let base = ImageBase(0x1_4000_0000);
        let va = base.checked_add_rva(Rva(0x1234)).expect("valid VA");
        assert_eq!(va, Va(0x1_4000_1234));
        assert_eq!(base.checked_rva_of(va), Some(Rva(0x1234)));
        assert_eq!(base.checked_rva_of(Va(0x1000)), None);
        assert_eq!(ImageBase(u64::MAX).checked_add_rva(Rva(1)), None);
        assert_eq!(va.checked_add_signed(-0x34), Some(Va(0x1_4000_1200)));
        assert_eq!(Va(0).checked_add_signed(-1), None);
    }
}
