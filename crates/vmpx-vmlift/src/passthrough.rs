//! Intrinsics lowered by re-emitting the native instruction they denote.
//!
//! The symbolic executor cannot compute `cpuid`, a `gs`-relative load, a double
//! shift or `popfq` statically, so it names each with an opaque intrinsic. A
//! replacement may emit the very same instruction instead of a value, which is
//! behaviour-preserving by identity. `cpuid` and `gs` loads read machine state,
//! so they are environment boundaries: correct by identity, not provable from the
//! expression alone, and reported separately for the differential check.

use vmpx_vm::BitWidth;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CpuidOutput {
    Eax,
    Ebx,
    Ecx,
    Edx,
}

impl CpuidOutput {
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Eax => 0,
            Self::Ebx => 1,
            Self::Ecx => 2,
            Self::Edx => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Passthrough {
    Cpuid(CpuidOutput),
    GsLoad,
    DoubleShift {
        left: bool,
        width: BitWidth,
        flags: bool,
    },
    PopFlags,
}

fn index_suffix(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

fn parse_width(text: &str) -> Option<BitWidth> {
    match text {
        "w8" => Some(BitWidth::W8),
        "w16" => Some(BitWidth::W16),
        "w32" => Some(BitWidth::W32),
        "w64" => Some(BitWidth::W64),
        _ => None,
    }
}

fn parse_double_shift(rest: &str, flags: bool) -> Option<Passthrough> {
    let mut parts = rest.split('_');
    let left = match parts.next()? {
        "left" => true,
        "right" => false,
        _ => return None,
    };
    let width = parse_width(parts.next()?)?;
    let index = parts.next()?;
    (index_suffix(index) && parts.next().is_none()).then_some(Passthrough::DoubleShift {
        left,
        width,
        flags,
    })
}

pub(crate) fn classify(name: &str) -> Option<Passthrough> {
    if let Some(rest) = name.strip_prefix("x86_cpuid_") {
        let (register, index) = rest.split_once('_')?;
        let output = match register {
            "eax" => CpuidOutput::Eax,
            "ebx" => CpuidOutput::Ebx,
            "ecx" => CpuidOutput::Ecx,
            "edx" => CpuidOutput::Edx,
            _ => return None,
        };
        return index_suffix(index).then_some(Passthrough::Cpuid(output));
    }
    if let Some(index) = name.strip_prefix("x86_gs_load_") {
        return index_suffix(index).then_some(Passthrough::GsLoad);
    }
    if let Some(index) = name.strip_prefix("x86_popfq_") {
        return index_suffix(index).then_some(Passthrough::PopFlags);
    }
    if let Some(rest) = name.strip_prefix("x86_double_shift_") {
        return parse_double_shift(rest, false);
    }
    if let Some(rest) = name.strip_prefix("vm_flags_double_shift_") {
        return parse_double_shift(rest, true);
    }
    None
}

/// True for an intrinsic the lowering re-emits as a native instruction.
#[must_use]
pub fn is_native_passthrough(name: &str) -> bool {
    classify(name).is_some()
}

/// True for a passthrough that reads machine state (`cpuid`, `gs`-relative
/// loads). Such a value is correct by identity re-emission but is a trusted
/// boundary, not something a static value-equivalence proof can establish.
#[must_use]
pub fn is_environment_boundary(name: &str) -> bool {
    matches!(
        classify(name),
        Some(Passthrough::Cpuid(_) | Passthrough::GsLoad)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_exactly_the_five_executor_names() {
        assert_eq!(
            classify("x86_cpuid_ebx_12"),
            Some(Passthrough::Cpuid(CpuidOutput::Ebx))
        );
        assert_eq!(classify("x86_gs_load_3"), Some(Passthrough::GsLoad));
        assert_eq!(classify("x86_popfq_0"), Some(Passthrough::PopFlags));
        assert_eq!(
            classify("x86_double_shift_right_w32_4"),
            Some(Passthrough::DoubleShift {
                left: false,
                width: BitWidth::W32,
                flags: false
            })
        );
        assert_eq!(
            classify("vm_flags_double_shift_left_w64_9"),
            Some(Passthrough::DoubleShift {
                left: true,
                width: BitWidth::W64,
                flags: true
            })
        );
        for rejected in [
            "x86_cpuid_esi_1",
            "x86_cpuid_eax_",
            "x86_gs_load_x",
            "x86_popfq",
            "x86_double_shift_up_w32_1",
            "x86_double_shift_left_w32_1_2",
            "vm_flags_add_w32",
            "read_timestamp_counter_1",
        ] {
            assert_eq!(classify(rejected), None, "{rejected}");
        }
        assert!(is_environment_boundary("x86_gs_load_1"));
        assert!(!is_environment_boundary("x86_popfq_1"));
    }
}
