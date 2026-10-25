use vmpx_vm::{BitWidth, Transform, TransformChain, TransformOp};

fn decode_slot(encoded: u64, key: u64) -> u64 {
    TransformChain {
        transforms: vec![
            Transform {
                op: TransformOp::Xor,
                width: BitWidth::W8,
                operand: key,
            },
            Transform {
                op: TransformOp::Sub,
                width: BitWidth::W8,
                operand: 0x6D,
            },
            Transform {
                op: TransformOp::Rol,
                width: BitWidth::W8,
                operand: 1,
            },
            Transform {
                op: TransformOp::Add,
                width: BitWidth::W8,
                operand: 1,
            },
            Transform {
                op: TransformOp::Neg,
                width: BitWidth::W8,
                operand: 0,
            },
        ],
    }
    .apply(encoded)
    .expect("all transforms support eight bits")
}

fn decode_rel32(encoded: u64, key: u64) -> u64 {
    TransformChain {
        transforms: vec![
            Transform {
                op: TransformOp::Xor,
                width: BitWidth::W32,
                operand: key,
            },
            Transform {
                op: TransformOp::Sub,
                width: BitWidth::W32,
                operand: 1,
            },
            Transform {
                op: TransformOp::Rol,
                width: BitWidth::W32,
                operand: 1,
            },
            Transform {
                op: TransformOp::Sub,
                width: BitWidth::W32,
                operand: 1,
            },
            Transform {
                op: TransformOp::Ror,
                width: BitWidth::W32,
                operand: 1,
            },
        ],
    }
    .apply(encoded)
    .expect("all transforms support 32 bits")
}

fn update_low_byte(key: u64, value: u64) -> u64 {
    (key & !u64::from(u8::MAX)) | ((key ^ value) & u64::from(u8::MAX))
}

fn update_low_dword(key: u64, value: u64) -> u64 {
    (key & !u64::from(u32::MAX)) | ((key ^ value) & u64::from(u32::MAX))
}

#[test]
fn reproduces_ida_verified_p06_h0_h1_state_transitions() {
    // Sample-specific oracle values remain in this regression test and RE notes,
    // never in the generic decoder or classifier.
    let mut key = 0x3_8572_2D91;

    let h0_slot = decode_slot(0xD9, key);
    assert_eq!(h0_slot, 0x48);
    key = update_low_byte(key, h0_slot);
    assert_eq!(key, 0x3_8572_2DD9);
    let h0_rel = decode_rel32(0x0562_C6CB, key);
    assert_eq!(h0_rel, 0x0010_EB11);
    key = update_low_dword(key, h0_rel);
    assert_eq!(key, 0x3_8562_C6C8);

    let h1_slot = decode_slot(0xEC, key);
    assert_eq!(h1_slot, 0x90);
    key = update_low_byte(key, h1_slot);
    assert_eq!(key, 0x3_8562_C658);
    let h1_rel = decode_rel32(0x0560_C09F, key);
    assert_eq!(h1_rel, 0x0002_06C6);
    key = update_low_dword(key, h1_rel);
    assert_eq!(key, 0x3_8560_C09E);
}
