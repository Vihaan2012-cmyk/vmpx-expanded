#![cfg(feature = "vtil")]

use vmpx_vtil_sys::{RegisterId, Routine, Status, revision};

#[test]
fn optimizes_and_serializes() {
    assert_eq!(revision().len(), 40);
    let mut routine = Routine::new_amd64(0x1000).unwrap();
    let reg = routine.alloc_register(64).unwrap();
    routine.emit_mov_imm(reg, 1).unwrap();
    routine.emit_add_imm(reg, 2).unwrap();
    routine.emit_add_imm(reg, 3).unwrap();
    routine.finish(reg).unwrap();

    let before = routine.stats().unwrap();
    assert_eq!(before.blocks, 1);
    assert!(before.instructions >= 5);
    routine.optimize().unwrap();
    let after = routine.stats().unwrap();
    assert!(after.instructions > 0 && after.instructions <= before.instructions);
    assert!(!routine.serialize().unwrap().is_empty());
}

#[test]
fn reports_errors() {
    let mut routine = Routine::new_amd64(0x1000).unwrap();
    let err = routine.emit_mov_imm(RegisterId(7), 1).unwrap_err();
    assert_eq!(err.status, Status::InvalidArgument);
    assert!(routine.optimize().is_err());
}
