//! Raw declarations matching `native/vtil-bridge/include/vmpx_vtil.h`.

use std::os::raw::c_char;

use crate::Stats;

#[repr(C)]
pub struct RawRoutine {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn vmpx_vtil_revision() -> *const c_char;
    pub fn vmpx_vtil_create_amd64(
        entry_vip: u64,
        output: *mut *mut RawRoutine,
        error: *mut c_char,
        error_capacity: usize,
    ) -> u32;
    pub fn vmpx_vtil_destroy(routine: *mut RawRoutine);
    pub fn vmpx_vtil_alloc_register(
        routine: *mut RawRoutine,
        bit_width: u16,
        register_id: *mut u64,
        error: *mut c_char,
        error_capacity: usize,
    ) -> u32;
    pub fn vmpx_vtil_emit_mov_imm(
        routine: *mut RawRoutine,
        register_id: u64,
        value: u64,
        error: *mut c_char,
        error_capacity: usize,
    ) -> u32;
    pub fn vmpx_vtil_emit_add_imm(
        routine: *mut RawRoutine,
        register_id: u64,
        value: u64,
        error: *mut c_char,
        error_capacity: usize,
    ) -> u32;
    pub fn vmpx_vtil_finish(
        routine: *mut RawRoutine,
        output_register_id: u64,
        error: *mut c_char,
        error_capacity: usize,
    ) -> u32;
    pub fn vmpx_vtil_optimize(
        routine: *mut RawRoutine,
        error: *mut c_char,
        error_capacity: usize,
    ) -> u32;
    pub fn vmpx_vtil_get_stats(
        routine: *const RawRoutine,
        stats: *mut Stats,
        error: *mut c_char,
        error_capacity: usize,
    ) -> u32;
    pub fn vmpx_vtil_serialize(
        routine: *const RawRoutine,
        output: *mut u8,
        output_capacity: usize,
        required_size: *mut usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> u32;
}
