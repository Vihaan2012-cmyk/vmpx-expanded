#ifndef VMPX_VTIL_H
#define VMPX_VTIL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vmpx_vtil_routine vmpx_vtil_routine;

typedef enum vmpx_vtil_status {
    VMPX_VTIL_OK = 0,
    VMPX_VTIL_INVALID_ARGUMENT = 1,
    VMPX_VTIL_INVALID_STATE = 2,
    VMPX_VTIL_BUFFER_TOO_SMALL = 3,
    VMPX_VTIL_INTERNAL_ERROR = 4
} vmpx_vtil_status;

typedef struct vmpx_vtil_stats {
    uint64_t blocks;
    uint64_t instructions;
    uint64_t branches;
    uint64_t local_optimizations;
} vmpx_vtil_stats;

const char* vmpx_vtil_revision(void);

vmpx_vtil_status vmpx_vtil_create_amd64(
    uint64_t entry_vip,
    vmpx_vtil_routine** output,
    char* error,
    size_t error_capacity
);

void vmpx_vtil_destroy(vmpx_vtil_routine* routine);

vmpx_vtil_status vmpx_vtil_alloc_register(
    vmpx_vtil_routine* routine,
    uint16_t bit_width,
    uint64_t* register_id,
    char* error,
    size_t error_capacity
);

vmpx_vtil_status vmpx_vtil_emit_mov_imm(
    vmpx_vtil_routine* routine,
    uint64_t register_id,
    uint64_t value,
    char* error,
    size_t error_capacity
);

vmpx_vtil_status vmpx_vtil_emit_add_imm(
    vmpx_vtil_routine* routine,
    uint64_t register_id,
    uint64_t value,
    char* error,
    size_t error_capacity
);

vmpx_vtil_status vmpx_vtil_finish(
    vmpx_vtil_routine* routine,
    uint64_t output_register_id,
    char* error,
    size_t error_capacity
);

vmpx_vtil_status vmpx_vtil_optimize(
    vmpx_vtil_routine* routine,
    char* error,
    size_t error_capacity
);

vmpx_vtil_status vmpx_vtil_get_stats(
    const vmpx_vtil_routine* routine,
    vmpx_vtil_stats* stats,
    char* error,
    size_t error_capacity
);

vmpx_vtil_status vmpx_vtil_serialize(
    const vmpx_vtil_routine* routine,
    uint8_t* output,
    size_t output_capacity,
    size_t* required_size,
    char* error,
    size_t error_capacity
);

#ifdef __cplusplus
}
#endif

#endif
