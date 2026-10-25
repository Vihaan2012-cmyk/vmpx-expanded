#include "vmpx_vtil.h"

#include <cstdint>
#include <cstring>
#include <vector>

#define CHECK(expression)      \
    do {                       \
        if (!(expression)) {   \
            return __LINE__;   \
        }                      \
    } while (false)

int main() {
    char error[512] = {};
    vmpx_vtil_routine* routine = nullptr;
    CHECK(std::strlen(vmpx_vtil_revision()) == 40);
    CHECK(vmpx_vtil_create_amd64(0x1000, &routine, error, sizeof(error)) == VMPX_VTIL_OK);
    CHECK(routine != nullptr);

    uint64_t value = 0;
    CHECK(vmpx_vtil_alloc_register(routine, 64, &value, error, sizeof(error)) == VMPX_VTIL_OK);
    CHECK(vmpx_vtil_emit_mov_imm(routine, value, 1, error, sizeof(error)) == VMPX_VTIL_OK);
    CHECK(vmpx_vtil_emit_add_imm(routine, value, 2, error, sizeof(error)) == VMPX_VTIL_OK);
    CHECK(vmpx_vtil_emit_add_imm(routine, value, 3, error, sizeof(error)) == VMPX_VTIL_OK);
    CHECK(vmpx_vtil_finish(routine, value, error, sizeof(error)) == VMPX_VTIL_OK);

    vmpx_vtil_stats before = {};
    CHECK(vmpx_vtil_get_stats(routine, &before, error, sizeof(error)) == VMPX_VTIL_OK);
    CHECK(before.blocks == 1);
    CHECK(before.instructions >= 5);

    CHECK(vmpx_vtil_optimize(routine, error, sizeof(error)) == VMPX_VTIL_OK);
    vmpx_vtil_stats after = {};
    CHECK(vmpx_vtil_get_stats(routine, &after, error, sizeof(error)) == VMPX_VTIL_OK);
    CHECK(after.blocks == 1);
    CHECK(after.instructions > 0);
    CHECK(after.instructions <= before.instructions);

    size_t required = 0;
    CHECK(vmpx_vtil_serialize(routine, nullptr, 0, &required, error, sizeof(error)) == VMPX_VTIL_OK);
    CHECK(required > 0);
    std::vector<uint8_t> bytes(required);
    CHECK(
        vmpx_vtil_serialize(
            routine,
            bytes.data(),
            bytes.size(),
            &required,
            error,
            sizeof(error)
        ) == VMPX_VTIL_OK
    );

    vmpx_vtil_destroy(routine);
    return 0;
}
