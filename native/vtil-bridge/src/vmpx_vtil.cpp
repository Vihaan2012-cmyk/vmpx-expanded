#include "vmpx_vtil.h"

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <vtil/vtil>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#ifndef VMPX_VTIL_REVISION
#define VMPX_VTIL_REVISION "unknown"
#endif

struct vmpx_vtil_routine {
    vtil::routine* routine = nullptr;
    vtil::basic_block* block = nullptr;
    std::vector<vtil::register_desc> registers;
    bool finished = false;
};

namespace {

void clear_error(char* error, size_t capacity) {
    if (error != nullptr && capacity != 0) {
        error[0] = '\0';
    }
}

void set_error(char* error, size_t capacity, const char* message) {
    if (error == nullptr || capacity == 0) {
        return;
    }
    const char* source = message == nullptr ? "unknown error" : message;
    const size_t length = std::min(capacity - 1, std::strlen(source));
    std::memcpy(error, source, length);
    error[length] = '\0';
}

bool valid_bit_width(uint16_t width) {
    return width == 1 || (width >= 8 && width <= 64 && width % 8 == 0);
}

vtil::register_desc* find_register(vmpx_vtil_routine* handle, uint64_t id) {
    if (handle == nullptr || id == 0 || id > handle->registers.size()) {
        return nullptr;
    }
    return &handle->registers[static_cast<size_t>(id - 1)];
}

uint64_t truncate_to_width(uint64_t value, int width) {
    if (width >= 64) {
        return value;
    }
    return value & ((uint64_t{1} << width) - 1);
}

template <typename Function>
vmpx_vtil_status guard(char* error, size_t capacity, Function&& function) {
    clear_error(error, capacity);
    try {
        function();
        return VMPX_VTIL_OK;
    } catch (const std::exception& exception) {
        set_error(error, capacity, exception.what());
        return VMPX_VTIL_INTERNAL_ERROR;
    } catch (...) {
        set_error(error, capacity, "unknown C++ exception");
        return VMPX_VTIL_INTERNAL_ERROR;
    }
}

}  // namespace

extern "C" const char* vmpx_vtil_revision(void) {
    return VMPX_VTIL_REVISION;
}

extern "C" vmpx_vtil_status vmpx_vtil_create_amd64(
    uint64_t entry_vip,
    vmpx_vtil_routine** output,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr || entry_vip == vtil::invalid_vip) {
        set_error(error, error_capacity, "invalid output pointer or entry VIP");
        return VMPX_VTIL_INVALID_ARGUMENT;
    }
    *output = nullptr;
    return guard(error, error_capacity, [&] {
        auto* handle = new vmpx_vtil_routine{};
        try {
            handle->block = vtil::basic_block::begin(entry_vip, vtil::architecture_amd64);
            handle->routine = handle->block->owner;
            *output = handle;
        } catch (...) {
            delete handle;
            throw;
        }
    });
}

extern "C" void vmpx_vtil_destroy(vmpx_vtil_routine* handle) {
    if (handle == nullptr) {
        return;
    }
    // The pinned VTIL destructor clears its owning unordered_map while each
    // basic_block destructor asserts that it has already been detached.  Use
    // the public deletion path explicitly so that invariant holds on MSVC.
    for (auto& [vip, block] : handle->routine->explored_blocks) {
        (void)vip;
        block->next.clear();
        block->prev.clear();
    }
    while (!handle->routine->explored_blocks.empty()) {
        handle->routine->delete_block(handle->routine->explored_blocks.begin()->second.get());
    }
    delete handle->routine;
    delete handle;
}

extern "C" vmpx_vtil_status vmpx_vtil_alloc_register(
    vmpx_vtil_routine* handle,
    uint16_t bit_width,
    uint64_t* register_id,
    char* error,
    size_t error_capacity
) {
    if (handle == nullptr || register_id == nullptr || !valid_bit_width(bit_width)) {
        set_error(error, error_capacity, "invalid routine, output pointer, or bit width");
        return VMPX_VTIL_INVALID_ARGUMENT;
    }
    if (handle->finished) {
        set_error(error, error_capacity, "cannot allocate a register after finishing the block");
        return VMPX_VTIL_INVALID_STATE;
    }
    return guard(error, error_capacity, [&] {
        handle->registers.push_back(handle->block->tmp(bit_width));
        *register_id = handle->registers.size();
    });
}

extern "C" vmpx_vtil_status vmpx_vtil_emit_mov_imm(
    vmpx_vtil_routine* handle,
    uint64_t register_id,
    uint64_t value,
    char* error,
    size_t error_capacity
) {
    auto* destination = find_register(handle, register_id);
    if (destination == nullptr) {
        set_error(error, error_capacity, "unknown register id");
        return VMPX_VTIL_INVALID_ARGUMENT;
    }
    if (handle->finished) {
        set_error(error, error_capacity, "cannot emit after finishing the block");
        return VMPX_VTIL_INVALID_STATE;
    }
    return guard(error, error_capacity, [&] {
        handle->block->mov(
            *destination,
            vtil::operand(truncate_to_width(value, destination->bit_count), destination->bit_count)
        );
    });
}

extern "C" vmpx_vtil_status vmpx_vtil_emit_add_imm(
    vmpx_vtil_routine* handle,
    uint64_t register_id,
    uint64_t value,
    char* error,
    size_t error_capacity
) {
    auto* destination = find_register(handle, register_id);
    if (destination == nullptr) {
        set_error(error, error_capacity, "unknown register id");
        return VMPX_VTIL_INVALID_ARGUMENT;
    }
    if (handle->finished) {
        set_error(error, error_capacity, "cannot emit after finishing the block");
        return VMPX_VTIL_INVALID_STATE;
    }
    return guard(error, error_capacity, [&] {
        handle->block->add(
            *destination,
            vtil::operand(truncate_to_width(value, destination->bit_count), destination->bit_count)
        );
    });
}

extern "C" vmpx_vtil_status vmpx_vtil_finish(
    vmpx_vtil_routine* handle,
    uint64_t output_register_id,
    char* error,
    size_t error_capacity
) {
    const auto* output = find_register(handle, output_register_id);
    if (output == nullptr) {
        set_error(error, error_capacity, "unknown output register id");
        return VMPX_VTIL_INVALID_ARGUMENT;
    }
    if (handle->finished) {
        set_error(error, error_capacity, "block is already finished");
        return VMPX_VTIL_INVALID_STATE;
    }
    return guard(error, error_capacity, [&] {
        handle->block->vpinr(*output);
        handle->block->vexit(uint64_t{0});
        handle->finished = true;
    });
}

extern "C" vmpx_vtil_status vmpx_vtil_optimize(
    vmpx_vtil_routine* handle,
    char* error,
    size_t error_capacity
) {
    if (handle == nullptr) {
        set_error(error, error_capacity, "routine is null");
        return VMPX_VTIL_INVALID_ARGUMENT;
    }
    if (!handle->finished) {
        set_error(error, error_capacity, "finish the block before optimization");
        return VMPX_VTIL_INVALID_STATE;
    }
    return guard(error, error_capacity, [&] { vtil::optimizer::apply_all(handle->routine); });
}

extern "C" vmpx_vtil_status vmpx_vtil_get_stats(
    const vmpx_vtil_routine* handle,
    vmpx_vtil_stats* stats,
    char* error,
    size_t error_capacity
) {
    if (handle == nullptr || stats == nullptr) {
        set_error(error, error_capacity, "routine or stats pointer is null");
        return VMPX_VTIL_INVALID_ARGUMENT;
    }
    return guard(error, error_capacity, [&] {
        stats->blocks = handle->routine->num_blocks();
        stats->instructions = handle->routine->num_instructions();
        stats->branches = handle->routine->num_branches();
        stats->local_optimizations = handle->routine->local_opt_count.load();
    });
}

extern "C" vmpx_vtil_status vmpx_vtil_serialize(
    const vmpx_vtil_routine* handle,
    uint8_t* output,
    size_t output_capacity,
    size_t* required_size,
    char* error,
    size_t error_capacity
) {
    if (handle == nullptr || required_size == nullptr) {
        set_error(error, error_capacity, "routine or required-size pointer is null");
        return VMPX_VTIL_INVALID_ARGUMENT;
    }
    clear_error(error, error_capacity);
    try {
        std::ostringstream stream(std::ios::out | std::ios::binary);
        vtil::serialize(stream, handle->routine);
        const std::string bytes = stream.str();
        *required_size = bytes.size();
        if (output == nullptr) {
            return VMPX_VTIL_OK;
        }
        if (output_capacity < bytes.size()) {
            set_error(error, error_capacity, "serialization output buffer is too small");
            return VMPX_VTIL_BUFFER_TOO_SMALL;
        }
        std::memcpy(output, bytes.data(), bytes.size());
        return VMPX_VTIL_OK;
    } catch (const std::exception& exception) {
        set_error(error, error_capacity, exception.what());
        return VMPX_VTIL_INTERNAL_ERROR;
    } catch (...) {
        set_error(error, error_capacity, "unknown C++ exception");
        return VMPX_VTIL_INTERNAL_ERROR;
    }
}
