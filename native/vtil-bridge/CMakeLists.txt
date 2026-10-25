cmake_minimum_required(VERSION 3.20)

project(vmpx-vtil-bridge LANGUAGES CXX)

include(FetchContent)

set(
    VMPX_VTIL_REVISION
    "9ebee8e890a3a8923cc41f07821f740910e7c9b2"
    CACHE STRING
    "Pinned VTIL-Core revision"
)
set(VMPX_VTIL_SOURCE_DIR "" CACHE PATH "Optional existing VTIL-Core checkout")
option(VMPX_VTIL_BRIDGE_BUILD_TESTS "Build the bridge smoke test" ON)

set(VTIL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(VTIL_UNITY_BUILD ON CACHE BOOL "" FORCE)

if(VMPX_VTIL_SOURCE_DIR)
    add_subdirectory(
        "${VMPX_VTIL_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/vtil-core"
        EXCLUDE_FROM_ALL
    )
else()
    FetchContent_Declare(
        vtil_core
        GIT_REPOSITORY https://github.com/vtil-project/VTIL-Core.git
        GIT_TAG "${VMPX_VTIL_REVISION}"
        GIT_SHALLOW FALSE
    )
    FetchContent_MakeAvailable(vtil_core)
endif()

add_library(vmpx-vtil-bridge STATIC src/vmpx_vtil.cpp)
target_include_directories(vmpx-vtil-bridge PUBLIC include)
target_link_libraries(vmpx-vtil-bridge PRIVATE VTIL)
target_compile_features(vmpx-vtil-bridge PUBLIC cxx_std_20)
target_compile_definitions(
    vmpx-vtil-bridge
    PRIVATE VMPX_VTIL_REVISION="${VMPX_VTIL_REVISION}"
)

if(MSVC)
    # VTIL template instantiation currently produces C4702 from its enumerators.
    target_compile_options(vmpx-vtil-bridge PRIVATE /W4 /WX /wd4702)
else()
    target_compile_options(vmpx-vtil-bridge PRIVATE -Wall -Wextra -Werror)
endif()

if(VMPX_VTIL_BRIDGE_BUILD_TESTS)
    enable_testing()
    add_executable(vmpx-vtil-bridge-smoke tests/smoke.cpp)
    target_link_libraries(vmpx-vtil-bridge-smoke PRIVATE vmpx-vtil-bridge)
    add_test(NAME vmpx-vtil-bridge-smoke COMMAND vmpx-vtil-bridge-smoke)
endif()
