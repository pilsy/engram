cmake_minimum_required(VERSION 3.20)
project(engram VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# O3 optimization for release builds
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3)

include(FetchContent)

# ──────────────────────────────────────────────
# llama.cpp (latest master)
# ──────────────────────────────────────────────
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

# ── GPU backend selection ──────────────────────────────────────────────────
# Metal is enabled by default on Apple; CUDA is opt-in.
if(APPLE)
    set(GGML_METAL          ON  CACHE BOOL "Enable Metal GPU backend (macOS/iOS)" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed Metal library into binary"  FORCE)
endif()

option(GGML_CUDA "Enable CUDA GPU backend (requires CUDA toolkit)" OFF)
if(GGML_CUDA)
    message(STATUS "engram: CUDA backend enabled")
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(llama_cpp)

# ──────────────────────────────────────────────
# cpp-httplib (latest)
# ──────────────────────────────────────────────
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(httplib)

# ──────────────────────────────────────────────
# nlohmann/json v3.11.3
# ──────────────────────────────────────────────
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ──────────────────────────────────────────────
# Engram binary
# ──────────────────────────────────────────────
add_executable(engram
    src/main.cpp
    src/session_store.cpp
    src/api.cpp
)

target_include_directories(engram PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

target_link_libraries(engram PRIVATE
    llama
    httplib::httplib
    nlohmann_json::nlohmann_json
)

# Threads required by httplib
find_package(Threads REQUIRED)
target_link_libraries(engram PRIVATE Threads::Threads)
