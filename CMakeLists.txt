cmake_minimum_required(VERSION 3.29)
project(allocator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
include(CTest)  # Include CTest for testing

function(add_compiler_flag flag)
    check_cxx_compiler_flag(${flag} FLAG_${flag})
    if(FLAG_${flag})
        add_compile_options(${flag})
    endif()
endfunction()

add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -ffast-math
)

if(APPLE)
    add_compile_options(
            -arch arm64
            -mtune=native
    )

    add_compiler_flag(-mcpu=apple-m1)
    add_compiler_flag(-mvsx)
    add_compiler_flag(-maltivec)

elseif(UNIX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
        -O3 \
        -DNDEBUG \
        -fomit-frame-pointer \
        -flto=thin \
        -fno-signed-zeros \
        -fno-trapping-math \
        -freciprocal-math \
        -ffinite-math-only \
        -fno-math-errno \
        -fno-rtti \
        -fno-exceptions \
        -ffunction-sections \
        -fdata-sections \
    ")
    add_compile_options(-march=native -mtune=native)

elseif(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
        /O3 \
        /DNDEBUG \
        /Ob2 \
        /Oi \
        /Gy \
        /GL \
    ")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
    -g3 \
    -fno-omit-frame-pointer \
    -fsanitize=address,undefined \
")

add_executable(allocator tests/main.cpp jalloc.hpp
        .dev-container/jalloc-dev.hpp
        tests/testsuit.h
)

target_compile_features(allocator PRIVATE cxx_std_20)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(allocator PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

    option(ENABLE_PGO_GENERATE "Enable PGO generation phase" OFF)
    option(ENABLE_PGO_USE "Enable PGO use phase" OFF)

    if(ENABLE_PGO_GENERATE)
        target_compile_options(allocator PRIVATE
                -fprofile-instr-generate
        )
        target_link_options(allocator PRIVATE
                -fprofile-instr-generate
        )
    endif()

    if(ENABLE_PGO_USE)
        target_compile_options(allocator PRIVATE
                -fprofile-instr-use=${CMAKE_BINARY_DIR}/default.profdata
        )
        target_link_options(allocator PRIVATE
                -fprofile-instr-use=${CMAKE_BINARY_DIR}/default.profdata
        )
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(allocator PRIVATE
            -Rpass=loop-vectorize
            -Rpass-missed=loop-vectorize
            -Rpass-analysis=loop-vectorize
    )
endif()

enable_testing()

add_executable(jalloc-test tests/main.cpp jalloc.hpp)

add_test(NAME jalloc_test COMMAND jalloc-test)
