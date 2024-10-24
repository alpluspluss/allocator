cmake_minimum_required(VERSION 3.29)
project(allocator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
include(CTest)

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

if (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(STATUS "Configuring for macOS AArch64")

    add_compile_options(
            -arch arm64
            -mtune=native
            -O3
            -DNDEBUG
            -fomit-frame-pointer
    )

    add_compiler_flag(-mcpu=apple-m1)
    add_compiler_flag(-mvsx)
    add_compiler_flag(-maltivec)

elseif(UNIX)
    message(STATUS "Configuring for Linux")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
        -O3 \
        -DNDEBUG \
        -fomit-frame-pointer \
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
    message(STATUS "Configuring for Windows")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
        /O2 \
        /DNDEBUG \
        /Ob2 \
        /Oi \
        /Gy \
        /GL \
        /sdl \
    ")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
    -g3 \
    -fno-omit-frame-pointer \
    -fsanitize=address,undefined \
")

add_executable(allocator tests/main.cpp
        jalloc.hpp
        tests/testsuit.h
)

target_compile_features(allocator PRIVATE cxx_std_20)

enable_testing()
add_executable(jalloc-test tests/main.cpp jalloc.hpp)
add_test(NAME jalloc_test COMMAND jalloc-test)
