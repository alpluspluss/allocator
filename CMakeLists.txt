cmake_minimum_required(VERSION 3.29)
project(allocator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
include(CTest)
enable_testing()

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
        /EHsc \
    ")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
        -g3 \
        -fno-omit-frame-pointer \
        -fsanitize=address,undefined \
    ")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined")
    endif()
endif()

add_library(jalloc INTERFACE)
target_include_directories(jalloc INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(jalloc INTERFACE cxx_std_17)

add_executable(jalloc_tests tests/main.cpp)
target_link_libraries(jalloc_tests PRIVATE jalloc)
target_compile_features(jalloc_tests PRIVATE cxx_std_17)

add_test(
        NAME jalloc_unit_tests
        COMMAND jalloc_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(jalloc_unit_tests PROPERTIES LABELS "unit")

set(CTEST_OUTPUT_ON_FAILURE ON)
set(CTEST_TEST_TIMEOUT 120)