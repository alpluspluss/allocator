cmake_minimum_required(VERSION 3.29)
project(allocator)

set(CMAKE_CXX_STANDARD 17)

add_executable(allocator main.cpp
        jalloc.hpp)
