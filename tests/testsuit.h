#pragma once

#ifndef JALLOCATOR_TESTS_HPP
#define JALLOCATOR_TESTS_HPP

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>
#include "../jalloc.hpp"

constexpr auto RESET = "\033[0m";
constexpr auto RED = "\033[31m";
constexpr auto GREEN = "\033[32m";
constexpr auto YELLOW = "\033[33m";
constexpr auto BLUE = "\033[34m";
constexpr auto CYAN = "\033[36m";
constexpr auto MAGENTA = "\033[35m";

constexpr size_t NUM_THREADS = 16;
constexpr size_t NUM_ALLOCATIONS = 1000;

inline std::mutex output_mutex;
inline std::unordered_set<void*> allocated_ptrs;
inline std::mutex alloc_mutex;

inline void* track_allocation(void* ptr)
{
    std::lock_guard lock(alloc_mutex);
    if (ptr)
        allocated_ptrs.insert(ptr);
    return ptr;
}

inline void track_deallocation(void* ptr)
{
    std::lock_guard lock(alloc_mutex);
    allocated_ptrs.erase(ptr);
}

inline void report_memory_leaks()
{
    std::lock_guard lock(alloc_mutex);
    if (!allocated_ptrs.empty())
    {
        std::cout << RED << "Memory leaks detected! Leaked pointers:\n" << RESET;
        for (const auto& ptr : allocated_ptrs)
            std::cout << RED << ptr << "\n" << RESET;
    }
    else
    {
        std::cout << GREEN << "No memory leaks detected.\n" << RESET;
    }
}

inline void single_threaded_test(const size_t size)
{
    std::vector<void*> allocated_ptrs_local;
    allocated_ptrs_local.reserve(NUM_ALLOCATIONS);

    for (size_t i = 0; i < NUM_ALLOCATIONS; ++i)
    {
        void* ptr = track_allocation(Jallocator::allocate(size));
        allocated_ptrs_local.push_back(ptr);
    }

    for (auto ptr : allocated_ptrs_local)
    {
        Jallocator::deallocate(ptr);
        track_deallocation(ptr);
    }

    std::lock_guard lock(output_mutex);
}

inline void thread_function(const size_t size)
{
    std::vector<void*> allocated_ptrs_local;
    allocated_ptrs_local.reserve(NUM_ALLOCATIONS);

    for (size_t i = 0; i < NUM_ALLOCATIONS; ++i)
    {
        void* ptr = track_allocation(Jallocator::allocate(size));
        allocated_ptrs_local.push_back(ptr);
    }

    for (const auto ptr : allocated_ptrs_local)
    {
        Jallocator::deallocate(ptr);
        track_deallocation(ptr);
    }

    std::lock_guard lock(output_mutex);
    std::cout << CYAN << "Thread finished allocations and deallocations for size: " << size << RESET << "\n";
}

inline void test_allocator_thread_safety(size_t size)
{
    std::cout << YELLOW << "Testing multi-threaded allocations for size: " << size << RESET << "\n";

    std::vector<std::thread> threads;
    for (size_t i = 0; i < NUM_THREADS; ++i)
        threads.emplace_back(thread_function, size);

    for (auto& thread : threads)
        thread.join();

    std::cout << GREEN << "Test completed for size: " << size << RESET << "\n";
}

inline void run_edge_case_tests()
{
    if (void* zero_ptr = Jallocator::allocate(0); zero_ptr != nullptr)
    {
        std::cout << RED << "Zero size allocation returned a non-null pointer!\n" << RESET;
        Jallocator::deallocate(zero_ptr);
        track_deallocation(zero_ptr);
    }
    else
    {
        std::cout << GREEN << "Zero size allocation returned nullptr as expected.\n" << RESET;
    }

    if (void* large_ptr = Jallocator::allocate(SIZE_MAX); large_ptr != nullptr)
    {
        std::cout << RED << "Large size allocation returned a non-null pointer!\n" << RESET;
        Jallocator::deallocate(large_ptr);
        track_deallocation(large_ptr);
    }
    else
    {
        std::cout << GREEN << "Large size allocation returned nullptr as expected.\n" << RESET;
    }
}

inline void test_memory_leak(const size_t size)
{
    std::vector<void*> allocated_ptrs_local;
    allocated_ptrs_local.reserve(NUM_ALLOCATIONS);

    for (size_t i = 0; i < NUM_ALLOCATIONS; ++i)
    {
        void* ptr = track_allocation(Jallocator::allocate(size));
        allocated_ptrs_local.push_back(ptr);
    }

    std::cout << YELLOW << "Memory leak test completed for size: " << size << ". "
              << "Allocated " << allocated_ptrs_local.size() << " pointers.\n" << RESET;

    for (auto ptr : allocated_ptrs_local)
    {
        Jallocator::deallocate(ptr);
        track_deallocation(ptr);
    }
}

inline void run_all_tests()
{
    std::cout << BLUE << "Running single-threaded tests..." << std::endl;
    for (const size_t size : {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})
        single_threaded_test(size);

    std::cout << BLUE << "Running multi-threaded tests..." << std::endl;
    for (const size_t size : {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})
        test_allocator_thread_safety(size);

    std::cout << BLUE << "Running edge case tests..." << std::endl;
    run_edge_case_tests();

    std::cout << BLUE << "Running memory leak tests..." << std::endl;
    for (const size_t size : {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})
        test_memory_leak(size);

    report_memory_leaks();
}

#endif
