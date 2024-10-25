#pragma once
#ifndef JALLOCATOR_TESTS_HPP
#define JALLOCATOR_TESTS_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <ranges>
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
constexpr size_t STRESS_TEST_DURATION_SEC = 30;
constexpr size_t REALLOC_TEST_COUNT = 1000;
constexpr size_t FRAGMENTATION_TEST_ALLOCS = 10000;
constexpr size_t MAX_ALIGNMENT_TEST = 1024;

struct allocation_stats_t
{
    std::atomic<size_t> total_allocations{0};
    std::atomic<size_t> total_deallocations{0};
    std::atomic<size_t> failed_allocations{0};
    std::atomic<size_t> current_bytes{0};
    std::atomic<size_t> peak_bytes{0};
} inline stats;

inline std::mutex output_mutex;
inline std::unordered_set<void*> allocated_ptrs;
inline std::mutex alloc_mutex;

// Thread-local RNG
inline thread_local std::mt19937_64 rng{std::random_device{}()};

// Utility functions
ALWAYS_INLINE
static size_t random_size()
{
    static std::uniform_int_distribution<size_t> dist(1, 16384);
    return dist(rng);
}

inline void* track_allocation(void* ptr)
{
    std::lock_guard lock(alloc_mutex);
    if (ptr) allocated_ptrs.insert(ptr);
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

// Basic allocation tests
inline void single_threaded_test(const size_t size)
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
}

static void thread_function(const size_t size)
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
    std::cout << CYAN << "Thread finished allocations and deallocations for size: "
              << size << RESET << "\n";
}

static void test_allocator_thread_safety(size_t size)
{
    std::cout << YELLOW << "Testing multi-threaded allocations for size: "
              << size << RESET << "\n";

    std::vector<std::thread> threads;
    for (size_t i = 0; i < NUM_THREADS; ++i)
        threads.emplace_back(thread_function, size);

    for (auto& thread : threads)
        thread.join();

    std::cout << GREEN << "Test completed for size: " << size << RESET << "\n";
}

// Edge case tests
static void run_edge_case_tests()
{
    // Zero allocation
    if (void* zero_ptr = Jallocator::allocate(0))
    {
        std::cout << RED << "Zero size allocation returned non-null pointer!\n" << RESET;
        Jallocator::deallocate(zero_ptr);
        track_deallocation(zero_ptr);
    }
    else
    {
        std::cout << GREEN << "Zero size allocation returned nullptr as expected.\n" << RESET;
    }

    // Max size allocation
    if (void* large_ptr = Jallocator::allocate(SIZE_MAX))
    {
        std::cout << RED << "MAX_SIZE allocation returned non-null pointer!\n" << RESET;
        Jallocator::deallocate(large_ptr);
        track_deallocation(large_ptr);
    }
    else
    {
        std::cout << GREEN << "MAX_SIZE allocation returned nullptr as expected.\n" << RESET;
    }

    Jallocator::deallocate(nullptr);
    Jallocator::deallocate(reinterpret_cast<void*>(1));

    void* ptr = track_allocation(Jallocator::allocate(64));
    Jallocator::deallocate(ptr);
    track_deallocation(ptr);
    Jallocator::deallocate(ptr);
}

static void test_header_validation()
{
    std::cout << MAGENTA << "Running header validation tests...\n" << RESET;

    for (const size_t size : {8, 64, 256, 1024})
    {
        void* ptr = track_allocation(Jallocator::allocate(size));
        if (!ptr)
        {
            std::cout << RED << "Allocation failed for size " << size << "\n" << RESET;
            continue;
        }

        if ((reinterpret_cast<uintptr_t>(ptr) & (ALIGNMENT - 1)) != 0)
        {
            std::cout << RED << "Pointer not properly aligned\n" << RESET;
            Jallocator::deallocate(ptr);
            track_deallocation(ptr);
            continue;
        }

        constexpr unsigned char pattern = 0xAA;
        std::memset(ptr, pattern, size);

        bool data_intact = true;
        for (size_t i = 0; i < size; i++)
        {
            if (static_cast<unsigned char*>(ptr)[i] != pattern)
            {
                data_intact = false;
                break;
            }
        }

        if (!data_intact)
            std::cout << RED << "Memory corruption detected\n" << RESET;

        void* new_ptr = track_allocation(Jallocator::reallocate(ptr, size * 2));
        track_deallocation(ptr);

        if (new_ptr)
        {
            data_intact = true;
            for (size_t i = 0; i < size; i++)
            {
                if (static_cast<unsigned char*>(new_ptr)[i] != pattern)
                {
                    data_intact = false;
                    break;
                }
            }

            if (!data_intact)
                std::cout << RED << "Data corruption after reallocation\n" << RESET;

            std::memset(static_cast<char*>(new_ptr) + size, pattern, size);

            Jallocator::deallocate(new_ptr);
            track_deallocation(new_ptr);
        }
    }

    Jallocator::deallocate(nullptr);

    // Test 2: zero-size allocation
    void* zero_ptr = Jallocator::allocate(0);
    if (zero_ptr != nullptr)
    {
        std::cout << RED << "Zero size allocation returned non-null\n" << RESET;
        Jallocator::deallocate(zero_ptr);
    }

    void* large_ptr = Jallocator::allocate(1ULL << 48);
    if (large_ptr != nullptr)
    {
        std::cout << RED << "Oversized allocation succeeded\n" << RESET;
        Jallocator::deallocate(large_ptr);
    }

    void* ptr = track_allocation(Jallocator::allocate(64));
    if (ptr)
    {
        Jallocator::deallocate(ptr);
        track_deallocation(ptr);
        Jallocator::deallocate(ptr);
    }

    if (ptr)
    {
        void* misaligned = static_cast<char*>(ptr) + 1;
        Jallocator::deallocate(misaligned);
    }

    std::cout << GREEN << "Header validation tests completed\n" << RESET;
}

static void test_memory_coalescing()
{
    //std::cout << MAGENTA << "Running memory coalescing tests...\n" << RESET;

    constexpr size_t block_size = 1024; // Medium size for coalescing
    std::vector<void*> blocks;

    // Allocate contiguous blocks
    for (size_t i = 0; i < 5; ++i)
    {
        void* ptr = track_allocation(Jallocator::allocate(block_size));
        blocks.push_back(ptr);
    }

    // Create fragmentation
    for (size_t i = 0; i < blocks.size(); i += 2)
    {
        Jallocator::deallocate(blocks[i]);
        track_deallocation(blocks[i]);
    }

    // Try coalesced allocation
    void* large_ptr = track_allocation(Jallocator::allocate(block_size * 2));

    if (!large_ptr)
        std::cout << RED << "Coalescing test - failed to allocate large block\n" << RESET;

    // Cleanup
    if (large_ptr)
    {
        Jallocator::deallocate(large_ptr);
        track_deallocation(large_ptr);
    }

    for (size_t i = 1; i < blocks.size(); i += 2)
    {
        Jallocator::deallocate(blocks[i]);
        track_deallocation(blocks[i]);
    }
}

// Thread cache tests
static void test_thread_cache()
{
    std::cout << MAGENTA << "Running thread cache tests...\n" << RESET;

    struct cache_stats_t
    {
        std::atomic<size_t> cache_hits{0};
        std::atomic<size_t> cache_misses{0};
    };

    constexpr size_t test_size = 64;
    std::vector<void*> ptrs;
    ptrs.reserve(CACHE_SIZE * 2);

    // Fill cache
    for (size_t i = 0; i < CACHE_SIZE; ++i)
    {
        void* ptr = track_allocation(Jallocator::allocate(test_size));
        ptrs.push_back(ptr);
    }

    // Return to cache
    for (const auto ptr : ptrs)
    {
        Jallocator::deallocate(ptr);
        track_deallocation(ptr);
    }
    ptrs.clear();

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < CACHE_SIZE; ++i)
    {
        void* ptr = track_allocation(Jallocator::allocate(test_size));
        ptrs.push_back(ptr);
    }

    const auto cached_time = std::chrono::high_resolution_clock::now() - start_time;

    for (const auto ptr : ptrs)
    {
        Jallocator::deallocate(ptr);
        track_deallocation(ptr);
    }
    ptrs.clear();
    Jallocator::cleanup();

    start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < CACHE_SIZE; ++i)
    {
        void* ptr = track_allocation(Jallocator::allocate(test_size));
        ptrs.push_back(ptr);
    }

    const auto uncached_time = std::chrono::high_resolution_clock::now() - start_time;

    for (auto ptr : ptrs)
    {
        Jallocator::deallocate(ptr);
        track_deallocation(ptr);
    }

    auto cached_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(cached_time).count();
    auto uncached_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(uncached_time).count();

    std::cout << CYAN << "Cache Performance:\n"
              << "Cached allocations: " << cached_ns << "ns\n"
              << "Uncached allocations: " << uncached_ns << "ns\n"
              << "Speedup: " << static_cast<double>(uncached_ns) / cached_ns << "x\n"
              << RESET;
}

static void test_pool_management()
{
    std::cout << MAGENTA << "Running pool management tests...\n" << RESET;

    std::vector<std::vector<void*>> pool_ptrs(8); // MAX_POOLS

    for (size_t pool = 0; pool < pool_ptrs.size(); ++pool)
    {
        constexpr size_t test_size = 256;
        for (size_t i = 0; i < PG_SIZE/test_size; ++i)
        {
            if (void* ptr = track_allocation(Jallocator::allocate(test_size)))
            {
                pool_ptrs[pool].push_back(ptr);
            }
            else
            {
                break;
            }
        }

        if (pool_ptrs[pool].empty())
        {
            std::cout << YELLOW << "Allocated " << pool << " pools\n" << RESET;
            break;
        }
    }

    for (auto& pool : pool_ptrs)
    {
        for (const auto ptr : pool)
        {
            Jallocator::deallocate(ptr);
            track_deallocation(ptr);
        }
    }
}

static void test_alignment()
{
    std::cout << MAGENTA << "Running alignment tests...\n" << RESET;

    for (size_t alignment = 8; alignment <= MAX_ALIGNMENT_TEST; alignment *= 2)
    {
        std::vector<void*> ptrs;
        ptrs.reserve(100);

        for (size_t i = 0; i < 100; ++i)
        {
            void* ptr = track_allocation(Jallocator::allocate(alignment));
            if (!ptr) continue;

            if (reinterpret_cast<uintptr_t>(ptr) % alignment != 0)
            {
                std::cout << RED << "Alignment failure: " << ptr
                         << " not aligned to " << alignment << "\n" << RESET;
            }
            ptrs.push_back(ptr);
        }

        for (auto ptr : ptrs)
        {
            Jallocator::deallocate(ptr);
            track_deallocation(ptr);
        }
    }
}

// Reallocation tests
static void test_realloc()
{
    std::cout << MAGENTA << "Running reallocation tests...\n" << RESET;

    struct test_block_t
    {
        size_t magic;
        std::array<char, 1024> data;
    };

    for (size_t i = 0; i < REALLOC_TEST_COUNT; ++i)
    {
        auto* ptr = static_cast<test_block_t*>(
            track_allocation(Jallocator::allocate(sizeof(test_block_t))));
        if (!ptr)
            continue;

        ptr->magic = 0xDEADBEEF;
        std::fill(ptr->data.begin(), ptr->data.end(), 'A');

        void* new_ptr = Jallocator::reallocate(ptr, sizeof(test_block_t) * 2);
        track_deallocation(ptr);

        if (new_ptr)
        {
            track_allocation(new_ptr);

            if (const auto* test_ptr = static_cast<test_block_t*>(new_ptr); test_ptr->magic != 0xDEADBEEF || test_ptr->data[0] != 'A') {
                std::cout << RED << "Reallocation data corruption detected!\n" << RESET;
            }

            Jallocator::deallocate(new_ptr);
            track_deallocation(new_ptr);
        }
    }
}

// Fragmentation tests
static void test_fragmentation()
{
    //std::cout << MAGENTA << "Running fragmentation tests...\n" << RESET;

    struct alloc_info_t
    {
        void* ptr;
        size_t size;
    };

    std::vector<alloc_info_t> allocations;
    allocations.reserve(FRAGMENTATION_TEST_ALLOCS);

    // Create fragmentation pattern
    for (size_t i = 0; i < FRAGMENTATION_TEST_ALLOCS; ++i)
    {
        const size_t size = random_size();
        if (void* ptr = track_allocation(Jallocator::allocate(size)))
            allocations.push_back({ptr, size});
    }

    std::shuffle(allocations.begin(), allocations.end(), rng);
    for (size_t i = 0; i < allocations.size() / 2; ++i)
    {
        Jallocator::deallocate(allocations[i].ptr);
        track_deallocation(allocations[i].ptr);
    }

    // Large allocation
    std::vector<void*> large_blocks;
    for (size_t size = 1024; size <= 1024 * 1024; size *= 2)
    {
        if (void* ptr = track_allocation(Jallocator::allocate(size)))
            large_blocks.push_back(ptr);
    }

    // Cleanup
    for (size_t i = allocations.size() / 2; i < allocations.size(); ++i)
    {
        Jallocator::deallocate(allocations[i].ptr);
        track_deallocation(allocations[i].ptr);
    }
    for (const auto ptr : large_blocks)
    {
        Jallocator::deallocate(ptr);
        track_deallocation(ptr);
    }
}

// Stress test thread function
static void stress_test_thread(const std::chrono::steady_clock::time_point end_time)
{
    struct alloc_record_t
    {
        void* ptr;
        size_t size;
    };

    std::vector<alloc_record_t> thread_allocs;
    thread_allocs.reserve(1000);

    while (std::chrono::steady_clock::now() < end_time)
    {
        switch (random_size() % 3)
        {
            case 0: // Allocate
            {
                size_t size = random_size();
                if (void* ptr = track_allocation(Jallocator::allocate(size)))
                {
                    thread_allocs.push_back({ptr, size});
                    ++stats.total_allocations;
                    stats.current_bytes += size;
                    stats.peak_bytes = std::max(stats.peak_bytes.load(),
                                              stats.current_bytes.load());
                }
                else
                {
                    ++stats.failed_allocations;
                }
                break;
            }
            case 1:  // Deallocate
            {
                if (!thread_allocs.empty())
                {
                    auto [ptr, size] = thread_allocs.back();
                    thread_allocs.pop_back();
                    Jallocator::deallocate(ptr);
                    track_deallocation(ptr);
                    ++stats.total_deallocations;
                    stats.current_bytes -= size;
                }
                break;
            }
            case 2: // Reallocate
            {
                if (!thread_allocs.empty())
                {
                    auto [ptr, size] = thread_allocs.back();
                    thread_allocs.pop_back();
                    const size_t new_size = random_size();
                    void* new_ptr = Jallocator::reallocate(ptr, new_size);
                    if (new_ptr) // Only track the new allocation if reallocation succeeded
                    {
                        track_deallocation(ptr);
                        track_allocation(new_ptr);
                        thread_allocs.push_back({new_ptr, new_size});
                        stats.current_bytes += (new_size - size);
                        stats.peak_bytes = std::max(stats.peak_bytes.load(),
                                                    stats.current_bytes.load());
                    }
                    else
                    {
                        // Reallocation failed, handle gracefully
                        track_deallocation(ptr);
                        Jallocator::deallocate(ptr); // Still free the original allocation
                    }
                }
                break;
            }
            default:
                break;
        }
    }

    for (const auto&[ptr, size] : thread_allocs)
    {
        Jallocator::deallocate(ptr);
        track_deallocation(ptr);
        stats.current_bytes -= size;
    }
}

// Main stress test
static void run_stress_test()
{
    std::cout << MAGENTA << "Running stress test...\n" << RESET;

    auto end_time = std::chrono::steady_clock::now() +
        std::chrono::seconds(STRESS_TEST_DURATION_SEC);

    std::vector<std::thread> threads;
    for (size_t i = 0; i < NUM_THREADS; ++i)
        threads.emplace_back(stress_test_thread, end_time);

    for (auto& thread : threads)
        thread.join();

    std::cout << CYAN << "Stress Test Results:\n"
              << "Total Allocations: " << stats.total_allocations << "\n"
              << "Total Deallocations: " << stats.total_deallocations << "\n"
              << "Failed Allocations: " << stats.failed_allocations << "\n"
              << "Peak Memory Usage: " << stats.peak_bytes << " bytes\n"
              << RESET;
}

// Memory leak test
static void test_memory_leak(const size_t size)
{
    std::vector<void*> allocated_ptrs_local;
    allocated_ptrs_local.reserve(NUM_ALLOCATIONS);

    for (size_t i = 0; i < NUM_ALLOCATIONS; ++i) {
        void* ptr = track_allocation(Jallocator::allocate(size));
        allocated_ptrs_local.push_back(ptr);
    }

    std::cout << YELLOW << "Memory leak test completed for size: " << size << ". "
              << "Allocated " << allocated_ptrs_local.size() << " pointers.\n" << RESET;

    for (const auto ptr : allocated_ptrs_local)
    {
        Jallocator::deallocate(ptr);
        track_deallocation(ptr);
    }
}

// SIMD tests
static void test_simd_capabilities()
{
    std::cout << MAGENTA << "Testing SIMD capabilities...\n" << RESET;

    // Test large memset performance which should use SIMD
    constexpr size_t test_size = 1024 * 1024;

    if (void* ptr = track_allocation(Jallocator::callocate(1, test_size)))
    {
        // Verify zero initialization
        auto all_zero = true;
        for (size_t i = 0; i < test_size; i += 8)
        {
            if (*reinterpret_cast<uint64_t*>(static_cast<char*>(ptr) + i) != 0)
            {
                all_zero = false;
                break;
            }
        }

        if (!all_zero)
            std::cout << RED << "SIMD zero initialization failed\n" << RESET;

        Jallocator::deallocate(ptr);
        track_deallocation(ptr);
    }
}

inline void run_all_tests()
{
    std::cout << BLUE << "Running basic tests..." << RESET << "\n";
    for (const size_t size : { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 })
        single_threaded_test(size);

    std::cout << BLUE << "Running multi-threaded tests..." << RESET << "\n";
    for (const size_t size : {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})
        test_allocator_thread_safety(size);

    std::cout << BLUE << "Running header validation tests..." << RESET << "\n";
    test_header_validation();

    std::cout << BLUE << "Running SIMD capability tests..." << RESET << "\n";
    test_simd_capabilities();

    std::cout << BLUE << "Running memory coalescing tests..." << RESET << "\n";
    test_memory_coalescing();

    std::cout << BLUE << "Running thread cache tests..." << RESET << "\n";
    test_thread_cache();

    std::cout << BLUE << "Running pool management tests..." << RESET << "\n";
    test_pool_management();

    // std::cout << BLUE << "Running alignment tests..." << RESET << "\n";
    // test_alignment();

    std::cout << BLUE << "Running reallocation tests..." << RESET << "\n";
    test_realloc();

    std::cout << BLUE << "Running fragmentation tests..." << RESET << "\n";
    test_fragmentation();

    std::cout << BLUE << "Running edge case tests..." << RESET << "\n";
    run_edge_case_tests();

    std::cout << BLUE << "Running stress test..." << RESET << "\n";
    run_stress_test();

    std::cout << BLUE << "Running memory leak tests..." << RESET << "\n";
    for (const size_t size : { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 })
        test_memory_leak(size);

    report_memory_leaks();
}

#endif