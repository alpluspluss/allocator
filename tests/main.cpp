#include <cassert>
#include <chrono>
#include <thread>
#include <vector>
#include "jalloc.hpp"

void single_threaded_test(size_t size, int num_ops = 1000)
{
    std::vector<void *> ptrs;
    ptrs.reserve(num_ops);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_ops; ++i)
    {
        ptrs.push_back(Jallocator::allocate(size));
    }

    for (std::vector<void *>::size_type i = 0; i < ptrs.size(); ++i)
    {
        Jallocator::deallocate(ptrs[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double ops_per_sec = (num_ops * 2.0) / (duration.count() / 1000000.0);

    printf("Size: %6zu bytes | Ops: %6d | Time: %8.3f ms | Ops/sec: %10.0f\n",
           size, num_ops, duration.count() / 1000.0, ops_per_sec);
}

void multi_threaded_test(size_t size, int num_threads = 16, int ops_per_thread = 1000)
{
    const auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int32_t i = 0; i < num_threads; ++i)
    {
        threads.push_back(std::thread([size, ops_per_thread]
        {
            std::vector<void *> ptrs;
            ptrs.reserve(ops_per_thread);

            for (auto j = 0; j < ops_per_thread; ++j)
                ptrs.push_back(Jallocator::allocate(size));

            for (std::vector<void *>::size_type j = 0; j < ptrs.size(); ++j)
                Jallocator::deallocate(ptrs[j]);
        }));
    }

    for (std::vector<std::thread>::size_type i = 0; i < threads.size(); ++i)
    {
        threads[i].join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double ops_per_sec = (num_threads * ops_per_thread * 2.0) / (duration.count() / 1000000.0);

    printf("Size: %6zu bytes | Threads: %2d | Time: %8.3f ms | Ops/sec: %10.0f\n",
           size, num_threads, duration.count() / 1000.0, ops_per_sec);
}

void memory_leak_test(size_t size, int num_allocs = 1000)
{
    std::vector<void *> ptrs;
    ptrs.reserve(num_allocs);

    for (auto i = 0; i < num_allocs; ++i)
    {
        ptrs.push_back(Jallocator::allocate(size));
        assert(ptrs.back() != nullptr);
    }

    for (std::vector<void *>::size_type i = 0; i < ptrs.size(); ++i)
    {
        Jallocator::deallocate(ptrs[i]);
    }

    printf("Memory leak test completed for size: %zu. Allocated %d pointers.\n",
           size, num_allocs);
}

void edge_case_tests()
{
    void *ptr = Jallocator::allocate(0);
    assert(ptr == nullptr);
    printf("Zero size allocation returned nullptr as expected.\n");

    ptr = Jallocator::allocate(SIZE_MAX);
    assert(ptr == nullptr);
    printf("Large size allocation returned nullptr as expected.\n");
}

void reallocation_test()
{
    // Test growing reallocation
    void *ptr = Jallocator::allocate(128);
    assert(ptr != nullptr);
    ptr = Jallocator::reallocate(ptr, 256);
    assert(ptr != nullptr);

    // Test shrinking reallocation
    ptr = Jallocator::reallocate(ptr, 64);
    assert(ptr != nullptr);

    // Cleanup
    Jallocator::deallocate(ptr);
}

int main()
{
#ifdef __APPLE__
    setenv("MallocNanoZone", "0", 1);
#endif

    printf("Running single-threaded tests...\n");
    std::vector<size_t> sizes = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

    for (auto size: sizes)
    {
        single_threaded_test(size);
    }

    printf("\nRunning multi-threaded tests...\n");
    for (auto size: sizes)
    {
        multi_threaded_test(size);
    }

    printf("\nRunning edge case tests...\n");
    edge_case_tests();

    printf("\nRunning memory leak tests...\n");
    for (auto size: sizes)
    {
        memory_leak_test(size);
    }
    printf("No memory leaks detected.\n");

    return 0;
}