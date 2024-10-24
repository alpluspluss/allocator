#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include "./.dev-container/jalloc-dev.hpp"

struct AllocationSize
{
    const char* name;
    size_t size;
};

constexpr AllocationSize SIZES[] = {
    {"Tiny-8", 8},
    {"Tiny-16", 16},
    {"Tiny-24", 24},
    {"Small-32", 32},
    {"Small-64", 64},
    {"Small-128", 128},
    {"Medium-256", 256},
    {"Medium-512", 512},
    {"Medium-1K", 1024},
    {"Large-2K", 2048},
    {"Large-4K", 4096},
    {"Large-8K", 8192}
};

constexpr size_t NUM_THREADS = 64;
constexpr size_t NUM_ALLOCATIONS = 100000;

std::mutex output_mutex;

void thread_function(const size_t size)
{
    std::vector<void*> allocated_ptrs;
    allocated_ptrs.reserve(NUM_ALLOCATIONS);

    for (size_t i = 0; i < NUM_ALLOCATIONS; ++i)
    {
        void* ptr = Jallocator::allocate(size);
        allocated_ptrs.push_back(ptr);
    }

    for (const auto ptr : allocated_ptrs)
        Jallocator::deallocate(ptr);

    std::lock_guard lock(output_mutex);
}

void test_allocator_thread_safety(size_t size)
{
    std::cout << "Testing multi-threaded allocations for size: " << size << std::endl;

    std::vector<std::thread> threads;
    for (size_t i = 0; i < NUM_THREADS; ++i)
        threads.emplace_back(thread_function, size);

    for (auto& thread : threads)
        thread.join();

    std::cout << "Test completed for size: " << size << std::endl;
}

int main()
{
    for (const auto& alloc_size : SIZES)
        test_allocator_thread_safety(alloc_size.size);

    std::cout << "All multi-threaded safety tests completed successfully." << std::endl;
    return 0;
}
