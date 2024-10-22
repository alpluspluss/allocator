#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "allocator.hpp"

void BENCH_MALLOC(const size_t num_allocations, const size_t block_size)
{
    std::vector<void*> ptrs;
    ptrs.reserve(num_allocations);

    const auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_allocations; ++i)
    {
        void* ptr = malloc(block_size);

        if (i % 10 == 0)
        {
            ptrs.emplace_back(ptr);
        }
        else
        {
            free(ptr);
        }

        if (i % 1000 == 0 && !ptrs.empty())
        {
            const size_t mid = ptrs.size() / 2;
            for (size_t j = 0; j < mid; ++j)
                free(ptrs[j]);
            ptrs.erase(ptrs.begin(), ptrs.begin() + mid);
        }
    }

    for (void* ptr : ptrs)
    {
        free(ptr);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - start;

    std::cout << "Standard malloc: " << std::fixed << std::setprecision(6)
              << duration.count() << " seconds" << std::endl;
}

void BENCH_CUSTOM(const size_t num_allocations, const size_t block_size)
{
    std::vector<void*> ptrs;
    ptrs.reserve(num_allocations);

    const auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_allocations; ++i)
    {
        void* ptr = Allocator::allocate(block_size);

        if (i % 10 == 0)
        {
            ptrs.emplace_back(ptr);
        }
        else
        {
            Allocator::deallocate(ptr);
        }

        if (i % 1000 == 0 && !ptrs.empty())
        {
            const size_t mid = ptrs.size() / 2;
            for (size_t j = 0; j < mid; ++j)
            {
                if (ptrs[j])
                    Allocator::deallocate(ptrs[j]);
            }
            ptrs.erase(ptrs.begin(), ptrs.begin() + mid);
        }
    }

    for (void* ptr : ptrs)
    {
        if (ptr)
        {
            Allocator::deallocate(ptr);
        }
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - start;

    std::cout << "Custom Allocator: " << std::fixed << std::setprecision(6)
              << duration.count() << " seconds" << std::endl;
}

int main()
{
    for (const std::vector<size_t> sizes = { 8, 16, 32, 64, 128, 256, 512, 1024, 4096 }; const size_t block_size : sizes)
    {
        constexpr static size_t num_allocations = 1024;
        std::cout << "\nBenchmarking " << num_allocations << " allocations of "
                  << block_size << " bytes each\n";

        BENCH_MALLOC(num_allocations, block_size);
        BENCH_CUSTOM(num_allocations, block_size);
    }

    return 0;
}