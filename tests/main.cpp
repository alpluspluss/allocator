#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include "jalloc.hpp"

template<typename Allocator>
void benchmark_single_threaded(const char *name, size_t size, int num_ops = 10000)
{
    std::vector<void *> ptrs;
    ptrs.reserve(num_ops);

    auto start = std::chrono::high_resolution_clock::now();

    for (auto i = 0; i < num_ops; ++i)
    {
        if constexpr (std::is_same_v<Allocator, Jallocator>)
        {
            ptrs.push_back(Allocator::allocate(size));
        } else
        {
            ptrs.push_back(malloc(size));
        }
    }

    for (auto ptr: ptrs)
    {
        if constexpr (std::is_same_v<Allocator, Jallocator>)
        {
            Allocator::deallocate(ptr);
        } else
        {
            free(ptr);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double ops_per_sec = (num_ops * 2.0) / (duration.count() / 1000000.0);

    std::cout << std::left << std::setw(12) << name
            << "| Single | Size: " << std::setw(6) << size
            << "| Ops: " << std::setw(6) << num_ops
            << "| Time: " << std::fixed << std::setprecision(3) << std::setw(8)
            << duration.count() / 1000.0
            << "ms | Ops/sec: " << std::scientific << std::setprecision(2)
            << ops_per_sec << '\n';
}

template<typename Allocator>
void benchmark_multi_threaded(const char *name, size_t size, int num_threads = 32, int ops_per_thread = 10000)
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    const auto start = std::chrono::high_resolution_clock::now();

    for (auto i = 0; i < num_threads; ++i)
    {
        threads.emplace_back([size, ops_per_thread]
        {
            std::vector<void *> ptrs;
            ptrs.reserve(ops_per_thread);

            for (auto j = 0; j < ops_per_thread; ++j)
            {
                if constexpr (std::is_same_v<Allocator, Jallocator>)
                {
                    ptrs.push_back(Allocator::allocate(size));
                } else
                {
                    ptrs.push_back(malloc(size));
                }
            }

            for (auto ptr: ptrs)
            {
                if constexpr (std::is_same_v<Allocator, Jallocator>)
                {
                    Allocator::deallocate(ptr);
                } else
                {
                    free(ptr);
                }
            }
        });
    }

    for (std::iterator_traits<std::__pointer<std::thread, std::allocator<std::thread> >::type>::reference thread:
         threads)
    {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double ops_per_sec = (num_threads * ops_per_thread * 2.0) / (duration.count() / 1000000.0);

    std::cout << std::left << std::setw(12) << name
            << "| Multi  | Size: " << std::setw(6) << size
            << "| Threads: " << std::setw(3) << num_threads
            << "| Time: " << std::fixed << std::setprecision(3) << std::setw(8)
            << duration.count() / 1000.0
            << "ms | Ops/sec: " << std::scientific << std::setprecision(2)
            << ops_per_sec << '\n';
}

struct StdAllocator
{
};

int main()
{
    std::cout << "Benchmarking Allocators\n";
    std::cout << std::string(std::string::size_type(80), '-') << "\n";

    std::vector<size_t> sizes = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

    // Single-threaded benchmarks
    std::cout << "\nSingle-threaded benchmarks:\n";
    std::cout << std::string(std::string::size_type(80), '-') << "\n";
    for (auto size: sizes)
    {
        benchmark_single_threaded<Jallocator>("Jallocator", size);
        benchmark_single_threaded<StdAllocator>("StdMalloc", size);
        std::cout << std::string(std::string::size_type(80), '-') << "\n";
    }

    std::cout << "\nMulti-threaded benchmarks:\n";
    std::cout << std::string(std::string::size_type(80), '-') << "\n";
    for (auto size: sizes)
    {
        benchmark_multi_threaded<Jallocator>("Jallocator", size);
        benchmark_multi_threaded<StdAllocator>("StdMalloc", size);
        std::cout << std::string(std::string::size_type(80), '-') << "\n";
    }

    return 0;
}
