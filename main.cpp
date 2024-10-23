#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "jalloc.hpp"

constexpr size_t NUM_ITERATIONS = 1000000;
constexpr size_t NUM_RUNS = 10;

struct AllocationSize
{
    const char* name;
    size_t size;
};

constexpr AllocationSize SIZES[] = {
    {"Tiny", 8},
    {"Small", 128},
    {"Medium", 512},
    {"Large", 4096}
};

template<typename F>
double measure_time_ms(F&& func)
{
    const auto start = std::chrono::high_resolution_clock::now();
    func();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct BenchmarkResult
{
    double alloc_time;
    double dealloc_time;
    double total_time;
    size_t memory_ops;
};

BenchmarkResult bench_jallocator(size_t size, size_t iterations)
{
    std::vector<void*> ptrs;
    ptrs.reserve(iterations);

    const double alloc_time = measure_time_ms([&]
    {
        for (size_t i = 0; i < iterations; ++i)
            ptrs.push_back(Jallocator::allocate(size));
    });

    const double dealloc_time = measure_time_ms([&]
    {
        for (const auto ptr : ptrs)
            Jallocator::deallocate(ptr);
    });

    return {alloc_time, dealloc_time, alloc_time + dealloc_time, iterations * 2};
}

BenchmarkResult bench_malloc(size_t size, const size_t iterations)
{
    std::vector<void*> ptrs;
    ptrs.reserve(iterations);

    const double alloc_time = measure_time_ms([&]()
    {
        for (size_t i = 0; i < iterations; ++i)
            ptrs.push_back(malloc(size));
    });

    const double dealloc_time = measure_time_ms([&]()
    {
        for (const auto ptr: ptrs)
            free(ptr);
    });

    return {alloc_time, dealloc_time, alloc_time + dealloc_time, iterations * 2};
}

void print_results(const char* name, const size_t size, const std::vector<BenchmarkResult>& jalloc_results,
                  const std::vector<BenchmarkResult>& malloc_results)
{

    auto calc_avg = [](const std::vector<BenchmarkResult>& results,
                      auto getter) -> double
    {
        double sum = 0;
        for (const auto& r : results)
            sum += getter(r);
        return sum / results.size();
    };

    double jalloc_avg_total = calc_avg(jalloc_results,
                                     [](const auto& r)
                                     {
                                         return r.total_time;
                                     });
    double malloc_avg_total = calc_avg(malloc_results,
                                     [](const auto& r)
                                     {
                                         return r.total_time;
                                     });

    double ops_per_sec_jalloc = (jalloc_results[0].memory_ops /
                                jalloc_avg_total) * 1000;
    double ops_per_sec_malloc = (malloc_results[0].memory_ops /
                                malloc_avg_total) * 1000;

    const double speedup = malloc_avg_total / jalloc_avg_total;

    std::cout << "\n=== " << name << " (" << size << " bytes) ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Jallocator: " << ops_per_sec_jalloc << " ops/sec\n";
    std::cout << "Malloc:     " << ops_per_sec_malloc << " ops/sec\n";
    std::cout << "Speedup:    " << speedup << "x\n";
}

int main()
{
    std::cout << "Running benchmarks (" << NUM_ITERATIONS
              << " iterations, " << NUM_RUNS << " runs)...\n";

    for (const auto& [name, size] : SIZES)
    {
        std::vector<BenchmarkResult> jalloc_results;
        std::vector<BenchmarkResult> malloc_results;

        for (size_t run = 0; run < NUM_RUNS; ++run)
        {
            jalloc_results.push_back(bench_jallocator(size, NUM_ITERATIONS));
            malloc_results.push_back(bench_malloc(size, NUM_ITERATIONS));
        }

        print_results(name, size, jalloc_results, malloc_results);
    }

    return 0;
}