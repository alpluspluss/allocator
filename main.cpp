#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "jalloc.hpp"

constexpr size_t NUM_ITERATIONS = 100;
constexpr size_t NUM_RUNS = 100;
constexpr size_t WARMUP_RUNS = 10;

struct AllocationSize
{
    const char* name;
    size_t size;
};

constexpr AllocationSize SIZES[] =
{
    {"Tiny-8", 8}, // Tiny allocations (small POD types)
    {"Tiny-16", 16}, // Common small string size
    {"Tiny-24", 24}, // Small class/struct size
    {"Small-32", 32}, // Common allocation size
    {"Small-64", 64}, // Cache line size
    {"Small-128", 128}, // Medium struct/class
    {"Medium-256", 256}, // Larger objects
    {"Medium-512", 512}, // Buffer sizes
    {"Medium-1K", 1024}, // Common buffer size
    {"Large-2K", 2048}, // Large objects
    {"Large-4K", 4096}, // Page size
    {"Large-8K", 8192} // Multiple pages
};

struct BenchmarkStats
{
    double mean;
    double median;
    double stddev;
    double min;
    double max;
    double ops_per_sec;
};

struct BenchmarkResult
{
    double alloc_time;
    double dealloc_time;
    double total_time;
    size_t memory_ops;
};

template<typename F>
double measure_time_ms(F&& func)
{
    const auto start = std::chrono::high_resolution_clock::now();
    func();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

BenchmarkResult bench_jallocator(const size_t size, const size_t iterations)
{
    std::vector<void *> ptrs;
    ptrs.reserve(iterations);

    const double alloc_time = measure_time_ms([&]
    {
        for (size_t i = 0; i < iterations; ++i)
            ptrs.emplace_back(Jallocator::allocate(size));
    });

    const double dealloc_time = measure_time_ms([&]
    {
        for (const auto ptr : ptrs)
            Jallocator::deallocate(ptr);
    });

    return {alloc_time, dealloc_time, alloc_time + dealloc_time, iterations * 2};
}

BenchmarkResult bench_malloc(const size_t size, const size_t iterations)
{
    std::vector<void *> ptrs;
    ptrs.reserve(iterations);

    const double alloc_time = measure_time_ms([&]
    {
        for (size_t i = 0; i < iterations; ++i)
            ptrs.emplace_back(malloc(size));
    });

    const double dealloc_time = measure_time_ms([&]
    {
        for (const auto ptr : ptrs)
            free(ptr);
    });

    return {alloc_time, dealloc_time, alloc_time + dealloc_time, iterations * 2};
}

BenchmarkStats calculate_stats(const std::vector<BenchmarkResult>& results)
{
    std::vector<double> times;
    times.reserve(results.size());

    for (const auto&[alloc_time, dealloc_time, total_time, memory_ops] : results)
        times.emplace_back(total_time);

    const double sum = std::accumulate(times.begin(), times.end(), 0.0);
    const double mean = sum / times.size();

    std::vector<double> diff(times.size());
    std::ranges::transform(times, diff.begin(),
                           [mean](const double x)
                           {
                               return x - mean;
                           });

    const double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    const double stddev = std::sqrt(sq_sum / (times.size() - 1));

    std::vector<double> sorted_times = times;
    std::ranges::sort(sorted_times);
    const double median = sorted_times[sorted_times.size() / 2];

    const double min = *std::ranges::min_element(times);
    const double max = *std::ranges::max_element(times);

    const double ops_per_sec = results[0].memory_ops / mean * 1000;

    return {mean, median, stddev, min, max, ops_per_sec};
}

void print_detailed_results(const char* name, size_t size,
                          const BenchmarkStats& jalloc_stats,
                          const BenchmarkStats& malloc_stats)
{
    const double speedup = malloc_stats.mean / jalloc_stats.mean;

    std::cout << "\n=== " << std::left << std::setw(12) << name
              << " (" << std::right << std::setw(6) << size << " bytes) ===\n";

    std::cout << std::fixed << std::setprecision(2);

    auto print_stats = [](const char* allocator_name, const BenchmarkStats& stats)
    {
        std::cout << allocator_name << ":\n"
                  << "  Ops/sec:  " << std::setw(12) << stats.ops_per_sec << "\n"
                  << "  Mean:     " << std::setw(8) << stats.mean << " ms\n"
                  << "  Median:   " << std::setw(8) << stats.median << " ms\n"
                  << "  StdDev:   " << std::setw(8) << stats.stddev << " ms\n"
                  << "  Min/Max:  " << std::setw(8) << stats.min << "/"
                  << std::setw(8) << stats.max << " ms\n";
    };

    print_stats("Jallocator", jalloc_stats);
    print_stats("Malloc    ", malloc_stats);

    std::cout << "\nSpeedup: " << std::setw(6) << speedup << "x\n";

    constexpr auto bar_width = 50;
    constexpr int jalloc_bar = bar_width;
    const auto malloc_bar = static_cast<int>(bar_width / speedup);

    std::cout << "\nPerformance comparison (longer is better):\n";
    std::cout << "Jallocator: [" << std::string(jalloc_bar, '=') << "]\n";
    std::cout << "Malloc:     [" << std::string(malloc_bar, '=')
              << std::string(jalloc_bar - malloc_bar, ' ') << "]\n";
}

int main()
{
    std::cout << "Running benchmarks with:\n"
              << "- " << NUM_ITERATIONS << " iterations per run\n"
              << "- " << NUM_RUNS << " measured runs\n"
              << "- " << WARMUP_RUNS << " warmup runs\n\n";

    for (const auto& [name, size] : SIZES)
    {
        for (size_t i = 0; i < WARMUP_RUNS; ++i)
        {
            bench_jallocator(size, NUM_ITERATIONS);
            bench_malloc(size, NUM_ITERATIONS);
        }

        std::vector<BenchmarkResult> jalloc_results;
        std::vector<BenchmarkResult> malloc_results;

        jalloc_results.reserve(NUM_RUNS);
        malloc_results.reserve(NUM_RUNS);

        for (size_t run = 0; run < NUM_RUNS; ++run)
        {
            jalloc_results.push_back(bench_jallocator(size, NUM_ITERATIONS));
            malloc_results.push_back(bench_malloc(size, NUM_ITERATIONS));
        }

        const auto jalloc_stats = calculate_stats(jalloc_results);
        const auto malloc_stats = calculate_stats(malloc_results);

        print_detailed_results(name, size, jalloc_stats, malloc_stats);
    }
    return 0;
}