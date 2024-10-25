#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
#include "testsuit.h"

constexpr size_t NUM_ITERATIONS = 100000;
constexpr size_t NUM_RUNS = 100;
constexpr size_t WARMUP_RUNS = 10;
constexpr size_t THREAD_COUNTS[] = { 1, 2, 4, 8, 16, 32 };

struct AllocationSize
{
    const char* name;
    size_t size;
};

constexpr AllocationSize SIZES[] =
{
    {"Tiny-8", 8}, {"Tiny-16", 16}, {"Tiny-24", 24},
    {"Small-32", 32}, {"Small-64", 64}, {"Small-128", 128},
    {"Medium-256", 256}, {"Medium-512", 512}, {"Medium-1K", 1024},
    {"Large-2K", 2048}, {"Large-4K", 4096}, {"Large-8K", 8192}
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

BenchmarkResult run_benchmark(const size_t size, const size_t iterations, bool use_jallocator)
{
    std::vector<void *> ptrs;
    ptrs.reserve(iterations);

    const double alloc_time = measure_time_ms([&]
    {
        for (size_t i = 0; i < iterations; ++i)
            ptrs.emplace_back(use_jallocator ? Jallocator::allocate(size) : malloc(size));
    });

    const double dealloc_time = measure_time_ms([&]
    {
        for (const auto ptr : ptrs)
            use_jallocator ? Jallocator::deallocate(ptr) : free(ptr);
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
    std::transform(times.begin(), times.end(), diff.begin(),
               [mean](const double x) { return x - mean; });

    const double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    const double stddev = std::sqrt(sq_sum / (times.size() - 1));

    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    const double median = sorted_times[sorted_times.size() / 2];
    const double min = *std::min_element(times.begin(), times.end());
    const double max = *std::max_element(times.begin(), times.end());
    const double ops_per_sec = results[0].memory_ops / mean * 1000;

    return {mean, median, stddev, min, max, ops_per_sec};
}

void run_multi_threaded_benchmarks(const char* name, size_t size, size_t num_threads)
{
    std::vector<std::thread> threads;
    std::vector<BenchmarkResult> jalloc_results;
    std::vector<BenchmarkResult> malloc_results;

    auto thread_func = [&](std::vector<BenchmarkResult>& results, bool use_jallocator)
    {
        results.push_back(run_benchmark(size, NUM_ITERATIONS / num_threads, use_jallocator));
    };

    for (size_t run = 0; run < NUM_RUNS; ++run)
    {
        jalloc_results.clear();
        malloc_results.clear();

        for (size_t t = 0; t < num_threads; ++t)
        {
            threads.emplace_back(thread_func, std::ref(jalloc_results), true);
            threads.emplace_back(thread_func, std::ref(malloc_results), false);
        }

        for (auto& thread : threads)
            thread.join();

        threads.clear();
    }

    const auto jalloc_stats = calculate_stats(jalloc_results);
    const auto malloc_stats = calculate_stats(malloc_results);

    const double speedup = malloc_stats.mean / jalloc_stats.mean;

    std::cout << "\n=== " << name << " - " << size << " bytes - "
              << num_threads << " Threads ===\n";

    auto print_stats = [](const char* allocator_name, const BenchmarkStats& stats)
    {
        std::cout << allocator_name << ":\n"
                  << "  Ops/sec:  " << stats.ops_per_sec << "\n"
                  << "  Mean:     " << stats.mean << " ms\n"
                  << "  Median:   " << stats.median << " ms\n"
                  << "  StdDev:   " << stats.stddev << " ms\n"
                  << "  Min/Max:  " << stats.min << "/" << stats.max << " ms\n";
    };

    print_stats("Jallocator", jalloc_stats);
    print_stats("Malloc    ", malloc_stats);
    std::cout << "Speedup: " << speedup << "x\n";
}

int main()
{
    std::cout << "Running multi-threaded benchmarks:\n"
              << "- " << NUM_ITERATIONS << " iterations per run\n"
              << "- " << NUM_RUNS << " runs\n"
              << "- " << WARMUP_RUNS << " warmup runs\n\n";

    for (const auto& [name, size] : SIZES)
    {
        for (size_t num_threads : THREAD_COUNTS)
        {
            for (size_t i = 0; i < WARMUP_RUNS; ++i)
            {
                run_benchmark(size, NUM_ITERATIONS / num_threads, true);
                run_benchmark(size, NUM_ITERATIONS / num_threads, false);
            }
            run_multi_threaded_benchmarks(name, size, num_threads);
        }
    }
    return 0;
}
