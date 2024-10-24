// jalloc.hpp - Just an Allocator™
// A high-performance, thread-safe memory allocator for C/C++
//
// Features:
// - Thread-safe & high-performance memory allocation
// - Multi-tiered allocation strategy for different sizes
// - SIMD-optimized memory operations
// - Automatic memory coalescing and return-to-OS
//
// Version: 0.1.1-unsafe
// Author: alpluspluss
// Created: 10/22/2024
// License: MIT

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <thread>

// Architecture-specific SIMD support detection and configuration
#if defined(__x86_64__)
    #include <immintrin.h>
    #include <emmintrin.h>
    #include <xmmintrin.h>
    #include <smmintrin.h>
    #include <tmmintrin.h>
    #ifdef __AVX2__
        #include <avx2intrin.h>
    #endif
    #ifdef __AVX512F__
        #include <avx512fintrin.h>
    #endif

    #ifdef _MSC_VER
        #include <intrin.h>
    #endif

    // CPU Feature Detection functions
    //--------------------------------------------------------------------------
    // Detects availability of AVX2 and AVX-512 instructions at runtime
    // Returns: true if the feature is available, false otherwise
    //--------------------------------------------------------------------------
    #ifdef __GNUC__
        #include <cpuid.h>
        ALWAYS_INLINE static bool cpu_has_avx2()
        {
            unsigned int eax, ebx, ecx, edx;
            __get_cpuid(7, &eax, &ebx, &ecx, &edx);
            return (ebx & bit_AVX2) != 0;
        }

        ALWAYS_INLINE static bool cpu_has_avx512f()
        {
            unsigned int eax, ebx, ecx, edx;
            __get_cpuid(7, &eax, &ebx, &ecx, &edx);
            return (ebx & bit_AVX512F) != 0;
        }
    #elif defined(_MSC_VER)
        ALWAYS_INLINE static bool cpu_has_avx2()
        {
            int cpuInfo[4];
            __cpuid(cpuInfo, 7);
            return (cpuInfo[1] & (1 << 5)) != 0;
        }

        ALWAYS_INLINE static bool cpu_has_avx512f()
        {
            int cpuInfo[4];
            __cpuid(cpuInfo, 7);
            return (cpuInfo[1] & (1 << 16)) != 0;
        }
    #endif

    //--------------------------------------------------------------------------
    // SIMD Operation Definitions
    // Provides unified interface for different SIMD instruction sets
    //--------------------------------------------------------------------------
    #ifdef __AVX512F__
        // AVX-512: 64-byte vector operations
        #define VECTOR_WIDTH 64
        #define STREAM_STORE_VECTOR(addr, val) _mm512_stream_si512((__m512i*)(addr), val)
        #define LOAD_VECTOR(addr) _mm512_loadu_si512((const __m512i*)(addr))
        #define STORE_VECTOR(addr, val) _mm512_storeu_si512((__m512i*)(addr), val)
        #define SET_ZERO_VECTOR() _mm512_setzero_si512()
    #elif defined(__AVX2__)
        // AVX2: 32-byte vector operations
        #define VECTOR_WIDTH 32
        #define STREAM_STORE_VECTOR(addr, val) _mm256_stream_si256((__m256i*)(addr), val)
        #define LOAD_VECTOR(addr) _mm256_loadu_si256((const __m256i*)(addr))
        #define STORE_VECTOR(addr, val) _mm256_storeu_si256((__m256i*)(addr), val)
        #define SET_ZERO_VECTOR() _mm256_setzero_si256()
    #else
        // SSE: 16-byte vector operations (fallback)
        #define VECTOR_WIDTH 16
        #define STREAM_STORE_VECTOR(addr, val) _mm_stream_si128((__m128i*)(addr), val)
        #define LOAD_VECTOR(addr) _mm_loadu_si128((const __m128i*)(addr))
        #define STORE_VECTOR(addr, val) _mm_storeu_si128((__m128i*)(addr), val)
        #define SET_ZERO_VECTOR() _mm_setzero_si128()
    #endif

    // Common operations available across all x86_64 platforms
    #define STREAM_STORE_64(addr, val) _mm_stream_si64((__int64*)(addr), val)
    #define CPU_PAUSE() _mm_pause()
    #define MEMORY_FENCE() _mm_sfence()
    #define CUSTOM_PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
    #define PREFETCH_WRITE(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
    #define PREFETCH_READ(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_NTA)

#elif defined(__arm__) || defined(__aarch64__)
    // ARM/AArch64 SIMD support using NEON
    #include <arm_neon.h>
    #ifdef __clang__
        #include <arm_acle.h>
    #endif

    // ARM NEON: 16-byte vector operations
    #define VECTOR_WIDTH 16

    #if defined(__aarch64__)
        // 64-bit ARM specific optimizations
        #define STREAM_STORE_VECTOR(addr, val) vst1q_u8((uint8_t*)(addr), val)
        #define LOAD_VECTOR(addr) vld1q_u8((const uint8_t*)(addr))
        #define STORE_VECTOR(addr, val) vst1q_u8((uint8_t*)(addr), val)
        #define SET_ZERO_VECTOR() vdupq_n_u8(0)
        #define STREAM_STORE_64(addr, val) vst1_u64((uint64_t*)(addr), vcreate_u64(val))
        #define MEMORY_FENCE() __dmb(SY)

        // ARM-specific prefetch hints
        #define CUSTOM_PREFETCH(addr) __asm__ volatile("prfm pldl1keep, [%0]" : : "r" (addr))
        #define PREFETCH_WRITE(addr) __asm__ volatile("prfm pstl1keep, [%0]" : : "r" (addr))
        #define PREFETCH_READ(addr) __asm__ volatile("prfm pldl1strm, [%0]" : : "r" (addr))
    #else
        // 32-bit ARM fallbacks
        #define STREAM_STORE_VECTOR(addr, val) vst1q_u8((uint8_t*)(addr), val)
        #define LOAD_VECTOR(addr) vld1q_u8((const uint8_t*)(addr))
        #define STORE_VECTOR(addr, val) vst1q_u8((uint8_t*)(addr), val)
        #define SET_ZERO_VECTOR() vdupq_n_u8(0)
        #define STREAM_STORE_64(addr, val) *((int64_t*)(addr)) = val
        #define MEMORY_FENCE() __dmb(SY)
        #define CUSTOM_PREFETCH(addr) __pld(reinterpret_cast<const char*>(addr))
        #define PREFETCH_WRITE(addr) __pld(reinterpret_cast<const char*>(addr))
        #define PREFETCH_READ(addr) __pld(reinterpret_cast<const char*>(addr))
    #endif

    #define CPU_PAUSE() __yield()

#else
    // Generic fallback for unsupported architectures
    // Provides basic functionality without SIMD optimizations
    #define VECTOR_WIDTH 8
    #define STREAM_STORE_VECTOR(addr, val) *((int64_t*)(addr)) = val
    #define LOAD_VECTOR(addr) *((const int64_t*)(addr))
    #define STORE_VECTOR(addr, val) *((int64_t*)(addr)) = val
    #define SET_ZERO_VECTOR() 0
    #define STREAM_STORE_64(addr, val) *((int64_t*)(addr)) = val
    #define CPU_PAUSE() ((void)0)
    #define MEMORY_FENCE() std::atomic_thread_fence(std::memory_order_seq_cst)
    #define CUSTOM_PREFETCH(addr) ((void)0)
    #define PREFETCH_WRITE(addr) ((void)0)
    #define PREFETCH_READ(addr) ((void)0)
#endif

// Compiler-specific optimizations and attributes
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang specific optimizations
    #define LIKELY(x) __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define ALWAYS_INLINE [[gnu::always_inline]] inline
    #define ALIGN_TO(x) __attribute__((aligned(x)))

    #if defined(__clang__)
        // Clang-specific optimizations
        #define HAVE_BUILTIN_ASSUME(x) __builtin_assume(x)
        #define HAVE_BUILTIN_ASSUME_ALIGNED(x, a) __builtin_assume_aligned(x, a)
        #define NO_SANITIZE_ADDRESS __attribute__((no_sanitize("address")))
        #define VECTORIZE_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
    #else
        // GCC-specific fallbacks
        #define HAVE_BUILTIN_ASSUME(x) ((void)0)
        #define HAVE_BUILTIN_ASSUME_ALIGNED(x, a) (x)
        #define NO_SANITIZE_ADDRESS
        #define VECTORIZE_LOOP _Pragma("GCC ivdep")
    #endif
#elif defined(_MSC_VER)
    // MSVC specific optimizations
    #define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
    #define ALWAYS_INLINE __forceinline
    #define ALIGN_TO(x) __declspec(align(x))
    #define HAVE_BUILTIN_ASSUME(x) __assume(x)
    #define HAVE_BUILTIN_ASSUME_ALIGNED(x, a) (x)
    #define NO_SANITIZE_ADDRESS
    #define VECTORIZE_LOOP
    #define CUSTOM_PREFETCH(addr) ((void)0)
#else
    // Generic fallbacks for other compilers
    #define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
    #define ALWAYS_INLINE inline
    #define ALIGN_TO(x)
    #define HAVE_BUILTIN_ASSUME(x) ((void)0)
    #define HAVE_BUILTIN_ASSUME_ALIGNED(x, a) (x)
    #define NO_SANITIZE_ADDRESS
    #define VECTORIZE_LOOP
    #define CUSTOM_PREFETCH(addr) ((void)0)
#endif

// Platform-specific memory management operations
#ifdef _WIN32
    #include <malloc.h>
    #include <windows.h>
    // Windows virtual memory management
    #define MAP_MEMORY(size) \
        VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    #define UNMAP_MEMORY(ptr, size) VirtualFree(ptr, 0, MEM_RELEASE)
    #ifndef MAP_FAILED
        #define MAP_FAILED nullptr
    #endif
    #define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#elif defined(__APPLE__)
    // macOS memory management
    #include <mach/mach.h>
#include <sys/mman.h>

    #define MAP_MEMORY(size) \
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    #define UNMAP_MEMORY(ptr, size) munmap(ptr, size)
    #define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
    #define ALIGNED_FREE(ptr) free(ptr)

#else
    // POSIX-compliant systems (Linux, BSD, etc.)
    #include <sched.h>
    #include <unistd.h>
    #include <sys/mman.h>

    #define MAP_MEMORY(size) \
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    #define UNMAP_MEMORY(ptr, size) munmap(ptr, size)
    #define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif


// Allocator constants
// Hardware constants
static constexpr size_t CACHE_LINE_SIZE = 64;
static constexpr size_t PG_SIZE = 4096;

// Block header
static constexpr size_t TINY_LARGE_THRESHOLD = 64;
static constexpr size_t SMALL_LARGE_THRESHOLD = 256;
static constexpr size_t ALIGNMENT = CACHE_LINE_SIZE;
static constexpr size_t LARGE_THRESHOLD = 1024 * 1024;

static constexpr size_t CACHE_SIZE = 32;
static constexpr size_t SIZE_CLASSES = 32;

// Safety flags
static constexpr uint64_t SIZE_MASK = 0x0000FFFFFFFFFFFF;
static constexpr uint64_t CLASS_MASK = 0x00FF000000000000;
static constexpr uint64_t MMAP_FLAG = 1ULL << 62;
static constexpr uint64_t COALESCED_FLAG = 1ULL << 61;
static constexpr uint64_t MAGIC_MASK = 0xF000000000000000;
static constexpr uint64_t MAGIC_VALUE = 0xA000000000000000;
static constexpr uint64_t THREAD_OWNER_MASK = 0xFFFF000000000000;

struct size_class
{
    uint16_t size;
    uint16_t slot_size;
    uint16_t blocks;
    uint16_t slack;
};

constexpr std::array<size_class, 32> size_classes = []
{
    std::array<size_class, 32> classes{};
    for (size_t i = 0; i < 32; ++i)
    {
        const size_t size = 1ULL << (i + 3);
        const size_t slot = size + ALIGNMENT - 1 & ~(ALIGNMENT - 1);
        classes[i] = {
            static_cast<uint16_t>(size),
            static_cast<uint16_t>(slot),
            static_cast<uint16_t>(PG_SIZE / slot),
            static_cast<uint16_t>(slot - size)
        };
    }
    return classes;
}();

struct thread_cache_t
{
    struct cached_block
    {
        void* ptr;
        uint8_t size_class;
    };

    struct size_class_cache
    {
        cached_block blocks[CACHE_SIZE];
        size_t count;
    };

    alignas(CACHE_LINE_SIZE) size_class_cache caches[SIZE_CLASSES]{};

    ALWAYS_INLINE
    void* get(const uint8_t size_class) noexcept
    {
        if (auto&[blocks, count] = caches[size_class]; LIKELY(count > 0))
            return blocks[--count].ptr;
        return nullptr;
    }

    ALWAYS_INLINE
    bool put(void* ptr, const uint8_t size_class) noexcept
    {
        if (auto&[blocks, count] = caches[size_class]; LIKELY(count < CACHE_SIZE))
        {
            blocks[count].ptr = ptr;
            blocks[count].size_class = size_class;
            ++count;
            return true;
        }
        return false;
    }

    ALWAYS_INLINE
    void clear() noexcept
    {
        for (auto&[blocks, count] : caches)
            count = 0;
    }
};

ALWAYS_INLINE
static void prefetch_for_read(const void* addr) noexcept
{
    CUSTOM_PREFETCH(addr);
}

template<size_t stride = 64>
ALWAYS_INLINE
static void prefetch_range(const void* addr, const size_t size) noexcept
{
    auto ptr = static_cast<const char*>(addr);

    for (const char* end = ptr + size; ptr < end; ptr += stride)
        CUSTOM_PREFETCH(ptr);
}

template<typename T>
ALWAYS_INLINE
static void stream_store(void* dst, const T& value) noexcept
{
    STREAM_STORE_64(dst, static_cast<int64_t>(value));
}

#if defined(__x86_64__)
    static ALWAYS_INLINE size_t count_trailing_zeros(uint64_t x)
    {
        #ifdef _MSC_VER
            return _tzcnt_u64(x);
        #else
             return __builtin_ctzll(x);
        #endif
    }

    static ALWAYS_INLINE void memory_fence()
    {
    _mm_mfence();
    }

    static ALWAYS_INLINE void prefetch(const void* addr)
    {
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
    }
#elif defined(__arm__) || defined(__aarch64__)
    ALWAYS_INLINE
    static size_t count_trailing_zeros(const uint64_t x)
    {
        return __builtin_ctzll(x);
    }
    ALWAYS_INLINE
    static void memory_fence()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    ALWAYS_INLINE
    static void prefetch(const void* addr)
    {
        prefetch_for_read(addr);
    }
#else
    static ALWAYS_INLINE size_t count_trailing_zeros(uint64_t x)
    {
        return __builtin_ctzll(x);
    }

    static ALWAYS_INLINE void memory_fence()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static ALWAYS_INLINE void prefetch(const void*) {}
#endif

class Jallocator
{
    struct bitmap
    {
        static constexpr size_t bits_per_word = 64;
        static constexpr size_t words_per_bitmap = PG_SIZE / (CACHE_LINE_SIZE * 8);

        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> words[words_per_bitmap];

        bitmap() noexcept
        {
            for (auto& word : words)
                word = ~0ULL;
        }

        ALWAYS_INLINE
        size_t find_free_block() noexcept
        {
            #if defined(__AVX512F__)
                for (size_t i = 0; i < words_per_bitmap; i += 8)
                {
                    __m512i v = _mm512_loadu_si512((__m512i*)(words + i));
                    uint64_t mask = _mm512_test_epi64_mask(v, v);
                    if (mask != 0)
                        return i * 64 + __builtin_ctzll(_mm512_mask2int(mask));
                }
            #elif defined(__AVX2__)
                for (size_t i = 0; i < words_per_bitmap; i += 4)
                {
                    __m256i v = _mm256_loadu_si256((__m256i*)(words + i));
                    __m256i zero = _mm256_setzero_si256();
                    __m256i cmp = _mm256_cmpeq_epi64(v, zero);
                    uint32_t mask = ~_mm256_movemask_epi8(cmp);
                    if (mask != 0)
                        return i * 64 + __builtin_ctzll(mask);
                }
            #elif defined(__aarch64__)
                for (size_t i = 0; i < words_per_bitmap; i += 2)
                {
                    uint64x2_t v = vld1q_u64(reinterpret_cast<const uint64_t *>(words + i));
                    uint64x2_t zero = vdupq_n_u64(0);
                    if (vgetq_lane_u64(vceqq_u64(v, zero), 0) != -1ULL)
                        return i * 64 + __builtin_ctzll(words[i].load(std::memory_order_relaxed));
                    if (vgetq_lane_u64(vceqq_u64(v, zero), 1) != -1ULL)
                        return (i + 1) * 64 + __builtin_ctzll(words[i+1].load(std::memory_order_relaxed));
                }
            #endif
            for (size_t i = 0; i < words_per_bitmap; ++i)
            {
                if (i + 1 < words_per_bitmap)
                    prefetch(&words[i + 1]);

                uint64_t expected = words[i].load(std::memory_order_relaxed);
                while (expected != 0)
                {
                    const size_t bit = count_trailing_zeros(expected);

                    if (const uint64_t desired = expected & ~(1ULL << bit); words[i].compare_exchange_weak(
                        expected, desired,
                        std::memory_order_acquire,
                        std::memory_order_relaxed))
                    {
                        memory_fence();
                        return i * bits_per_word + bit;
                    }
                }
            }
            return ~static_cast<size_t>(0);
        }

        ALWAYS_INLINE
        void mark_free(const size_t index) noexcept
        {
            const size_t word_idx = index / bits_per_word;
            const size_t bit_idx = index % bits_per_word;
            prefetch(&words[word_idx]);
            words[word_idx].fetch_or(1ULL << bit_idx, std::memory_order_release);
        }

        ALWAYS_INLINE
        bool is_completely_free() const noexcept
        {
            for (size_t i = 0; i < words_per_bitmap; ++i)
            {
                if (i + 1 < words_per_bitmap)
                    prefetch(&words[i + 1]);
                if (words[i].load(std::memory_order_relaxed) != ~0ULL)
                    return false;
            }
            return true;
        }
    };

    struct alignas(ALIGNMENT) block_header
    {
        // Bit field layout:
        // [63]    - Free flag
        // [62]    - Memory mapped flag
        // [61]    - Coalesced flag
        // [56-48] - Size class
        // [47-0]  - Block size
        uint64_t data;
        block_header* prev_physical;
        block_header* next_physical;

        ALWAYS_INLINE
        void init(const size_t sz, const uint8_t size_class, const bool is_free,
             block_header* prev = nullptr, block_header* next = nullptr) noexcept
        {
            encode(sz, size_class, is_free);
            prev_physical = prev;
            next_physical = next;
        }

        ALWAYS_INLINE
        void encode(const size_t size, const uint8_t size_class, const bool is_free) noexcept
        {
            data = (size & SIZE_MASK) |
               (static_cast<uint64_t>(size_class) << 48) |
               (static_cast<uint64_t>(is_free) << 63) |
               MAGIC_VALUE;
        }

        ALWAYS_INLINE
        bool is_valid() const noexcept
        {
            return (data & MAGIC_MASK) == MAGIC_VALUE;
        }

        ALWAYS_INLINE
        void set_free(const bool is_free) noexcept
        {
            data = data & ~(1ULL << 63) | static_cast<uint64_t>(is_free) << 63;
        }

        ALWAYS_INLINE
        void set_memory_mapped(const bool is_mmap) noexcept
        {
            data = data & ~MMAP_FLAG | static_cast<uint64_t>(is_mmap) << 62;
        }

        size_t size() const noexcept
        {
            return data & SIZE_MASK;
        }
        uint8_t size_class() const noexcept
        {
            return (data & CLASS_MASK) >> 48;
        }
        bool is_free() const noexcept
        {
            return data & 1ULL << 63;
        }
        bool is_memory_mapped() const noexcept
        {
            return data & MMAP_FLAG;
        }

        // Check if the block is perfectly aligned to the cache line size (64B)
        // and verify if the pointer is corrupted or not
        // The performance trade-offs are worth it
        // Please note in mind that this DOES NOT check
        // 1. Perfectly-aligned corrupted pointers
        // 2. Maliciously-aligned pointers
        // 3.
        static bool is_aligned(const void* ptr) noexcept
        {
            return (reinterpret_cast<uintptr_t>(ptr) & ALIGNMENT - 1) == 0;
        }

        bool try_coalesce() noexcept
        {
            if (is_memory_mapped() || size_class() < tiny_block_manager::num_tiny_classes)
                return false;

            auto coalesced = false;

            if (next_physical && next_physical->is_free())
            {
                const size_t combined_size = size() + next_physical->size() + sizeof(block_header);
                next_physical = next_physical->next_physical;
                if (next_physical)
                    next_physical->prev_physical = this;
                encode(combined_size, size_class(), true);
                set_coalesced(true);
                coalesced = true;
            }

            if (prev_physical && prev_physical->is_free())
            {
                const size_t combined_size = size() + prev_physical->size() + sizeof(block_header);
                prev_physical->next_physical = next_physical;
                if (next_physical)
                    next_physical->prev_physical = prev_physical;
                prev_physical->encode(combined_size, prev_physical->size_class(), true);
                prev_physical->set_coalesced(true);
                coalesced = true;
            }
            return coalesced;
        }

        ALWAYS_INLINE
        void set_coalesced(const bool is_coalesced) noexcept
        {
            data = data & ~COALESCED_FLAG | (static_cast<uint64_t>(is_coalesced) << 61);
        }

        ALWAYS_INLINE
        bool is_coalesced() const noexcept
        {
            return data & COALESCED_FLAG;
        }
    };

    struct alignas(PG_SIZE) pool
    {
        static constexpr size_t MIN_RETURN_SIZE = 64 * 1024;
        static constexpr auto MEM_USAGE_THRESHOLD = 0.2;
        bitmap bitmap;
        uint8_t memory[PG_SIZE - sizeof(bitmap)]{};

        ALWAYS_INLINE
        void* allocate(const size_class& sc) noexcept
        {
            if (const size_t index = bitmap.find_free_block();
                index != ~static_cast<size_t>(0))
            {
                return memory + index * sc.slot_size;
            }
            return nullptr;
        }

        ALWAYS_INLINE
        void deallocate(void* ptr, const size_class& sc) noexcept
        {
            const size_t offset = static_cast<uint8_t*>(ptr) - memory;
            bitmap.mark_free(offset / sc.slot_size);
        }

        ALWAYS_INLINE
        bool is_completely_free() const noexcept
        {
            return bitmap.is_completely_free();
        }

        ALWAYS_INLINE
        void return_memory() noexcept
        {
            size_t free_space = 0;
            #if defined(__AVX512F__)
                const auto* current = reinterpret_cast<block_header*>(memory);
                __m512i sum = _mm512_setzero_si512();

                while (current)
                {
                    if (current->is_free())
                        sum = _mm512_add_epi64(sum, _mm512_set1_epi64(current->size()));
                    current = current->next_physical;
                }
                free_space = _mm512_reduce_add_epi64(sum);

            #elif defined(__AVX2__)
                const auto* current = reinterpret_cast<block_header*>(memory);
                // 4 accumulators gives the best throughput here
                __m256i sum1 = _mm256_setzero_si256();
                __m256i sum2 = _mm256_setzero_si256();
                __m256i sum3 = _mm256_setzero_si256();
                __m256i sum4 = _mm256_setzero_si256();

                while (current)
                {
                    block_header* next = current->next_physical;
                    if (next)
                        _mm_prefetch(reinterpret_cast<const char*>(next), _MM_HINT_T0);
                    // this is to reduce depend on chains
                    if (current->is_free())
                    {
                        __m256i size_vec = _mm256_set1_epi64x(current->size());
                        sum1 = _mm256_add_epi64(sum1, size_vec);
                        size_vec = _mm256_set1_epi64x(current->size() >> 1);
                        sum2 = _mm256_add_epi64(sum2, size_vec);
                        size_vec = _mm256_set1_epi64x(current->size() >> 2);
                        sum3 = _mm256_add_epi64(sum3, size_vec);
                        size_vec = _mm256_set1_epi64x(current->size() >> 3);
                        sum4 = _mm256_add_epi64(sum4, size_vec);
                    }
                    current = next;
                }

                // mash the accumulators together
                sum1 = _mm256_add_epi64(sum1, _mm256_slli_epi64(sum2, 1));
                sum3 = _mm256_add_epi64(sum3, _mm256_slli_epi64(sum4, 3));
                sum1 = _mm256_add_epi64(sum1, _mm256_slli_epi64(sum3, 2));

                // caclulate the total horizontal sum with least possible instructions
                __m128i sum_low = _mm256_extracti128_si256(sum1, 0);
                __m128i sum_high = _mm256_extracti128_si256(sum1, 1);
                __m128i sum = _mm_add_epi64(sum_low, sum_high);
                sum = _mm_add_epi64(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1,0,3,2)));
                free_space = _mm_cvtsi128_si64(sum);

            #elif defined(__aarch64__)
                auto* current = reinterpret_cast<block_header*>(memory);
                uint64x2_t sum1 = vdupq_n_u64(0);
                uint64x2_t sum2 = vdupq_n_u64(0);
                uint64x2_t sum3 = vdupq_n_u64(0);
                uint64x2_t sum4 = vdupq_n_u64(0);

                while (current)
                {
                    block_header* next = current->next_physical;
                    if (next)
                    {
                       prefetch(next);
                    }

                    if (current->is_free())
                    {
                        const uint64_t size = current->size();
                        sum1 = vaddq_u64(sum1, vdupq_n_u64(size));
                        sum2 = vaddq_u64(sum2, vdupq_n_u64(size >> 1));
                        sum3 = vaddq_u64(sum3, vdupq_n_u64(size >> 2));
                        sum4 = vaddq_u64(sum4, vdupq_n_u64(size >> 3));
                    }
                    current = next;
                }

                const uint64x2_t sum_12 = vaddq_u64(sum1, vshlq_n_u64(sum2, 1));
                const uint64x2_t sum_34 = vaddq_u64(sum3, vshlq_n_u64(sum4, 3));
                const uint64x2_t final_sum = vaddq_u64(sum_12, vshlq_n_u64(sum_34, 2));

                free_space = vgetq_lane_u64(final_sum, 0) + vgetq_lane_u64(final_sum, 1);

            #else
                const auto* current = reinterpret_cast<block_header*>(memory);
                while (current)
                {
                    if (current->is_free())
                        free_space += current->size();
                    current = current->next_physical;
                }
            #endif

            if (free_space >= MIN_RETURN_SIZE &&
                static_cast<double>(free_space) / PG_SIZE >= (1.0 - MEM_USAGE_THRESHOLD))
            {

                current = reinterpret_cast<block_header*>(memory);
                while (current)
                {
                    if (current->is_free() && current->is_coalesced())
                    {
                        void* block_start = current + 1;
                        auto page_start = reinterpret_cast<void*>(
                            (reinterpret_cast<uintptr_t>(block_start) + PG_SIZE - 1) & ~(PG_SIZE - 1));
                        auto page_end = reinterpret_cast<void*>(
                            reinterpret_cast<uintptr_t>(block_start) + current->size() & ~(PG_SIZE - 1));

                        if (page_end > page_start)
                        {
                            #ifdef _WIN32
                                VirtualAlloc(page_start,
                                           reinterpret_cast<char*>(page_end) -
                                           reinterpret_cast<char*>(page_start),
                                           MEM_RESET,
                                           PAGE_READWRITE);
                            #elif defined(__APPLE__)
                                madvise(page_start,
                                       static_cast<char*>(page_end) -
                                       static_cast<char*>(page_start),
                                       MADV_FREE);
                            #else
                                madvise(page_start,
                                       reinterpret_cast<char*>(page_end) -
                                       reinterpret_cast<char*>(page_start),
                                       MADV_DONTNEED);
                            #endif
                        }
                    }
                    current = current->next_physical;
                }
            }
        }
    };

    struct tiny_block_manager
    {
        static constexpr size_t num_tiny_classes = 8;

        struct alignas(PG_SIZE) tiny_pool
        {
            bitmap bitmap;
            uint8_t memory[PG_SIZE - sizeof(bitmap)]{};

            ALWAYS_INLINE
            void* allocate_tiny(const uint8_t size_class) noexcept
            {
                if (const size_t index = bitmap.find_free_block();
                    index != ~static_cast<size_t>(0))
                {
                    return memory + index * ((size_class + 1) << 3);
                }
                return nullptr;
            }

            ALWAYS_INLINE
            void deallocate_tiny(void* ptr, const uint8_t size_class) noexcept
            {
                const size_t offset = static_cast<uint8_t*>(ptr) - memory;
                bitmap.mark_free(offset / ((size_class + 1) << 3));
            }
        };
    };

    struct pool_manager
    {
        static constexpr size_t MAX_POOLS = 8;
        static constexpr size_t SIZE_CLASSES = 32;

        struct pool_entry
        {
            pool* p;
            size_t used_blocks;
        };

        alignas(CACHE_LINE_SIZE) pool_entry pools[SIZE_CLASSES][MAX_POOLS]{};
        size_t pool_count[SIZE_CLASSES]{};

        ALWAYS_INLINE
        void* allocate(const uint8_t size_class) noexcept
        {
            const auto& sc = size_classes[size_class];

            for (size_t i = 0; i < pool_count[size_class]; ++i)
            {
                auto&[p, used_blocks] = pools[size_class][i];
                if (void* ptr = p->allocate(sc))
                {
                    ++used_blocks;
                    return ptr;
                }
            }

            if (pool_count[size_class] < MAX_POOLS)
            {
                auto* new_pool = new (std::align_val_t{PG_SIZE}) pool();

                auto&[p, used_blocks] = pools[size_class][pool_count[size_class]];
                p = new_pool;
                used_blocks = 1;

                if (void* ptr = new_pool->allocate(sc))
                {
                    ++pool_count[size_class];
                    return ptr;
                }
                delete new_pool;
            }

            return nullptr;
        }

        ALWAYS_INLINE
        void deallocate(void* ptr, const uint8_t size_class) noexcept
        {
            if (UNLIKELY((reinterpret_cast<uintptr_t>(ptr) & ~(PG_SIZE-1)) == 0))
                return;

            const auto& sc = size_classes[size_class];
            if (UNLIKELY(size_class >= SIZE_CLASSES))
                return;

            for (size_t i = 0; i < pool_count[size_class]; ++i)
            {
                auto& entry = pools[size_class][i];
                const auto* pool_start = reinterpret_cast<const char*>(entry.p);

                if (const auto* pool_end = pool_start + PG_SIZE; ptr >= pool_start && ptr < pool_end)
                {
                    entry.p->deallocate(ptr, sc);
                    if (--entry.used_blocks == 0)
                    {
                        delete entry.p;
                        entry = pools[size_class][--pool_count[size_class]];
                    }
                    return;
                }
            }
        }

        ~pool_manager()
        {
            for (size_t sc = 0; sc < SIZE_CLASSES; ++sc)
            {
                const size_t count = pool_count[sc];
                VECTORIZE_LOOP
                for (size_t i = 0; i < count; ++i)
                    delete pools[sc][i].p;
            }
        }
    };

    static thread_local thread_cache_t thread_cache_;
    static thread_local pool_manager pool_manager_;
    static thread_local std::array<tiny_block_manager::tiny_pool*,
                                 tiny_block_manager::num_tiny_classes> tiny_pools_;

    ALWAYS_INLINE
    static void* allocate_tiny(const size_t size) noexcept
    {
        const uint8_t size_class = (size - 1) >> 3;

        if (auto* tiny_pool = tiny_pools_[size_class]; LIKELY(tiny_pool != nullptr))
        {
            if (void* ptr = tiny_pool->allocate_tiny(size_class); LIKELY(ptr))
            {
                stream_store(ptr,
                    (static_cast<uint64_t>(size) & SIZE_MASK) |
                    static_cast<uint64_t>(size_class) << 48);
                return static_cast<char*>(ptr) + sizeof(block_header);
            }
        }

        static std::mutex init_mutex;
        std::lock_guard lock(init_mutex);

        if (!tiny_pools_[size_class])
        {
            tiny_pools_[size_class] = new (std::align_val_t{PG_SIZE})
                tiny_block_manager::tiny_pool();
            return allocate_tiny(size);
        }

        return nullptr;
    }

    ALWAYS_INLINE
    static void* allocate_small(const size_t size) noexcept
    {
        const uint8_t size_class = (size - 1) >> 3;

        if (void* cached = thread_cache_.get(size_class))
        {
            auto* header = reinterpret_cast<block_header*>(
                static_cast<char*>(cached) - sizeof(block_header));

            if (LIKELY(header->is_valid()))
            {
                header->set_free(false);
                return cached;
            }
            return nullptr;
        }

        if (void* ptr = pool_manager_.allocate(size_class); LIKELY(ptr))
        {
            auto* header = new (ptr) block_header();
            header->encode(size, size_class, false);
            return static_cast<char*>(ptr) + sizeof(block_header);
        }

        return nullptr;
    }

    ALWAYS_INLINE
    static void* allocate_medium(const size_t size, const uint8_t size_class) noexcept
    {
        if (void* cached = thread_cache_.get(size_class))
        {
            auto* header = reinterpret_cast<block_header*>(
                static_cast<char*>(cached) - sizeof(block_header));
            header->set_free(false);
            return cached;
        }

        if (void* ptr = pool_manager_.allocate(size_class); LIKELY(ptr))
        {
            auto* header = new (ptr) block_header();
            header->encode(size, size_class, false);
            return static_cast<char*>(ptr) + sizeof(block_header);
        }

        return nullptr;
    }

    ALWAYS_INLINE
    static void* allocate_large(const size_t size) noexcept
    {
        const size_t total_size = size + sizeof(block_header);
        const size_t aligned_size = 1ULL << (64 - __builtin_clzll(total_size - 1));

        void* ptr = MAP_MEMORY(aligned_size);

        if (UNLIKELY(!ptr))
            return nullptr;

        auto* header = new (ptr) block_header();
        header->encode(size, 255, false);
        header->set_memory_mapped(true);
        return static_cast<char*>(ptr) + sizeof(block_header);
    }

public:
    ALWAYS_INLINE
    static void* allocate(const size_t size) noexcept
    {
        if (UNLIKELY(size == 0 || size > (1ULL << 47)))
            return nullptr;

        if (LIKELY(size <= TINY_LARGE_THRESHOLD))
            return allocate_tiny(size);

        if (size <= SMALL_LARGE_THRESHOLD)
            return allocate_small(size);

        if (size < LARGE_THRESHOLD)
        {
            const size_t size_class = 31 - __builtin_clz(size - 1);
            return allocate_medium(size, size_class);
        }

        return allocate_large(size);
    }

    ALWAYS_INLINE
    static void deallocate(void* ptr) noexcept
    {
        if (!ptr)
            return;
        if (UNLIKELY(!block_header::is_aligned(ptr)))
            return;

        auto* header = reinterpret_cast<block_header*>(
            static_cast<char*>(ptr) - sizeof(block_header));

        if (UNLIKELY(!header->is_valid()))
            return;

        if (UNLIKELY((reinterpret_cast<uintptr_t>(ptr) & ~(PG_SIZE-1)) == 0))
            return;

        const uint8_t size_class = header->size_class();
        if (UNLIKELY(size_class >= SIZE_CLASSES && size_class != 255))
            return;

        if (size_class < tiny_block_manager::num_tiny_classes)
        {
            if (header->is_free())
                return;

            if (auto* tiny_pool = tiny_pools_[size_class])
            {
                header->set_free(true);
                tiny_pool->deallocate_tiny(
                    static_cast<char*>(ptr) - sizeof(block_header),
                    size_class
                );
            }
            return;
        }

        if (UNLIKELY(size_class == 255))
        {
            void* block = static_cast<char*>(ptr) - sizeof(block_header);
            if (header->is_memory_mapped())
            {
                const size_t total_size = header->size() + sizeof(block_header);
                const size_t aligned_size = total_size + PG_SIZE - 1 & ~(PG_SIZE - 1);
                UNMAP_MEMORY(block, aligned_size);
            }
            else
            {
                free(block);
            }
            return;
        }

        if (header->is_free())
            return;

        if (thread_cache_.put(ptr, size_class))
        {
            header->set_free(true);
            return;
        }

        header->set_free(true);
        if (header->try_coalesce())
        {
            auto* pool_start = reinterpret_cast<void*>(
                reinterpret_cast<uintptr_t>(header) & ~(PG_SIZE - 1));
            static_cast<pool*>(pool_start)->return_memory();
        }

        void* block = static_cast<char*>(ptr) - sizeof(block_header);
        pool_manager_.deallocate(block, size_class);
    }

    ALWAYS_INLINE NO_SANITIZE_ADDRESS
    static void* reallocate(void* ptr, const size_t new_size) noexcept
    {
        if (UNLIKELY(!ptr))
            return allocate(new_size);

        if (UNLIKELY(!block_header::is_aligned(ptr)))
            return nullptr;

        if (UNLIKELY(new_size == 0))
        {
            deallocate(ptr);
            return nullptr;
        }
        // cool thing I just learned
        // header is 100% not null if the ptr is aligned!
        const auto* header = reinterpret_cast<block_header*>(
            static_cast<char*>(ptr) - sizeof(block_header));

        if (UNLIKELY(!header))
            return nullptr;

        if (UNLIKELY(!header->is_valid()))
            return nullptr;

        // Add size validation
        if (UNLIKELY(header->size() > (1ULL << 47)))
            return nullptr;

        const size_t old_size = header->size();
        const uint8_t old_class = header->size_class();

        #if defined(__clang__)
            HAVE_BUILTIN_ASSUME(old_class <= SIZE_CLASSES);
        #endif

        if (old_class < tiny_block_manager::num_tiny_classes)
        {
            if (const size_t max_tiny_size = (old_class + 1) << 3; new_size <= max_tiny_size)
                return ptr;
        }
        else if (old_class < SIZE_CLASSES)
        {
            if (const size_t max_size = size_classes[old_class].size; new_size <= max_size)
                return ptr;
        }

        if (UNLIKELY(header->is_memory_mapped()))
        {
            #ifdef __linux__
                void* block = static_cast<char*>(ptr) - sizeof(block_header);
                const size_t new_total = new_size + sizeof(block_header);
                void* new_block = mremap(block, old_total, new_total, MREMAP_MAYMOVE);
                if (new_block != MAP_FAILED)
                {
                    auto* new_header = reinterpret_cast<block_header*>(new_block);
                    new_header->encode(new_size, 255, false);
                    new_header->set_memory_mapped(true);
                    return static_cast<char*>(new_block) + sizeof(block_header);
                }
            #endif
        }

        void* new_ptr = allocate(new_size);
        if (UNLIKELY(!new_ptr))
            return nullptr;

        if (const size_t copy_size = old_size < new_size ? old_size : new_size; copy_size <= 32)
        {
            std::memcpy(new_ptr, ptr, copy_size);
        }
        else if (copy_size >= 4096)
        {
            prefetch_range(ptr, copy_size);
            prefetch_range(new_ptr, copy_size);

            const auto dst = static_cast<char*>(new_ptr);
            const auto src = static_cast<const char*>(ptr);

            #if defined(__AVX512F__)
                for (size_t i = 0; i < copy_size; i += 64)
                {
                    __m512i v = _mm512_loadu_si512((const __m512i*)(src + i));
                    _mm512_stream_si512((__m512i*)(dst + i), v);
                }
            #elif defined(__AVX2__)
                for (size_t i = 0; i < copy_size; i += 32)
                {
                    __m256i v = _mm256_loadu_si256((const __m256i*)(src + i));
                    _mm256_stream_si256((__m256i*)(dst + i), v);
                }
            #elif defined(__aarch64__)
                for (size_t i = 0; i < copy_size; i += 64)
                {
                    auto src_bytes = reinterpret_cast<const uint8_t*>(src + i);
                    auto dst_bytes = reinterpret_cast<uint8_t*>(dst + i);
                    uint8x16x4_t v = vld4q_u8(src_bytes);
                    vst4q_u8(dst_bytes, v);
                }
            #endif
            memory_fence();

            for (size_t i = 0; i < copy_size; i += 8)
                stream_store(dst + i,
                    *reinterpret_cast<const int64_t*>(src + i));
            memory_fence();
        }
        else
        {
            #if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
                #if __GLIBC_PREREQ(2, 14)
                    __memcpy_chk(new_ptr, ptr, copy_size, copy_size);
                #else
                    std::memcpy(new_ptr, ptr, copy_size);
                #endif
            #else
                std::memcpy(new_ptr, ptr, copy_size);
            #endif
        }

        deallocate(ptr);
        return new_ptr;
    }

    ALWAYS_INLINE NO_SANITIZE_ADDRESS
    static void* callocate(const size_t num, const size_t size) noexcept
    {
        if (UNLIKELY(num == 0 || size == 0))
            return nullptr;

        if (UNLIKELY(num > SIZE_MAX / size))
            return nullptr;

        size_t total_size = num * size;
        void* ptr = allocate(total_size);

        if (UNLIKELY(!ptr))
            return nullptr;

        if (total_size <= 32)
        {
            std::memset(ptr, 0, total_size);
            return ptr;
        }

        #ifdef __linux__
            if (total_size >= 4096)
            {
                char* page_aligned = reinterpret_cast<char*>(
                    (reinterpret_cast<uintptr_t>(ptr) + 4095) & ~4095ULL);
                size_t prefix = page_aligned - static_cast<char*>(ptr);
                size_t suffix = (total_size - prefix) & 4095;
                size_t middle = total_size - prefix - suffix;

                auto zero_range = [](void* dst, size_t size)
                {
                    #if defined(__AVX512F__)
                        auto* aligned_dst = reinterpret_cast<__m512i*>((reinterpret_cast<uintptr_t>(dst) + 63) & ~63ULL);
                        size_t pre = reinterpret_cast<char*>(aligned_dst) - static_cast<char*>(dst);
                        if (pre) std::memset(dst, 0, pre);

                        __m512i zero = _mm512_setzero_si512();
                        for (size_t i = 0; i < (size - pre) / 64; ++i)
                            _mm512_stream_si512(aligned_dst + i, zero);

                        size_t post = (size - pre) & 63;
                        if (post) std::memset(reinterpret_cast<char*>(aligned_dst + (size - pre) / 64), 0, post);

                    #elif defined(__AVX2__)
                        auto* aligned_dst = reinterpret_cast<__m256i*>((reinterpret_cast<uintptr_t>(dst) + 31) & ~31ULL);
                        size_t pre = reinterpret_cast<char*>(aligned_dst) - static_cast<char*>(dst);
                        if (pre) std::memset(dst, 0, pre);

                        __m256i zero = _mm256_setzero_si256();
                        for (size_t i = 0; i < (size - pre) / 32; ++i)
                            _mm256_stream_si256(aligned_dst + i, zero);

                        size_t post = (size - pre) & 31;
                        if (post) std::memset(reinterpret_cast<char*>(aligned_dst + (size - pre) / 32), 0, post);

                    #elif defined(__aarch64__)
                        auto* aligned_dst = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(dst) + 15) & ~15ULL);
                        size_t pre = aligned_dst - static_cast<uint8_t*>(dst);
                        if (pre) std::memset(dst, 0, pre);

                        uint8x16x4_t zero = { vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0) };
                        for (size_t i = 0; i < (size - pre) / 64; ++i)
                            vst4q_u8(aligned_dst + i * 64, zero);

                        size_t post = (size - pre) & 63;
                        if (post) std::memset(aligned_dst + ((size - pre) / 64) * 64, 0, post);
                    #else
                        std::memset(dst, 0, size);
                    #endif
                };

                if (prefix)
                    zero_range(ptr, prefix);

                if (middle)
                {
                    if (madvise(page_aligned, middle, MADV_DONTNEED) == 0)
                    {
                        if (suffix)
                            zero_range(page_aligned + middle, suffix);
                        memory_fence();
                        return ptr;
                    }
                }
            }
        #endif

        auto dst = static_cast<char*>(ptr);
        #if defined(__AVX512F__)
            __m512i zero = _mm512_setzero_si512();
            auto* aligned_dst = reinterpret_cast<__m512i*>((reinterpret_cast<uintptr_t>(dst) + 63) & ~63ULL);
            size_t pre = reinterpret_cast<char*>(aligned_dst) - dst;
            if (pre) std::memset(dst, 0, pre);

            size_t blocks = (total_size - pre) / 64;
            for (size_t i = 0; i < blocks; ++i)
                _mm512_stream_si512(aligned_dst + i, zero);

            size_t remain = (total_size - pre) & 63;
            if (remain)
                std::memset(reinterpret_cast<char*>(aligned_dst + blocks), 0, remain);

        #elif defined(__AVX2__)
            __m256i zero = _mm256_setzero_si256();
            auto* aligned_dst = reinterpret_cast<__m256i*>((reinterpret_cast<uintptr_t>(dst) + 31) & ~31ULL);
            size_t pre = reinterpret_cast<char*>(aligned_dst) - dst;
            if (pre) std::memset(dst, 0, pre);

            size_t blocks = (total_size - pre) / 32;
            for (size_t i = 0; i < blocks; ++i)
                _mm256_stream_si256(aligned_dst + i, zero);

            size_t remain = (total_size - pre) & 31;
            if (remain)
                std::memset(reinterpret_cast<char*>(aligned_dst + blocks), 0, remain);

        #elif defined(__aarch64__)
            uint8x16x4_t zero = { vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0) };
            auto* dst_bytes = reinterpret_cast<uint8_t*>(dst);
            auto* aligned_dst = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(dst_bytes) + 15) & ~15ULL);
            size_t pre = aligned_dst - dst_bytes;
            if (pre) std::memset(dst_bytes, 0, pre);

            size_t blocks = (total_size - pre) / 64;
            for (size_t i = 0; i < blocks; ++i)
                vst4q_u8(aligned_dst + i * 64, zero);

        if (size_t remain = (total_size - pre) & 63)
                std::memset(aligned_dst + blocks * 64, 0, remain);

        #else
            if (const size_t align = reinterpret_cast<uintptr_t>(dst) & 7)
            {
                std::memset(dst, 0, align);
                dst += align;
                total_size -= align;
            }

            const size_t blocks = total_size >> 3;
            for (size_t i = 0; i < blocks; ++i)
                stream_store(dst + (i << 3), 0LL);

            if (const size_t remain = total_size & 7)
                std::memset(dst + (blocks << 3), 0, remain);
        #endif

        memory_fence();
        return ptr;
    }

    ALWAYS_INLINE
    static void cleanup() noexcept
    {
        thread_cache_.clear();
        for (auto*& pool : tiny_pools_)
        {
            delete pool;
            pool = nullptr;
        }
    }
};

// Initialization
thread_local thread_cache_t Jallocator::thread_cache_{};
thread_local Jallocator::pool_manager Jallocator::pool_manager_{};
thread_local std::array<Jallocator::tiny_block_manager::tiny_pool*,
                       Jallocator::tiny_block_manager::num_tiny_classes>
    Jallocator::tiny_pools_{};

inline void* operator new(const size_t __sz)
{
    return Jallocator::allocate(__sz);
}

inline void* operator new[](const size_t __sz)
{
    return Jallocator::allocate(__sz);
}

inline void operator delete(void* __p) noexcept
{
    Jallocator::deallocate(__p);
}

inline void operator delete[](void* __p) noexcept
{
    Jallocator::deallocate(__p);
}

// C API
#ifndef __cplusplus
{
    extern  "C"
{

}
    ALWAYS_INLINE
    void* malloc(const size_t size)
    {
        return Jallocator::allocate(size);
    }

    ALWAYS_INLINE
    void free(void* ptr)
    {
        Jallocator::deallocate(ptr);
    }

    ALWAYS_INLINE
    void* realloc(void* ptr, const size_t new_size)
    {
        return Jallocator::reallocate(ptr, new_size);
    }

    ALWAYS_INLINE
    void* calloc(const size_t num, const size_t size)
    {
        return Jallocator::callocate(num, size);
    }

    ALWAYS_INLINE
    void cleanup()
    {
        Jallocator::cleanup();
    }
}
#endif