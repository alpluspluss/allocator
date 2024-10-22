//
// Just an Allocator™
// Author: alpluspluss 10/22/2024 00:27 AM
// License: MIT
//

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>

#if defined(__x86_64__)
    #include <immintrin.h>
    #include <emmintrin.h>
    #include <xmmintrin.h>

    #ifdef _MSC_VER
        #include <intrin.h>
    #endif

    #define STREAM_STORE_64(addr, val) _mm_stream_si64((__int64*)(addr), val)
    #define CPU_PAUSE() _mm_pause()
    #define CUSTOM_PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)

#elif defined(__arm__) || defined(__aarch64__)
    #include <arm_neon.h>
    #ifdef __clang__
        #include <arm_acle.h>
    #endif

    #define STREAM_STORE_64(addr, val) *((int64_t*)(addr)) = val
    #define CPU_PAUSE() __yield()
    #if defined(__clang__) && (defined(__aarch64__) || defined(__arm64__))
        #define CUSTOM_PREFETCH(addr) __pld(reinterpret_cast<const char*>(addr))
    #else
        #define CUSTOM_PREFETCH(addr) ((void)0)
    #endif

#else
    #define STREAM_STORE_64(addr, val) *((int64_t*)(addr)) = val
    #define CPU_PAUSE() ((void)0)
    #define CUSTOM_PREFETCH(addr) ((void)0)
#endif

#ifdef _WIN32
    #include <malloc.h>
    #include <windows.h>

    #define MAP_MEMORY(size) \
        VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    #define UNMAP_MEMORY(ptr, size) VirtualFree(ptr, 0, MEM_RELEASE)
    #define MAP_FAILED nullptr

#elif defined(__APPLE__)
    #include <mach/mach.h>
    #include <sys/mman.h>

    #define MAP_MEMORY(size) \
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    #define UNMAP_MEMORY(ptr, size) munmap(ptr, size)

#else
    #include <sched.h>
    #include <unistd.h>
    #include <sys/mman.h>

    #define MAP_MEMORY(size) \
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    #define UNMAP_MEMORY(ptr, size) munmap(ptr, size)
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define LIKELY(x) __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define ALWAYS_INLINE [[gnu::always_inline]] inline

    #if defined(__clang__)
        #define HAVE_BUILTIN_ASSUME(x) __builtin_assume(x)
        #define HAVE_BUILTIN_ASSUME_ALIGNED(x, a) __builtin_assume_aligned(x, a)
        #define NO_SANITIZE_ADDRESS __attribute__((no_sanitize("address")))
    #else
        #define HAVE_BUILTIN_ASSUME(x) ((void)0)
        #define HAVE_BUILTIN_ASSUME_ALIGNED(x, a) (x)
        #define NO_SANITIZE_ADDRESS
    #endif

#elif defined(_MSC_VER)
    #define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
    #define ALWAYS_INLINE __forceinline
    #define HAVE_BUILTIN_ASSUME(x) ((void)0)
    #define HAVE_BUILTIN_ASSUME_ALIGNED(x, a) (x)
    #define NO_SANITIZE_ADDRESS
    #define CUSTOM_PREFETCH(addr) ((void)0)

#else
    #define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
    #define ALWAYS_INLINE inline
    #define HAVE_BUILTIN_ASSUME(x) ((void)0)
    #define HAVE_BUILTIN_ASSUME_ALIGNED(x, a) (x)
    #define NO_SANITIZE_ADDRESS
    #define CUSTOM_PREFETCH(addr) ((void)0)
#endif

static constexpr size_t CACHE_LINE_SIZE = 64;
static constexpr size_t TINY_LARGE_THRESHOLD = 64;
static constexpr size_t SMALL_LARGE_THRESHOLD = 256;
static constexpr size_t ALIGNMENT = CACHE_LINE_SIZE;
static constexpr size_t PG_SIZE = 4096;
static constexpr size_t LARGE_THRESHOLD = 1024 * 1024;
static constexpr size_t CACHE_SIZE = 32;
static constexpr size_t SIZE_CLASSES = 32;

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

class Allocator
{
    struct bitmap
    {
        static constexpr size_t bits_per_word = 64;
        static constexpr size_t words_per_bitmap = PG_SIZE / (CACHE_LINE_SIZE * 8);

        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> words[words_per_bitmap];

        bitmap() noexcept
        {
            for (auto& word : words)
                word.store(~0ULL, std::memory_order_relaxed);
        }

        ALWAYS_INLINE
        size_t find_free_block() noexcept
        {
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
            memory_fence();
            words[word_idx].fetch_or(1ULL << bit_idx, std::memory_order_release);
            memory_fence();
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
        uint64_t data;

        static constexpr uint64_t size_mask = 0x0000FFFFFFFFFFFF;
        static constexpr uint64_t class_mask = 0x00FF000000000000;
        static constexpr uint64_t flags_mask = 0xFF00000000000000;
        static constexpr uint64_t mmap_flag = 1ULL << 62;

        ALWAYS_INLINE
        void encode(const size_t size, const uint8_t size_class, const bool is_free) noexcept
        {
            data = size & size_mask |
                   static_cast<uint64_t>(size_class) << 48 |
                   static_cast<uint64_t>(is_free) << 63;
        }

        ALWAYS_INLINE
        void set_free(const bool is_free) noexcept
        {
            data = data & ~(1ULL << 63) | static_cast<uint64_t>(is_free) << 63;
        }

        ALWAYS_INLINE
        void set_memory_mapped(const bool is_mmap) noexcept
        {
            data = data & ~mmap_flag | static_cast<uint64_t>(is_mmap) << 62;
        }

        size_t size() const noexcept
        {
            return data & size_mask;
        }
        uint8_t size_class() const noexcept
        {
            return (data & class_mask) >> 48;
        }
        bool is_free() const noexcept
        {
            return data & 1ULL << 63;
        }
        bool is_memory_mapped() const noexcept
        {
            return data & mmap_flag;
        }

        static bool is_aligned(const void* ptr) noexcept
        {
            return (reinterpret_cast<uintptr_t>(ptr) & ALIGNMENT - 1) == 0;
        }
    };

    struct alignas(PG_SIZE) pool
    {
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

        bool is_completely_free() const noexcept
        {
            return bitmap.is_completely_free();
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
            const auto& sc = size_classes[size_class];

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
                for (size_t i = 0; i < pool_count[sc]; ++i)
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
                    static_cast<uint64_t>(size) & block_header::size_mask |
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
        if (UNLIKELY(size == 0))
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
        if (UNLIKELY(!ptr || !block_header::is_aligned(ptr)))
            return;

        auto* header = reinterpret_cast<block_header*>(
            static_cast<char*>(ptr) - sizeof(block_header));

        if (UNLIKELY(!header))
            return;

        const uint8_t size_class = header->size_class();

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

        void* block = static_cast<char*>(ptr) - sizeof(block_header);
        pool_manager_.deallocate(block, size_class);
        header->set_free(true);
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

        const auto* header = reinterpret_cast<block_header*>(
            static_cast<char*>(ptr) - sizeof(block_header));

        if (UNLIKELY(!header))
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
    static void* calloc(const size_t num, const size_t size) noexcept
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

            // Zero non-aligned prefix
            if (prefix)
            {
                MemoryHelper::stream_store(ptr, 0LL);
                prefix = (prefix + 7) & ~7ULL;
                for (size_t i = 8; i < prefix; i += 8)
                {
                    MemoryHelper::stream_store(
                        static_cast<char*>(ptr) + i, 0LL);
                }
            }

            if (middle)
            {
                if (madvise(page_aligned, middle, MADV_DONTNEED) == 0)
                {
                    if (suffix)
                    {
                        char* suffix_ptr = page_aligned + middle;
                        for (size_t i = 0; i < suffix; i += 8)
                        {
                            MemoryHelper::stream_store(suffix_ptr + i, 0LL);
                        }
                    }
                    MemoryHelper::memory_fence();
                    return ptr;
                }
            }
        }
        #endif

        auto dst = static_cast<char*>(ptr);

        if (const size_t align = reinterpret_cast<uintptr_t>(dst) & 7)
        {
            std::memset(dst, 0, align);
            dst += align;
            total_size -= align;
        }

        const size_t blocks = total_size >> 3;
        for (size_t i = 0; i < blocks; ++i)
        {
            stream_store(dst + (i << 3), 0LL);
        }

        if (const size_t remain = total_size & 7)
        {
            std::memset(dst + (blocks << 3), 0, remain);
        }

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

thread_local thread_cache_t Allocator::thread_cache_{};
thread_local Allocator::pool_manager Allocator::pool_manager_{};
thread_local std::array<Allocator::tiny_block_manager::tiny_pool*,
                       Allocator::tiny_block_manager::num_tiny_classes>
    Allocator::tiny_pools_{};