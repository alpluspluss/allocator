#include "../jalloc.hpp"
#include "../extern/catch_amalgamated.hpp"

namespace
{
    bool verify_block_alignment(void* ptr)
    {
        if (!ptr)
            return false;
        return (reinterpret_cast<uintptr_t>(ptr) & (ALIGNMENT - 1)) == 0;
    }

    bool verify_memory(void* ptr, const size_t size)
    {
        if (!ptr) return false;
        auto* bytes = static_cast<uint8_t*>(ptr);
        try
        {
            for (size_t i = 0; i < size; i++)
            {
                bytes[i] = static_cast<uint8_t>(i & 0xFF);
            }

            for (size_t i = 0; i < size; i++)
            {
                if (bytes[i] != static_cast<uint8_t>(i & 0xFF))
                {
                    return false;
                }
            }
            return true;
        } catch (...) {
            return false;
        }
    }
}

TEST_CASE("Basic allocation functionality", "[memory]")
{
    SECTION("Allocation of zero bytes returns nullptr")
    {
        void* ptr = Jallocator::allocate(0);
        REQUIRE(ptr == nullptr);
    }

    SECTION("Small allocation (<=64 bytes)")
    {
        constexpr size_t sizes[] = {1, 8, 16, 32, 64};
        for (auto size : sizes)
        {
            INFO("Testing allocation of size: " << size);
            void* ptr = Jallocator::allocate(size);
            REQUIRE(ptr != nullptr);
            REQUIRE(verify_block_alignment(ptr));
            REQUIRE(verify_memory(ptr, size));
            Jallocator::deallocate(ptr);
        }
    }

    SECTION("Medium allocation (65-256 bytes)")
    {
        constexpr size_t sizes[] = { 65, 128, 200, 256 };
        for (auto size : sizes)
        {
            INFO("Testing allocation of size: " << size);
            void* ptr = Jallocator::allocate(size);
            REQUIRE(ptr != nullptr);
            REQUIRE(verify_block_alignment(ptr));
            REQUIRE(verify_memory(ptr, size));
            Jallocator::deallocate(ptr);
        }
    }

    SECTION("Large allocation (>256 bytes)")
    {
        constexpr size_t sizes[] = { 257, 512, 1024, 4096 };
        for (auto size : sizes)
        {
            INFO("Testing allocation of size: " << size);
            void* ptr = Jallocator::allocate(size);
            REQUIRE(ptr != nullptr);
            REQUIRE(verify_block_alignment(ptr));
            REQUIRE(verify_memory(ptr, size));
            Jallocator::deallocate(ptr);
        }
    }
}

TEST_CASE("Memory safety and edge cases", "[memory]")
{
    SECTION("Double free has no effect")
    {
        void* ptr = Jallocator::allocate(64);
        REQUIRE(ptr != nullptr);
        Jallocator::deallocate(ptr);
        Jallocator::deallocate(ptr);
    }

    SECTION("Null pointer deallocation is safe")
    {
        Jallocator::deallocate(nullptr);
    }

    SECTION("Very large allocation")
    {
        constexpr size_t large_size = 1024 * 1024;
        if (void* ptr = Jallocator::allocate(large_size))
        {
            REQUIRE(verify_block_alignment(ptr));
            REQUIRE(verify_memory(ptr, large_size));
            Jallocator::deallocate(ptr);
        }
    }
}

TEST_CASE("Reallocation behavior", "[memory]")
{
    SECTION("Reallocate null pointer")
    {
        void* ptr = Jallocator::reallocate(nullptr, 64);
        REQUIRE(ptr != nullptr);
        REQUIRE(verify_block_alignment(ptr));
        Jallocator::deallocate(ptr);
    }

    SECTION("Reallocate to zero bytes")
    {
        void* ptr = Jallocator::allocate(64);
        REQUIRE(ptr != nullptr);
        ptr = Jallocator::reallocate(ptr, 0);
        REQUIRE(ptr == nullptr);
    }

    SECTION("Grow allocation")
    {
        void* ptr = Jallocator::allocate(64);
        REQUIRE(ptr != nullptr);
        REQUIRE(verify_block_alignment(ptr));

        auto bytes = static_cast<uint8_t*>(ptr);
        for (auto i = 0; i < 64; i++)
        {
            bytes[i] = static_cast<uint8_t>(i);
        }

        // Grow
        void* new_ptr = Jallocator::reallocate(ptr, 128);
        REQUIRE(new_ptr != nullptr);
        REQUIRE(verify_block_alignment(new_ptr));

        // Verify pattern preserved
        bytes = static_cast<uint8_t*>(new_ptr);
        for (auto i = 0; i < 64; i++)
        {
            REQUIRE(bytes[i] == static_cast<uint8_t>(i));
        }

        Jallocator::deallocate(new_ptr);
    }
}

TEST_CASE("Calloc behavior", "[memory]")
{
    SECTION("Basic calloc")
    {
        void* ptr = Jallocator::callocate(16, 4);
        REQUIRE(ptr != nullptr);
        REQUIRE(verify_block_alignment(ptr));

        auto bytes = static_cast<uint8_t*>(ptr);
        for (size_t i = 0; i < 64; i++)
        {
            REQUIRE(bytes[i] == 0);
        }
        Jallocator::deallocate(ptr);
    }

    SECTION("Calloc with zero size")
    {
        REQUIRE(Jallocator::callocate(0, 4) == nullptr);
        REQUIRE(Jallocator::callocate(4, 0) == nullptr);
        REQUIRE(Jallocator::callocate(0, 0) == nullptr);
    }
}

TEST_CASE("Thread cache behavior", "[memory]")
{
    SECTION("Cache invalidation")
    {
        std::vector<void*> pointers;
        for (auto i = 0; i < 100; i++)
        {
            void* ptr = Jallocator::allocate(64);
            REQUIRE(ptr != nullptr);
            REQUIRE(verify_block_alignment(ptr));
            pointers.push_back(ptr);
        }

        for (void* ptr : pointers)
        {
            Jallocator::deallocate(ptr);
        }

        Jallocator::cleanup();

        void* ptr = Jallocator::allocate(64);
        REQUIRE(ptr != nullptr);
        REQUIRE(verify_block_alignment(ptr));
        Jallocator::deallocate(ptr);
    }
}
