#include "jalloc.hpp"
#include <iostream>

int main()
{
    for (auto i = 0; i < 100; ++i)
    {
        if (void* tiny = Allocator::allocate(8); tiny == nullptr)
        {
            std::cout << "Tiny allocation (8B) failed\n";
            return 1;
        }
        else
        {
            Allocator::deallocate(tiny);
        }

        if (void* small = Allocator::allocate(128); small == nullptr)
        {
            std::cout << "Small allocation (128B) failed\n";
            return 1;
        }
        else
        {
            Allocator::deallocate(small);
        }
        if (void* medium = Allocator::allocate(512); medium == nullptr)
        {
            std::cout << "Medium allocation (512B) failed\n";
            return 1;
        }
        else
        {
            Allocator::deallocate(medium);
        }

        if (void* large = Allocator::allocate(4096); large == nullptr)
        {
            std::cout << "Large allocation (4KB) failed\n";
            return 1;
        }
        else
        {
            Allocator::deallocate(large);
        }
    }

    std::cout << "All allocations successful\n";
    return 0;
}