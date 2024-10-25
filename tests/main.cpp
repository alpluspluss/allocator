#include <assert.h>

#include "../jalloc.hpp"

int main()
{
    void *ptr = Jallocator::allocate(16);
    assert(ptr != nullptr);
    Jallocator::deallocate(ptr);
    return 0;
}
