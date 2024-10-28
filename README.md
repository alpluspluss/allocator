# Archived Status

This project is archived and no longer maintained. It is kept here for historical purposes only.
However, feel free to fork and continue development on your own.

# Just an Allocator™
It allocates memory... efficiently™

## What is it?
It's just a header file. That's it. No really:

```c++
#include "jalloc.hpp"
```
Done. You now have:
- Thread-caching
- Lock-free operations
- SIMD-optimized memory operations
- Bitmap-based block management
- Three-tiered allocation strategy

all in one header file.

## Usage

```c++
#include "jalloc.hpp"

int main() 
{
    void* ptr = jalloc::allocate(64);        // Allocate 64 bytes
    jalloc::deallocate(ptr);                 // Free memory

    // Or with new/delete operators
    int* arr = new int[100];
    delete[] arr;

    void* buf = jalloc::callocate(1, 64);    // Zero-initialized allocation
    void* ptr2 = jalloc::reallocate(ptr, 128); // Resize allocation
    
    return 0;
}
```

## Supported Platform Status
| Platform | Architecture          | Status     |
|----------|-----------------------|------------|
| macOS    | Apple Silicon (ARM64) | Tested     |
| macOS    | x86_64                | Not tested |
| Linux    | x86_64                | Not tested |
| Linux    | ARM64                 | Not tested |
| Windows  | x86_64                | Not tested |

### Note: GNU and Clang will be the only supported compilers.

## Requirements
- C++17 or later
- C++ Compiler

## Performance

- SIMD instructions on all architectures enable bulk memory operations.
- Bitmap-based block management minimizes overhead.
- Three-tiered allocation strategy optimizes for small, medium, and large allocations.

See the [benchmarks](benches/benchmark.md) for performance data.

## Current State
Currently, the allocator is in the development phase. It is not recommended for production use as some platform-specific
features are not yet fully tested and unsafe. Only ARM-based architecture is supported single-threaded.

## Note
This is a work in progress. Please feel free to [contribute](.github/CONTRIBUTING.md).

## License
MIT. FYI, please see the [License](LICENSE).

---
*Remember: It's Just an Allocator™ - Any resemblance to a sophisticated memory management system is purely coincidental.*
---
