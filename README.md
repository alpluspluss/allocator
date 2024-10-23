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
`jalloc::allocate` and `jalloc::deallocate` to allocate and deallocate memory. 
Extras: `jalloc::reallocate` and `jalloc::callocate` for reallocation and zero-initialization.

## Platforms
- Windows
- Unix
- x86_64 (for now)

## Technical Details
For those who insist on knowing more:

### Memory Layout
```
Tiny blocks:  [Header(8B)][Data(≤64B)]
Small blocks: [Header(8B)][Data(≤256B)]
Large blocks: [Header(8B)][Data(>256B)]
```

### Thread Safety
- Thread-local caches
- Atomic operations
- Lock-free fast paths

# Requirements
- C++17 or later
- C++ Compiler

# Performance
- SIMD instructions on every architecture to perform bulk operations.
- Bitmap-based block management.
- Three-tiered allocation strategy.

See the [bechmarks](benches/benchmark.md) for more details.

# Current State
Currently, the allocator is in the development phase. It is not recommended for production use as some platform-specific
features are not yet fully tested and unsafe. Only ARM-based architecture is supported single-threaded.

# Note
This is a work in progress. Please feel free to [contribute](.github/CONTRIBUTING.md).

# License
MIT. FYI, please see the [License](LICENSE).

---
*Remember: It's Just an Allocator™ - Any resemblance to a sophisticated memory management system is purely coincidental.*
---
