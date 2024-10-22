# Just an Allocator™

It allocates memory... efficiently™

## What is it?

It's just a header file. That's it. No really:

```c++
#include "justalloc.hpp"
```
Done. You now have:

- Thread-caching
- Lock-free operations
- SIMD-optimized memory operations
- Bitmap-based block management
- Three-tiered allocation strategy

all in one header file. It's just an allocator™.

## Usage

`jalloc::allocate` and `jalloc::deallocate` to allocate and deallocate memory. That's it.

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

### Requirements
- C++17 or later
- C++ Compiler

## License

MIT. See more here `LICENSE.md`.

---
*Remember: It's Just an Allocator™ - Any resemblance to a sophisticated memory management system is purely coincidental.*
---
