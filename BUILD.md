# Build Instructions

## Prerequisites

- **CUDA Toolkit** 12.x+
- **CMake** 3.22+ (3.24+ recommended for native GPU detection)
- **C++20 compiler** (GCC 11+ or Clang 14+)
- **OpenMP**

## x86 / Discrete GPU (RTX, etc.)

```bash
cmake -B build -DGPU_TARGET=native
cmake --build build -j$(nproc)
```

## Jetson Orin Nano (SM 8.7)

```bash
export PATH=/usr/local/cuda/bin:$PATH  # if nvcc not in PATH
cmake -B build -DGPU_TARGET=Orin
cmake --build build -j4
```

## GPU_TARGET Options

| Value | Architecture |
|-------|-------------|
| `native` | Auto-detect (CMake 3.24+, nvidia-smi fallback for 3.22) |
| `Orin` / `87` | Jetson AGX Orin / Orin NX / Orin Nano |
| `Turing` / `75` | RTX 20xx, GTX 16xx |
| `Ampere` / `80` | RTX 30xx, A100 |
| `Ada` / `89` | RTX 40xx |
| `Hopper` / `90` | H100 |
| `Blackwell` / `120` | B200 |
| `all` | All major architectures (slow build) |
| Any numeric SM | e.g., `86` for GA102 |

## Quick Test

```bash
./build/tests/cuda-mpqs --verbose              # Default ~80-digit composite
./build/tests/cuda-mpqs --RSA100 --verbose     # RSA-100 (~2 min on RTX 5070 Ti)
```

## Cross-compilation (Jetson binary on x86)

```bash
cmake -B build-orin -DGPU_TARGET=Orin
cmake --build build-orin -j$(nproc)
# Binary won't run on x86 but verifies compilation for SM 87
```

## Notes

- **CMake 3.22 compatibility:** Native GPU detection uses `CMAKE_CUDA_ARCHITECTURES=native` (requires 3.24). On CMake 3.22 (common on Jetson), the build falls back to parsing `nvidia-smi` output for GPU arch detection.
- **Submodules:** Clone with `--recurse-submodules` or run `git submodule update --init --recursive` after cloning. The `src/sieve/` and `src/linalg/` directories are git submodules.
- **Separable compilation (RDC):** All CUDA libraries use relocatable device code. The final executable resolves device symbols at link time via `CUDA_RESOLVE_DEVICE_SYMBOLS ON`.
