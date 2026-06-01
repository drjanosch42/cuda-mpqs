# Changelog

All notable changes to cuda-mpqs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-01
### Added
- Initial public release.
- Full SIQS/MPQS pipeline: parameter tuning, optional autotuning, GPU sieving,
  sparse GF(2) matrix construction, Block Wiedemann linear algebra, and square
  root refinement.
- Single-node and distributed cluster execution across heterogeneous multi-node
  setups (explicitly tested with more than two nodes).
- Autotuning: 4-stage joint (F, L) optimizer with persistent history.
- GPU preprocessing: packed sparse GF(2) matrix with compact-merge cycles.
- Large-prime variant: single large prime via GPU slab hash table.
- Supports NVIDIA Turing, Ampere, Hopper, and Blackwell GPU architectures,
  including the A100 (Ampere) and H100 (Hopper) data-center accelerators.
- Supports the Jetson Orin Nano Super 8 GB embedded platform.
