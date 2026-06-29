# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
# This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
# See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

set(CUDAMPQS_VERSION_MAJOR 1)
set(CUDAMPQS_VERSION_MINOR 0)
set(CUDAMPQS_VERSION_PATCH 4)
# Numeric MAJOR.MINOR.PATCH only — consumed by project(... VERSION ...), so it must
# stay strictly numeric. The non-numeric pre-release suffix (e.g. "a") lives in
# CUDAMPQS_VERSION_SUFFIX and is appended in the human-facing display string below.
set(CUDAMPQS_VERSION "${CUDAMPQS_VERSION_MAJOR}.${CUDAMPQS_VERSION_MINOR}.${CUDAMPQS_VERSION_PATCH}")
set(CUDAMPQS_VERSION_SUFFIX "")
# Display string with suffix — this is what the binary prints (--version / banner).
set(CUDAMPQS_VERSION_STRING "${CUDAMPQS_VERSION}${CUDAMPQS_VERSION_SUFFIX}")

set(CUDAMPQS_LINALG_VERSION_MAJOR 1)
set(CUDAMPQS_LINALG_VERSION_MINOR 0)
set(CUDAMPQS_LINALG_VERSION_PATCH 0)
set(CUDAMPQS_LINALG_VERSION "${CUDAMPQS_LINALG_VERSION_MAJOR}.${CUDAMPQS_LINALG_VERSION_MINOR}.${CUDAMPQS_LINALG_VERSION_PATCH}")
