#!/bin/bash
# Build metal-eq. Needs only the Xcode Command Line Tools: the shaders compile at runtime.
set -euo pipefail
cd "$(dirname "$0")"
xxd -i equihash.metal > kernel_src.h
xxd -i equihash_v2.metal > kernel_v2_src.h
clang -fobjc-arc -O3 -mcpu=apple-m1 -Wall -Wno-unused-function -include stdint.h -include stddef.h \
    -o metal-eq metal_eq.m blake.c sha256.c cpuhash.c -framework Metal -framework Foundation
echo "built ./metal-eq"
