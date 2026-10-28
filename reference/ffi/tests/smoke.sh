#!/usr/bin/env bash
# Compiles tests/smoke.c against the committed header and the static library, runs it on the test model, and checks that the header also compiles as C++. CI runs this; it is also the way to check the C side locally.
set -euo pipefail
cd "$(dirname "$0")/.."

out=target/smoke
mkdir -p "$out"

# Building through `cargo rustc` both produces the static library and makes rustc print the system libraries a C program must link alongside it, which differ between Linux and macOS.
libs=$(cargo rustc --locked --color never --lib -- --print native-static-libs 2>&1 | sed -n 's/.*native-static-libs: //p' | tail -n 1)
cargo run --locked --quiet --example write_fixture -- "$out/model.safetensors"

${CC:-cc} -std=c11 -Wall -Wextra -Werror -pedantic -Iinclude tests/smoke.c target/debug/libchinese_ime_lm_ffi.a $libs -o "$out/smoke"
"$out/smoke" "$out/model.safetensors"

echo '#include "chinese_ime_lm.h"' | ${CXX:-c++} -std=c++17 -Wall -Wextra -Werror -pedantic -Iinclude -x c++ -fsyntax-only -
echo "header compiles as C++"
