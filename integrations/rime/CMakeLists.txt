# librime plugin: place this directory (or a symlink to it) at librime/plugins/ime-lm and build librime as usual.
#
# The plugin links chinese-ime-lm's C ABI as a static library. By default it is built here with cargo from reference/ffi in this repository; set CIMELM_FFI_LIBRARY and CIMELM_FFI_INCLUDE_DIR to use a prebuilt one instead, which is how a build without a Rust toolchain links it.
project(rime-ime-lm)
cmake_minimum_required(VERSION 3.10)

set(CIMELM_FFI_LIBRARY "" CACHE FILEPATH "Prebuilt static library of chinese-ime-lm-ffi; built with cargo when empty")
set(CIMELM_FFI_INCLUDE_DIR "" CACHE PATH "Directory containing chinese_ime_lm.h, for a prebuilt CIMELM_FFI_LIBRARY")

if(CIMELM_FFI_LIBRARY)
  if(NOT CIMELM_FFI_INCLUDE_DIR)
    message(FATAL_ERROR "CIMELM_FFI_LIBRARY is set but CIMELM_FFI_INCLUDE_DIR is not")
  endif()
  set(ime_lm_ffi_library "${CIMELM_FFI_LIBRARY}")
  set(ime_lm_ffi_include "${CIMELM_FFI_INCLUDE_DIR}")
else()
  # Resolved through symlinks, because the plugin is usually linked into librime's plugins directory and the crate is found relative to where this file really lives.
  get_filename_component(ime_lm_real_dir "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
  set(ime_lm_ffi_dir "${ime_lm_real_dir}/../../reference/ffi")
  if(NOT EXISTS "${ime_lm_ffi_dir}/Cargo.toml")
    message(FATAL_ERROR "chinese-ime-lm-ffi not found at ${ime_lm_ffi_dir}; set CIMELM_FFI_LIBRARY and CIMELM_FFI_INCLUDE_DIR to a prebuilt library")
  endif()
  find_program(CARGO_EXECUTABLE cargo)
  if(NOT CARGO_EXECUTABLE)
    message(FATAL_ERROR "cargo not found; install a Rust toolchain or set CIMELM_FFI_LIBRARY and CIMELM_FFI_INCLUDE_DIR to a prebuilt library")
  endif()
  set(ime_lm_cargo_target "${CMAKE_CURRENT_BINARY_DIR}/cargo")
  set(ime_lm_ffi_library "${ime_lm_cargo_target}/release/${CMAKE_STATIC_LIBRARY_PREFIX}chinese_ime_lm_ffi${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(ime_lm_ffi_include "${ime_lm_ffi_dir}/include")
  # Always run: cargo is a no-op when nothing changed, and CMake cannot see the Rust sources this depends on.
  add_custom_target(rime-ime-lm-ffi
    COMMAND "${CARGO_EXECUTABLE}" build --release --locked --lib --manifest-path "${ime_lm_ffi_dir}/Cargo.toml" --target-dir "${ime_lm_cargo_target}"
    BYPRODUCTS "${ime_lm_ffi_library}"
    COMMENT "Building chinese-ime-lm-ffi with cargo"
    USES_TERMINAL)
endif()

aux_source_directory(src ime_lm_src)

add_library(rime-ime-lm-objs OBJECT ${ime_lm_src})
target_include_directories(rime-ime-lm-objs PRIVATE "${ime_lm_ffi_include}")
if(TARGET rime-ime-lm-ffi)
  add_dependencies(rime-ime-lm-objs rime-ime-lm-ffi)
endif()
if(BUILD_SHARED_LIBS)
  set_target_properties(rime-ime-lm-objs
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
endif()

# A Rust static library needs the platform's thread and dynamic-loading libraries linked beside it. These are plain flags rather than imported targets because librime links plugin_deps from its own directory, where this directory's imported targets are not visible.
find_package(Threads REQUIRED)

set(plugin_name rime-ime-lm PARENT_SCOPE)
set(plugin_objs $<TARGET_OBJECTS:rime-ime-lm-objs> PARENT_SCOPE)
set(plugin_deps ${rime_library} "${ime_lm_ffi_library}" ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} PARENT_SCOPE)
set(plugin_modules "ime_lm" PARENT_SCOPE)
