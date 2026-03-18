cmake_minimum_required(VERSION 3.25)
project(miia LANGUAGES CXX)

# Runtime library
add_library(miia_runtime
    src/runtime/engine.cpp
)

target_include_directories(miia_runtime
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(miia_runtime PUBLIC cxx_std_23)