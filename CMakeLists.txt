cmake_minimum_required(VERSION 3.25)
project(miia LANGUAGES CXX)

add_library(miia_runtime
    src/runtime/numerical_runtime.cpp
    src/runtime/symbolic_runtime.cpp
    src/runtime/procedural_runtime.cpp
    src/runtime/inference_orchestrator.cpp
    src/runtime/inference_provider.cpp
)

target_include_directories(miia_runtime
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(miia_runtime PUBLIC cxx_std_23)