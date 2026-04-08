cmake_minimum_required(VERSION 3.25)
project(miia LANGUAGES CXX)

# 1. Cria a biblioteca (target) primeiro
add_library(miia_runtime
    src/runtime/numerical_runtime.cpp
    src/runtime/symbolic_runtime.cpp
    src/runtime/procedural_runtime.cpp
    src/runtime/inference_orchestrator.cpp
    src/runtime/inference_provider.cpp
)

# 2. Define os diretórios de include públicos
target_include_directories(miia_runtime
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

# 3. Define o padrão de C++ utilizado
target_compile_features(miia_runtime PUBLIC cxx_std_23)

# 4. Encontra a dependência pybind11
find_package(pybind11 REQUIRED)

# 5. Faz o link correto com pybind11 (modo embed para rodar Python)
target_link_libraries(miia_runtime
    PUBLIC pybind11::embed
)