# ============================================
# cpp/client/meson.build
# ============================================

client_inc = include_directories('include')

# ============================================
# Biblioteca cliente
#
# onnxruntime_dep é listado aqui porque inference_client.cpp
# inclui headers do ONNX Runtime internamente. É uma dependência
# de compilação/link da lib, mas NÃO é exposta no declare_dependency
# abaixo — quem usa inference_client_dep não precisa saber disso.
# ============================================
inference_client_lib = library('asa_miia_client',
    files('src/inference_client.cpp'),
    include_directories: [client_inc, proto_inc],
    dependencies: [
        proto_dep,
        grpc_dep,
        onnxruntime_dep,
        thread_dep,
    ],
    install:     true,
    install_dir: get_option('libdir'),
)

# Apenas dependências que fazem parte da interface pública da lib.
# onnxruntime é um detalhe de implementação — não vaza para fora.
inference_client_dep = declare_dependency(
    link_with:           inference_client_lib,
    include_directories: [client_inc],
    dependencies: [
        proto_dep,
        grpc_dep,
        thread_dep,
    ],
)

# ============================================
# CLI de exemplo
# ============================================
client_example = executable('AsaMiiaClient',
    files('src/main.cpp'),
    dependencies:  [inference_client_dep],
    install_rpath: '$ORIGIN/../lib',
    install:       true,
    install_dir:   get_option('bindir'),
)

install_headers(
    'include/client/inference_client.hpp',
    subdir: 'asa-poc-miia/client',
)
