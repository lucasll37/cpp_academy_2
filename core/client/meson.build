# ============================================
# cpp/client/meson.build
#
# client_inc vem do escopo raiz.
# inference_client.cpp inclui inprocess_backend.hpp que usa inference_engine.hpp,
# portanto worker_lib_dep (que encapsula inference_inc) é necessário.
# ============================================

inference_client_lib = library('asa_miia_client',
    files(
        'src/inference_client.cpp',
        'src/grpc_client_backend.cpp',
        'src/inprocess_backend.cpp',
    ),
    include_directories: [client_inc, inference_inc, utils_inc, proto_inc],
    dependencies: [
        proto_dep,
        grpc_dep,
        onnxruntime_dep,
        thread_dep,
        worker_lib_dep,
    ],
    install:     true,
    install_dir: get_option('libdir'),
)

inference_client_dep = declare_dependency(
    link_with:           inference_client_lib,
    include_directories: [client_inc, utils_inc],
    dependencies: [
        proto_dep,
        grpc_dep,
        thread_dep,
    ],
)

client_example = executable('AsaMiiaClient',
    files('src/main.cpp'),
    include_directories: [client_inc, utils_inc],
    dependencies:  [inference_client_dep],
    install_rpath: '$ORIGIN/../lib',
    install:       true,
    install_dir:   get_option('bindir'),
)

install_headers(
    'include/client/inference_client.hpp',
    subdir: 'asa-poc-miia/client',
)