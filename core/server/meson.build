# ============================================
# cpp/server/meson.build
#
# Apenas o executável AsaMiia.
# worker_lib_dep, inference_inc e client_inc vêm do escopo raiz/inference.
# worker_server.hpp inclui inference/inference_engine.hpp e client/value_convert.hpp,
# por isso ambos os include dirs são necessários aqui.
# ============================================

worker_server_inc = include_directories('include')

worker_server_lib = static_library('asa_miia_worker_server',
    files('src/worker_server.cpp'),
    include_directories: [worker_server_inc, inference_inc, client_inc, utils_inc],
    dependencies: [
        proto_dep,
        thread_dep,
        worker_lib_dep,
    ],
    install: false,
)

worker_server_dep = declare_dependency(
    link_with:           worker_server_lib,
    include_directories: [worker_server_inc, inference_inc, client_inc, utils_inc],
    dependencies:        [worker_lib_dep],
)

worker = executable('AsaMiia',
    files('src/main.cpp'),
    include_directories: [worker_server_inc, inference_inc, client_inc, utils_inc],
    dependencies:  [worker_server_dep],
    install:       true,
    install_dir:   get_option('bindir'),
)
