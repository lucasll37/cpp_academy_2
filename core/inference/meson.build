# ============================================
# cpp/inference/meson.build
#
# Constrói a biblioteca estática com InferenceEngine e backends.
# O executable AsaMiia fica em cpp/worker/meson.build.
# inference_inc e client_inc vêm do escopo raiz.
# ============================================

worker_lib = static_library('asa_miia_worker_lib',
    files(
        'src/inference_engine.cpp',
        'src/onnx_backend.cpp',
        'src/model_backend.cpp',
        'src/python_backend.cpp'
    ),
    include_directories: [inference_inc, client_inc, utils_inc],
    dependencies: [
        proto_dep,
        thread_dep,
        onnxruntime_dep,
        python3_dep,
        python_extra_dep
    ],
    install: false,
)

worker_lib_dep = declare_dependency(
    link_with:           worker_lib,
    include_directories: [inference_inc, client_inc, utils_inc],
    dependencies: [
        proto_dep,
        thread_dep,
        onnxruntime_dep,
        python3_dep,
        python_extra_dep
    ]
)
