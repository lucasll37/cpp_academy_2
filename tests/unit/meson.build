# ============================================
# tests/unit/meson.build
#
# Testes unitários — não requerem servidor.
# ============================================

project_root = meson.project_source_root()

_integration_env = {
    'MODELS_DIR':     project_root / 'models',
    'LOG_LEVEL':      'DEBUG'
}

gtest_main_lib = static_library(
    'gtest_main',
    'gtest_main.cpp',
    dependencies: [worker_server_dep, gtest_dep],
)

gtest_main_dep = declare_dependency(
    link_with: gtest_main_lib
)

test('unit_python_backend',
    executable('test_python_backend',
        'test_python_backend.cpp',
        include_directories: [inference_inc, client_inc],
        dependencies: [worker_lib_dep, proto_dep, gtest_dep],
        install: false,
    ),
    suite:       'unit',
    timeout:     20,
    is_parallel: false,   # CPython é single-instance por processo
    env:         _integration_env,
)

test('unit_onnx_backend',
    executable('test_unit_onnx_backend',
        'test_onnx_backend.cpp',
        include_directories: [inference_inc, client_inc],
        dependencies: [worker_lib_dep, proto_dep, gtest_main_dep, gtest_dep],
        install: false,
    ),
    suite:       'unit',
    timeout:     20,
    is_parallel: false,
    env:         _integration_env,
)

test('unit_inference_engine',
    executable('test_unit_inference_engine',
        'test_inference_engine.cpp',
        include_directories: [inference_inc, client_inc],
        dependencies: [worker_lib_dep, proto_dep, gtest_main_dep, gtest_dep],
        install: false,
    ),
    suite:       'unit',
    timeout:     20,
    is_parallel: false,
    env:         _integration_env,
)

test('unit_backend_registry',
    executable('test_unit_backend_registry',
        'test_backend_registry.cpp',
        include_directories: [inference_inc, client_inc],
        dependencies: [worker_lib_dep, proto_dep, gtest_main_dep, gtest_dep],
        install: false,
    ),
    suite:       'unit',
    timeout:     20,
    is_parallel: false,
    env:         _integration_env,
)

test('unit_value',
    executable('test_unit_value',
        'test_value.cpp',
        include_directories: [client_inc],
        dependencies: [gtest_main_dep, gtest_dep],
        install: false,
    ),
    suite:       'unit',
    timeout:     20,
    is_parallel: true,
)

test('unit_value_convert',
    executable('test_unit_value_convert',
        'test_value_convert.cpp',
        include_directories: [client_inc],
        dependencies: [worker_lib_dep, gtest_main_dep, gtest_dep],
        install: false,
    ),
    suite:       'unit',
    timeout:     20,
    is_parallel: true,
)

test('unit_server',
    executable('test_unit_server',
        'test_server.cpp',
        include_directories: [server_inc, inference_inc, client_inc],
        dependencies: [worker_server_dep, proto_dep,
                       gtest_main_dep, gtest_dep],
        install: false,
    ),
    suite:       'unit',
    timeout:     20,
    is_parallel: false,
    env:         _integration_env
)