# ============================================
# tests/integration/meson.build
#
# Testes de integração — requerem worker rodando.
# Todos pulam automaticamente (GTEST_SKIP) se
# o worker estiver indisponível.
#
# Variáveis de ambiente respeitadas:
#   WORKER_ADDRESS  (padrão: localhost:50052)
#   MODELS_DIR      (padrão: ./models)
#
# Para rodar:
#   make run-worker &
#   make test-integration
# ============================================

project_root = meson.project_source_root()

_integration_env = {
    'WORKER_ADDRESS': 'localhost:50052',
    'MODELS_DIR':     project_root / 'models',
    'LOG_LEVEL':      'DEBUG'
}

gtest_main_lib = static_library(
    'gtest_main',
    'gtest_main.cpp',
    dependencies: [gtest_dep],
)

gtest_main_dep = declare_dependency(
    link_with: gtest_main_lib,
    dependencies: gtest_dep
)

# --------------------------------------------
# Geral — conectividade, ciclo de vida, batch
# --------------------------------------------

test('integration_inprocess_onnx',
    executable('test_inprocess_onnx',
        'test_inprocess_onnx.cpp',
        include_directories: [client_inc],
        dependencies: [inference_client_dep, gtest_main_dep],
    ),
    suite:       'integration',
    timeout:     20,
    is_parallel: false,
    env:         _integration_env,
)

test('integration_grpc_onnx',
    executable('test_grpc_onnx',
        'test_grpc_onnx.cpp',
        include_directories: [client_inc],
        dependencies: [inference_client_dep, gtest_main_dep],
    ),
    suite:       'integration',
    timeout:     20,
    is_parallel: false,
    env:         _integration_env,
)

test('integration_inprocess_python',
    executable('test_inprocess_python',
        'test_inprocess_python.cpp',
        include_directories: [client_inc],
        dependencies: [inference_client_dep, gtest_main_dep]
    ),
    suite:       'integration',
    timeout:     20,
    is_parallel: false,
    env:         _integration_env,
)

test('integration_grpc_python',
    executable('test_grpc_python',
        'test_grpc_python.cpp',
        include_directories: [client_inc],
        dependencies: [inference_client_dep, gtest_main_dep],
    ),
    suite:       'integration',
    timeout:     20,
    is_parallel: false,
    env:         _integration_env,
)

test('integration_server',
    executable('test_server',
        'test_server.cpp',
        include_directories: [server_inc, inference_inc, client_inc],
        dependencies: [worker_server_dep, proto_dep, inference_client_dep, gtest_dep],
        install: false,
    ),
    suite:       'integration',
    timeout:     20,
    is_parallel: false,
    env:         _integration_env,
)

test('sandbox',
    executable('sandbox',
        'sandbox.cpp',
        include_directories: [client_inc],
        dependencies: [inference_client_dep, gtest_main_dep],
    ),
    suite:       'integration',
    timeout:     30,
    is_parallel: false,
    env:         _integration_env,
)