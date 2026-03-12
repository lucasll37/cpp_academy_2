# ============================================
# cpp/tests/meson.build
# ============================================

gtest_dep = dependency('gtest',
    fallback: ['gtest', 'gtest_dep'],
    required: true,
)

gtest_main_dep = dependency('gtest_main',
    fallback: ['gtest', 'gtest_main_dep'],
    required: true,
)

# Usado apenas em env: strings, nunca em include_directories()
project_root = meson.project_source_root()

# ============================================
# Suite: unit
# ============================================

# Teste 1 — InferenceClient
# Não requer worker rodando nem modelos em disco.
test_client_exe = executable('test_unit_client',
    'unit/test_inference_client.cpp',
    dependencies: [
        inference_client_dep,
        gtest_main_dep,
        thread_dep,
    ],
    install: false,
)

test('unit_client',
    test_client_exe,
    suite:       'unit',
    timeout:     60,
    is_parallel: true,
)

# Teste 2 — InferenceEngine
# Teste 3 — BackendRegistry
#
# Dependem de worker_lib_dep e de modelos em disco (make create-models).
if get_option('enable_worker_tests')

    test_engine_exe = executable('test_unit_engine',
        'unit/test_inference_engine.cpp',
        dependencies: [
            worker_lib_dep,
            gtest_main_dep,
            thread_dep,
        ],
        install: false,
    )

    test('unit_engine',
        test_engine_exe,
        suite:       'unit',
        timeout:     120,
        is_parallel: false,   # BackendRegistry é singleton global
        env: {
            'MODELS_DIR': project_root / 'models',
        },
    )

    test_registry_exe = executable('test_unit_registry',
        'unit/test_backend_registry.cpp',
        dependencies: [
            worker_lib_dep,
            gtest_main_dep,
            thread_dep,
        ],
        install: false,
    )

    test('unit_registry',
        test_registry_exe,
        suite:       'unit',
        timeout:     30,
        is_parallel: true,
    )

endif  # enable_worker_tests

# ============================================
# Suite: integration
# Requer worker rodando em localhost:50052.
# Execute com: make test-integration
# ============================================
test_integration_exe = executable('test_integration',
    'integration/test_worker_integration.cpp',
    dependencies: [
        inference_client_dep,
        gtest_main_dep,
        thread_dep,
    ],
    install: false,
)

test('integration',
    test_integration_exe,
    suite:       'integration',
    timeout:     180,
    is_parallel: false,
    env: {
        'WORKER_ADDRESS': 'localhost:50052',
        'MODELS_DIR':     project_root / 'models',
    },
)
