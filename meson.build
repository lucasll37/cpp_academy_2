project('miia', 'cpp',
    version: '1.0.0',
    license: 'Copyright © 2026 ASA. All rights reserved.',
    default_options: [
        'c_std=c11',
        'cpp_std=c++17',
        'warning_level=3',
        'werror=false',
    ],
)

message('ML Inference System - Meson Build')
message('Version: ' + meson.project_version())
message('Build Type: ' + get_option('buildtype'))

enable_gpu = get_option('enable_gpu')

cpp = meson.get_compiler('cpp')

add_project_arguments(
    cpp.get_supported_arguments(['-Wno-pedantic', '-Wno-unused-parameter']),
    language: 'cpp',
)

add_project_link_arguments(
    '-Wl,-z,noexecstack',
    language: 'cpp',
)

# ============================================
# Dependências
# ============================================
thread_dep = dependency('threads')

protobuf_dep = dependency('protobuf',
    method:  'pkg-config',
    version: '>=3.21.0',
    required: true,
)
grpc_dep = dependency('grpc++',
    method:  'pkg-config',
    version: '>=1.54.0',
    required: true,
)
grpc_reflection_dep = dependency('grpc++_reflection',
    method:  'pkg-config',
    required: true,
)
onnxruntime_dep = dependency('onnxruntime',
    method:  'pkg-config',
    version: '>=1.18.1',
    required: true,
)

gtest_dep      = dependency('gtest',      fallback: ['gtest', 'gtest_dep'],      required: true)
gtest_main_dep = dependency('gtest_main', fallback: ['gtest', 'gtest_main_dep'], required: true)

# ============================================
# protoc / grpc_cpp_plugin
# ============================================
protoc          = find_program(protobuf_dep.get_variable('bindir') / 'protoc',          required: true)
grpc_cpp_plugin = find_program(grpc_dep.get_variable('bindir')     / 'grpc_cpp_plugin', required: true)

# ============================================
# doxygen
# ============================================
doxygen = find_program('doxygen', required: false)

# ============================================
# Geração de código Protobuf / gRPC
# ============================================
subdir('proto')

# ============================================
# Includes globais (declarados antes dos subdirs)
# ============================================
client_inc    = include_directories('core/client/include')
inference_inc = include_directories('core/inference/include')
utils_inc     = include_directories('core/utils/include')
server_inc    = include_directories('core/server/include')

# ============================================
# Detecção Python / numpy
# ============================================
subdir('python')

# ============================================
# Subdiretórios
# Ordem: inference → worker → client → tests
# inference deve vir primeiro pois worker e client dependem de worker_lib_dep
# ============================================
subdir('core/inference')
subdir('core/server')
subdir('core/client')
subdir('core/utils')

# ============================================
# Documentação
# ============================================
if get_option('enable_docs')
    subdir('docs/doxygen')
endif

# ============================================
# Testes
# ============================================
if get_option('build_tests')
  subdir('tests')
endif

# ============================================
# pkg-config
# ============================================
pkg_conf = configuration_data()
pkg_conf.set('PREFIX',     get_option('prefix'))
pkg_conf.set('LIBDIR',     get_option('prefix') / 'lib')
pkg_conf.set('INCLUDEDIR', get_option('prefix') / 'include')
pkg_conf.set('VERSION',    meson.project_version())

configure_file(
    input:         'miia.pc.in',
    output:        'miia.pc',
    configuration: pkg_conf,
    install:       true,
    install_dir:   get_option('libdir') / 'pkgconfig',
)

summary({
    'Build Type':   get_option('buildtype'),
    'C++ Standard': get_option('cpp_std'),
    'GPU Support':  enable_gpu,
    'Prefix':       get_option('prefix'),
    'Libdir':       get_option('libdir'),
}, section: 'Configuration')

summary({
    'Worker':         worker.full_path(),
    'Client Library': inference_client_lib.full_path(),
    'Client CLI':     client_example.full_path(),
}, section: 'Build Artifacts')
