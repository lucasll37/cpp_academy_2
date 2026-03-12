project('asa-poc-miia', 'cpp',
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

# ============================================
# Opções do Projeto
# ============================================
enable_gpu = get_option('enable_gpu')

# ============================================
# Compilador e Flags
# ============================================
cpp = meson.get_compiler('cpp')

add_project_arguments(
    cpp.get_supported_arguments([
        '-Wno-pedantic',
        '-Wno-unused-parameter',
    ]),
    language: 'cpp',
)

# ============================================
# Dependências via pkg-config
# ============================================
thread_dep = dependency('threads')

protobuf_dep = dependency('protobuf',
    method: 'pkg-config',
    version: '>=3.21.0',
    required: true,
)

grpc_dep = dependency('grpc++',
    method: 'pkg-config',
    version: '>=1.54.0',
    required: true,
)

grpc_reflection_dep = dependency('grpc++_reflection',
    method: 'pkg-config',
    required: true,
)

onnxruntime_dep = dependency('onnxruntime',
    method: 'pkg-config',
    version: '>=1.18.1',
    required: true,
)

gtest_dep = dependency('gtest',
    fallback: ['gtest', 'gtest_dep'],
    required: true,
)

# ============================================
# protoc e grpc_cpp_plugin
# Resolução a partir dos bindir de cada pacote,
# garantindo consistência de versão com as libs.
# ============================================
grpc_bindir     = grpc_dep.get_variable('bindir')
protobuf_bindir = protobuf_dep.get_variable('bindir')

protoc          = find_program(protobuf_bindir / 'protoc',       required: true)
grpc_cpp_plugin = find_program(grpc_bindir     / 'grpc_cpp_plugin', required: true)

# ============================================
# Geração de código Protobuf / gRPC
# ============================================
proto_dir        = meson.project_source_root() / 'proto'
proto_output_dir = meson.current_build_dir()

proto_files = [
    'common.proto',
    'worker.proto',
]

generated_sources = []
generated_headers = []

foreach proto_file : proto_files
    proto_name = proto_file.replace('.proto', '')
    proto_path = proto_dir / proto_file

    pb_gen = custom_target(proto_name + '_pb',
        input:   proto_path,
        output:  [proto_name + '.pb.cc', proto_name + '.pb.h'],
        command: [
            protoc,
            '--cpp_out=' + proto_output_dir,
            '-I' + proto_dir,
            '@INPUT@',
        ],
        install:     true,
        install_dir: [false, get_option('includedir')],
    )

    grpc_gen = custom_target(proto_name + '_grpc',
        input:   proto_path,
        output:  [proto_name + '.grpc.pb.cc', proto_name + '.grpc.pb.h'],
        depends: pb_gen,
        command: [
            protoc,
            '--grpc_out=' + proto_output_dir,
            '--plugin=protoc-gen-grpc=' + grpc_cpp_plugin.full_path(),
            '--proto_path=' + proto_dir,
            '@INPUT@',
        ],
        install:     true,
        install_dir: [false, get_option('includedir')],
    )

    generated_sources += [pb_gen[0],  grpc_gen[0]]
    generated_headers += [pb_gen[1],  grpc_gen[1]]
endforeach

# ============================================
# Biblioteca Proto
#
# proto_inc aponta para o diretório de build onde os
# headers gerados são emitidos. É mantida local ao
# meson.build raiz e propagada apenas via proto_dep,
# evitando expor o build dir para outros targets.
# ============================================
proto_inc = include_directories('.')

proto_lib = static_library('proto_lib',
    generated_sources,
    include_directories: proto_inc,
    dependencies: [protobuf_dep, grpc_dep, grpc_reflection_dep],
    install: false,
)

proto_dep = declare_dependency(
    link_with:         proto_lib,
    include_directories: proto_inc,
    dependencies:      [protobuf_dep, grpc_dep, grpc_reflection_dep],
    sources:           generated_headers,
)

# ============================================
# Subdiretórios
# Ordem: worker → client → tests
# (tests depende de worker_lib_dep e inference_client_dep,
#  declarados nos respectivos meson.build)
# ============================================
subdir('cpp/worker')
subdir('cpp/client')
subdir('cpp/tests')

# ============================================
# pkg-config
# ============================================
pkg = import('pkgconfig')

pkg_conf = configuration_data()
pkg_conf.set('PREFIX',     get_option('prefix'))
pkg_conf.set('LIBDIR',     get_option('prefix') / 'lib')
pkg_conf.set('INCLUDEDIR', get_option('prefix') / 'include')
pkg_conf.set('VERSION',    meson.project_version())

configure_file(
    input:       'asa-poc-miia.pc.in',
    output:      'asa-poc-miia.pc',
    configuration: pkg_conf,
    install:     true,
    install_dir: get_option('libdir') / 'pkgconfig',
)

# ============================================
# Sumário — consolidado aqui para visão única
# (worker, client e tests não chamam summary())
# ============================================
summary({
    'Build Type': get_option('buildtype'),
    'C++ Standard': get_option('cpp_std'),
    'GPU Support': enable_gpu,
    'Worker Tests': get_option('enable_worker_tests'),
    'Prefix': get_option('prefix'),
    'Libdir': get_option('libdir'),
}, section: 'Configuration')

summary({
    'Worker':         worker.full_path(),
    'Client Library': inference_client_lib.full_path(),
    'Client CLI':     client_example.full_path(),
}, section: 'Build Artifacts')
