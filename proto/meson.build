# ============================================
# Geração de código Protobuf / gRPC
# ============================================
proto_dir        = meson.project_source_root() / 'proto'
proto_output_dir = meson.current_build_dir()

generated_sources = []
generated_headers = []

foreach proto_file : ['common.proto', 'server.proto']
    proto_name = proto_file.replace('.proto', '')

    pb_gen = custom_target(proto_name + '_pb',
        input:   proto_dir / proto_file,
        output:  [proto_name + '.pb.cc', proto_name + '.pb.h'],
        command: [protoc, '--cpp_out=' + proto_output_dir, '-I' + proto_dir, '@INPUT@'],
        install:     true,
        install_dir: [false, get_option('includedir')],
    )
    grpc_gen = custom_target(proto_name + '_grpc',
        input:   proto_dir / proto_file,
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

proto_inc = include_directories('.')

proto_lib = static_library('proto_lib',
    generated_sources,
    include_directories: proto_inc,
    dependencies: [protobuf_dep, grpc_dep, grpc_reflection_dep],
    install: false,
)
proto_dep = declare_dependency(
    link_with:           proto_lib,
    include_directories: proto_inc,
    dependencies:        [protobuf_dep, grpc_dep, grpc_reflection_dep],
    sources:             generated_headers,
)