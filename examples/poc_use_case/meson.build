project('asa-tutorial', 'cpp',
    license: 'Copyright © 2021-2023 ASA. All rights reserved.',
    default_options : [
        'c_std=c11',
        'cpp_std=c++17'
    ]
)

script = run_command('./version.sh', check: true)
version = script.stdout().strip()

buildtype = get_option('buildtype')

message('Building version:', version)
message('Building type:', buildtype)

# Dependencies

pkg = import('pkgconfig')

# 2nd party dependencies
mixr_dep             = dependency('mixr')
asa_libs_dep         = dependency('asa-libs')
asa_models_dep       = dependency('asa-models')

# 3rd party dependencies
behavior_tree_dep    = dependency('behaviortree.cpp.asa')
boost_chrono_dep     = dependency('boost-chrono')
boost_filesystem_dep = dependency('boost-filesystem')
fmt_dep              = dependency('fmt')
gtest_dep            = dependency('gtest')
protobuf_dep         = dependency('protobuf')
rapidjson_dep        = dependency('rapidjson')
asa_poc_miia_dep     = dependency('asa-poc-miia')

protoc = '@0@/bin/protoc'.format(protobuf_dep.get_variable('prefix'))

public_headers = include_directories('include')

subdir('./src')
# subdir('./test')

# Installation

conf_data = configuration_data({
    'prefix' : get_option('prefix'),
    'libdir' : get_option('prefix') / 'usr/extensions/AsaTutorial/bin',
    'description' : 'Asa tutorial model.',
    'version' : version,
})

configure_file(
    input : 'asa-tutorial.pc.in',
    output : 'asa-tutorial.pc',
    configuration : conf_data,
    install : true,
    install_dir : get_option('prefix') / 'lib/pkgconfig'
)

install_subdir(
    'include/asa/tutorial',
    install_dir : 'include/asa',
)

install_subdir(
    'resources/icons',
    install_dir : 'usr/extensions/AsaTutorial',
)

install_subdir(
    'resources/templates',
    install_dir : 'usr/extensions/AsaTutorial',
)

install_data(
    'resources/config.json',
    install_dir : 'usr/extensions/AsaTutorial'
)
