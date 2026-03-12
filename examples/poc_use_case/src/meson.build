source_files = [
    './HazardSource.cpp',
    './Hydrodynamics.cpp',
    './Scanner.cpp',
    './Ship.cpp',
    './ShipAction.cpp',
    './ShipAgent.cpp',
    './ShipState.cpp',
    './export.cpp',
]

asa_tutorial = library(
    'AsaTutorial',
    [source_files],
    include_directories : public_headers,
    dependencies : [
        # 2nd party
        asa_libs_dep,
        asa_models_dep,
        mixr_dep,
        asa_poc_miia_dep,
        # 3rd party
        behavior_tree_dep,
        boost_chrono_dep,
        boost_filesystem_dep,
        protobuf_dep,
        fmt_dep,
        rapidjson_dep,
    ],
    install : true,
    install_dir: get_option('prefix') / 'usr/extensions/AsaTutorial/bin',
    cpp_args : '-DCOMPILATION_BUILD_TYPE="@0@"'.format(buildtype)
)

asa_models_r_dep = declare_dependency(
    link_with : asa_tutorial,
    include_directories : public_headers
)
