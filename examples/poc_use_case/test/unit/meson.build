sources = [
  './example.cpp'
]

example_test = executable(
  'unit_tests',
  [sources, test_main],
  include_directories : public_headers,
  dependencies : [
    test_deps,
    asa_models_r_dep,
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
  link_with : [asa_tutorial])

test('AsaTutorial test', example_test)