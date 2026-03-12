sources = files('main.cpp')

client_example = executable('apaga',
  sources,
  include_directories: [],
  dependencies: [mixr_dep, asa_poc_miia_dep],
  install: true,
  install_dir: get_option('bindir')
)