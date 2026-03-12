# ============================================
# cpp/worker/meson.build
# ============================================

# ============================================
# Dependência Python — detecção de numpy
#
# Ordem de preferência para o interpretador:
#   1. $VENV_PYTHON (env var, usada pelo Docker e pelo make package)
#   2. python/.venv do projeto (desenvolvimento local)
#   3. python3 do sistema (fallback)
# ============================================
python3_dep = dependency('python3-embed', required: true)

_extra_compile_args = []

# Multiarch include — necessário quando o compilador é o do conda/cross,
# que não busca /usr/include/<triplet>/ por padrão.
_triplet_result = run_command('dpkg-architecture', '-qDEB_HOST_MULTIARCH', check: false)
if _triplet_result.returncode() == 0
    _multiarch_inc = '/usr/include' / _triplet_result.stdout().strip()
    if run_command('test', '-d', _multiarch_inc, check: false).returncode() == 0
        _extra_compile_args += ['-isystem', _multiarch_inc]
        message('Python multiarch include: ' + _multiarch_inc)
    endif
endif

# Localizar numpy
_venv_python_env = run_command('sh', '-c', 'echo $VENV_PYTHON', check: false).stdout().strip()

if _venv_python_env != ''
    _numpy_result = run_command(
        _venv_python_env, '-c', 'import numpy; print(numpy.get_include())',
        check: false,
    )
else
    _venv_python = meson.source_root() / 'python' / '.venv' / 'bin' / 'python'
    _numpy_result = run_command(
        _venv_python, '-c', 'import numpy; print(numpy.get_include())',
        check: false,
    )
endif

if _numpy_result.returncode() != 0
    _numpy_result = run_command(
        'python3', '-c', 'import numpy; print(numpy.get_include())',
        check: false,
    )
endif

if _numpy_result.returncode() == 0
    _extra_compile_args += ['-isystem', _numpy_result.stdout().strip()]
    message('numpy include: ' + _numpy_result.stdout().strip())
else
    error('numpy não encontrado. Execute: make python-install')
endif

# Agrega os compile_args extras numa dependência reutilizável
python_extra_dep = declare_dependency(compile_args: _extra_compile_args)

# ============================================
# Worker — biblioteca estática
#
# Separada do executável para que cpp/tests/ possa linkar
# diretamente com InferenceEngine, OnnxBackend etc.
# sem precisar subir o processo completo.
# ============================================
worker_inc = include_directories('include')

worker_lib = static_library('asa_miia_worker_lib',
    files(
        'src/worker_server.cpp',
        'src/inference_engine.cpp',
        'src/onnx_backend.cpp',
        'src/model_backend.cpp',
        'src/python_backend.cpp',
    ),
    include_directories: worker_inc,
    dependencies: [
        proto_dep,
        thread_dep,
        onnxruntime_dep,
        python3_dep,
        python_extra_dep,
    ],
    install: false,
)

# Dependency object reutilizável por cpp/tests/meson.build
worker_lib_dep = declare_dependency(
    link_with:           worker_lib,
    include_directories: worker_inc,
    dependencies: [
        proto_dep,
        thread_dep,
        onnxruntime_dep,
        python3_dep,
        python_extra_dep,
    ],
)

# ============================================
# Worker — executável
# Usa worker_lib_dep, que já encapsula includes,
# links e dependências transitivas.
# ============================================
worker = executable('AsaMiia',
    files('src/main.cpp'),
    dependencies: [worker_lib_dep],
    install:     true,
    install_dir: get_option('bindir'),
)
