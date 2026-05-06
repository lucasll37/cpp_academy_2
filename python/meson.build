# ============================================
# Detecção Python / numpy
# ============================================

python3_dep = dependency('python3-embed', required: true)

_extra_compile_args = []

_triplet = run_command('dpkg-architecture', '-qDEB_HOST_MULTIARCH', check: false)
if _triplet.returncode() == 0
    _multiarch_inc = '/usr/include' / _triplet.stdout().strip()
    if run_command('test', '-d', _multiarch_inc, check: false).returncode() == 0
        _extra_compile_args += ['-isystem', _multiarch_inc]
    endif
endif

_numpy = run_command('python3', '-c', 'import numpy; print(numpy.get_include())', check: false)

if _numpy.returncode() == 0
    _extra_compile_args += ['-isystem', _numpy.stdout().strip()]
    message('numpy include: ' + _numpy.stdout().strip())
else
    error('numpy não encontrado. Execute: make python-install')
endif

python_extra_dep = declare_dependency(compile_args: _extra_compile_args)