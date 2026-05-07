if doxygen.found()
    custom_target('docs',
        input:   'Doxyfile',
        output:  'doxygen',
        command: [doxygen, '@INPUT@'],
        build_by_default: false,
    )
endif