from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import os
import numpy

common_sources = ['cputime.c', 'hash.c', 'heap.c', 'hook.c', 'iolp.c',
                  'linalg.c', 'main.c', 'solve.c', 'strdup.c', 'tree.c',
                  'noamplio.c']
ipo_sources= ['ldlt.c']

all_sources = ['pycllp/cyipo.pyx']
all_sources += [os.path.join('pycllp','common',src) for src in common_sources]
all_sources += [os.path.join('pycllp','ipo',src) for src in ipo_sources]

cyipo = Extension('pycllp._ipo',
    sources = all_sources,
    include_dirs = ['pycllp/common', 'pycllp/ipo', numpy.get_include()]
    )

glpk = Extension('pycllp.solvers.cython_glpk', ['pycllp/solvers/cython_glpk.pyx'], libraries=['glpk'],)

ldl = Extension('pycllp._ldl', ['pycllp/_ldl.pyx'], include_dirs=[numpy.get_include()])

setup(name='pycllp',
      packages=['pycllp', 'pycllp.solvers'],
      install_requires=['numpy>=1.7', 'scipy>=0.14', 'pyopencl>=2015.0',
                        'cython>0.17'],
      ext_modules=[cyipo, glpk, ldl],
      cmdclass = {'build_ext': build_ext},
      package_data={'pycllp': ['cl/*.h', 'cl/*.cl']}
      )
