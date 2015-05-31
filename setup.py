from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import os

common_sources = ['cputime.c', 'hash.c', 'heap.c', 'hook.c', 'iolp.c',
                  'linalg.c', 'main.c', 'solve.c', 'strdup.c', 'tree.c',
                  'noamplio.c']
ipo_sources= ['ldlt.c']

all_sources = ['pycllp/cyipo.pyx']
all_sources += [os.path.join('pycllp','common',src) for src in common_sources]
all_sources += [os.path.join('pycllp','ipo',src) for src in ipo_sources]

cyipo = Extension('pycllp._ipo',
    sources = all_sources,
    include_dirs = ['pycllp/common', 'pycllp/ipo']
    )

setup(name='pycllp',
      packages=['pycllp'],
      install_requires=['cython>0.17'],
      ext_modules=[cyipo],
      cmdclass = {'build_ext': build_ext})
