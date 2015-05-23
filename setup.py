from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import os

common_sources = ['cputime.c', 'hash.c', 'heap.c', 'hook.c', 'iolp.c',
                  'linalg.c', 'main.c', 'solve.c', 'strdup.c', 'tree.c',
                  'noamplio.c']
ipo_sources= ['ldlt.c']

all_sources = ['pyipo/cyipo.pyx']
all_sources += [os.path.join('pyipo','common',src) for src in common_sources]
all_sources += [os.path.join('pyipo','ipo',src) for src in ipo_sources]

cyipo = Extension('pyipo._ipo',
    sources = all_sources,
    include_dirs = ['pyipo/common', 'pyipo/ipo']
    )

setup(name='pyipo',
      packages=['pyipo'],
      install_requires=['cython>0.17'],
      ext_modules=[cyipo],
      cmdclass = {'build_ext': build_ext})
