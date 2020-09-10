# coding:utf-8

from setuptools import find_packages, Extension

from distutils.core import setup
from Cython.Distutils import build_ext
import numpy as np

# python setup.py build_ext --inplace (make sure using gcc 7.1.0)

extra_compile_args = []
extra_link_args = []

ext_modules = [
    Extension(
      "pyhawkes.internals.parent_updates",
      ["internals/parent_updates.pyx"],
      define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    ),
    Extension(
      "pyhawkes.internals.continuous_time_helpers",
      ["internals/continuous_time_helpers.pyx"],
      define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    ),
    Extension(
      "pyhawkes.internals.weight_updates",
      ["internals/weight_updates.pyx"],
      define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    ),
]

for e in ext_modules:
    e.extra_compile_args.extend(extra_compile_args)
    e.extra_link_args.extend(extra_link_args)

setup(
      name='pyhawkes',
      version='1.0.1',
      description='pyhawkes library',
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules,
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'joblib',
                        'scikit-learn',
                        'pybasicbayes'],
      include_dirs=[np.get_include()],
      packages=find_packages()
     )
