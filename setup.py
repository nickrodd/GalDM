from __future__ import print_function

import logging
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import cython_gsl

extensions = [
    Extension("GalDM.dm_int", ["GalDM/dm_int.pyx"],
              include_dirs=[numpy.get_include(),cython_gsl.get_include()], library_dirs=[cython_gsl.get_library_dir()],libraries=["m","gsl","gslcblas"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp","-lcblas","-lgslcblas","-lgsl"],
              extra_link_args=['-fopenmp']),
    Extension("GalDM.dm_profiles", ["GalDM/dm_profiles.pyx"],
              include_dirs=[numpy.get_include(),cython_gsl.get_include()], library_dirs=[cython_gsl.get_library_dir()],libraries=["m","gsl","gslcblas"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp","-lcblas","-lgslcblas","-lgsl"],
              extra_link_args=['-fopenmp'])
]

setup_args = {'name':'GalDM',
    'version':'0.0',
    'description':'Creating galactic dark matter maps and calculated J/D factors',
    'url':'https://github.com/nickrodd/GalDM',
    'author':'Nicholas L Rodd',
    'author_email':'nrodd@mit.edu',
    'license':'MIT',
    'install_requires':[
            'numpy',
            'matplotlib',
            'Cython',
            'jupyter',
            'CythonGSL',
        ]}

setup(packages=['GalDM'],
    ext_modules = cythonize(extensions),
    **setup_args
)

print("Compilation successful!")
