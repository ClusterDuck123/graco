from setuptools import setup
from Cython.Build import cythonize

setup(
    name='This_is_a_name',
    ext_modules=cythonize("cdistances.pyx"),
    zip_safe=False,
)
