# This file is part of PyCI.
#
# PyCI is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# PyCI is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCI. If not, see <http://www.gnu.org/licenses/>.

r"""
PyCI setup script.

Run `python setup.py --help` for help.

"""

from os import path

from setuptools import setup

import numpy


# Uncomment this to use exact (colexicographical order) hashing.
# This only supports determinant sets with |D| < 2 ** 63.
#PYCI_EXACT_HASH = True


# Uncomment this to use a specific non-negative integer seed for the SpookyHash algorithm.
#PYCI_SPOOKYHASH_SEED = 0


name = 'pyci'


version = '0.3.5'


license = 'GPLv3'


author = 'Michael Richer'


author_email = 'richerm@mcmaster.ca'


url = 'https://github.com/msricher/PyCI'


description = 'A flexible ab-initio quantum chemistry library for Configuration Interaction.'


long_description = open('README.rst', 'r', encoding='utf-8').read()


classifiers = [
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Science/Engineering :: Molecular Science',
        ]


install_requires = [
        'numpy>=1.13',
        ]


extras_require = {
        'build': ['cython'],
        'test': ['nose'],
        'doc': ['sphinx', 'sphinx_rtd_theme'],
        }


packages = [
        'pyci',
        'pyci.test',
        ]


package_data = {
        'pyci': ['pyci.pyx', 'pyci.cpp', 'include/*.h', 'include/*.pxd', 'src/*.cpp'],
        'pyci.test': ['data/*.fcidump', 'data/*.npy', 'data/*.npz'],
        }


sources = [
        'pyci/src/pyci.cpp',
        'pyci/pyci.cpp',
        ]


include_dirs = [
        numpy.get_include(),
        'lib/parallel-hashmap',
        'lib/eigen',
        'lib/spectra/include',
        'pyci/include',
        ]


extra_compile_args = [
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-Wall',
        '-fopenmp',
        ]


extra_link_args = [
        '-fopenmp',
        ]


cython_compile_time_env = {
        'PYCI_VERSION': version,
        }


cext = {
        'name': 'pyci.pyci',
        'language': 'c++',
        'sources': sources,
        'include_dirs': include_dirs,
        'extra_compile_args': extra_compile_args,
        'extra_link_args': extra_link_args,
        }


if __name__ == '__main__':


    try:

        from Cython.Distutils import Extension, build_ext

        sources.clear()
        sources.extend(('pyci/src/pyci.cpp', 'pyci/pyci.pyx'))
        cext.update(cython_compile_time_env=cython_compile_time_env)

    except ImportError:

        from setuptools import Extension
        from setuptools.command.build_ext import build_ext


    try:

        if PYCI_EXACT_HASH:
            extra_compile_args.append('-DPYCI_EXACT_HASH')

    except NameError:

        pass


    try:

        hex_seed = hex(abs(PYCI_SPOOKYHASH_SEED)) + 'UL'
        extra_compile_args.append('-DPYCI_SPOOKYHASH_SEED=' + hex_seed)

    except NameError:

        pass


    pyci_extension = Extension(**cext)


    ext_modules = [
            pyci_extension,
            ]


    cmdclass = {
            'build_ext': build_ext,
            }


    setup(
            name=name,
            version=version,
            license=license,
            author=author,
            author_email=author_email,
            url=url,
            description=description,
            long_description=long_description,
            classifiers=classifiers,
            install_requires=install_requires,
            extras_require=extras_require,
            packages=packages,
            package_data=package_data,
            include_package_data=True,
            ext_modules=ext_modules,
            cmdclass=cmdclass,
            )
