..
    : This file is part of PyCI.
    :
    : PyCI is free software: you can redistribute it and/or modify it under
    : the terms of the GNU General Public License as published by the Free
    : Software Foundation, either version 3 of the License, or (at your
    : option) any later version.
    :
    : PyCI is distributed in the hope that it will be useful, but WITHOUT
    : ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    : FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    : for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with PyCI. If not, see <http://www.gnu.org/licenses/>.

|Python 3.8|

PyCI
====

PyCI_ is a flexible *ab-initio* quantum chemistry library for Configuration Interaction.

PyCI is distributed under the GNU General Public License version 3 (GPLv3).

See http://www.gnu.org/licenses/ for more information.

Dependencies
------------

The following programs/libraries are required to run PyCI:

-  Python_ (≥3.4)
-  NumPy_ (≥1.13)
-  SciPy_ (≥1.0)
-  Pytest_ (optional: to run tests)

The following programs/libraries are required to build PyCI:

-  GCC_ (≥4.8) or `Clang/LLVM`_ (≥3.3) C++ compiler
-  Python_ (≥3.4, including C headers)
-  NumPy_ (≥1.13, including C headers)

The following programs/libraries are required to build the PyCI documentation:

-  Sphinx_
-  `Read the Docs Sphinx Theme`__

__ Sphinx-RTD-Theme_

The following header-only libraries are included as git submodules:

-  `Parallel Hashmap`__
-  Pybind11_

__ Parallel-Hashmap_

The SpookyHash_ library source files (Public Domain) are included directly.

Installation
------------

Basic Compilation and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following in your shell to download PyCI and its submodules via git:

.. code:: shell

    git clone --recursive https://github.com/msricher/pyci.git

Then, run the following to build and install PyCI:

.. code:: shell

    python setup.py install --user

Compiling on OSX
~~~~~~~~~~~~~~~~

Since the default Xcode Clang compiler for OSX does not support OpenMP, the C and C++ compilers must
be specified by using the ``CC`` and ``CXX`` environment variables:

.. code:: shell

    CC=/path/to/your/gcc/or/clang CXX=/path/to/your/g++/or/clang++ python setup.py install --user

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Run the following in your shell to install the Read the Docs Sphinx theme via pip:

.. code:: shell

    pip install sphinx-rtd-theme --user

Then, after installing PyCI, run the following to build the HTML documentation:

.. code:: shell

    cd doc && make html

.. _PyCI:               http://github.com/msricher/PyCI/
.. _Python:             http://python.org/
.. _NumPy:              http://numpy.org/
.. _SciPy:              http://docs.scipy.org/doc/scipy/reference/
.. _GCC:                http://gcc.gnu.org/
.. _`Clang/LLVM`:       http://clang.llvm.org/
.. _Sphinx:             http://sphinx-doc.org/
.. _Sphinx-RTD-Theme:   http://sphinx-rtd-theme.readthedocs.io/
.. _Parallel-Hashmap:   http://github.com/greg7mdp/parallel-hashmap/
.. _Pybind11:           http://pybind11.readthedocs.io/en/stable/
.. _Pytest:             http://docs.pytest.org/en/latest/
.. _SpookyHash:         http://www.burtleburtle.net/bob/hash/spooky.html

.. |Python 3.8| image:: http://img.shields.io/badge/python-3.8-blue.svg
   :target: https://docs.python.org/3.8/
