# PyFJCore - Python wrapper of FJCore functionality
# Copyright (C) 2020 Patrick T. Komiske III
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import platform
import re
import subprocess
import sys

from setuptools import setup
from setuptools.extension import Extension

import numpy as np

with open(os.path.join('pyfjcore', '__init__.py'), 'r') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

# run swig to generate pyfjcore.py and pyfjcore.cpp from pyfjcore.i
if sys.argv[1] == 'swig':
    command = 'swig -python -c++ -fastproxy -w509,511 -keyword -py3 -Ipyfjcore -o pyfjcore/pyfjcore.cpp pyfjcore/swig/pyfjcore.i'
    print(command)
    subprocess.run(command.split())

# build extension
else:

    # compiler flags, libraries, etc
    cxxflags = ['-std=c++14', '-g0']
    if platform.system() == 'Windows':
        cxxflags = ['/std:c++14']

    pyfjcore = Extension('pyfjcore._pyfjcore',
                         sources=[os.path.join('pyfjcore', 'pyfjcore.cpp'), os.path.join('pyfjcore', 'fjcore.cc')],
                         define_macros=[('SWIG', None), ('SWIG_TYPE_TABLE', 'fjcore')],
                         include_dirs=[np.get_include(), 'pyfjcore'],
                         extra_compile_args=cxxflags)

    setup(
        ext_modules=[pyfjcore],
        version=__version__
    )