// -*- C++ -*-
//
// PyFJCore - Python wrapper of FJCore functionality
// Copyright (C) 2020 Patrick T. Komiske III
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

%{
// needed by numpy.i, harmless otherwise
#define SWIG_FILE_WITH_INIT

// standard library headers we need
#include <cstdlib>
#include <cstring>
%}

// include numpy typemaps
%include numpy.i
%init %{
import_array();
%}

%pythoncode %{
import numpy as _np
%}

// numpy typemaps
/*%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {
  (double** arr_out0, int* n0),
  (double** arr_out1, int* n1)
}
%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {
  (double** arr_out0, int* n0, int* n1)
}
%apply (double** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {
  (double** arr_out0, int* n0, int* n1, int* n2),
  (double** arr_out1, int* m0, int* m1, int* m2)
}

// mallocs a 1D array of doubles of the specified size
%define MALLOC_1D_DOUBLE_ARRAY(arr_out, n, size, nbytes)
  *n = size;
  size_t nbytes = size_t(*n)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL)
    throw std::runtime_error("failed to allocate " + std::to_string(nbytes) + " bytes");
%enddef

// mallocs a 3D array of doubles of the specified size
%define MALLOC_2D_DOUBLE_ARRAY(arr_out, n0, n1, size0, size1, nbytes)
  *n0 = size0;
  *n1 = size1;
  size_t nbytes = size_t(*n0)*size_t(*n1)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL)
    throw std::runtime_error("failed to allocate " + std::to_string(nbytes) + " bytes");
%enddef

// mallocs a 3D array of doubles of the specified size
%define MALLOC_3D_DOUBLE_ARRAY(arr_out, n0, n1, n2, size0, size1, size2, nbytes)
  *n0 = size0;
  *n1 = size1;
  *n2 = size2;
  size_t nbytes = size_t(*n0)*size_t(*n1)*size_t(*n2)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL)
    throw std::runtime_error("failed to allocate " + std::to_string(nbytes) + " bytes");
%enddef

%define COPY_1DARRAY_TO_NUMPY(arr_out0, n0, size, nbytes, ptr)
  MALLOC_1D_DOUBLE_ARRAY(arr_out0, n0, size, nbytes)
  memcpy(*arr_out0, ptr, nbytes);
%enddef

%define RETURN_1DNUMPY_FROM_VECTOR(pyname, cppname, size)
void pyname(double** arr_out0, int* n0) {
  COPY_1DARRAY_TO_NUMPY(arr_out0, n0, size, nbytes, $self->cppname().data())
}
%enddef
*/