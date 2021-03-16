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

%module pyfjcore
%feature("autodoc", "1");

// C++ standard library wrappers
%include <exception.i>
%include <std_string.i>
%include <std_vector.i>

// this can be used to ensure that swig parses classes correctly
#define SWIG_PREPROCESSOR

// include numpy typemaps
%{
#define SWIG_FILE_WITH_INIT
%}
%include numpy.i
%fragment("NumPy_Macros");

%{
// include these to avoid needing to define them at compile time 
#ifndef SWIG
#define SWIG
#endif

// C++ library headers
#include <cstdlib>
#include <cstring>

// PyFJCore headers
#include "fjcore.hh"
#include "PyFJCoreExtensions.hh"

// using namespaces
using namespace fastjet;

// Python class for representing errors from FastJet
static PyObject * FastJetError_;

%}

// this gets placed in the SWIG_init function
%init %{

  // for numpy
  import_array();

  // setup error class
  fastjet::Error::set_print_errors(false);
  FastJetError_ = PyErr_NewException("pyfjcore.FastJetError", NULL, NULL);
  Py_INCREF(FastJetError_);
  if (PyModule_AddObject(m, "FastJetError", FastJetError_) < 0) {
    Py_DECREF(m);
    Py_DECREF(FastJetError_);
  }

  // turn off printing banner
  fastjet::ClusterSequence::set_fastjet_banner_stream(new std::ostringstream());

  // default pseudojet printing
  fastjet::set_pseudojet_format(fastjet::PseudoJetRepresentation::ptyphim);
%}

// additional numpy typemaps
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* particles, int mult, int nfeatures)}
%apply (int** ARGOUTVIEWM_ARRAY1, int* DIM1) {(int** inds, int* mult)}
%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(double** particles, int* mult, int* nfeatures)}

%pythoncode %{
FastJetError = _pyfjcore.FastJetError;
%}

// vector templates
%template(vectorPseudoJet) std::vector<fastjet::PseudoJet>;

%typemap(in) const std::vector<fastjet::PseudoJet> & (int res = SWIG_OLDOBJ) {
    // ptk: convert PseudoJetContainer to const std::vector<PseudoJet> &
    void* argp = 0;
    res = SWIG_ConvertPtr($input, &argp, SWIGTYPE_p_fastjet__PseudoJetContainer, 0);
    if (SWIG_IsOK(res) && argp) {
      $1 = reinterpret_cast< fastjet::PseudoJetContainer * >(argp)->as_ptr();
      res = SWIG_OLDOBJ;
    }
    else {
      std::vector<PseudoJet> *ptr = (std::vector<PseudoJet> *) 0;
      res = swig::asptr($input, &ptr);
      if (SWIG_IsOK(res) && ptr)
        $1 = ptr;
      else {
        SWIG_exception_fail(SWIG_ArgError(res), "in method '$symname', argument $argnum of type '$type'");
      }
    }
  }

// basic exception handling for all functions
%exception {
  try { $action }
  catch (fastjet::Error & e) {
    PyErr_SetString(FastJetError_, e.message().c_str());
    SWIG_fail;
  }
  catch (std::exception & e) {
    SWIG_exception(SWIG_SystemError, e.what());
  }
}

// a macro to get support for description through __str__ method
%define FASTJET_SWIG_ADD_STR(Class)
%extend Class {
  std::string __str__() const {return $self->description();}
}
%enddef

// ignore variables
%ignore _INCLUDE_FJCORE_CONFIG_AUTO_H;
%ignore FJCORE_HAVE_DLFCN_H;
%ignore FJCORE_HAVE_INTTYPES_H;
%ignore FJCORE_HAVE_LIBM;
%ignore FJCORE_HAVE_MEMORY_H;
%ignore FJCORE_HAVE_STDINT_H;
%ignore FJCORE_HAVE_STDLIB_H;
%ignore FJCORE_HAVE_STRINGS_H;
%ignore FJCORE_HAVE_STRING_H;
%ignore FJCORE_HAVE_SYS_STAT_H;
%ignore FJCORE_HAVE_SYS_TYPES_H;
%ignore FJCORE_HAVE_UNISTD_H;
%ignore FJCORE_LT_OBJDIR;
%ignore FJCORE_PACKAGE;
%ignore FJCORE_PACKAGE_BUGREPORT;
%ignore FJCORE_PACKAGE_NAME;
%ignore FJCORE_PACKAGE_STRING;
%ignore FJCORE_PACKAGE_TARNAME;
%ignore FJCORE_PACKAGE_URL;
%ignore FJCORE_PACKAGE_VERSION;
%ignore FJCORE_STDC_HEADERS;

namespace fastjet {

// ignore functions that otherwise get wrapped
%ignore LimitedWarning;
%ignore Error;
%ignore InternalError;
%ignore IndexedSortHelper;
%ignore SelectorWorker;
%ignore _NoInfo;
%ignore operator+;
%ignore operator-;
%ignore operator*;
%ignore operator/;
%ignore operator==;
%ignore operator!=;
%ignore operator!;
%ignore operator||;
%ignore operator&&;
%ignore ClusterSequence::print_banner;
%ignore ClusterSequence::fastjet_banner_stream;
%ignore ClusterSequence::set_fastjet_banner_stream;
%ignore ClusterSequence::ClusterSequence();
%ignore ClusterSequence::ClusterSequence(const ClusterSequence &);
%ignore UserInfoPython;

} // namespace fastjet

// include EECHist and declare templates
%include "fjcore.hh"
%include "PyFJCoreExtensions.hh"

// makes python class printable from a description method
%define ADD_REPR_FROM_DESCRIPTION
%pythoncode %{
  def __repr__(self):
      return self.description()
%}
%enddef

namespace fastjet {

  %extend PseudoJetContainer {
    %pythoncode {
      def __len__(self):
          return len(self.vector)

      def __iter__(self):
          return self.vector.__iter__();

      def __repr__(self):
          s = ['PseudoJetContainer[' + str(len(self)) + '](']
          for pj in self:
              s.append('  ' + repr(pj) + ',')
          s.append(')')
          return '\n'.join(s)

      def __delitem__(self, key):
          self.vector.__delitem__(key)

      def __getitem__(self, key):
          return self.vector.__getitem__(key)

      def __setitem__(self, key, val):
          self.vector.__setitem__(key, val)

      @property
      def vector(self):
          if not hasattr(self, '_vector'):
              self._vector = self.as_vector()
          return self._vector
    }
  }

  %extend PseudoJet {

    std::string __repr__() {
      const unsigned len_max = 512;
      char temp[len_max];
      if (PseudoJetRep_ == PseudoJetRepresentation::ptyphim)
        snprintf(temp, len_max, "PseudoJet(pt=%.6g, y=%.6g, phi=%.6g, m=%.6g)",
                 $self->pt(), $self->rap(), $self->phi(),
                 [](double m){return std::fabs(m) < 1e-6 ? 0 : m;}($self->m()));
      else if (PseudoJetRep_ == PseudoJetRepresentation::epxpypz)
        snprintf(temp, len_max, "PseudoJet(e=%.6g, px=%.6g, py=%.6g, pz=%.6g)",
                 $self->e(), $self->px(), $self->py(), $self->pz());
      else
        snprintf(temp, len_max, "PseudoJet(pt=%.6g, y=%.6g, phi=%.6g)",
                 $self->pt(), $self->rap(), $self->phi());
      return std::string(temp);
    }

    void set_python_info(PyObject * pyobj) {
      fastjet::UserInfoPython * new_python_info = new fastjet::UserInfoPython(pyobj);
      $self->set_user_info(new_python_info);
    }

    PyObject * python_info() const {
      return $self->user_info<fastjet::UserInfoPython>().get_pyobj();
    }
    
    // these C++ operators are not automatically handled by SWIG (would only
    // be handled if there were part of the class)
    PseudoJet __add__ (const PseudoJet & p) { return *($self) + p; }
    PseudoJet __sub__ (const PseudoJet & p) { return *($self) - p; }
    bool      __eq__  (const PseudoJet & p) { return *($self) == p; }
    bool      __ne__  (const PseudoJet & p) { return *($self) != p; }
    PseudoJet __mul__ (double x) { return *($self) * x; }
    PseudoJet __rmul__(double x) { return *($self) * x; }
    PseudoJet __div__ (double x) { return *($self) / x; }
    bool      __eq__  (double x) { return *($self) == x; }
    bool      __ne__  (double x) { return *($self) != x; }
  }

  // extend JetDefinition
  %extend JetDefinition {
    ADD_REPR_FROM_DESCRIPTION
    PseudoJetContainer __call__(const std::vector<PseudoJet> & particles) {
      return (*self)(particles);
    }
  }

  %extend Selector {
    ADD_REPR_FROM_DESCRIPTION

    // The C++ operators [* && || !] map to [* & | ~] in python
    Selector __mul__   (const Selector & other) { return *($self) *  other; }
    Selector __and__   (const Selector & other) { return *($self) && other; }
    Selector __or__    (const Selector & other) { return *($self) || other; }
    Selector __invert__()                       { return !(*($self)); }
  }

} // namespace fastjet
