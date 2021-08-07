// -*- C++ -*-
//
// PyFJCore - Python wrapper of FJCore functionality
// Copyright (C) 2020-2021 Patrick T. Komiske III
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
# define SWIG
#endif

// C++ library headers
#include <cstdlib>
#include <cstring>

// PyFJCore headers
#include "pyfjcore/fjcore.hh"
#include "pyfjcore/PyFJCoreExtensions.hh"

using namespace fastjet;

%}

// define as macro for use in contrib files
%define FASTJET_ERRORS_AS_PYTHON_EXCEPTIONS(module)
%{
// Python class for representing errors from FastJet
static PyObject * FastJetError_;
%}

// this gets placed in the SWIG_init function
%init %{
  // setup error class
  fastjet::Error::set_print_errors(false);
  char * msg_pyfjcore = (char*) calloc(strlen(`module`)+15, sizeof(char));
  strcpy(msg_pyfjcore, `module`);
  strcat(msg_pyfjcore, ".FastJetError");
  FastJetError_ = PyErr_NewException(msg_pyfjcore, NULL, NULL);
  Py_INCREF(FastJetError_);
  if (PyModule_AddObject(m, "FastJetError", FastJetError_) < 0) {
    Py_DECREF(m);
    Py_DECREF(FastJetError_);
  }
%}
%enddef

FASTJET_ERRORS_AS_PYTHON_EXCEPTIONS(pyfjcore)

// this gets placed in the SWIG_init function
%init %{

// for numpy
import_array();

// turn off printing banner
fastjet::ClusterSequence::set_fastjet_banner_stream(new std::ostringstream());
%}

%numpy_typemaps(float, NPY_FLOAT, std::ptrdiff_t)
%numpy_typemaps(double, NPY_DOUBLE, std::ptrdiff_t)
%numpy_typemaps(int, NPY_INT, std::ptrdiff_t)

// additional numpy typemaps
%apply (double* IN_ARRAY2, std::ptrdiff_t DIM1, std::ptrdiff_t DIM2) {
  (double* particles, std::ptrdiff_t mult, std::ptrdiff_t nfeatures)
}
%apply (int** ARGOUTVIEWM_ARRAY1, std::ptrdiff_t* DIM1) {
  (int** inds, std::ptrdiff_t* mult)
}
%apply (double** ARGOUTVIEWM_ARRAY2, std::ptrdiff_t* DIM1, std::ptrdiff_t* DIM2) {
  (double** particles, std::ptrdiff_t* mult, std::ptrdiff_t* nfeatures)
}
%apply (float** ARGOUTVIEWM_ARRAY2, std::ptrdiff_t* DIM1, std::ptrdiff_t* DIM2) {
  (float** particles, std::ptrdiff_t* mult, std::ptrdiff_t* nfeatures)
}

%pythoncode %{
  FastJetError = _pyfjcore.FastJetError;

  import platform
  if platform.system() != 'Windows':
      import ctypes
      ctypes.CDLL(_pyfjcore.__file__, mode=ctypes.RTLD_GLOBAL)
%}

// vector templates
%template(vectorPseudoJet) std::vector<fastjet::PseudoJet>;

// ensure that we move-construct the heap container from the stack container
%typemap(out) fastjet::PseudoJetContainer %{
  $result = SWIG_NewPointerObj(new fastjet::PseudoJetContainer(std::move($1)),
                               SWIGTYPE_p_fastjet__PseudoJetContainer, SWIG_POINTER_OWN | 0);
%}

// basic exception handling for all functions
%exception {
  try { $action }
  catch (fastjet::Error & e) {
    PyErr_SetString(FastJetError_, e.message().c_str());
    SWIG_fail;
  }
  SWIG_CATCH_STDEXCEPT
  catch (...) {
    SWIG_exception_fail(SWIG_UnknownError, "unknown exception");
  }
}

// makes python class printable from a description method
%define ADD_REPR_FROM_DESCRIPTION
  std::string __repr__() const {
    return $self->description();
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
  %ignore ArrayToPseudoJets;
  %ignore ConstructPtYPhiM;
  %ignore ConstructPtYPhi;
  %ignore ConstructEPxPyPz;

  %rename(passes) Selector::pass;

  %copyctor PseudoJetContainer;

  // needs to be here so that templated version is defined for Recluster wrapper
  template<typename TOut>
  class FunctionOfPseudoJet{
  public:
    FunctionOfPseudoJet(){}
    virtual ~FunctionOfPseudoJet(){}
    virtual std::string description() const;
    virtual TOut result(const PseudoJet &pj) const = 0;
    TOut operator()(const PseudoJet &pj) const;
    std::vector<TOut> operator()(const std::vector<PseudoJet> &pjs) const;
  };

  %template(FunctionOfPseudoJet_PseudoJet) FunctionOfPseudoJet<fastjet::PseudoJet>;

  // ensure methods that modify container delete _vector
  %extend PseudoJetContainer {

    %feature("shadow") clear %{
      def clear(self):
          if hasattr(self, '_vector'):
              del self._vector
          $action(self)
    %}

    %feature("shadow") push_back %{
      def push_back(self, val):
          if hasattr(self, '_vector'):
              del self._vector
          $action(self, val)
    %}

    %feature("shadow") resize %{
      def resize(self, size):
          if hasattr(self, '_vector'):
              del self._vector
          $action(self, size)
    %}
  }

} // namespace fastjet

// include EECHist and declare templates
%include "pyfjcore/fjcore.hh"
%include "pyfjcore/PyFJCoreExtensions.hh"

namespace fastjet {

  // template SharedPtr
  %template(sharedPtrPseudoJetStructureBase) SharedPtr<fastjet::PseudoJetStructureBase>;

  // functions accepting a vector of pseudojets and returning a numpy array
  %template(pjs_to_epxpypz_array_float64) pjs_to_epxpypz_array<double>;
  %template(pjs_to_epxpypz_array_float32) pjs_to_epxpypz_array<float>;
  %template(pjs_to_ptyphim_array_float64) pjs_to_ptyphim_array<double>;
  %template(pjs_to_ptyphim_array_float32) pjs_to_ptyphim_array<float>;
  %template(pjs_to_array_float64) pjs_to_array<double>;
  %template(pjs_to_array_float32) pjs_to_array<float>;

  // functions accepting a pseudojet container and returning a numpy array
  %template(pjc_to_epxpypz_array_float64) pjc_to_epxpypz_array<double>;
  %template(pjc_to_epxpypz_array_float32) pjc_to_epxpypz_array<float>;
  %template(pjc_to_ptyphim_array_float64) pjc_to_ptyphim_array<double>;
  %template(pjc_to_ptyphim_array_float32) pjc_to_ptyphim_array<float>;  
  %template(pjc_to_array_float64) pjc_to_array<double>; 
  %template(pjc_to_array_float32) pjc_to_array<float>;

  // member functions of pseudojet container that return a numpy array
  %template(epxpypz_array_float64) PseudoJetContainer::epxpypz_array<double>;
  %template(epxpypz_array_float32) PseudoJetContainer::epxpypz_array<float>;
  %template(ptyphim_array_float64) PseudoJetContainer::ptyphim_array<double>;
  %template(ptyphim_array_float32) PseudoJetContainer::ptyphim_array<float>;
  %template(array_float64) PseudoJetContainer::array<double>;
  %template(array_float32) PseudoJetContainer::array<float>;

  %extend PseudoJetContainer {

    void _setitem(std::ptrdiff_t key, const PseudoJet & val) {
      if (key < 0) key += $self->size();
      if (std::size_t(key) >= $self->size())
        throw std::length_error("index out of bounds");

      (*$self)[key] = val;
    }

    void _delitem(std::ptrdiff_t key) {
      if (key < 0) key += $self->size();
      if (std::size_t(key) >= $self->size())
        throw std::length_error("index out of bounds");

      $self->as_vector().erase($self->as_vector().begin() + key);
    }

    void _delitems(std::ptrdiff_t start, std::ptrdiff_t stop) {
      if (start < 0) start += $self->size();
      if (stop < 0) stop += $self->size();
      if (std::size_t(start) >= $self->size())
        throw std::length_error("start index out of bounds");
      if (std::size_t(stop) > $self->size())
        throw std::length_error("stop index out of bounds");

      if (start < stop)
        $self->as_vector().erase($self->as_vector().begin() + start, $self->as_vector().begin() + stop);
    }

    %pythoncode %{

      def epxpypz_array(self, float32=False):
          return self.epxpypz_array_float32() if float32 else self.epxpypz_array_float64()

      def ptyphim_array(self, mass=True, phi_std=False, phi_ref=None, float32=False):
          return (self.ptyphim_array_float32(mass, phi_std, phi_ref) if float32 else
                  self.ptyphim_array_float64(mass, phi_std, phi_ref))

      def array(self, pjrep=PseudoJetRepresentation_ptyphim, float32=False):
          return self.array_float32(pjrep) if float32 else self.array_float64(pjrep)

      def append(self, val):
          self.push_back(val)

      @property
      def vector(self):
          if not hasattr(self, '_vector'):
              self._vector = self.as_vector()
          return self._vector

      def __len__(self):
          return self.size()

      def __iter__(self):
          return self.vector.__iter__();

      # read only
      def __getitem__(self, key):
          return self.vector[key]

      # invalidates _vector
      def __setitem__(self, key, val):
          if hasattr(self, '_vector'):
              del self._vector

          if isinstance(key, slice):
              for k,v in zip(range(*key.indices(len(self))), val):
                  self._setitem(k, v)
          else:
              self._setitem(key, val)

      # invalidates _vector
      def __delitem__(self, key):
          if hasattr(self, '_vector'):
              del self._vector

          if isinstance(key, slice):
              if key.step is None or key.step == 1:
                  self._delitems(key.start, key.stop)
              else:
                  for k in sorted(range(*key.indices(len(self))))[::-1]:
                      self._delitem(k)
          else:
              self._delitem(key)

      def __repr__(self):
          s = ['PseudoJetContainer[' + str(len(self)) + '](']
          for i,pj in enumerate(self):
              if i >= 25:
                  s.append('  ...')
                  break
              s.append('  ' + repr(pj) + ',')
          s.append(')')
          return '\n'.join(s)
    %}
  }

  %extend PseudoJet {

    std::string __repr__() {
      const unsigned len_max = 512;
      char temp[len_max];
      if (PseudoJetRep_ == PseudoJetRepresentation::ptyphim)
        snprintf(temp, len_max, "PseudoJet(pt=%.6g, y=%.6g, phi=%.6g, m=%.6g, index=%i)",
                 $self->pt(), $self->rap(), $self->phi_std(),
                 [](double m){return std::fabs(m) < 1e-5 ? 0 : m;}($self->m()), $self->user_index());
      else if (PseudoJetRep_ == PseudoJetRepresentation::epxpypz)
        snprintf(temp, len_max, "PseudoJet(e=%.6g, px=%.6g, py=%.6g, pz=%.6g, index=%i)",
                 $self->e(), $self->px(), $self->py(), $self->pz(), $self->user_index());
      else
        snprintf(temp, len_max, "PseudoJet(pt=%.6g, y=%.6g, phi=%.6g, index=%i)",
                 $self->pt(), $self->rap(), $self->phi_std(), $self->user_index());
      return std::string(temp);
    }

    void set_python_info(PyObject * pyobj) {
      UserInfoPython * new_python_info = new UserInfoPython(pyobj);
      $self->set_user_info(new_python_info);
    }

    PyObject * python_info() const {
      if ($self->has_user_info())
        return $self->user_info<UserInfoPython>().get_pyobj();
      Py_RETURN_NONE;
    }
    
    // these C++ operators are not automatically handled by SWIG (would only
    // be handled if they were part of the class)
    PseudoJet __add__ (const PseudoJet & p) { return (*$self) + p; }
    PseudoJet __sub__ (const PseudoJet & p) { return (*$self) - p; }
    bool      __eq__  (const PseudoJet & p) { return (*$self) == p; }
    bool      __ne__  (const PseudoJet & p) { return (*$self) != p; }
    PseudoJet __mul__ (double x) { return (*$self) * x; }
    PseudoJet __rmul__(double x) { return (*$self) * x; }
    PseudoJet __truediv__ (double x) { return (*$self) / x; }
    bool      __eq__  (double x) { return (*$self) == x; }
    bool      __ne__  (double x) { return (*$self) != x; }

    %pythoncode {
      def __getitem__(self, key):
          return self(key)
    }
  }

  // extend JetDefinition
  %extend JetDefinition {
    ADD_REPR_FROM_DESCRIPTION
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

%pythoncode %{

  def pseudojets_to_epxpypz_array(arg, float32=False):
      is_pjc = isinstance(arg, PseudoJetContainer)
      if float32:
          return pjc_to_epxpypz_array_float32(arg) if is_pjc else pjs_to_epxpypz_array_float32(arg)
      return pjc_to_epxpypz_array_float64(arg) if is_pjc else pjs_to_epxpypz_array_float64(arg)

  def pseudojets_to_ptyphim_array(arg, mass=True, phi_std=False, phi_ref=None, float32=False):
      phi_ref = phi_ref or pseudojet_invalid_phi
      is_pjc = isinstance(arg, PseudoJetContainer)
      if float32:
          return (pjc_to_ptyphim_array_float32(arg, mass, phi_std, phi_ref) if is_pjc else
                  pjs_to_ptyphim_array_float32(arg, mass, phi_std, phi_ref))
      return (pjc_to_ptyphim_array_float64(arg, mass, phi_std, phi_ref) if is_pjc else
              pjs_to_ptyphim_array_float64(arg, mass, phi_std, phi_ref))

  def pseudojets_to_array(arg, pjrep=PseudoJetRepresentation_ptyphim, float32=False):
      is_pjc = isinstance(arg, PseudoJetContainer)
      if float32:
          return pjc_to_array_float32(arg, pjrep) if is_pjc else pjs_to_array_float32(arg, pjrep)
      return pjc_to_array_float64(arg, pjrep) if is_pjc else pjs_to_array_float64(arg, pjrep)

  import copyreg

  def _pickle_jet_definition(obj):
      jet_alg = obj.jet_algorithm()
      R = obj.R()
      extra = obj.extra_param()
      recomb = obj.recombination_scheme()
      nparams = obj.n_parameters_for_algorithm(jet_alg)

      return _unpickle_jet_definition, (jet_alg, R, extra, recomb, nparams)

  def _unpickle_jet_definition(jet_alg, R, extra, recomb, nparams):
      if nparams == 1:
          return JetDefinition(jet_alg, R, recomb)
      else:
          return JetDefinition(jet_alg, R, extra, recomb)

  copyreg.pickle(JetDefinition, _pickle_jet_definition)

  def _pickle_pseudojet(obj):
      return _unpickle_pseudojet, (obj.px(), obj.py(), obj.pz(), obj.E())

  def _unpickle_pseudojet(*args):
      return PseudoJet(*args)

  copyreg.pickle(PseudoJet, _pickle_pseudojet)

%}