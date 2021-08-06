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

#ifndef PYFJCOREEXTENSIONS
#define PYFJCOREEXTENSIONS

#include "fjcore.hh"

FJCORE_BEGIN_NAMESPACE

class UserInfoPython : public PseudoJet::UserInfoBase {
public:
  UserInfoPython(PyObject * pyobj) : _pyobj(pyobj) {
    Py_INCREF(_pyobj);
  }

  ~UserInfoPython() {
    Py_DECREF(_pyobj);
  }

  PyObject * get_pyobj() const {
    // since there's going to be an extra reference to this object
    // one must increase the reference count; it seems that this
    // is _our_ responsibility
    Py_INCREF(_pyobj);
    return _pyobj;
  }
  
private:
  PyObject * _pyobj;
};

//----------------------------------------------------------------------
// Since Python handles enum types as int, there can be some confusion
// between different JetDefinition ctors, where a int param (intended
// as a double, like using R=1 or p=-1 for the genkt algorithm) is
// actually interpreted an te enum (for the recombination scheme).
//
// We therefore provide a few helpers to force the construction of a
// Jet Definition with a fied number of parameters (+recombiner+strategy)
//
// JetDefinition0Param(algorithm, recomb_scheme, strategy)
JetDefinition JetDefinition0Param(JetAlgorithm jet_algorithm, 
                                  RecombinationScheme recomb_scheme = E_scheme,
                                  Strategy strategy = Best) {
  return JetDefinition(jet_algorithm, recomb_scheme, strategy);
}

// JetDefinition1Param(algorithm, R, recomb_scheme, strategy)
JetDefinition JetDefinition1Param(JetAlgorithm jet_algorithm, 
                                  double R_in, 
                                  RecombinationScheme recomb_scheme = E_scheme,
                                  Strategy strategy = Best) {
  return JetDefinition(jet_algorithm, R_in, recomb_scheme, strategy);
}

// JetDefinition2Param(algorithm, R, extrarecomb_scheme, strategy)
JetDefinition JetDefinition2Param(JetAlgorithm jet_algorithm, 
                                  double R_in, 
                                  double xtra_param,
                                  RecombinationScheme recomb_scheme = E_scheme,
                                  Strategy strategy = Best) {
  return JetDefinition(jet_algorithm, R_in, xtra_param, recomb_scheme, strategy);
}

static PseudoJetRepresentation PseudoJetRep_ = PseudoJetRepresentation::ptyphim;
void set_pseudojet_format(PseudoJetRepresentation rep) {
  if (int(rep) > 2 || int(rep) < 0)
    throw Error("invalid PseudoJetRepresentation");

  PseudoJetRep_ = rep;
}

struct ArrayToPseudoJets {

  template<typename A> static
  std::vector<PseudoJet> convert(typename A::Float * particles, std::ptrdiff_t mult, std::ptrdiff_t nfeatures) {
    std::vector<PseudoJet> pjs;
    pjs.reserve(mult);

    std::size_t k(0);
    for (std::ptrdiff_t i = 0; i < mult; i++, k+=nfeatures) {
      A::construct(pjs, particles, k);
      pjs.back().set_user_index(i);
    }

    return pjs;
  }

  template<typename A> static
  std::vector<PseudoJet> convert_with_info(typename A::Float * particles, std::ptrdiff_t mult, std::ptrdiff_t nfeatures) {
    std::vector<PseudoJet> pjs;
    pjs.reserve(mult);

    npy_intp dims[1] = {nfeatures - 4};
    std::size_t nfbytes(dims[0] * sizeof(typename A::Float)), k(0);
    for (std::ptrdiff_t i = 0; i < mult; i++, k+=nfeatures) {
      A::construct(pjs, particles, k);
      pjs.back().set_user_index(i);

      PyObject* user_features(PyArray_SimpleNew(1, dims, sizeof(typename A::Float) == 4 ? NPY_FLOAT : NPY_DOUBLE));
      if (!user_features)
        throw Error("cannot allocate array for user features");

      memcpy(array_data(user_features), particles + k + 4, nfbytes);
      pjs.back().set_user_info(new UserInfoPython(user_features));
      Py_DECREF(user_features);
    }

    return pjs;
  }
};

template<typename F>
struct ConstructPtYPhiM {
  typedef F Float;
  static void construct(std::vector<PseudoJet> & pjs, Float* particles, std::size_t k) {
    pjs.push_back(PtYPhiM(particles[k], particles[k+1], particles[k+2], particles[k+3]));
  }
};

template<typename F>
struct ConstructPtYPhi {
  typedef F Float;
  static void construct(std::vector<PseudoJet> & pjs, Float* particles, std::size_t k) {
    pjs.push_back(PtYPhiM(particles[k], particles[k+1], particles[k+2]));
  }
};

template<typename F>
struct ConstructEPxPyPz {
  typedef F Float;
  static void construct(std::vector<PseudoJet> & pjs, Float* particles, std::size_t k) {
    pjs.emplace_back(particles[k+1], particles[k+2], particles[k+3], particles[k]);
  }
};

// convert numpy array to PseudoJets
PseudoJetContainer ptyphim_array_to_pseudojets(double* particles, std::ptrdiff_t mult, std::ptrdiff_t nfeatures) {

  // array is pt, y, phi, m
  if (nfeatures == 4)
    return ArrayToPseudoJets::convert<ConstructPtYPhiM<double>>(particles, mult, 4);

  // array is pt, y, phi
  else if (nfeatures == 3)
    return ArrayToPseudoJets::convert<ConstructPtYPhi<double>>(particles, mult, 3);

  // array is pt, y, phi, m, [more features]
  else if (nfeatures > 4)
    return ArrayToPseudoJets::convert_with_info<ConstructPtYPhiM<double>>(particles, mult, nfeatures);
  
  throw Error("array must have at least 3 columns");
}

// convert numpy array to PseudoJets
PseudoJetContainer epxpypz_array_to_pseudojets(double* particles, std::ptrdiff_t mult, std::ptrdiff_t nfeatures) {

  // array is e, px, py, pz
  if (nfeatures == 4)
    return ArrayToPseudoJets::convert<ConstructEPxPyPz<double>>(particles, mult, 4);

  // array is pt, y, phi, m, [more features]
  else if (nfeatures > 4)
    return ArrayToPseudoJets::convert_with_info<ConstructEPxPyPz<double>>(particles, mult, nfeatures);

  throw Error("array must have at least 4 columns");
}

// function that selects representation based on enum
PseudoJetContainer array_to_pseudojets(double* particles, std::ptrdiff_t mult, std::ptrdiff_t nfeatures,
                                           PseudoJetRepresentation pjrep = PseudoJetRepresentation::ptyphim) {

  if (pjrep == PseudoJetRepresentation::ptyphim || pjrep == PseudoJetRepresentation::ptyphi)
    return ptyphim_array_to_pseudojets(particles, mult, nfeatures);

  else if (pjrep == PseudoJetRepresentation::epxpypz)
    return epxpypz_array_to_pseudojets(particles, mult, nfeatures);

  throw Error("unknown pseudojet representation");
}

void user_indices(int** inds, std::ptrdiff_t* mult, const PseudoJetContainer & pjc) {
  user_indices(inds, mult, pjc.as_vector());
}

FJCORE_END_NAMESPACE

#endif // PYFJCOREEXTENSIONS
