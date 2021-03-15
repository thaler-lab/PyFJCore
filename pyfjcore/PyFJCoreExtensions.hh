#ifndef PYFJCOREEXTENSIONS
#define PYFJCOREEXTENSIONS

#include "fjcore.hh"

FJCORE_BEGIN_NAMESPACE

class UserInfoPython : public PseudoJet::UserInfoBase {
public:
  UserInfoPython(PyObject * pyobj) : _pyobj(pyobj) {
    Py_XINCREF(_pyobj);
  }

  PyObject * get_pyobj() const {
    // since there's going to be an extra reference to this object
    // one must increase the reference count; it seems that this
    // is _our_ responsibility
    Py_XINCREF(_pyobj);
    return _pyobj;
  }
  //const PyObject * get_pyobj() const {return _pyobj;}
  
  ~UserInfoPython() {
    Py_XDECREF(_pyobj);
  }
private:
  PyObject * _pyobj;
};

class PJVector : public std::vector<PseudoJet> {

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
                                  Strategy strategy = Best){
  return JetDefinition(jet_algorithm, recomb_scheme, strategy);
}

// JetDefinition1Param(algorithm, R, recomb_scheme, strategy)
JetDefinition JetDefinition1Param(JetAlgorithm jet_algorithm, 
                                  double R_in, 
                                  RecombinationScheme recomb_scheme = E_scheme,
                                  Strategy strategy = Best){
  return JetDefinition(jet_algorithm, R_in, recomb_scheme, strategy);
}

// JetDefinition2Param(algorithm, R, extrarecomb_scheme, strategy)
JetDefinition JetDefinition2Param(JetAlgorithm jet_algorithm, 
                                  double R_in, 
                                  double xtra_param,
                                  RecombinationScheme recomb_scheme = E_scheme,
                                  Strategy strategy = Best){
  return JetDefinition(jet_algorithm, R_in, xtra_param, recomb_scheme, strategy);
}

// convert numpy array to PseudoJets
PJVector ptyphim_array_to_pseudojets(double* particles, int mult, int nfeatures) {
  PJVector pjs;
  pjs.reserve(mult);

  // array is pt, y, phi
  std::size_t k(0);
  if (nfeatures == 3)
    for (int i = 0; i < mult; i++, k += 3) {
      pjs.push_back(PtYPhiM(particles[k], particles[k+1], particles[k+2]));
      pjs.back().set_user_index(i);
    }

  // array is pt, y, phi, m
  else if (nfeatures == 4)
    for (int i = 0; i < mult; i++, k += 4) {
      pjs.push_back(PtYPhiM(particles[k], particles[k+1], particles[k+2], particles[k+3]));
      pjs.back().set_user_index(i);
    }

  // array is pt, y, phi, m, [more features]
  else if (nfeatures > 4) {
    npy_intp dims[1] = {nfeatures - 4};
    std::size_t nfbytes(dims[0] * sizeof(double));
    for (int i = 0; i < mult; i++, k += nfeatures) {
      pjs.push_back(PtYPhiM(particles[k], particles[k+1], particles[k+2], particles[k+3]));
      pjs.back().set_user_index(i);

      PyObject* user_features(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
      if (!user_features)
        throw Error("cannot allocate array for user features");

      memcpy(array_data(user_features), particles + k + 4, nfbytes);
      pjs.back().set_user_info(new fastjet::UserInfoPython(user_features));
      Py_DECREF(user_features);
    }
  }
  else throw Error("array must have at least 3 columns");

  return pjs;
}

// convert numpy array to PseudoJets
PJVector epxpypz_array_to_pseudojets(double* particles, int mult, int nfeatures) {
  PJVector pjs;
  pjs.reserve(mult);

  // array is pt, y, phi, m
  std::size_t k(0);
  if (nfeatures == 4)
    for (int i = 0; i < mult; i++, k += 3) {
      pjs.emplace_back(particles[k+1], particles[k+2], particles[k+3], particles[k]);
      pjs.back().set_user_index(i);
    }

  // array is pt, y, phi, m, [more features]
  else if (nfeatures > 4) {
    npy_intp dims[1] = {nfeatures - 4};
    std::size_t nfbytes(dims[0] * sizeof(double));
    for (int i = 0; i < mult; i++, k += nfeatures) {
      pjs.emplace_back(particles[k+1], particles[k+2], particles[k+3], particles[k]);
      pjs.back().set_user_index(i);

      PyObject* user_features(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
      if (!user_features)
        throw Error("cannot allocate array for user features");

      memcpy(array_data(user_features), particles + k + 4, nfbytes);
      pjs.back().set_user_info(new fastjet::UserInfoPython(user_features));
      Py_DECREF(user_features);
    }
  }
  else throw Error("array must have at least 4 columns");

  return pjs;
}

enum class PJRep { epxpypz, ptyphim, ptyphi };

static PJRep PseudoJetRep_;
void set_pseudojet_format(PJRep rep) {
  PseudoJetRep_ = rep;
}

// convert pseudojets to numpy array
void pseudojets_to_array(double** particles, int* mult, int* nfeatures,
                         const std::vector<PseudoJet> & pjs, PJRep pjrep = PJRep::ptyphim) {
  *mult = pjs.size();
  *nfeatures = (pjrep == PJRep::ptyphi ? 3 : 4);
  std::size_t nbytes = (*nfeatures)*(*mult)*sizeof(double);
  *particles = (double *) malloc(nbytes);
  if (*particles == NULL)
    throw std::runtime_error("failed to allocate " + std::to_string(nbytes) + " bytes");

  std::size_t k(0);
  if (pjrep == PJRep::ptyphim)
    for (const auto & pj : pjs) {
      (*particles)[k++] = pj.pt();
      (*particles)[k++] = pj.rap();
      (*particles)[k++] = pj.phi();
      (*particles)[k++] = pj.m();
    }
  else if (pjrep == PJRep::epxpypz)
    for (const auto & pj : pjs) {
      (*particles)[k++] = pj.e();
      (*particles)[k++] = pj.px();
      (*particles)[k++] = pj.py();
      (*particles)[k++] = pj.pz();
    }
  else if (pjrep == PJRep::ptyphi)
    for (const auto & pj : pjs) {
      (*particles)[k++] = pj.pt();
      (*particles)[k++] = pj.rap();
      (*particles)[k++] = pj.phi();
    }
}

FJCORE_END_NAMESPACE

#endif // PYFJCOREEXTENSIONS