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

// to select between different representations of PseudoJets
enum PseudoJetRepresentation { epxpypz = 0, ptyphim = 1, ptyphi = 2 };

static PseudoJetRepresentation PseudoJetRep_;
void set_pseudojet_format(PseudoJetRepresentation rep) {
  if (rep > 2 || rep < 0)
    throw Error("invalid PseudoJetRepresentation");

  PseudoJetRep_ = rep;
}

struct ArrayToPseudoJets {

  template<typename A> static
  std::vector<PseudoJet> convert(double* particles, int mult, int nfeatures) {
    std::vector<PseudoJet> pjs;
    pjs.reserve(mult);

    std::size_t k(0);
    for (int i = 0; i < mult; i++, k+=nfeatures) {
      A::construct(pjs, particles, k);
      pjs.back().set_user_index(i);
    }

    return pjs;
  }

  template<typename A> static
  std::vector<PseudoJet> convert_with_info(double* particles, int mult, int nfeatures) {
    std::vector<PseudoJet> pjs;
    pjs.reserve(mult);

    npy_intp dims[1] = {nfeatures - 4};
    std::size_t nfbytes(dims[0] * sizeof(double)), k(0);
    for (int i = 0; i < mult; i++, k+=nfeatures) {
      A::construct(pjs, particles, k);
      pjs.back().set_user_index(i);

      PyObject* user_features(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
      if (!user_features)
        throw Error("cannot allocate array for user features");

      memcpy(array_data(user_features), particles + k + 4, nfbytes);
      pjs.back().set_user_info(new UserInfoPython(user_features));
      Py_DECREF(user_features);
    }

    return pjs;
  }
};

struct ConstructPtYPhiM {
  static void construct(std::vector<PseudoJet> & pjs, double* particles, std::size_t k) {
    pjs.push_back(PtYPhiM(particles[k], particles[k+1], particles[k+2], particles[k+3]));
  }
};

struct ConstructPtYPhi {
  static void construct(std::vector<PseudoJet> & pjs, double* particles, std::size_t k) {
    pjs.push_back(PtYPhiM(particles[k], particles[k+1], particles[k+2]));
  }
};

struct ConstructEPxPyPz {
  static void construct(std::vector<PseudoJet> & pjs, double* particles, std::size_t k) {
    pjs.emplace_back(particles[k+1], particles[k+2], particles[k+3], particles[k]);
  }
};

// convert numpy array to PseudoJets
std::vector<PseudoJet> ptyphim_array_to_pseudojets(double* particles, int mult, int nfeatures) {

  // array is pt, y, phi, m
  if (nfeatures == 4)
    return ArrayToPseudoJets::convert<ConstructPtYPhiM>(particles, mult, 4);

  // array is pt, y, phi
  else if (nfeatures == 3)
    return ArrayToPseudoJets::convert<ConstructPtYPhi>(particles, mult, 3);

  // array is pt, y, phi, m, [more features]
  else if (nfeatures > 4)
    return ArrayToPseudoJets::convert_with_info<ConstructPtYPhiM>(particles, mult, nfeatures);
  
  throw Error("array must have at least 3 columns");
}

// convert numpy array to PseudoJets
std::vector<PseudoJet> epxpypz_array_to_pseudojets(double* particles, int mult, int nfeatures) {

  // array is e, px, py, pz
  if (nfeatures == 4)
    return ArrayToPseudoJets::convert<ConstructEPxPyPz>(particles, mult, 4);

  // array is pt, y, phi, m, [more features]
  else if (nfeatures > 4)
    return ArrayToPseudoJets::convert_with_info<ConstructEPxPyPz>(particles, mult, nfeatures);

  throw Error("array must have at least 4 columns");
}

// function that selects representation based on enum
std::vector<PseudoJet> array_to_pseudojets(double* particles, int mult, int nfeatures,
                                       PseudoJetRepresentation pjrep = ptyphim) {

  if (pjrep == ptyphim || pjrep == ptyphi)
    return ptyphim_array_to_pseudojets(particles, mult, nfeatures);

  else if (pjrep == epxpypz)
    return epxpypz_array_to_pseudojets(particles, mult, nfeatures);

  throw Error("unknown pseudojet representation");
}

// convert pseudojets to numpy array of e, px, py, pz values
void pseudojets_to_epxpypz_array(double** particles, int* mult, int* nfeatures,
                                 const std::vector<PseudoJet> & pjs) {
  *mult = pjs.size();
  *nfeatures = 4;
  std::size_t nbytes = 4 * pjs.size() * sizeof(double);
  *particles = (double *) malloc(nbytes);
  if (*particles == NULL)
    throw Error("failed to allocate " + std::to_string(nbytes) + " bytes");

  std::size_t k(0);
  for (const auto & pj : pjs) {
    (*particles)[k++] = pj.e();
    (*particles)[k++] = pj.px();
    (*particles)[k++] = pj.py();
    (*particles)[k++] = pj.pz();
  }
}

// convert pseudojets to numpy array of e, px, py, pz values
void pseudojets_to_ptyphim_array(double** particles, int* mult, int* nfeatures,
                                 const std::vector<PseudoJet> & pjs, bool mass = true) {
  *mult = pjs.size();
  *nfeatures = (mass ? 4 : 3);
  std::size_t nbytes = (*nfeatures) * pjs.size() * sizeof(double);
  *particles = (double *) malloc(nbytes);
  if (*particles == NULL)
    throw Error("failed to allocate " + std::to_string(nbytes) + " bytes");

  std::size_t k(0);
  if (mass)
    for (const auto & pj : pjs) {
      (*particles)[k++] = pj.pt();
      (*particles)[k++] = pj.rap();
      (*particles)[k++] = pj.phi();
      (*particles)[k++] = pj.m();
    }
  else
    for (const auto & pj : pjs) {
      (*particles)[k++] = pj.pt();
      (*particles)[k++] = pj.rap();
      (*particles)[k++] = pj.phi();
    }
}

// function that selects representation based on enum
void pseudojets_to_array(double** particles, int* mult, int* nfeatures,
                         const std::vector<PseudoJet> & pjs,
                         PseudoJetRepresentation pjrep = ptyphim) {

  if (pjrep == ptyphim)
    pseudojets_to_ptyphim_array(particles, mult, nfeatures, pjs, true);

  else if (pjrep == ptyphi)
    pseudojets_to_ptyphim_array(particles, mult, nfeatures, pjs, false);

  else if (pjrep == epxpypz)
    pseudojets_to_epxpypz_array(particles, mult, nfeatures, pjs);

  else throw Error("unknown pseudojet representation");
}

// function that extracts user indices to a numpy array
void user_indices(int** inds, int* mult, const std::vector<PseudoJet> & pjs) {
  *mult = pjs.size();
  std::size_t nbytes = pjs.size() * sizeof(int);
  *inds = (int *) malloc(nbytes);
  if (*inds == NULL)
    throw Error("failed to allocate " + std::to_string(nbytes) + " bytes");

  std::size_t k(0);
  for (const auto & pj : pjs)
    (*inds)[k++] = pj.user_index();
}

FJCORE_END_NAMESPACE

#endif // PYFJCOREEXTENSIONS