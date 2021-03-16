#ifndef PYFJCOREEXTENSIONS
#define PYFJCOREEXTENSIONS

#include "fjcore.hh"

FJCORE_BEGIN_NAMESPACE

class UserInfoPython : public PseudoJet::UserInfoBase {
public:
  UserInfoPython(PyObject * pyobj) : _pyobj(pyobj) {
    Py_XINCREF(_pyobj);
  }

  ~UserInfoPython() {
    Py_XDECREF(_pyobj);
  }

  PyObject * get_pyobj() const {
    // since there's going to be an extra reference to this object
    // one must increase the reference count; it seems that this
    // is _our_ responsibility
    Py_XINCREF(_pyobj);
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
enum class PseudoJetRepresentation { epxpypz, ptyphim, ptyphi };

static PseudoJetRepresentation PseudoJetRep_;
void set_pseudojet_format(PseudoJetRepresentation rep) {
  PseudoJetRep_ = rep;
}

#define BEGIN_CONVERT_TO_PJ(nf, kinematics) \
for (int i = 0; i < mult; i++, k += nf) {   \
  kinematics;                               \
  pjs.back().set_user_index(i);
#define END_CONVERT_TO_PJ }

#define CONVERT_TO_PJS_WITH_INFO(kinematics)                            \
npy_intp dims[1] = {nfeatures - 4};                                     \
std::size_t nfbytes(dims[0] * sizeof(double));                          \
BEGIN_CONVERT_TO_PJ(nfeatures, kinematics)                              \
  PyObject* user_features(PyArray_SimpleNew(1, dims, NPY_DOUBLE));      \
  if (!user_features)                                                   \
    throw Error("cannot allocate array for user features");             \
  memcpy(array_data(user_features), particles + k + 4, nfbytes);        \
  pjs.back().set_user_info(new fastjet::UserInfoPython(user_features)); \
  Py_DECREF(user_features);                                             \
END_CONVERT_TO_PJ

// convert numpy array to PseudoJets
PseudoJetContainer ptyphim_array_to_pseudojets(double* particles, int mult, int nfeatures) {
  std::vector<PseudoJet> pjs;
  pjs.reserve(mult);

  // array is pt, y, phi, m
  std::size_t k(0);
  if (nfeatures == 4) {
    BEGIN_CONVERT_TO_PJ(4, pjs.push_back(PtYPhiM(particles[k], particles[k+1], particles[k+2], particles[k+3])))
    END_CONVERT_TO_PJ
  }

  // array is pt, y, phi
  else if (nfeatures == 3) {
    BEGIN_CONVERT_TO_PJ(3, pjs.push_back(PtYPhiM(particles[k], particles[k+1], particles[k+2])))
    END_CONVERT_TO_PJ
  }

  // array is pt, y, phi, m, [more features]
  else if (nfeatures > 4) {
    CONVERT_TO_PJS_WITH_INFO(pjs.push_back(PtYPhiM(particles[k], particles[k+1], particles[k+2], particles[k+3])))
  }
  else throw Error("array must have at least 3 columns");

  return pjs;
}

// convert numpy array to PseudoJets
PseudoJetContainer epxpypz_array_to_pseudojets(double* particles, int mult, int nfeatures) {
  std::vector<PseudoJet> pjs;
  pjs.reserve(mult);

  // array is e, px, py, pz
  std::size_t k(0);
  if (nfeatures == 4) {
    BEGIN_CONVERT_TO_PJ(4, pjs.emplace_back(particles[k+1], particles[k+2], particles[k+3], particles[k]))
    END_CONVERT_TO_PJ
  }

  // array is pt, y, phi, m, [more features]
  else if (nfeatures > 4) {
    CONVERT_TO_PJS_WITH_INFO(pjs.emplace_back(particles[k+1], particles[k+2], particles[k+3], particles[k]))
  }

  else throw Error("array must have at least 4 columns");

  return pjs;
}

// function that selects representation based on enum
PseudoJetContainer array_to_pseudojets(double* particles, int mult, int nfeatures,
                                       PseudoJetRepresentation pjrep = PseudoJetRepresentation::ptyphim) {

  if (pjrep == PseudoJetRepresentation::ptyphim || pjrep == PseudoJetRepresentation::ptyphi)
    return ptyphim_array_to_pseudojets(particles, mult, nfeatures);

  else if (pjrep == PseudoJetRepresentation::epxpypz)
    return epxpypz_array_to_pseudojets(particles, mult, nfeatures);

  else throw Error("unknown pseudojet representation");

  return std::vector<PseudoJet>();
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
                         PseudoJetRepresentation pjrep = PseudoJetRepresentation::ptyphim) {

  if (pjrep == PseudoJetRepresentation::ptyphim)
    pseudojets_to_ptyphim_array(particles, mult, nfeatures, pjs, true);

  else if (pjrep == PseudoJetRepresentation::ptyphi)
    pseudojets_to_ptyphim_array(particles, mult, nfeatures, pjs, false);

  else if (pjrep == PseudoJetRepresentation::epxpypz)
    pseudojets_to_epxpypz_array(particles, mult, nfeatures, pjs);

  else throw Error("unknown pseudojet representation");
}

// function that extracts user indices to a numpy array
void user_indices(int** inds, int* mult, const std::vector<PseudoJet> & pjs) {
  EXTRACT_USER_INDICES
}

FJCORE_END_NAMESPACE

#endif // PYFJCOREEXTENSIONS