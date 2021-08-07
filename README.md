# PyFJCore

[![build-wheels](https://github.com/pkomiske/Wasserstein/actions/workflows/build-wheels.yml/badge.svg)](https://github.com/pkomiske/Wasserstein/actions)
[![PyPI version](https://badge.fury.io/py/PyFJCore.svg)](https://pypi.org/project/PyFJCore/)
[![python versions](https://img.shields.io/pypi/pyversions/PyFJCore)](https://pypi.org/project/PyFJCore/)

PyFJCore is a Python wrapper of FastJet Core functionality with additional NumPy support. In contrast with the [`pyjet`](https://github.com/scikit-hep/pyjet) package, PyFJCore wraps all the methods/functions in fjcore and works with regular NumPy arrays instead of structured one. In contrast with the Python extension to the main [FastJet library](http://fastjet.fr), this package can be built in a portable manner, including on Windows, and is available on PyPI.

Current version of fjcore: 3.4.0

## Documentation

The FastJet [documentation](http://fastjet.fr/repo/doxygen-3.4.0/) and [manual](http://fastjet.fr/repo/fastjet-doc-3.4.0.pdf) contain helpful information for the classes and methods in PyFJCore. Not all FastJet classes are wrapped in PyFJCore, primarily just `PseudoJet`, `JetDefinition`, `ClusterSequence`, and `Selector`.

### `PseudoJet`

Particles are represented in FastJet by `PseudoJet` objects. PseudoJets can be constructed from Cartesian momenta using the constructor, `pyfjcore.PseudoJet(px, py, pz, e)`. They can be constructed from hadronic momenta as `pyfjcore.PtYPhiM(pt, y, phi, [mass])`, where the mass is optional and is taken to be zero if not present.

PseudoJets have a user index which is set to -1 by default. It can be set using the `set_user_index(index)` method and accessed with the `user_index()` method. An arbitrary Python object can also be associated with a PseudoJet using the `set_python_info(pyobj)` method, and accessed as `python_info()`.

### `PseudoJetContainer`

A [`PseudoJetContainer`](https://github.com/pkomiske/PyFJCore/blob/main/pyfjcore/fjcore.hh#L1098-L1151) is useful for efficiently working with a list/vector of PseudoJets. In C++, it is a struct holding a `std::vector<PseudoJet>`, which allows the user to control SWIG's automatic coercion to/from native Python containers. Such coercion can be useful, but can also be inefficient since it requires a lot of copying. This coercion is still possible if desired by explicit tuple/list construction, e.g. `tuple(pjcontainer)` or `list(pjcontainer)`. A `PseudoJetContainer` can be indexed, assigned to, or modified (by deleting elements) as if it were a list of PseudoJets. The wrapper code has been modified so that methods that accept `const std::vector<PseudoJet> &` will accept a `PseudoJetContainer` without any copying. The `vector` property can be used to access the underlying `vectorPseudoJet` (SWIG's wrapper of `std::vector<fastjet::PseudoJet>`) directly.

`PseudoJetContainer`s con be constructed directly from an iterable of PseudoJets, or more commonly from NumPy arrays of particle kinematics (see the functions `ptyphim_array_to_pseudojets`, `epxpypz_array_to_pseudojets`, `array_to_pseudojets` below). Given a `PseudoJetContainer`, a NumPy array of the particle kinematics can be obtained using the methods `ptyphim_array`, `epxpypz_array`, and `array`, which correspond to the functions `pseudojets_to_ptyphim_array`, `pseudojets_to_epxpypz_array`, and `pseudojets_to_array` below. The user indices of the PseudoJets can be obtained as an integer NumPy array using the `user_indices()` method.

### NumPy conversion functions

```python
pyfjcore.ptyphim_array_to_pseudojets(ptyphims)
```

Converts a 2D array of particles, each as `(pt, y, phi, [mass])`, to PseudoJets (the mass is optional). Any additional features (columns after the initial four) of the array are set as the Python user info of the PseudoJets. This also sets the `user_index` of the PseudoJets to their position in the input array.

```python
pyfjcore.epxpypz_array_to_pseudojets(epxpypzs)
```

Converts a 2D array of particles, each as `(E, px, py, pz)`, to PseudoJets. Any additional features (columns after the initial four) of the array are set as the Python user info of the PseudoJets. This also sets the `user_index` of the PseudoJets to their position in the input array.

```python
pyfjcore.array_to_pseudojets(particles, pjrep=pyfjcore.ptyphim)
```

Converts a 2D array of particles to PseudoJets. The format of the particles kinematics is determined by the `pjrep` argument. The `PseudoJetRepresentation` enum can take the values `ptyphim`, `ptyphi`, `epxpypz`. The first two values cause this function to invoke `ptyphim_array_to_pseudojets` and the third invokes `epxpypz_array_to_pseudojets`. Any additional features (columns) of the array are set as the Python user info of the PseudoJets. This also sets the `user_index` of the PseudoJets to their position in the input array.

```python
pyfjcore.pseudojets_to_ptyphim_array(pseudojets, mass=True, phi_std=False, phi_ref=None, float32=False)
```

Converts a collection of PseudoJets (`PseudoJetContainer` or a Python iterable) to a 2D NumPy array of `(pt, y, phi, [mass])` values, where the presence of the mass is determine by the keyword argument. `phi_std` determines if the phi values will be in the range $[-\pi,\pi)$ or $[0,2\pi)$. `phi_ref`, if not `None`, the phi values will lie within $\pi$ of `phi_ref`. The `float32` argument controls if the resulting array will be single-precision (can be useful to avoid extraneous copying, if 32-bit floats will be ultimately used).

```python
pyfjcore.pseudojets_to_epxpypz_array(pseudojets, float32=False)
```

Converts a collection of PseudoJets (`PseudoJetContainer` or a Python iterable) to a 2D NumPy array of `(E, px, py, pz)` values. The `float32` argument controls if the resulting array will be single-precision (can be useful to avoid extraneous copying, if 32-bit floats will be ultimately used).

```python
pyfjcore.pseudojets_to_array(pseudojets, pjrep=pyfjcore.ptyphim, float32=False)
```

Converts a collection of PseudoJets (`PseudoJetContainer` or a Python iterable) to a 2D NumPy array of particles in the representation determined by the `pjrep` keyword argument (valid options are `pyfjcore.ptyphim`, `pyfjcore.epxpypz`, `pyfjcore.ptyphi`). The `float32` argument controls if the resulting array will be single-precision (can be useful to avoid extraneous copying, if 32-bit floats will be ultimately used).

```python
pyfjcore.user_indices(pseudojets)
```

Extracts the user indices from a collection of PseudoJets (`PseudoJetContainer` or a Python iterable) and returns them as a NumPy array of integers.

## Version History

### 1.0.x

**1.0.0**

- Restored `PseudoJetContainer` by explicitly overloading methods where `const std::vector<PseudoJet> &` is accepted as an argument. Methods that previously returned `std::vector<PseudoJet>` now return `PseudoJetContainer`.
- [`EventGeometry`](https://github.com/pkomiske/EventGeometry) and [`Piranha`](https://github.com/pkomiske/Piranha) packages are now built using pyfjcore to provide basic FastJet classes.
- fjcore.cc is compiled into a shared library that is linked into the Python extension.

### 0.4.x

**0.4.0**

- Incompatibility with FastJet Python extension fixed by adding virtual methods to `PseudoJet`, `PseudoJetStructureBase`, `CompositeJetStructure`, and `ClusterSequenceStructure` that were removed in fjcore due to lack of area support.

### 0.3.x

**0.3.0**

- Memory leak (and subsequent crash) detected in EnergyFlow testing of PyFJCore. Removing `PseudoJetContainer` for now.

### 0.2.x

**0.2.1**

- Fixed typechecking so that PseudoJetContainer is accepted in overloaded functions such as `Selector::operator()`.

**0.2.0**

- Built against older NumPy properly; added `pyproject.toml` file.

### 0.1.x

**0.1.2**

- Renamed some `PseudoJetRepresentation` constants.
- Updated documentation.

**0.1.1**

- Fixed several bugs, including an inability to pass a `PseudoJetContainer` to the `ClusterSequence` constructor due to SWIG's typechecking.

**0.1.0**

- First version released on PyPI.

## References

PyFJCore relies critically on the fjcore [header](https://github.com/pkomiske/PyFJCore/blob/main/pyfjcore/fjcore.hh) and [source](https://github.com/pkomiske/PyFJCore/blob/main/pyfjcore/fjcore.cc) files, which in turn are created from the main FastJet library. So if you use this package in your research, please cite the [FastJet package and publications](http://fastjet.fr/about.html).

### Summary of changes to fjcore

- **fjcore.hh** 
    - Changed namespace from `fjcore` to `fastjet` to facilitate interoperability with the FastJet Python extension.
    - Added back virtual methods to `PseudoJet`, `PseudoJetStructureBase`, `CompositeJetStructure`, and `ClusterSequenceStructure` that were removed in fjcore due to lack of area support. This is critical for ensuring compatibility with the FastJet Python extension.
    - Wrapped some code in `IsBaseAndDerived` that SWIG cannot parse with `#ifndef SWIG_PREPROCESSOR` and `#endif`. Since SWIG doesn't need this code for anything, it parses the file correctly without affecting the actual compilation.
    - Changed templated `ClusterSequence` constructor to an untemplated version using `PseudoJet` as the former template type.
    - Added methods that accept `PseudoJetContainer` anywhere that `const std::vector<PseudoJet> &` is an argument.

## fjcore README

```text
// fjcore -- extracted from FastJet v3.4.0 (http://fastjet.fr)
//
// fjcore constitutes a digest of the main FastJet functionality.
// The files fjcore.hh and fjcore.cc are meant to provide easy access to these 
// core functions, in the form of single files and without the need of a full 
// FastJet installation:
//
//     g++ main.cc fjcore.cc
// 
// with main.cc including fjcore.hh.
//
// A fortran interface, fjcorefortran.cc, is also provided. See the example 
// and the Makefile for instructions.
//
// The results are expected to be identical to those obtained by linking to
// the full FastJet distribution.
//
// NOTE THAT, IN ORDER TO MAKE IT POSSIBLE FOR FJCORE AND THE FULL FASTJET
// TO COEXIST, THE FORMER USES THE "fjcore" NAMESPACE INSTEAD OF "fastjet". 
//
// In particular, fjcore provides:
//
//   - access to all native pp and ee algorithms, kt, anti-kt, C/A.
//     For C/A, the NlnN method is available, while anti-kt and kt
//     are limited to the N^2 one (still the fastest for N < 100k particles)
//   - access to selectors, for implementing cuts and selections
//   - access to all functionalities related to pseudojets (e.g. a jet's
//     structure or user-defined information)
//
// Instead, it does NOT provide:
//
//   - jet areas functionality
//   - background estimation
//   - access to other algorithms via plugins
//   - interface to CGAL
//   - fastjet tools, e.g. filters, taggers
//
// If these functionalities are needed, the full FastJet installation must be
// used. The code will be fully compatible, with the sole replacement of the
// header files and of the fjcore namespace with the fastjet one.
//
// fjcore.hh and fjcore.cc are not meant to be human-readable.
// For documentation, see the full FastJet manual and doxygen at http://fastjet.fr
//
// Like FastJet, fjcore is released under the terms of the GNU General Public
// License version 2 (GPLv2). If you use this code as part of work towards a
// scientific publication, whether directly or contained within another program
// (e.g. Delphes, MadGraph, SpartyJet, Rivet, LHC collaboration software frameworks, 
// etc.), you should include a citation to
// 
//   EPJC72(2012)1896 [arXiv:1111.6097] (FastJet User Manual)
//   and, optionally, Phys.Lett.B641 (2006) 57 [arXiv:hep-ph/0512210]
//
//FJSTARTHEADER
// $Id$
//
// Copyright (c) 2005-2021, Matteo Cacciari, Gavin P. Salam and Gregory Soyez
//
//----------------------------------------------------------------------
// This file is part of FastJet (fjcore).
//
//  FastJet is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  The algorithms that underlie FastJet have required considerable
//  development. They are described in the original FastJet paper,
//  hep-ph/0512210 and in the manual, arXiv:1111.6097. If you use
//  FastJet as part of work towards a scientific publication, please
//  quote the version you use and include a citation to the manual and
//  optionally also to hep-ph/0512210.
//
//  FastJet is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with FastJet. If not, see <http://www.gnu.org/licenses/>.
//----------------------------------------------------------------------
//FJENDHEADER
```
