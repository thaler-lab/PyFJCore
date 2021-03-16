# PyFJCore

Python wrapper of FastJet Core functionality with additional NumPy support. In contrast with the [`pyjet`](https://github.com/scikit-hep/pyjet) package, `pyfjcore` wraps all the methods/functions in fjcore and also does not require structured NumPy arrays. In contrast with the Python extension to the main [FastJet library](http://fastjet.fr), this package can be built in a portable manner, including on Windows.

Current version of fjcore: 3.3.4

## Documentation

The [FastJet documentation](http://fastjet.fr/repo/doxygen-3.3.4/), despite being for the C++ package, contains helpful information for the classes and methods in PyFJCore. Not all FastJet classes are wrapped in PyFJCore, primarily just `PseudoJet`, `JetDefinition`, `ClusterSequence`, and `Selector`.

A few modifcations have been made to fjcore to make it more amenable to wrapping in Python. SWIG automatically converts return values of `std::vector<PseudoJet>` to Python tuples, making a copy in the process. Another copy is be required to pass a Python iterable to methods that accept `const std::vector<PseudoJet> &`.

### `PseudoJetContainer`

To avoid unnecessary copying, fjcore has been modified to return a `PseudoJetContainer` any time `std::vector<PseudoJet>` is normally returned. `PseudoJetContainer` holds a `vectorPseudoJet`, which is the Python wrapper around `std::vector<PseudoJet>`. The wrapper code has been modified so that methods that accept `const std::vector<PseudoJet> &` will accept a `PseudoJetContainer` without any copying.

`PseudoJetContainer` is convertible to a Python iterable like a list or tuple (by using the `__iter__` method from `vectorPseudoJet`). It can be indexed, assigned to, and modified (by deleting elements) as if it were a vector of PseudoJets. The `vector` property can be used to access the underlying `vectorPseudoJet` directly.

A Python object can be attached to a `PseudoJet` using the `.set_python_info()` method. It can be accessed as `.python_info()`.

### NumPy conversion functions

```python
pyfjcore.ptyphim_array_to_pseudojets(ptyphims)
```

Converts an array of particles, as `(pt, y, phi, [mass])`, to PseudoJets. Any additional features (columns) of the array are set as the Python user info of the PseudoJets. This also sets the `user_index` of the PseudoJets to their position in the input array. Returns a `PseudoJetContainer`.

```python
pyfjcore.epxpypz_array_to_pseudojets(epxpypzs)
```

Converts an array of particles, as `(E, px, py, pz)`, to PseudoJets. Any additional features (columns) of the array are set as the Python user info of the PseudoJets. This also sets the `user_index` of the PseudoJets to their position in the input array. Returns a `PseudoJetContainer`.

```python
pyfjcore.array_to_pseudojets(particles, pjrep=PseudoJetRepresentation_ptyphim)
```

Converts an array of particles to PseudoJets. The format of the particles kinematics is determined by the `pjrep` argument. The `PseudoJetRepresentation` enum can take the values `PseudoJetRepresentation_ptyphim`, `PseudoJetRepresentation_ptyphi`, `PseudoJetRepresentation_epxpypz`. The first two values cause this function to invoke `ptyphim_array_to_pseudojets` and the third invokes `epxpypz_array_to_pseudojets`. Any additional features (columns) of the array are set as the Python user info of the PseudoJets. This also sets the `user_index` of the PseudoJets to their position in the input array. Returns a `PseudoJetContainer`.

```python
pyfjcore.pseudojets_to_ptyphim_array(pseudojets, mass=True)
```

Converts a vector of PseudoJets (equivalently, `PseudoJetContainer`), to a 2D NumPy array of `(pt, y, phi, [mass])` values, where the presence of the mass is determine by the keyword argument.

```python
pyfjcore.pseudojets_to_epxpypz_array(pseudojets, mass=True)
```

Converts a vector of PseudoJets (equivalently, `PseudoJetContainer`), to a 2D NumPy array of `(E, px, py, pz)` values.

```python
pyfjcore.pseudojets_to_array(pseudojets, pjrep=PseudoJetRepresentation_ptyphim)
```

Converts a vector of PseudoJets (equivalently, `PseudoJetContainer`), to a 2D NumPy array of particles in the representation determined by the `pjrep` keyword argument.

```python
pyfjcore.user_indices(pseudojets)
```

Extracts the user indices from a vector of PseudoJets (equivalently, `PseudoJetContainer`) and returns the as a NumPy array. There is also a `user_indices` method of `PseudoJetContainer` that has the same effect.

## References

PyFJCore relies critically on the fjcore header and source files, which in turn are created from the main FastJet library. So if you use this package in your research, please cite the [FastJet package and publications](http://fastjet.fr/about.html).

## fjcore README

```text
// fjcore -- extracted from FastJet v3.3.4 (http://fastjet.fr)
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
// Copyright (c) 2005-2020, Matteo Cacciari, Gavin P. Salam and Gregory Soyez
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
