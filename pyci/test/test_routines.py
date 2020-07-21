# This file is part of PyCI.
#
# PyCI is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# PyCI is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCI. If not, see <http://www.gnu.org/licenses/>.

from nose.tools import assert_raises

import numpy as np
import numpy.testing as npt

from scipy.special import comb

import pyci
from pyci.test import datafile

import itertools
def parity2(p):
    return sum(
        1 for (x,px) in enumerate(p)
          for (y,py) in enumerate(p)
          if x<y and px>py
        )%2==0


def parity(p):
    par = parity2(p)
    if par:
        return 1.0
    return -1.0


class TestRoutines:

    CASES = [
        ('he_ccpvqz',  pyci.fullci_wfn, (1, 1),  -2.886809116),
        ('li2_ccpvdz', pyci.doci_wfn,   (3,),   -14.878455349),
        ('be_ccpvdz',  pyci.doci_wfn,   (2,),   -14.600556994),
        ('he_ccpvqz',  pyci.doci_wfn,   (1,),    -2.886809116),
        ('be_ccpvdz',  pyci.fullci_wfn, (2, 2), -14.600556994),
        ('h2o_ccpvdz', pyci.doci_wfn,   (5,),   -75.634588422),
        ]

    def test_solve_sparse(self):
        for filename, wfn_type, occs, energy in self.CASES:
            yield self.run_solve_sparse, filename, wfn_type, occs, energy

    def test_sparse_rectangular(self):
        for filename, wfn_type, occs, energy in self.CASES:
            yield self.run_sparse_rectangular, filename, wfn_type, occs, energy

    def test_compute_rdms(self):
        for filename, wfn_type, occs, energy in self.CASES:
            yield self.run_compute_rdms, filename, wfn_type, occs, energy

    def test_run_hci(self):
        for filename, wfn_type, occs, energy in self.CASES[:4]:
            yield self.run_run_hci, filename, wfn_type, occs, energy

    def run_solve_sparse(self, filename, wfn_type, occs, energy):
        ham = pyci.restricted_ham(datafile('{0:s}.fcidump'.format(filename)))
        wfn = wfn_type(ham.nbasis, *occs)
        wfn.add_all_dets()
        op = pyci.sparse_op(ham, wfn)
        es, cs = op.solve(n=1, ncv=30, tol=1.0e-6)
        npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-9)

    def run_sparse_rectangular(self, filename, wfn_type, occs, energy):
        ham = pyci.restricted_ham(datafile('{0:s}.fcidump'.format(filename)))
        wfn = wfn_type(ham.nbasis, *occs)
        wfn.add_all_dets()
        nrow = len(wfn) - 10
        op = pyci.sparse_op(ham, wfn, nrow)
        assert op.shape == (nrow, len(wfn))
        y = op(np.ones(op.shape[1], dtype=pyci.c_double))
        assert y.ndim == 1
        assert y.shape[0] == op.shape[0]

    def run_compute_rdms(self, filename, wfn_type, occs, energy):
        def true_two_rdm_two_spin_up_two_down(wf, coeff):
            r""" Get the true rdm2 from straight brute force calculation by going through
            all slater determinants for each row/column."""
            true = np.zeros((wf.nbasis * 2, wf.nbasis * 2, wf.nbasis * 2, wf.nbasis * 2))
            true_onedm = np.zeros((wf.nbasis * 2, wf.nbasis * 2))
            # Go Through Rows.
            for i, occ_dual in enumerate(wf.to_occ_array()):
                dspin_up, dspin_dn = occ_dual[0], wfn.nbasis + occ_dual[1]
                concatenate = np.hstack((dspin_up, dspin_dn))
                permutate = itertools.permutations(list(concatenate))
                print(i)
                print("Start concatenate", concatenate)
                for p in permutate:
                    p = np.array(p)
                    sign_p = parity(p)
                    # Go Through Columns.
                    for j, occ in enumerate(wf.to_occ_array()):
                        spin_up, spin_dn = occ[0], wfn.nbasis + occ[1]
                        concatenate_row = np.hstack((spin_up, spin_dn))
                        permutate2 = itertools.permutations(list(concatenate_row))

                        for d in permutate2:
                            d = np.array(d)

                            if np.all(np.abs(p[2:] - d[2:]) < 1e-5):
                                sign_d = parity(d)
                                true[p[0], p[1], d[0], d[1]] += sign_p * sign_d * coeff[i] * coeff[
                                    j]

                                if np.all(np.abs(p[1:] - d[1:]) < 1e-5):
                                    true_onedm[p[0], d[0]] += sign_p * sign_d * coeff[i] * coeff[j]
                print("\n")
            return true, true_onedm


        ham = pyci.restricted_ham(datafile('{0:s}.fcidump'.format(filename)))
        wfn = wfn_type(ham.nbasis, *occs)
        wfn.add_all_dets()
        op = pyci.sparse_op(ham, wfn)
        es, cs = op.solve(n=1, ncv=30, tol=1.0e-6)
        if isinstance(wfn, pyci.doci_wfn):
            d0, d2 = pyci.compute_rdms(wfn, cs[0])
            npt.assert_allclose(np.trace(d0), wfn.nocc_up, rtol=0, atol=1.0e-9)
            npt.assert_allclose(np.sum(d2), wfn.nocc_up * (wfn.nocc_up - 1), rtol=0, atol=1.0e-9)
            k0, k2 = pyci.reduce_senzero_integrals(ham.h, ham.v, ham.w, wfn.nocc_up)
            energy = ham.ecore
            energy += np.einsum('ij,ij', k0, d0)
            energy += np.einsum('ij,ij', k2, d2)
            npt.assert_allclose(energy, es[0], rtol=0.0, atol=1.0e-9)
            rdm1, rdm2 = pyci.make_rdms(d0, d2)
        elif isinstance(wfn, pyci.fullci_wfn):
            d1, d2 = pyci.compute_rdms(wfn, cs[0])
            rdm1, rdm2 = pyci.make_rdms(d1, d2)
            # TODO: Ali commented this out because the code does it.
            # rdm1 = rdm1 + rdm1.T - np.diag(rdm1)

        else:
            rdm1, rdm2 = pyci.compute_rdms(wfn, cs[0])
        with np.load(datafile('{0:s}_spinres.npz'.format(filename))) as f:
            one_mo = f['one_mo']
            two_mo = f['two_mo']

        assert np.all(np.abs(rdm1 - rdm1.T) < 1e-5)

        # Test RDM2 is antisymmetric
        for i in range(0, wfn.nbasis * 2):
            for j in range(0, wfn.nbasis * 2):
                assert np.all(rdm2[i, j, :, :] + rdm2[i, j, :, :].T) < 1e-5
                for k in range(0, wfn.nbasis * 2):
                    for l in range(0, wfn.nbasis * 2):
                        assert np.abs(rdm2[i, j, k, l] - rdm2[k, l, i, j]) < 1e-5

        # "Testing that non Antiysmmetric parts are all zeros."
        for i in range(0, wfn.nbasis * 2):
            print("i,", i)
            assert np.all(np.abs(rdm2[i, i, :, :]) < 1e-5)
            assert np.all(np.abs(rdm2[:, :, i, i]) < 1e-5)


        energy = ham.ecore
        energy += np.einsum('ij,ij', one_mo, rdm1)
        energy += 0.25 * np.einsum('ijkl,ijkl', two_mo, rdm2)
        npt.assert_allclose(energy, es[0], rtol=0.0, atol=1.0e-9)

    def run_run_hci(self, filename, wfn_type, occs, energy):
        ham = pyci.restricted_ham(datafile('{0:s}.fcidump'.format(filename)))
        wfn = wfn_type(ham.nbasis, *occs)
        wfn.add_hartreefock_det()
        es, cs = pyci.sparse_op(ham, wfn).solve(n=1, tol=1.0e-6)
        dets_added = 1
        niter = 0
        while dets_added:
            dets_added = pyci.run_hci(ham, wfn, cs[0], eps=1.0e-5)
            es, cs = pyci.sparse_op(ham, wfn).solve(n=1, tol=1.0e-6)
            niter += 1
        assert niter > 1
        assert len(wfn) < np.prod([comb(wfn.nbasis, occ, exact=True) for occ in occs])
        npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-6)
        dets_added = 1
        while dets_added:
            dets_added = pyci.run_hci(ham, wfn, cs[0], eps=0.0)
            op = pyci.sparse_op(ham, wfn)
            es, cs = op.solve(n=1, tol=1.0e-6)
        assert len(wfn) == np.prod([comb(wfn.nbasis, occ, exact=True) for occ in occs])
        npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-9)


class TestRDMAnalyticExamples():
    r"""
    Test for rdm1, rdm2 of comparing against very simple, analytic examples.

    """

    def test_compute_rdm_two_particles_one_up_one_dn(self):
        wfn = pyci.fullci_wfn(2, 1, 1)
        wfn.add_all_dets()

        coeffs = np.sqrt(np.array([1., 2., 3., 4.]))
        coeffs /= np.linalg.norm(coeffs)

        d0, d1 = pyci.compute_rdms(wfn, coeffs)

        # Test diagonal of abab.
        assert np.abs(d1[2, 0, 0, 0, 0] - coeffs[0] ** 2.0) < 1e-5
        assert np.abs(d1[2, 0, 1, 0, 1] - coeffs[1] ** 2.0) < 1e-5
        assert np.abs(d1[2, 1, 0, 1, 0] - coeffs[2] ** 2.0) < 1e-5
        assert np.abs(d1[2, 1, 1, 1, 1] - coeffs[3] ** 2.0) < 1e-5

        # "Test Spin-Up off-diagonal of abab.
        assert np.abs(d1[2, 0, 0, 0, 1] - coeffs[0] * coeffs[1]) < 1e-5
        assert np.abs(d1[2, 0, 0, 1, 0] - coeffs[0] * coeffs[2]) < 1e-5
        assert np.abs(d1[2, 0, 0, 1, 1] - coeffs[0] * coeffs[3]) < 1e-5

        assert np.abs(d1[2, 0, 1, 0, 0] - coeffs[1] * coeffs[0]) < 1e-5
        assert np.abs(d1[2, 0, 1, 1, 0] - coeffs[1] * coeffs[2]) < 1e-5
        assert np.abs(d1[2, 0, 1, 1, 1] - coeffs[1] * coeffs[3]) < 1e-5

        assert np.abs(d1[2, 1, 0, 0, 0] - coeffs[2] * coeffs[0]) < 1e-5
        assert np.abs(d1[2, 1, 0, 0, 1] - coeffs[2] * coeffs[1]) < 1e-5
        assert np.abs(d1[2, 1, 0, 1, 1] - coeffs[2] * coeffs[3]) < 1e-5

        assert np.abs(d1[2, 1, 1, 0, 0] - coeffs[3] * coeffs[0]) < 1e-5
        assert np.abs(d1[2, 1, 1, 0, 1] - coeffs[3] * coeffs[1]) < 1e-5
        assert np.abs(d1[2, 1, 1, 1, 0] - coeffs[3] * coeffs[2]) < 1e-5

        # Testing that aaaa is all zeros.
        assert np.all(np.abs(d1[0, :, :, :, :]) < 1e-5)

        # Testing that bbbb is all zeros.
        assert np.all(np.abs(d1[1, :, :, :, :]) < 1e-5)

    def test_make_rdm_rdm2_two_particles_one_up_one_dn(self):
        wfn = pyci.fullci_wfn(2, 1, 1)
        wfn.add_all_dets()

        coeffs = np.sqrt(np.array([1., 2., 3., 4.]))
        coeffs /= np.linalg.norm(coeffs)

        d0, d1 = pyci.compute_rdms(wfn, coeffs)
        _, rdm2 = pyci.make_rdms(d0, d1)

        # "Test out the diagonal RDM2"
        assert np.abs(rdm2[0, 0, 0, 0]) < 1e-5
        assert np.abs(rdm2[0, 1, 0, 1]) < 1e-5  # Since no spin up spin up.and
        assert np.abs(rdm2[0, 2, 0, 2] - coeffs[0] ** 2.0) < 1e-5
        assert np.abs(rdm2[0, 3, 0, 3] - coeffs[1] ** 2.0) < 1e-5
        assert np.abs(rdm2[1, 0, 1, 0]) < 1e-5
        assert np.abs(rdm2[1, 1, 1, 1]) < 1e-5
        assert np.abs(rdm2[1, 2, 1, 2] - coeffs[2] ** 2.0) < 1e-5
        assert np.abs(rdm2[1, 3, 1, 3] - coeffs[3] ** 2.0) < 1e-5
        assert np.abs(rdm2[2, 0, 2, 0] - coeffs[0] ** 2.0) < 1e-5
        assert np.abs(rdm2[2, 1, 2, 1] - coeffs[2] ** 2.0) < 1e-5
        assert np.abs(rdm2[2, 2, 2, 2]) < 1e-5
        assert np.abs(rdm2[2, 3, 2, 3]) < 1e-5
        assert np.abs(rdm2[3, 0, 3, 0] - coeffs[1] ** 2.0) < 1e-5
        assert np.abs(rdm2[3, 1, 3, 1] - coeffs[3] ** 2.0) < 1e-5
        assert np.abs(rdm2[3, 2, 3, 2]) < 1e-5
        assert np.abs(rdm2[3, 3, 3, 3]) < 1e-5

        # "Testing that non Antiysmmetric parts are all zeros."
        for i in range(0, 4):
            assert np.all(rdm2[i, i, :, :] == 0)
            assert np.all(rdm2[:, :, i, i] == 0)

        # "Testing that One has to be Occupied Up and hte other Down."
        assert np.all(np.abs(rdm2[0, 1, :, :]) < 1e-5)
        assert np.all(np.abs(rdm2[:, :, 0, 1]) < 1e-5)
        assert np.all(np.abs(rdm2[1, 0, :, :]) < 1e-5)
        assert np.all(np.abs(rdm2[:, :, 1, 0]) < 1e-5)
        assert np.all(np.abs(rdm2[2, 3, :, :]) < 1e-5)
        assert np.all(np.abs(rdm2[3, 2, :, :]) < 1e-5)
        assert np.all(np.abs(rdm2[:, :, 2, 3]) < 1e-5)
        assert np.all(np.abs(rdm2[:, :, 3, 2]) < 1e-5)

        # Test out off-diagonal.
        assert np.abs(rdm2[0, 2, 0, 3] - coeffs[0] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[0, 2, 1, 2] - coeffs[0] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[0, 2, 1, 3] - coeffs[0] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[0, 2, 2, 0] + coeffs[0] ** 2.0) < 1e-5
        assert np.abs(rdm2[0, 2, 2, 1] + coeffs[0] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[0, 2, 2, 3]) < 1e-5
        assert np.abs(rdm2[0, 2, 3, 0] + coeffs[0] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[0, 2, 3, 1] + coeffs[0] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[0, 2, 3, 2]) < 1e-5

        assert np.abs(rdm2[2, 0, 0, 3] + coeffs[0] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[2, 0, 1, 2] + coeffs[0] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[2, 0, 1, 3] + coeffs[0] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[2, 0, 2, 0] - coeffs[0] ** 2.0) < 1e-5
        assert np.abs(rdm2[2, 0, 2, 1] - coeffs[0] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[2, 0, 2, 3]) < 1e-5
        assert np.abs(rdm2[2, 0, 3, 0] - coeffs[0] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[2, 0, 3, 1] - coeffs[0] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[2, 0, 3, 2]) < 1e-5

        assert np.abs(rdm2[0, 3, 0, 2] - coeffs[1] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[0, 3, 1, 2] - coeffs[1] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[0, 3, 1, 3] - coeffs[1] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[0, 3, 2, 0] + coeffs[1] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[0, 3, 2, 1] + coeffs[1] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[0, 3, 2, 3]) < 1e-5
        assert np.abs(rdm2[0, 3, 3, 0] + coeffs[1] ** 2.0) < 1e-5
        assert np.abs(rdm2[0, 3, 3, 1] + coeffs[1] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[0, 3, 3, 2]) < 1e-5

        assert np.abs(rdm2[3, 0, 0, 2] + coeffs[1] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[3, 0, 1, 2] + coeffs[1] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[3, 0, 1, 3] + coeffs[1] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[3, 0, 2, 0] - coeffs[1] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[3, 0, 2, 1] - coeffs[1] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[3, 0, 2, 3]) < 1e-5
        assert np.abs(rdm2[3, 0, 3, 0] - coeffs[1] ** 2.0) < 1e-5
        assert np.abs(rdm2[3, 0, 3, 1] - coeffs[1] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[3, 0, 3, 2]) < 1e-5

        assert np.abs(rdm2[1, 2, 0, 2] - coeffs[2] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[1, 2, 0, 3] - coeffs[2] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[1, 2, 1, 0]) < 1e-5
        assert np.abs(rdm2[1, 2, 1, 3] - coeffs[2] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[1, 2, 2, 0] + coeffs[2] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[1, 2, 2, 1] + coeffs[2] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[1, 2, 2, 3]) < 1e-5
        assert np.abs(rdm2[1, 2, 3, 0] + coeffs[2] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[1, 2, 3, 1] + coeffs[2] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[1, 2, 3, 2]) < 1e-5
        assert np.abs(rdm2[1, 2, 3, 3]) < 1e-5

        assert np.abs(rdm2[2, 1, 0, 2] + coeffs[2] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[2, 1, 0, 3] + coeffs[2] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[2, 1, 1, 0]) < 1e-5
        assert np.abs(rdm2[2, 1, 1, 3] + coeffs[2] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[2, 1, 2, 0] - coeffs[2] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[2, 1, 2, 1] - coeffs[2] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[2, 1, 2, 3]) < 1e-5
        assert np.abs(rdm2[2, 1, 3, 0] - coeffs[2] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[2, 1, 3, 1] - coeffs[2] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[2, 1, 3, 2]) < 1e-5
        assert np.abs(rdm2[2, 1, 3, 3]) < 1e-5

        assert np.abs(rdm2[1, 3, 0, 1]) < 1e-5
        assert np.abs(rdm2[1, 3, 0, 2] - coeffs[3] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[1, 3, 0, 3] - coeffs[3] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[1, 3, 1, 0]) < 1e-5
        assert np.abs(rdm2[1, 3, 1, 2] - coeffs[3] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[1, 3, 2, 0] + coeffs[3] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[1, 3, 2, 1] + coeffs[3] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[1, 3, 2, 3]) < 1e-5
        assert np.abs(rdm2[1, 3, 3, 0] + coeffs[3] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[1, 3, 3, 1] + coeffs[3] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[1, 3, 3, 2]) < 1e-5

        assert np.abs(rdm2[3, 1, 0, 1]) < 1e-5
        assert np.abs(rdm2[3, 1, 0, 2] + coeffs[3] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[3, 1, 0, 3] + coeffs[3] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[3, 1, 1, 0]) < 1e-5
        assert np.abs(rdm2[3, 1, 1, 2] + coeffs[3] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[3, 1, 2, 0] - coeffs[3] * coeffs[0]) < 1e-5
        assert np.abs(rdm2[3, 1, 2, 1] - coeffs[3] * coeffs[2]) < 1e-5
        assert np.abs(rdm2[3, 1, 2, 3]) < 1e-5
        assert np.abs(rdm2[3, 1, 3, 0] - coeffs[3] * coeffs[1]) < 1e-5
        assert np.abs(rdm2[3, 1, 3, 1] - coeffs[3] * coeffs[3]) < 1e-5
        assert np.abs(rdm2[3, 1, 3, 2]) < 1e-5

    def test_make_rdm_rdm1_two_particles_one_up_one_dn(self):
        wfn = pyci.fullci_wfn(2, 1, 1)
        wfn.add_all_dets()

        coeffs = np.sqrt(np.array([1., 2., 3., 4.]))
        coeffs /= np.linalg.norm(coeffs)

        d0, d1 = pyci.compute_rdms(wfn, coeffs)
        rdm1, _ = pyci.make_rdms(d0, d1)

        assert np.abs(rdm1[0, 0] - coeffs[0] ** 2.0 - coeffs[1] ** 2.0) < 1e-5
        assert np.abs(rdm1[0, 1] - coeffs[0] * coeffs[2] - coeffs[1] * coeffs[3]) < 1e-5
        assert np.abs(rdm1[0, 2]) < 1e-5
        assert np.abs(rdm1[0, 3]) < 1e-5

        assert np.abs(rdm1[1, 0] - coeffs[0] * coeffs[2] - coeffs[1] * coeffs[3]) < 1e-5
        assert np.abs(rdm1[1, 1] - coeffs[3] ** 2.0 - coeffs[2] ** 2.0) < 1e-5
        assert np.abs(rdm1[1, 2]) < 1e-5
        assert np.abs(rdm1[1, 3]) < 1e-5

        assert np.abs(rdm1[2, 0]) < 1e-5
        assert np.abs(rdm1[2, 1]) < 1e-5
        assert np.abs(rdm1[2, 2] - coeffs[0] ** 2.0 - coeffs[2] ** 2.0) < 1e-5
        assert np.abs(rdm1[2, 3] - coeffs[2] * coeffs[3] - coeffs[0] * coeffs[1]) < 1e-5

        assert np.abs(rdm1[3, 0]) < 1e-5
        assert np.abs(rdm1[3, 1]) < 1e-5
        assert np.abs(rdm1[3, 2] - coeffs[2] * coeffs[3] - coeffs[0] * coeffs[1]) < 1e-5
        assert np.abs(rdm1[3, 3] - coeffs[3] ** 2.0 - coeffs[1] ** 2.0) < 1e-5

    def test_make_rdm_rdm2_two_up_one_dn(self):
        wfn = pyci.fullci_wfn(3, 2, 1)
        wfn.add_all_dets()
        print("Number of Spatial Orbital Basis ", wfn.nbasis)
        print("Determinant Array in Binary")
        print(wfn.to_det_array())

        coeffs = np.sqrt(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]))
        coeffs /= np.linalg.norm(coeffs)

        d0, d1 = pyci.compute_rdms(wfn, coeffs)

        print(d1[0].shape)
        # assert antisymmetry aspec of aaaa is all zeros.
        for i in range(0, 3):
            assert np.all(np.abs(d1[0, i, i, :, :]) < 1e-5)
            assert np.all(np.abs(d1[0, :, :, i, i]) < 1e-5)

            for j in range(0, 3):
                assert np.all(np.abs(d1[0, i, j, i, j] + d1[0, i, j, j, i]) < 1e-5)
                assert np.all(np.abs(d1[0, j, i, i, j] - d1[0, i, j, j, i]) < 1e-5)
                assert np.all(np.abs(d1[0, i, j, i, j] + d1[0, j, i, i, j]) < 1e-5)

        aaaa = d1[0]
        # assert Diagonal elements of aaaa.
        assert np.abs(
            aaaa[0, 1, 0, 1] - coeffs[0] ** 2.0 - coeffs[1] ** 2.0 - coeffs[2] ** 2.0) < 1e-5
        assert np.abs(
            aaaa[0, 2, 0, 2] - coeffs[3] ** 2.0 - coeffs[4] ** 2.0 - coeffs[5] ** 2.0) < 1e-5
        assert np.abs(
            aaaa[1, 2, 1, 2] - coeffs[6] ** 2.0 - coeffs[7] ** 2.0 - coeffs[8] ** 2.0) < 1e-5

        # assert non-diagonal elements of aaaa
        assert np.abs(
            aaaa[0, 1, 0, 2] - coeffs[0] * coeffs[3] - coeffs[1] * coeffs[4] - coeffs[2] * coeffs[
                5]) < 1e-5
        assert np.abs(
            aaaa[0, 1, 1, 2] - coeffs[0] * coeffs[6] - coeffs[1] * coeffs[7] - coeffs[2] * coeffs[
                8]) < 1e-5
        assert np.abs(
            aaaa[0, 2, 1, 2] - coeffs[3] * coeffs[6] - coeffs[4] * coeffs[7] - coeffs[5] * coeffs[
                8]) < 1e-5

        # Assert that bbbb is all zeros.
        assert np.all(d1[1] < 1e-5)

        abab = d1[2]

        # Test antisymmetry of abab
        assert np.abs(abab[0, 0, 0, 0] - coeffs[0] ** 2.0 - coeffs[3] ** 2.0) < 1e-5
        assert np.abs(abab[0, 0, 0, 1] - coeffs[0] * coeffs[1] - coeffs[3] * coeffs[4]) < 1e-5
        assert np.abs(abab[0, 0, 0, 2] - coeffs[0] * coeffs[2] - coeffs[3] * coeffs[5]) < 1e-5
        assert np.abs(abab[0, 0, 1, 0] - coeffs[3] * coeffs[6]) < 1e-5
        assert np.abs(abab[0, 0, 1, 1] - coeffs[3] * coeffs[7]) < 1e-5
        assert np.abs(abab[0, 0, 1, 2] - coeffs[3] * coeffs[8]) < 1e-5
        assert np.abs(abab[0, 0, 2, 0] + coeffs[6] * coeffs[0]) < 1e-5
        assert np.abs(abab[0, 0, 2, 1] + coeffs[0] * coeffs[7]) < 1e-5
        assert np.abs(abab[0, 0, 2, 2] + coeffs[0] * coeffs[8]) < 1e-5

        assert np.abs(abab[0, 1, 0, 0] - coeffs[1] * coeffs[0] - coeffs[3] * coeffs[4]) < 1e-5
        assert np.abs(abab[0, 1, 0, 1] - coeffs[1] ** 2.0 - coeffs[4] ** 2.0) < 1e-5
        assert np.abs(abab[0, 1, 0, 2] - coeffs[4] * coeffs[5] - coeffs[1] * coeffs[2]) < 1e-5
        assert np.abs(abab[0, 1, 1, 0] - coeffs[4] * coeffs[6]) < 1e-5
        assert np.abs(abab[0, 1, 1, 1] - coeffs[4] * coeffs[7]) < 1e-5
        assert np.abs(abab[0, 1, 1, 2] - coeffs[4] * coeffs[8]) < 1e-5
        assert np.abs(abab[0, 1, 2, 0] + coeffs[1] * coeffs[6]) < 1e-5
        assert np.abs(abab[0, 1, 2, 1] + coeffs[1] * coeffs[7]) < 1e-5
        assert np.abs(abab[0, 1, 2, 2] + coeffs[1] * coeffs[8]) < 1e-5

        assert np.abs(abab[0, 2, 0, 0] - coeffs[2] * coeffs[0] - coeffs[3] * coeffs[5]) < 1e-5
        assert np.abs(abab[0, 2, 0, 1] - coeffs[2] * coeffs[1] - coeffs[5] * coeffs[4]) < 1e-5
        assert np.abs(abab[0, 2, 1, 0] - coeffs[5] * coeffs[6]) < 1e-5
        assert np.abs(abab[0, 2, 1, 1] - coeffs[5] * coeffs[7]) < 1e-5
        assert np.abs(abab[0, 2, 1, 2] - coeffs[5] * coeffs[8]) < 1e-5
        assert np.abs(abab[0, 2, 2, 0] + coeffs[2] * coeffs[6]) < 1e-5
        assert np.abs(abab[0, 2, 2, 1] + coeffs[2] * coeffs[7]) < 1e-5
        assert np.abs(abab[0, 2, 2, 2] + coeffs[2] * coeffs[8]) < 1e-5

        assert np.abs(abab[1, 0, 0, 0] - coeffs[6] * coeffs[3]) < 1e-5
        assert np.abs(abab[1, 0, 0, 1] - coeffs[6] * coeffs[4]) < 1e-5
        assert np.abs(abab[1, 0, 0, 2] - coeffs[6] * coeffs[5]) < 1e-5
        assert np.abs(abab[1, 0, 1, 1] - coeffs[0] * coeffs[1] - coeffs[6] * coeffs[7]) < 1e-5
        assert np.abs(abab[1, 0, 1, 2] - coeffs[0] * coeffs[2] - coeffs[6] * coeffs[8]) < 1e-5
        assert np.abs(abab[1, 0, 2, 0] - coeffs[0] * coeffs[3]) < 1e-5
        assert np.abs(abab[1, 0, 2, 1] - coeffs[0] * coeffs[4]) < 1e-5
        assert np.abs(abab[1, 0, 2, 2] - coeffs[0] * coeffs[5]) < 1e-5

        assert np.abs(abab[1, 1, 0, 0] - coeffs[3] * coeffs[7]) < 1e-5
        assert np.abs(abab[1, 1, 0, 1] - coeffs[4] * coeffs[7]) < 1e-5
        assert np.abs(abab[1, 1, 0, 2] - coeffs[5] * coeffs[7]) < 1e-5
        assert np.abs(abab[1, 1, 1, 0] - coeffs[0] * coeffs[1] - coeffs[7] * coeffs[6]) < 1e-5
        assert np.abs(abab[1, 1, 1, 2] - coeffs[1] * coeffs[2] - coeffs[7] * coeffs[8]) < 1e-5
        assert np.abs(abab[1, 1, 2, 0] - coeffs[1] * coeffs[3]) < 1e-5
        assert np.abs(abab[1, 1, 2, 1] - coeffs[1] * coeffs[4]) < 1e-5
        assert np.abs(abab[1, 1, 2, 2] - coeffs[1] * coeffs[5]) < 1e-5

        assert np.abs(abab[1, 2, 0, 0] - coeffs[8] * coeffs[3]) < 1e-5
        assert np.abs(abab[1, 2, 0, 1] - coeffs[8] * coeffs[4]) < 1e-5
        assert np.abs(abab[1, 2, 0, 2] - coeffs[8] * coeffs[5]) < 1e-5
        assert np.abs(abab[1, 2, 1, 0] - coeffs[0] * coeffs[2] - coeffs[6] * coeffs[8]) < 1e-5
        assert np.abs(abab[1, 2, 1, 1] - coeffs[8] * coeffs[7] - coeffs[2] * coeffs[1]) < 1e-5
        assert np.abs(abab[1, 2, 2, 0] - coeffs[2] * coeffs[3]) < 1e-5
        assert np.abs(abab[1, 2, 2, 1] - coeffs[2] * coeffs[4]) < 1e-5
        assert np.abs(abab[1, 2, 2, 2] - coeffs[2] * coeffs[5]) < 1e-5

        assert np.abs(abab[2, 2, 0, 0] + coeffs[0] * coeffs[8]) < 1e-5
        assert np.abs(abab[2, 2, 0, 1] + coeffs[8] * coeffs[1]) < 1e-5
        assert np.abs(abab[2, 2, 0, 2] + coeffs[8] * coeffs[2]) < 1e-5
        assert np.abs(abab[2, 2, 1, 0] - coeffs[0] * coeffs[5]) < 1e-5
        assert np.abs(abab[2, 2, 1, 1] - coeffs[1] * coeffs[5]) < 1e-5
        assert np.abs(abab[2, 2, 1, 2] - coeffs[2] * coeffs[5]) < 1e-5
        assert np.abs(abab[2, 2, 2, 0] - coeffs[5] * coeffs[3] - coeffs[6] * coeffs[8]) < 1e-5
        assert np.abs(abab[2, 2, 2, 1] - coeffs[5] * coeffs[4] - coeffs[8] * coeffs[7]) < 1e-5
        assert np.abs(abab[2, 2, 2, 2] - coeffs[8] ** 2.0 - coeffs[5] ** 2.0) < 1e-5


