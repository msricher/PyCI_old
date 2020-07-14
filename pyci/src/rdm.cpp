/* This file is part of PyCI.
 *
 * PyCI is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * PyCI is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PyCI. If not, see <http://www.gnu.org/licenses/>. */

#include <cstring>

#include <vector>

#include <iostream>
#include <pyci.h>


namespace pyci {


void OneSpinWfn::compute_rdms_doci(const double *coeffs, double *d0, double *d2) const {
    // prepare working vectors
    std::vector<uint_t> det(nword);
    std::vector<int_t> occs(nocc);
    std::vector<int_t> virs(nvir);
    // fill rdms with zeros
    int_t i = nbasis * nbasis, j = 0;
    while (j < i) {
        d0[j] = 0;
        d2[j++] = 0;
    }
    // iterate over determinants
    int_t idet, jdet, k, l;
    double val1, val2;
    for (idet = 0; idet < ndet; ++idet) {
        // fill working vectors
        copy_det(idet, &det[0]);
        fill_occs(nword, &det[0], &occs[0]);
        fill_virs(nword, nbasis, &det[0], &virs[0]);
        // diagonal elements
        val1 = coeffs[idet] * coeffs[idet];
        for (i = 0; i < nocc; ++i) {
            k = occs[i];
            d0[k * (nbasis + 1)] += val1;
            for (j = i + 1; j < nocc; ++j) {
                l = occs[j];
                d2[nbasis * k + l] += val1;
                d2[nbasis * l + k] += val1;
            }
            // pair excitation elements
            for (j = 0; j < nvir; ++j) {
                l = virs[j];
                excite_det(k, l, &det[0]);
                jdet = index_det(&det[0]);
                excite_det(l, k, &det[0]);
                // check if excited determinant is in wfn
                if (jdet > idet) {
                    val2 = coeffs[idet] * coeffs[jdet];
                    d0[nbasis * k + l] += val2;
                    d0[nbasis * l + k] += val2;
                }
            }
        }
    }
}


void TwoSpinWfn::compute_rdms_fullci(const double *coeffs, double *aa, double *bb,
        double *aaaa, double *bbbb, double *abab) const {
    // prepare working vectors
    std::vector<uint_t> det(nword2);
    std::vector<int_t> occs_up(nocc_up);
    std::vector<int_t> occs_dn(nocc_dn);
    std::vector<int_t> virs_up(nvir_up);
    std::vector<int_t> virs_dn(nvir_dn);
    const uint_t *rdet_up, *rdet_dn;
    uint_t *det_up = &det[0], *det_dn = &det[nword];
    // fill rdms with zeros
    int_t i = nbasis * nbasis, j = 0;
    while (j < i) {
        aa[j] = 0;
        bb[j] = 0;
        aaaa[j] = 0;
        bbbb[j] = 0;
        abab[j++] = 0;
    }
    i *= nbasis * nbasis;
    while (j < i) {
        aaaa[j] = 0;
        bbbb[j] = 0;
        abab[j++] = 0;
    }
    // iterate over determinants
    int_t k, l, ii, jj, kk, ll, jdet, ioffset, koffset, sign_up;
    int_t n1 = nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val1, val2;
    for (int_t idet = 0; idet < ndet; ++idet) {
        // fill working vectors
        rdet_up = &dets[idet * nword2];
        rdet_dn = rdet_up + nword;
        std::memcpy(det_up, rdet_up, sizeof(uint_t) * nword2);
        fill_occs(nword, rdet_up, &occs_up[0]);
        fill_occs(nword, rdet_dn, &occs_dn[0]);
        fill_virs(nword, nbasis, rdet_up, &virs_up[0]);
        fill_virs(nword, nbasis, rdet_dn, &virs_dn[0]);
        val1 = coeffs[idet] * coeffs[idet];
        // loop over spin-up occupied indices
        for (i = 0; i < nocc_up; ++i) {
            ii = occs_up[i];
            ioffset = n3 * ii;
            // compute 0-0 terms
            //aa(ii, ii) += val1;
            aa[(n1 + 1) * ii] += val1;
            for (k = i + 1; k < nocc_up; ++k) {
                kk = occs_up[k];
                koffset = ioffset + n2 * kk;
                //aaaa(ii, kk, ii, kk) += val1;
                aaaa[koffset + ii * n1 + kk] += val1;
                //aaaa(ii, kk, kk, ii) -= val1;
                aaaa[koffset + kk * n1 + ii] -= val1;
            }
            for (k = 0; k < nocc_dn; ++k) {
                kk = occs_dn[k];
                //abab(ii, kk, ii, kk) += val1;
                abab[ioffset + kk * n2 + ii * n1 + kk] += val1;
            }
            // loop over spin-up virtual indices
            for (j = 0; j < nvir_up; ++j) {
                jj = virs_up[j];
                // 1-0 excitation elements
                excite_det(ii, jj, det_up);
                sign_up = phase_single_det(nword, ii, jj, rdet_up);
                jdet = index_det(det_up);
                // check if 1-0 excited determinant is in wfn
                if (jdet > idet) {
                    // compute 1-0 terms
                    val2 = coeffs[idet] * coeffs[jdet] * sign_up;
                    //aa(ii, jj) += val2;
                    aa[ii * n1 + jj] += val2;
                    for (k = 0; k < nocc_up; ++k) {
                        kk = occs_up[k];
                        koffset = ioffset + n2 * kk;
                        //aaaa(ii, kk, jj, kk) += val2;
                        aaaa[koffset + jj * n1 + kk] += val2;
                        //aaaa(ii, kk, kk, jj) -= val2;
                        aaaa[koffset + kk * n1 + jj] -= val2;
                    }
                    for (k = 0; k < nocc_dn; ++k) {
                        kk = occs_dn[k];
                        //abab(ii, kk, jj, kk) += val2;
                        abab[ioffset + kk * n2 + jj * n1 + kk] += val2;
                    }
                }
                // loop over spin-down occupied indices
                for (k = 0; k < nocc_dn; ++k) {
                    kk = occs_dn[k];
                    koffset = ioffset + n2 * kk;
                    // loop over spin-down virtual indices
                    for (l = 0; l < nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 1-1 excitation elements
                        excite_det(kk, ll, det_dn);
                        jdet = index_det(det_up);
                        // check if 1-1 excited determinant is in wfn
                        if (jdet > idet) {
                            // compute 1-1 terms
                            val2 = coeffs[idet] * coeffs[jdet]
                                 * sign_up * phase_single_det(nword, kk, ll, rdet_dn);
                            //abab(ii, kk, jj, ll) += val2;
                            abab[koffset + jj * n1 + ll] += val2;
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                // loop over spin-up occupied indices
                for (k = i + 1; k < nocc_up; ++k) {
                    kk = occs_up[k];
                    koffset = ioffset + n2 * kk;
                    // loop over spin-up virtual indices
                    for (l = j + 1; l < nvir_up; ++l) {
                        ll = virs_up[l];
                        // 2-0 excitation elements
                        excite_det(kk, ll, det_up);
                        jdet = index_det(det_up);
                        // check if 2-0 excited determinant is in wfn
                        if (jdet > idet) {
                            // compute 2-0 terms
                            val2 = coeffs[idet] * coeffs[jdet]
                                 * phase_double_det(nword, ii, kk, jj, ll, rdet_up);
                            //aaaa(ii, kk, jj, ll) += val2;
                            aaaa[koffset + jj * n1 + ll] += val2;
                            //aaaa(ii, kk, ll, jj) -= val2;
                            aaaa[koffset + ll * n1 + jj] -= val2;
                        }
                        excite_det(ll, kk, det_up);
                    }
                }
                excite_det(jj, ii, det_up);
            }
        }
        // loop over spin-down occupied indices
        for (i = 0; i < nocc_dn; ++i) {
            ii = occs_dn[i];
            ioffset = n3 * ii;
            // compute 0-0 terms
            //bb(ii, ii) += val1;
            bb[(n1 + 1) * ii] += val1;
            for (k = i + 1; k < nocc_dn; ++k) {
                kk = occs_dn[k];
                koffset = ioffset + n2 * kk;
                //bbbb(ii, kk, ii, kk) += val1;
                bbbb[koffset + ii * n1 + kk] += val1;
                //bbbb(ii, kk, kk, ii) -= val1;
                bbbb[koffset + kk * n1 + ii] -= val1;
            }
            // loop over spin-down virtual indices
            for (j = 0; j < nvir_dn; ++j) {
                jj = virs_dn[j];
                // 0-1 excitation elements
                excite_det(ii, jj, det_dn);
                jdet = index_det(det_up);
                // check if 0-1 excited determinant is in wfn
                if (jdet > idet) {
                    // compute 0-1 terms
                    val2 = coeffs[idet] * coeffs[jdet]
                         * phase_single_det(nword, ii, jj, rdet_dn);
                    //bb(ii, jj) += val2;
                    bb[ii * n1 + jj] += val2;
                    for (k = 0; k < nocc_up; ++k) {
                        kk = occs_up[k];
                        //abab(ii, kk, jj, kk) += val2;
                        abab[ioffset + kk * n2 + jj * n1 + kk] += val2;
                    }
                    for (k = 0; k < nocc_dn; ++k) {
                        kk = occs_dn[k];
                        koffset = ioffset + n2 * kk;
                        //bbbb(ii, kk, jj, kk) += val2;
                        bbbb[koffset + jj * n1 + kk] += val2;
                        //bbbb(ii, kk, kk, jj) -= val2;
                        bbbb[koffset + kk * n1 + jj] -= val2;
                    }
                }
                // loop over spin-down occupied indices
                for (k = i + 1; k < nocc_dn; ++k) {
                    kk = occs_dn[k];
                    koffset = ioffset + n2 * kk;
                    // loop over spin-down virtual indices
                    for (l = j + 1; l < nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 0-2 excitation elements
                        excite_det(kk, ll, det_dn);
                        jdet = index_det(det_up);
                        // check if excited determinant is in wfn
                        if (jdet > idet) {
                            // compute 2-0 terms
                            val2 = coeffs[idet] * coeffs[jdet]
                                 * phase_double_det(nword, ii, kk, ll, jj, rdet_dn);
                            //bbbb(ii, kk, jj, ll) += val2;
                            bbbb[koffset + jj * n1 + ll] += val2;
                            //bbbb(ii, kk, ll, jj) -= val2;
                            bbbb[koffset + ll * n1 + jj] -= val2;
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                excite_det(jj, ii, det_dn);
            }
        }
    }
}

#include <pybind11/pybind11.h>
namespace py = pybind11;

void print_array(double *rdm2, int nbasis) {
    for (int i = 0; i < nbasis; i++)
    {
        for(int j = 0 ; j < nbasis; j++)
        {
            for(int k = 0; k < nbasis; k++)
            {
                for(int l = 0; l < nbasis; l++)
                {
                    int index = nbasis * nbasis * nbasis * i + nbasis * nbasis * j + nbasis * k + l;
                    py::print(i, j, k, l, rdm2[index]);
                }
            }
            py::print("\n");
        }
    }
}

int indices_to_index(int i, int j, int k, int l, int nbasis)
{
    return i * nbasis * nbasis * nbasis + j * nbasis * nbasis + k * nbasis + l;
}


void OneSpinWfn::compute_rdms_genci(const double *coeffs, double *rdm1, double *rdm2) const {
    // prepare working vectors
    std::vector<uint_t> det(nword);
    std::vector<int_t> occs(nocc);
    std::vector<int_t> virs(nvir);
    const uint_t *rdet;
    // fill rdms with zeros
    int_t i = nbasis * nbasis, j = 0;
    while (j < i) {
        rdm1[j] = 0;
        rdm2[j++] = 0;
    }
    i *= nbasis * nbasis;
    while (j < i)
        rdm2[j++] = 0;
    // loop over determinants
    int_t k, l, ii, jj, kk, ll, jdet, ioffset, koffset;
    int_t n1 = nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val1, val2;

    py::print("Number of occupation in C", nocc);
    py::print("Number of Virtual in C", nvir);

    for (int_t idet = 0; idet < ndet; ++idet) {
        // fill working vectors
        rdet = &dets[idet * nword];
        std::memcpy(&det[0], rdet, sizeof(uint_t) * nword);
        fill_occs(nword, rdet, &occs[0]);
        fill_virs(nword, nbasis, rdet, &virs[0]);
        val1 = coeffs[idet] * coeffs[idet];
        // loop over occupied indices
        for (i = 0; i < nocc; ++i) {
            ii = occs[i];
            ioffset = n3 * ii;
            // compute diagonal terms
            //rdm1(ii, ii) += val1;
            rdm1[(n1 + 1) * ii] += val1;
            for (k = i + 1; k < nocc; ++k) {
                kk = occs[k];
                koffset = ioffset + n2 * kk;
                //rdm2(ii, kk, ii, kk) += val1;
                //rdm2[koffset + ii * n1 + kk] += val1;
                py::print(indices_to_index(ii, kk, ii, kk, nbasis));
                py::print(koffset + ii * n1 + kk);
                py::print("\n");
                int index = indices_to_index(ii, kk, ii, kk, nbasis);
                rdm2[index] += val1;
                //rdm2(ii, kk, kk, ii) -= val1;
                rdm2[koffset + kk * n1 + ii] -= val1;
            }
            // loop over virtual indices
            for (j = 0; j < nvir; ++j) {
                jj = virs[j];
                // single excitation elements
                excite_det(ii, jj, &det[0]);
                jdet = index_det(&det[0]);
                // check if singly-excited determinant is in wfn
                if (jdet != -1) {
                    // compute single excitation terms
                    val2 = coeffs[idet] * coeffs[jdet] * phase_single_det(nword, ii, jj, rdet);
                    //rdm1(ii, jj) += val2;
                    rdm1[ii * n1 + jj] += val2;
                    for (k = 0; k < nocc; ++k) {
                        kk = occs[k];
                        //rdm2(ii, kk, jj, kk) += val2;
                        rdm2[koffset + jj * n1 + kk] += val2;
                        //rdm2(ii, kk, kk, jj) -= val2;
                        rdm2[koffset + kk * n1 + jj] -= val2;
                    }
                }
                // loop over occupied indices
                for (k = i + 1; k < nocc; ++k) {
                    kk = occs[k];
                    koffset = ioffset + n2 * kk;
                    // loop over virtual indices
                    for (l = j + 1; l < nvir; ++l) {
                        ll = virs[l];
                        // double excitation elements
                        excite_det(kk, ll, &det[0]);
                        jdet = index_det(&det[0]);
                        // check if double excited determinant is in wfn
                        if (jdet != -1) {
                            // compute double excitation terms
                            val2 = coeffs[idet] * coeffs[jdet]
                                 * phase_double_det(nword, ii, kk, jj, ll, rdet);
                            //rdm2(ii, kk, jj, ll) += val2;
                            rdm2[koffset + jj * n1 + ll] += val2;
                            //rdm2(ii, kk, ll, jj) -= val2;
                            rdm2[koffset + ll * n1 + jj] -= val2;
                        }
                        excite_det(ll, kk, &det[0]);
                    }
                }
                excite_det(jj, ii, &det[0]);
            }
        }
    }
}


} // namespace pyci
