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

void print_vector(std::vector<int_t> vec, int row)
{
    for(int m = 0 ; m < row ; m++)
    {
        py::print(vec[m], " ");
    }
}

void print_vector2(std::vector<uint_t> vec, int row)
{
    for(int m = 0 ; m < row ; m++)
    {
        py::print(vec[m], " ");
    }
}

int indices_to_index(int i, int j, int k, int l, int nbasis)
{
    return i * nbasis * nbasis * nbasis + j * nbasis * nbasis + k * nbasis + l;
}


double factorial(int_t x)
{
    if (x == 0)
    {
        return 1.0;
    }
    double output = 1.0;
    for(int i = 1; i <= x; ++i)
    {
        output *= (double) i;
    }
    return output;
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


    // ALI ADDED THESE CONSTANTS
    //double permutation_1rdm = factorial(nocc_up + nocc_dn - 1.0);
    double permutation_1rdm = 1.0;
    // This number is needed because all of the various ways of permutating the parts that are going to be traced out.,
    //double permutation_2rdm_aaaa_bbbb = factorial(nocc_up - 2 + nocc_dn);
    double permutation_2rdm_aaaa_bbbb = 1.0;
    py::print("Permutation ", permutation_1rdm);

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

        py::print("Determinant Number ", idet);
        print_vector2(det, det.size());
        py::print("Occupation Up Then Down. ");
        print_vector(occs_up, occs_up.size());
        print_vector(occs_dn, occs_dn.size());
        py::print("Virtual Up Then Down");
        print_vector(virs_up, virs_up.size());
        print_vector(virs_dn, virs_dn.size());


        // loop over spin-up occupied indices
        py::print("Loop over Up");
        for (i = 0; i < nocc_up; ++i) {
            ii = occs_up[i];
            ioffset = n3 * ii;
            // compute 0-0 terms
            //aa(ii, ii) += val1;
            aa[(n1 + 1) * ii] += val1 * permutation_1rdm;
            for (k = i + 1; k < nocc_up; ++k) {
                kk = occs_up[k];
                koffset = ioffset + n2 * kk;
                //aaaa(ii, kk, ii, kk) += val1;
                aaaa[koffset + ii * n1 + kk] += val1 * permutation_2rdm_aaaa_bbbb;
                //aaaa(ii, kk, kk, ii) -= val1;
                aaaa[koffset + kk * n1 + ii] -= val1 * permutation_2rdm_aaaa_bbbb;

                // TODO: Double check the indices work.
                aaaa[indices_to_index(kk, ii, ii, kk, nbasis)] -= val1 * permutation_2rdm_aaaa_bbbb;
                //rdm2(ii, kk, kk, ii) -= val1;
                aaaa[indices_to_index(kk, ii, kk, ii, nbasis)] += val1 * permutation_2rdm_aaaa_bbbb;
            }
            for (k = 0; k < nocc_dn; ++k) {
                kk = occs_dn[k];
                py::print("    abab, Diagonal, ii", ii, "kk", kk, "ii", ii, "kk", kk);
                //abab(ii, kk, ii, kk) += val1;
                abab[ioffset + kk * n2 + ii * n1 + kk] += val1;

            }

            // loop over spin-up virtual indices
            py::print("     Start Spin-up virtual ");
            for (j = 0; j < nvir_up; ++j) {
                jj = virs_up[j];
                // 1-0 excitation elements
                excite_det(ii, jj, det_up);
                py::print("    Excite ");
                print_vector2(det, nword);
                py::print("    Done ");
                sign_up = phase_single_det(nword, ii, jj, rdet_up);
                jdet = index_det(det_up);

                py::print("    jj ", jj, "jdet", jdet);

                // check if 1-0 excited determinant is in wfn
                if (jdet > idet) {
                    // compute 1-0 terms
                    val2 = coeffs[idet] * coeffs[jdet] * sign_up;
                    //aa(ii, jj) += val2;
                    aa[ii * n1 + jj] += val2 * permutation_1rdm;
                    aa[jj * n1 + ii] += val2 * permutation_1rdm;
                    for (k = 0; k < nocc_up; ++k) {
                        if (i != k)
                        {
                            kk = occs_up[k];
                            koffset = ioffset + n2 * kk;
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "ii", ii, "kk", kk, "jj", jj, "kk", kk);
                            //aaaa(ii, kk, jj, kk) += val2;
                            aaaa[koffset + jj * n1 + kk] += val2 * permutation_2rdm_aaaa_bbbb;
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "ii", ii, "kk", kk, "kk", kk, "jj", jj);
                            //aaaa(ii, kk, kk, jj) -= val2;
                            aaaa[koffset + kk * n1 + jj] -= val2 * permutation_2rdm_aaaa_bbbb;

                            // aaaa(kk, ii, kk, jj)
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "kk", kk, "ii", ii, "kk", kk, "jj", jj);
                            aaaa[kk * n3 + ii * n2 + kk * n1 + jj] += val2 * permutation_2rdm_aaaa_bbbb;
                            // aaaa(kk, ii, jj, kk)
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "kk", kk, "ii", ii, "jj", jj, "kk", kk);
                            aaaa[kk * n3 + ii * n2 + jj * n1 + kk] -= val2 * permutation_2rdm_aaaa_bbbb;



                            // Switch Particles
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "jj", jj, "kk", kk, "ii", ii, "kk", kk);
                            //aaaa(jj, kk, ii, kk)
                            aaaa[n3 * jj + n2 * kk + n1 * ii + kk] += val2 * permutation_2rdm_aaaa_bbbb;
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "jj", jj, "kk", kk, "kk", kk, "ii", ii);
                            //aaaa(jj, kk, kk, ii)
                            aaaa[n3 * jj + n2 * kk + n1 * kk + ii] -= val2 * permutation_2rdm_aaaa_bbbb;

                            //Switch Above
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "kk", kk, "jj", jj, "ii", ii, "kk", kk);
                            //aaaa(kk, jj, ii, kk)
                            aaaa[n3 * kk + n2 * jj + n1 * ii + kk] -= val2 * permutation_2rdm_aaaa_bbbb;
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "kk", kk, "jj", jj, "kk", kk, "ii", ii);
                            //aaaa(kk, jj, kk, ii)
                            aaaa[n3 * kk + n2 * jj + n1 * kk + ii] += val2 * permutation_2rdm_aaaa_bbbb;

                            py::print("    disti ", ii, kk, jj, kk);
                            py::print("    disti ", ii, kk, jj, jj);
                            py::print("    disti ", kk, ii, kk, jj);
                            py::print("    disti ", kk, ii, jj, kk);
                            py::print("    disti ", kk, jj, ii, kk);
                            py::print("    disti ", kk, jj, kk, ii);

                        }
                    }

                    for (k = 0; k < nocc_dn; ++k) {
                        kk = occs_dn[k];
                        py::print("    abab: idet", idet, "jdet,", jdet, "ii", ii, "kk", kk, "jj", jj, "kk", kk);
                        //abab(ii, kk, jj, kk) += val2;
                        abab[ioffset + kk * n2 + jj * n1 + kk] += val2;

                        // TODO: Ive added the enxt line
                        py::print("    abab: idet", idet, "jdet,", jdet, "jj", jj, "kk", kk, "ii", ii, "kk", kk);
                        //abab(jj, kk, ii, kk)
                        abab[n3 * jj + kk * n2 + ii * n1 + kk] += val2;
                    }

                }

                // loop over spin-down occupied indices
                py::print("    Start spin-down occupied down ");
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
                            py::print("    abab idet ", idet, "jdet ", jdet, "ii", ii, "kk", kk, "jj", jj, "ll", ll);

                            // compute 1-1 terms
                            val2 = coeffs[idet] * coeffs[jdet]
                                 * sign_up * phase_single_det(nword, kk, ll, rdet_dn);
                            //abab(ii, kk, jj, ll) += val2;
                            abab[koffset + jj * n1 + ll] += val2;

                            // ALI I've added the next line
                            //abab(jj, ll, ii, kk)
                            py::print("    abab idet ", idet, "jdet ", jdet, "jj", jj, "ll", ll, "ii", ii, "kk", kk);
                            abab[n3 * jj + n2 * ll + n1 * ii + kk] += val2;
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
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "ii", ii, "kk", kk, "jj", jj, "ll", ll);

                            //aaaa(ii, kk, jj, ll) += val2;
                            aaaa[koffset + jj * n1 + ll] += val2 * permutation_2rdm_aaaa_bbbb;
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "ii", ii, "kk", kk, "ll", ll, "jj", jj);
                            //aaaa(ii, kk, ll, jj) -= val2;
                            aaaa[koffset + ll * n1 + jj] -= val2 * permutation_2rdm_aaaa_bbbb;

                            //aaaa(kk, ii, jj, ll)
                            aaaa[n3 * kk + n2 * ii + n1 * jj + ll] -= val2 * permutation_2rdm_aaaa_bbbb;
                            //aaaa(kk, ii, ll, jj)
                            aaaa[n3 * kk + n2 * ii + n1 * ll + jj] += val2 * permutation_2rdm_aaaa_bbbb;


                           // ALI: I've added this may have to remove it.
                             //aaaa(jj, ll, ii, kk) += val2;
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "jj", jj, "ll", ll, "ii", ii, "kk", kk);
                            aaaa[jj * n3 + ll * n2 + ii * n1 + kk] += val2 * permutation_2rdm_aaaa_bbbb;
                            //aaaa(jj, ll, kk, ii)
                            aaaa[jj * n3 + ll * n2 + kk * n1 + ii] -= val2 * permutation_2rdm_aaaa_bbbb;
                            py::print("    aaaa: idet", idet, "jdet,", jdet, "ll", ll, "jj", jj, "ii", ii, "kk", kk );
                            //aaaa(ll, jj, ii, kk) -= val2;
                            aaaa[n3 * ll + n2 * jj + n1 * ii + kk] -= val2 * permutation_2rdm_aaaa_bbbb;
                            //aaaa(ll, jj, kk, ii) += val2;
                            aaaa[n3 * ll + n2 * jj + n1 * kk + ii] += val2 * permutation_2rdm_aaaa_bbbb;

                            py::print("    disti ", ii, kk, jj, ll);
                            py::print("    disti ", ii, kk, ll, jj);
                            py::print("    disti ", jj, ll, ii, kk);
                            py::print("    disti ", ll, jj, ii, kk);
                        }
                        excite_det(ll, kk, det_up);
                    }
                }

                // De-escalte the Determinant.
                excite_det(jj, ii, det_up);
            }
        }

        /*
        STOP IN THE NAME OF LOVE>

        */

        // loop over spin-down occupied indices
        py::print("Loop Over Spin-Down.");
        for (i = 0; i < nocc_dn; ++i) {
            ii = occs_dn[i];
            ioffset = n3 * ii;
            // compute 0-0 terms
            //bb(ii, ii) += val1;
            bb[(n1 + 1) * ii] += val1 * permutation_1rdm;
            py::print("    ii ", ii);

            for (k = i + 1; k < nocc_dn; ++k) {
                kk = occs_dn[k];
                koffset = ioffset + n2 * kk;

                py::print("    bbbb, idet ", idet, "ii", ii, "kk", kk, "ii", ii, "kk", kk);
                py::print("    bbbb, idet ", idet, "ii", ii, "kk", kk, "kk", kk, "ii", ii);
                //bbbb(ii, kk, ii, kk) += val1;
                bbbb[koffset + ii * n1 + kk] += val1 * permutation_2rdm_aaaa_bbbb;
                //bbbb(ii, kk, kk, ii) -= val1;
                bbbb[koffset + kk * n1 + ii] -= val1 * permutation_2rdm_aaaa_bbbb;

                // TODO: Double check the indices work.
                py::print("    bbbb, idet ", idet, "kk", kk, "ii", ii, "ii", ii, "kk", kk);
                py::print("    bbbb, idet ", idet, "kk", kk, "ii", ii, "kk", kk, "ii", ii);
                bbbb[indices_to_index(kk, ii, ii, kk, nbasis)] -= val1 * permutation_2rdm_aaaa_bbbb;
                //rdm2(ii, kk, kk, ii) -= val1;
                bbbb[indices_to_index(kk, ii, kk, ii, nbasis)] += val1 * permutation_2rdm_aaaa_bbbb;
            }


            // loop over spin-down virtual indices
            for (j = 0; j < nvir_dn; ++j) {
                jj = virs_dn[j];
                // 0-1 excitation elements
                excite_det(ii, jj, det_dn);
                // TODO : Ali chanaged det_up to det_dn. It gives lots of errors if I do so I lfet it as det_up!.
                jdet = index_det(det_up);

                py::print("    jj", jj);
                // check if 0-1 excited determinant is in wfn
                if (jdet > idet) {
                    // compute 0-1 terms
                    val2 = coeffs[idet] * coeffs[jdet]
                         * phase_single_det(nword, ii, jj, rdet_dn);
                    //bb(ii, jj) += val2;
                    bb[ii * n1 + jj] += val2 * permutation_1rdm;
                    bb[jj * n1 + ii] += val2 * permutation_1rdm;
                    for (k = 0; k < nocc_up; ++k) {
                        kk = occs_up[k];
                        //abab(ii, kk, jj, kk) += val2;
                        py::print("    abab: idet ", idet, "jdet", jdet, "ii", ii, "kk", kk, "kk", kk, "jj", jj);
                        // I'm switchking ii, kk the next line is the Legit one. This is the last change I made.
                        // The reason for the switch is because ii is spin down and kk is spin up but it is abab.
                        //abab[ioffset + kk * n2 + kk * n1 + jj] += val2;
                        abab[n3 * kk + n2 * ii + kk * n1 + jj] += val2;

                        // TODO Might have to remove this- next line->
                        //abab(kk, jj, kk, ii)
                        py::print("    abab: idet ", idet, "jdet", jdet, "kk", kk, "jj", jj, "kk", kk, "ii", ii);
                        abab[n3 * kk + jj * n2 + kk * n1 + ii] += val2;
                    }
                    for (k = 0; k < nocc_dn; ++k) {
                        // Two electrons in the same orbital should be zero hence the if statement.
                        if (i != k){
                            kk = occs_dn[k];

                            py::print("    bbbb: idet ", idet, "jdet", jdet, "ii", ii, "kk ", kk, "jj", jj, "kk", kk);
                            py::print("    bbbb: idet ", idet, "jdet", jdet, "ii", ii, "kk ", kk, "kk", kk, "jj", jj);
                            koffset = ioffset + n2 * kk;
                            //bbbb(ii, kk, jj, kk) += val2;
                            bbbb[koffset + jj * n1 + kk] += val2 * permutation_2rdm_aaaa_bbbb;
                            //bbbb(ii, kk, kk, jj) -= val2;
                            bbbb[koffset + kk * n1 + jj] -= val2 * permutation_2rdm_aaaa_bbbb;


                            py::print("    bbbb: idet ", idet, "jdet", jdet, "kk", kk, "ii ", ii, "kk", kk, "jj", jj);
                            py::print("    bbbb: idet ", idet, "jdet", jdet, "kk", kk, "ii ", ii, "jj", jj, "kk", kk);
                            // bbbb(kk, ii, kk, jj)
                            bbbb[kk * n3 + ii * n2 + kk * n1 + jj] += val2 * permutation_2rdm_aaaa_bbbb;
                            // bbbb(kk, ii, jj, kk)
                            bbbb[kk * n3 + ii * n2 + jj * n1 + kk] -= val2 * permutation_2rdm_aaaa_bbbb;


                            // Switch Particles
                            py::print("    bbbb: idet", idet, "jdet,", jdet, "jj", jj, "kk", kk, "ii", ii, "kk", kk);
                            //bbbb(jj, kk, ii, kk)
                            bbbb[n3 * jj + n2 * kk + n1 * ii + kk] += val2 * permutation_2rdm_aaaa_bbbb;
                            py::print("    bbbb: idet", idet, "jdet,", jdet, "jj", jj, "kk", kk, "kk", kk, "ii", ii);
                            //bbbb(jj, kk, kk, ii)
                            bbbb[n3 * jj + n2 * kk + n1 * kk + ii] -= val2 * permutation_2rdm_aaaa_bbbb;

                            //Switch Above
                            py::print("    bbbb: idet", idet, "jdet,", jdet, "kk", kk, "jj", jj, "ii", ii, "kk", kk);
                            //bbbb(kk, jj, ii, kk)
                            bbbb[n3 * kk + n2 * jj + n1 * ii + kk] -= val2 * permutation_2rdm_aaaa_bbbb;
                            py::print("    bbbb: idet", idet, "jdet,", jdet, "kk", kk, "jj", jj, "kk", kk, "ii", ii);
                            //bbbb(kk, jj, kk, ii)
                            bbbb[n3 * kk + n2 * jj + n1 * kk + ii] += val2 * permutation_2rdm_aaaa_bbbb;

                        }

                    }
                }
                // loop over spin-down occupied indices
                for (k = i + 1; k < nocc_dn; ++k) {
                    kk = occs_dn[k];
                    py::print("    k ", k, "kk", kk);
                    koffset = ioffset + n2 * kk;
                    // loop over spin-down virtual indices
                    for (l = j + 1; l < nvir_dn; ++l) {
                    //for(l = 0; l < nvir_dn; ++l){  // Ali -< Changed this. Real line is above.
                        ll = virs_dn[l];
                        // 0-2 excitation elements
                        excite_det(kk, ll, det_dn);
                        jdet = index_det(det_up); // ALI I changed this to det_dn.
                        py::print("    jdet ", jdet);
                        // check if excited determinant is in wfn
                        if (jdet > idet) {
                            // compute 2-0 terms
                            // TODO:ALi try out  phase_double_det(nword, ii, kk, jj, ll, rdet_up);
                            //val2 = coeffs[idet] * coeffs[jdet]
                            //     * phase_double_det(nword, ii, kk, ll, jj, rdet_dn);
                            val2 = coeffs[idet] * coeffs[jdet]
                                     * phase_double_det(nword, ii, kk, jj, ll, rdet_dn);
                            py::print("    bbbb: idet ", idet, "jdet", jdet, "ii", ii, "kk", kk, "jj", jj, "ll", ll);
                            //bbbb(ii, kk, jj, ll) += val2;
                            bbbb[koffset + jj * n1 + ll] += val2 * permutation_2rdm_aaaa_bbbb;
                            py::print("    bbbb: idet ", idet, "jdet", jdet, "ii", ii, "kk", kk, "ll", ll, "jj", jj);
                            //bbbb(ii, kk, ll, jj) -= val2;
                            bbbb[koffset + ll * n1 + jj] -= val2 * permutation_2rdm_aaaa_bbbb;

                            // ALI: I've added this may have to remove it.

                            //bbbb(kk, ii, jj, ll)
                            bbbb[n3 * kk + n2 * ii + n1 * jj + ll] -= val2 * permutation_2rdm_aaaa_bbbb;
                            //bbbb(kk, ii, ll, jj)
                            bbbb[n3 * kk + n2 * ii + n1 * ll + jj] += val2 * permutation_2rdm_aaaa_bbbb;

                             //bbbb(jj, ll, ii, kk) += val2;
                            py::print("    bbbb: idet", idet, "jdet,", jdet, "jj", jj, "ll", ll, "ii", ii, "kk", kk);
                            bbbb[jj * n3 + ll * n2 + ii * n1 + kk] += val2 * permutation_2rdm_aaaa_bbbb;
                            py::print("    bbbb: idet", idet, "jdet,", jdet, "ll", ll, "jj", jj, "ii", ii, "kk", kk );
                            //bbbb(ll, jj, ii, kk) -= val2;
                            bbbb[n3 * ll + n2 * jj + n1 * ii + kk] -= val2 * permutation_2rdm_aaaa_bbbb;

                            //bbbb(jj, ll, kk, ii)x`
                            bbbb[jj * n3 + ll * n2 + kk * n1 + ii] -= val2 * permutation_2rdm_aaaa_bbbb;
                            //bbbb(ll, jj, kk, ii)
                            bbbb[n3 * ll + n2 * jj + n1 * kk + ii] += val2 * permutation_2rdm_aaaa_bbbb;
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }


                // De-escalate the determinant.
                excite_det(jj, ii, det_dn);
            }

        }
        py::print("Next Determinant \n \n");

    }
}



void OneSpinWfn::compute_rdms_genci(const double *coeffs, double *rdm1, double *rdm2) const {
    // prepare working vectors, these are used so that indexing is not needed.
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
        rdet = &dets[idet * nword];   // Store the Reference point at the start of the ith determinant dets[idet * nword].
        std::memcpy(&det[0], rdet, sizeof(uint_t) * nword);  // Copy at the start of the det
        fill_occs(nword, rdet, &occs[0]);
        fill_virs(nword, nbasis, rdet, &virs[0]);
        val1 = coeffs[idet] * coeffs[idet];

        py::print("Determinant");
        print_vector2(det, det.size());
        py::print("Virtual ");
        print_vector(virs, virs.size());
        py::print("Occupancy ");
        print_vector(occs, occs.size());

        // loop over occupied indices
        for (i = 0; i < nocc; ++i) {
            ii = occs[i];
            ioffset = n3 * ii;
            // compute diagonal terms
            //rdm1(ii, ii) += val1;
            rdm1[(n1 + 1) * ii] += val1;

            // k = i + 1; because symmetric matrix and that when k == i, it is zero.
            py::print("Diagonal");
            for (k = i + 1; k < nocc; ++k) {

                py::print("idet ", idet, "ii", occs[i], "kk", occs[k]);

                //rdm2(ii, kk, ii, kk) += val1;
                rdm2[indices_to_index(occs[i], occs[k], occs[i], occs[k], nbasis)] += val1;
                //rdm2(ii, kk, kk, ii) -= val1;
                rdm2[indices_to_index(occs[i], occs[k], occs[k], occs[i], nbasis)] -= val1;

                //rdm2(ii, kk, ii, kk) += val1;
                rdm2[indices_to_index(occs[k], occs[i], occs[i], occs[k], nbasis)] -= val1;
                //rdm2(ii, kk, kk, ii) -= val1;
                rdm2[indices_to_index(occs[k], occs[i], occs[k], occs[i], nbasis)] += val1;
            }


            py::print("Over Vitual Indices");
            // loop over virtual indices
            for (j = 0; j < nvir; ++j) {
                jj = virs[j];
                // single excitation elements

                py::print("ii, jj, det", ii, jj, det[0]);
                excite_det(ii, jj, &det[0]);

                py::print("Excite ");
                print_vector2(det, det.size());
                py::print("Done ");


                jdet = index_det(&det[0]);
                // check if singly-excited determinant is in wfn
                if (jdet != -1) {
                    // compute single excitation terms
                    val2 = coeffs[idet] * coeffs[jdet] * phase_single_det(nword, ii, jj, rdet);
                    //rdm1(ii, jj) += val2;
                    rdm1[ii * n1 + jj] += val2;
                    for (k = 0; k < nocc; ++k) {
                        if (i != k)
                        {
                        kk = occs[k];

                        py::print("idet ", idet, "jdet ", jdet, "---> ", "ii", ii, "kk", kk, "jj", jj);

                        //rdm2(ii, kk, jj, kk) += val2;
                         rdm2[indices_to_index(ii, kk, jj, kk, nbasis)] += val2;
                        //rdm2(ii, kk, kk, jj) -= val2;
                         rdm2[indices_to_index(ii, kk, kk, jj, nbasis)] -= val2;


                        rdm2[indices_to_index(kk, ii, jj, kk, nbasis)] -= val2;
                        rdm2[indices_to_index(kk, ii, kk, jj, nbasis)] += val2;
                        }
                    }
                }


                // loop over occupied indices
                for (k = i + 1; k < nocc; ++k) {
                    kk = occs[k];
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
                            rdm2[indices_to_index(ii, kk, jj, ll, nbasis)] += val2;
                            //rdm2(ii, kk, ll, jj) -= val2;
                            rdm2[indices_to_index(ii, kk, ll, jj, nbasis)] -= val2;
                        }

                        excite_det(ll, kk, &det[0]);
                    }
                }

                excite_det(jj, ii, &det[0]);

                py::print("DeExcite? ");
                print_vector2(det, det.size());
                py::print("Done ");
            }
        py::print("Next occupancy i. \n ");
        }
    py::print("Next Determinant. \n \n ");
    }
}


} // namespace pyci
