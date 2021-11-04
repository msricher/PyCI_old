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

#include <pyci.h>

namespace pyci {

long APIG::compute_nparam(const long nbasis, const long nocc_up, const long nocc_dn) {
    return nbasis * nocc_up + 1;
}

APIG::APIG(const APIG &other) : FanCI(other), occs(other.occs) {
    return;
}

APIG::APIG(APIG &&other) noexcept : FanCI(other), occs(std::move(other.occs)) {
    return;
}

APIG::APIG(const Ham &ham, const DOCIWfn &wfn, const long nproj_)
    : FanCI(ham, wfn, compute_nparam(wfn.nbasis, wfn.nocc_up, wfn.nocc_dn),
            (nproj_ == -1) ? compute_nparam(wfn.nbasis, wfn.nocc_up, wfn.nocc_dn) : nproj_,
            (nproj_ == -1) ? compute_nparam(wfn.nbasis, wfn.nocc_up, wfn.nocc_dn) : nproj_),
      occs(wfn.ndet * wfn.nocc_up) {
    for (long idet = 0; idet != wfn.ndet; ++idet) {
        fill_occs(wfn.nword, wfn.det_ptr(idet), &occs[idet * wfn.nocc_up]);
    }
}

void APIG::initial_guess(double *x) const {
    long i, j, k = 0;
    for (i = 0; i != nocc_up; ++i) {
        for (j = 0; j != nbasis; ++j) {
            x[k++] = (i == j);
        }
    }
    x[nparam - 1] = op.get_element(0, 0);
}

void APIG::compute_overlap(const double *x, double *y, const long start, const long end) const {
    for (long idet = start; idet < end; ++idet) {
        y[idet] = permanent(x, &occs[idet * nocc_up], nocc_up, nbasis);
    }
}

void APIG::compute_overlap_deriv(const double *x, double *y, const long start,
                                 const long end) const {
    long ielem = 0, idet, i, j, k, l;
    long rows[64], cols[64];
    for (i = 0; i != nocc_up; ++i) {
        for (l = 0; l < i; ++l)
            rows[l] = l;
        for (l = i + 1; l < nocc_up; ++l)
            rows[l - 1] = l;
        for (j = 0; j != nbasis; ++j) {
            for (idet = start; idet < end; ++idet) {
                // Compute derivative
                for (k = 0; k != nocc_up; ++k) {
                    if (occs[idet * nocc_up + k] == j) {
                        for (l = 0; l < k; ++l)
                            cols[l] = occs[idet * nocc_up + l];
                        for (l = k + 1; l < nocc_up; ++l)
                            cols[l - 1] = occs[idet * nocc_up + l];
                        y[ielem] = permanent(x, rows, cols, nocc_up - 1, nbasis);
                        goto next_param;
                    }
                }
                y[ielem] = 0;
            next_param:
                ++ielem;
            }
        }
    }
}

void APIG::compute_cost(const double *x, double *y) const {
    double energy = x[nparam - 1];
    std::vector<double> olp(nconn);
    compute_overlap(x, &olp[0], 0, nconn);
    op.perform_op(&olp[0], y);
    for (long i = 0; i != nproj; ++i) {
        y[i] -= olp[i] * energy;
    }
}

void APIG::compute_cost_deriv(const double *x, double *y) const {
    double energy = x[nparam - 1];
    std::vector<double> olp((nparam - 1) * nconn);
    long i, j;
    compute_overlap(x, &olp[0], 0, nproj);
    for (j = 0; j != nproj; ++j) {
        y[nproj * (nparam - 1) + j] = -olp[j];
    }
    compute_overlap_deriv(x, &olp[0], 0, nconn);
    for (i = 0; i != nparam - 1; ++i) {
        op.perform_op(&olp[nconn * i], &y[nproj * i]);
        for (j = 0; j != nproj; ++j) {
            y[nproj * i + j] -= olp[nconn * i + j] * energy;
        }
    }
}

Array<double> APIG::py_initial_guess(void) const {
    Array<double> array({nparam});
    initial_guess(reinterpret_cast<double *>(array.request().ptr));
    return array;
}

Array<double> APIG::py_compute_overlap(Array<double> x, long start, long end) const {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = nconn;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    Array<double> array({end - start});
    compute_overlap(reinterpret_cast<double *>(x.request().ptr),
                    reinterpret_cast<double *>(array.request().ptr), start, end);
    return array;
}

FArray<double> APIG::py_compute_overlap_deriv(Array<double> x, long start, long end) const {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = nconn;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    FArray<double> array({end - start, nparam - 1});
    compute_overlap_deriv(reinterpret_cast<double *>(x.request().ptr),
                          reinterpret_cast<double *>(array.request().ptr), start, end);
    return array;
}

Array<double> APIG::py_compute_cost(Array<double> x) const {
    Array<double> array({nproj});
    compute_cost(reinterpret_cast<double *>(x.request().ptr),
                 reinterpret_cast<double *>(array.request().ptr));
    return array;
}

FArray<double> APIG::py_compute_cost_deriv(Array<double> x) const {
    FArray<double> array({nproj, nparam});
    compute_cost_deriv(reinterpret_cast<double *>(x.request().ptr),
                       reinterpret_cast<double *>(array.request().ptr));
    return array;
}

} // namespace pyci
