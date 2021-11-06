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

long pCCD::compute_nparam(const long nbasis, const long nocc_up, const long nocc_dn) {
    return nocc_up * (nbasis - nocc_up) + 1;
}

pCCD::pCCD(const pCCD &other) : FanCI(other), occs(other.occs) {
    return;
}

pCCD::pCCD(pCCD &&other) noexcept : FanCI(other), occs(std::move(other.occs)) {
    return;
}

pCCD::pCCD(const Ham &ham, const DOCIWfn &wfn, const long nproj_)
    : FanCI(ham, wfn, compute_nparam(wfn.nbasis, wfn.nocc_up, wfn.nocc_dn),
            (nproj_ == -1) ? compute_nparam(wfn.nbasis, wfn.nocc_up, wfn.nocc_dn) : nproj_,
            (nproj_ == -1) ? compute_nparam(wfn.nbasis, wfn.nocc_up, wfn.nocc_dn) : nproj_) {
    std::vector<ulong> ref(wfn.nword);
    std::vector<ulong> det(wfn.nword);
    fill_hartreefock_det(wfn.nocc_up, &ref[0]);
    int c, i, j = 0;
    for (long idet = 0; idet != wfn.ndet; ++idet) {
        // Get holes
        wfn.copy_det(idet, &det[0]);
        c = 0;
        for (i = 0; i != wfn.nword; ++i) {
            det[i] = ref[i] & (det[i] ^ ref[i]);
            c += Pop<ulong>(det[i]);
        }
        // Add [count] [holes] [particles]
        occs.resize(j + 1 + 2 * c);
        // Add [count]
        occs[j] = c;
        ++j;
        if (c == 0)
            continue;
        // Add [holes]
        fill_occs(wfn.nword, &det[0], &occs[j]);
        j += c;
        // Get particles
        wfn.copy_det(idet, &det[0]);
        for (i = 0; i != wfn.nword; ++i) {
            det[i] = det[i] & (det[i] ^ ref[i]);
        }
        // Add [particles]
        fill_occs(wfn.nword, &det[0], &occs[j]);
        j += c;
    }
}

void pCCD::initial_guess(double *x) const {
    long n = nocc_up * (nbasis - nocc_up);
    for (long i = 0; i != n; ++i)
        x[i] = 0;
    x[n] = op.get_element(0, 0);
}

void pCCD::compute_overlap(const double *x, double *y, const long start, const long end) const {
    long nv = nbasis - nocc_up, iocc = 0, c;
    for (long idet = start; idet < end; ++idet) {
        c = occs[iocc++];
        y[idet] = permanent(x, &occs[iocc], &occs[iocc + c], c, nv);
        iocc += 2 * c;
    }
}

void pCCD::compute_overlap_deriv(const double *x, double *y, const long start,
                                 const long end) const {
}

void pCCD::compute_cost(const double *x, double *y) const {
    double energy = x[nparam - 1];
    std::vector<double> olp(nconn);
    compute_overlap(x, &olp[0], 0, nconn);
    op.perform_op(&olp[0], y);
    for (long i = 0; i != nproj; ++i) {
        y[i] -= olp[i] * energy;
    }
}

void pCCD::compute_cost_deriv(const double *x, double *y) const {
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

Array<double> pCCD::py_initial_guess(void) const {
    Array<double> array({nparam});
    initial_guess(reinterpret_cast<double *>(array.request().ptr));
    return array;
}

Array<double> pCCD::py_compute_overlap(Array<double> x, long start, long end) const {
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

FArray<double> pCCD::py_compute_overlap_deriv(Array<double> x, long start, long end) const {
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

Array<double> pCCD::py_compute_cost(Array<double> x) const {
    Array<double> array({nproj});
    compute_cost(reinterpret_cast<double *>(x.request().ptr),
                 reinterpret_cast<double *>(array.request().ptr));
    return array;
}

FArray<double> pCCD::py_compute_cost_deriv(Array<double> x) const {
    FArray<double> array({nproj, nparam});
    compute_cost_deriv(reinterpret_cast<double *>(x.request().ptr),
                       reinterpret_cast<double *>(array.request().ptr));
    return array;
}

} // namespace pyci
