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

#include <new>
#include <stdexcept>
#include <vector>

namespace pyci {

int_t binomial(int_t n, int_t k) {
    if (k == 0)
        return 1;
    else if (k == 1)
        return n;
    else if (k >= n)
        return (k == n);
    else if (k > n / 2)
        k = n - k;
    int_t binom = 1;
    for (int_t d = 1; d <= k; ++d)
        binom = binom * n-- / d;
    return binom;
}

int_t binomial_cutoff(int_t n, int_t k) {
    if (k == 0)
        return 1;
    else if (k == 1)
        return n;
    else if (k >= n)
        return (k == n);
    else if (k > n / 2)
        k = n - k;
    int_t binom = 1;
    for (int_t d = 1; d <= k; ++d) {
        if (binom >= PYCI_INT_MAX / n)
            return PYCI_INT_MAX;
        binom = binom * n-- / d;
    }
    return binom;
}

void fill_det(const int_t nocc, const int_t *occs, uint_t *det) {
    int_t j;
    for (int_t i = 0; i < nocc; ++i) {
        j = occs[i];
        det[j / PYCI_UINT_SIZE] |= PYCI_UINT_ONE << (j % PYCI_UINT_SIZE);
    }
}

void fill_occs(const int_t nword, const uint_t *det, int_t *occs) {
    int_t p, j = 0, offset = 0;
    uint_t word;
    for (int_t i = 0; i < nword; ++i) {
        word = det[i];
        while (word) {
            p = PYCI_CTZ(word);
            occs[j++] = p + offset;
            word &= ~(PYCI_UINT_ONE << p);
        }
        offset += PYCI_UINT_SIZE;
    }
}

void fill_virs(const int_t nword, int_t nbasis, const uint_t *det, int_t *virs) {
    int_t p, j = 0, offset = 0;
    uint_t word, mask;
    for (int_t i = 0; i < nword; ++i) {
        mask = (nbasis < PYCI_UINT_SIZE) ? ((PYCI_UINT_ONE << nbasis) - 1) : PYCI_UINT_MAX;
        word = det[i] ^ mask;
        while (word) {
            p = PYCI_CTZ(word);
            virs[j++] = p + offset;
            word &= ~(PYCI_UINT_ONE << p);
        }
        offset += PYCI_UINT_SIZE;
        nbasis -= PYCI_UINT_SIZE;
    }
}

void next_colex(int_t *indices) {
    int_t i = 0;
    while (indices[i + 1] - indices[i] == 1) {
        indices[i] = i;
        ++i;
    }
    ++(indices[i]);
}

int_t rank_colex(const int_t nbasis, const int_t nocc, const uint_t *det) {
    int_t k = 0, binom = 1, rank = 0;
    for (int_t i = 0; i < nbasis; ++i) {
        if (k == nocc)
            break;
        else if (det[i / PYCI_UINT_SIZE] & (PYCI_UINT_ONE << (i % PYCI_UINT_SIZE))) {
            ++k;
            binom = (k == i) ? 1 : binom * i / k;
            rank += binom;
        } else
            binom = (k >= i) ? 1 : binom * i / (i - k);
    }
    return rank;
}

void unrank_colex(int_t nbasis, const int_t nocc, int_t rank, int_t *occs) {
    int_t i, j, k, binom;
    for (i = 0; i < nocc; ++i) {
        j = nocc - i;
        binom = binomial(nbasis, j);
        if (binom <= rank) {
            for (k = 0; k < j; ++k)
                occs[k] = k;
            break;
        }
        while (binom > rank)
            binom = binomial(--nbasis, j);
        occs[j - 1] = nbasis;
        rank -= binom;
    }
}

int_t phase_single_det(const int_t nword, const int_t i, const int_t a, const uint_t *det) {
    int_t j, k, l, m, n, high, low, nperm = 0;
    uint_t *mask = new (std::nothrow) uint_t[nword];
    if (i > a) {
        high = i;
        low = a;
    } else {
        high = a;
        low = i;
    }
    k = high / PYCI_UINT_SIZE;
    m = high % PYCI_UINT_SIZE;
    j = low / PYCI_UINT_SIZE;
    n = low % PYCI_UINT_SIZE;
    for (l = j; l < k; ++l)
        mask[l] = PYCI_UINT_MAX;
    mask[k] = (PYCI_UINT_ONE << m) - 1;
    mask[j] &= ~(PYCI_UINT_ONE << (n + 1)) + 1;
    for (l = j; l <= k; ++l)
        nperm += PYCI_POPCNT(det[l] & mask[l]);
    delete[] mask;
    return (nperm % 2) ? -1 : 1;
}

int_t phase_double_det(const int_t nword, const int_t i1, const int_t i2, const int_t a1,
                       const int_t a2, const uint_t *det) {
    int_t j, k, l, m, n, high, low, nperm = 0;
    uint_t *mask = new (std::nothrow) uint_t[nword];
    // first excitation
    if (i1 > a1) {
        high = i1;
        low = a1;
    } else {
        high = a1;
        low = i1;
    }
    k = high / PYCI_UINT_SIZE;
    m = high % PYCI_UINT_SIZE;
    j = low / PYCI_UINT_SIZE;
    n = low % PYCI_UINT_SIZE;
    for (l = j; l < k; ++l)
        mask[l] = PYCI_UINT_MAX;
    mask[k] = (PYCI_UINT_ONE << m) - 1;
    mask[j] &= ~(PYCI_UINT_ONE << (n + 1)) + 1;
    for (l = j; l <= k; ++l)
        nperm += PYCI_POPCNT(det[l] & mask[l]);
    // second excitation
    if (i2 > a2) {
        high = i2;
        low = a2;
    } else {
        high = a2;
        low = i2;
    }
    k = high / PYCI_UINT_SIZE;
    m = high % PYCI_UINT_SIZE;
    j = low / PYCI_UINT_SIZE;
    n = low % PYCI_UINT_SIZE;
    for (l = j; l < k; ++l)
        mask[l] = PYCI_UINT_MAX;
    mask[k] = (PYCI_UINT_ONE << m) - 1;
    mask[j] &= ~(PYCI_UINT_ONE << (n + 1)) + 1;
    for (l = j; l <= k; ++l)
        nperm += PYCI_POPCNT(det[l] & mask[l]);
    // order excitations properly
    if ((i2 < a1) || (i1 > a2))
        ++nperm;
    delete[] mask;
    return (nperm % 2) ? -1 : 1;
}

int_t popcnt_det(const int_t nword, const uint_t *det) {
    int_t popcnt = 0;
    for (int_t i = 0; i < nword; ++i)
        popcnt += PYCI_POPCNT(det[i]);
    return popcnt;
}

int_t ctz_det(const int_t nword, const uint_t *det) {
    uint_t word;
    for (int_t i = 0; i < nword; ++i) {
        word = det[i];
        if (word)
            return PYCI_CTZ(word) + i * PYCI_UINT_SIZE;
    }
    return 0;
}

Wfn::Wfn(const Wfn &wfn)
    : nbasis(wfn.nbasis), nocc(wfn.nocc), nocc_up(wfn.nocc_up), nocc_dn(wfn.nocc_dn),
      nvir(wfn.nvir), nvir_up(wfn.nvir_up), nvir_dn(wfn.nvir_dn), ndet(wfn.ndet), nword(wfn.nword),
      nword2(wfn.nword2), maxrank_up(wfn.maxrank_up), maxrank_dn(wfn.maxrank_dn), dets(wfn.dets),
      dict(wfn.dict) {
}

Wfn::Wfn(const Wfn &&wfn) noexcept
    : nbasis(std::exchange(wfn.nbasis, 0)), nocc(std::exchange(wfn.nocc, 0)),
      nocc_up(std::exchange(wfn.nocc_up, 0)), nocc_dn(std::exchange(wfn.nocc_dn, 0)),
      nvir(std::exchange(wfn.nvir, 0)), nvir_up(std::exchange(wfn.nvir_up, 0)),
      nvir_dn(std::exchange(wfn.nvir_dn, 0)), ndet(std::exchange(wfn.ndet, 0)),
      nword(std::exchange(wfn.nword, 0)), nword2(std::exchange(wfn.nword2, 0)),
      maxrank_up(std::exchange(wfn.maxrank_up, 0)), maxrank_dn(std::exchange(wfn.maxrank_dn, 0)),
      dets(std::move(wfn.dets)), dict(std::move(wfn.dict)) {
}

void Wfn::squeeze(void) {
    dets.shrink_to_fit();
}

void Wfn::init(const int_t nb, const int_t nu, const int_t nd) {
    if (nd < 0)
        throw std::runtime_error("nocc_dn is < 0");
    else if (nu < nd)
        throw std::runtime_error("nocc_up is < nocc_dn");
    else if (nb < nu)
        throw std::runtime_error("nbasis is < nocc_up");
    nbasis = nb;
    nocc = nu + nd;
    nocc_up = nu;
    nocc_dn = nd;
    nvir = nb * 2 - nu - nd;
    nvir_up = nb - nu;
    nvir_dn = nb - nd;
    ndet = 0;
    nword = nword_det(nb);
    nword2 = nword * 2;
    maxrank_up = binomial_cutoff(nb, nu);
    maxrank_dn = binomial_cutoff(nb, nd);
}

} // namespace pyci
