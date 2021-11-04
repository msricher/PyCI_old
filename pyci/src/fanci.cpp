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

FanCI::FanCI(const FanCI &other)
    : nbasis(other.nbasis), nocc_up(other.nocc_up), nocc_dn(other.nocc_dn), nparam(other.nparam),
      nequation(other.nequation), nproj(other.nproj), nconn(other.nconn), op(other.op) {
    return;
}

FanCI::FanCI(FanCI &&other) noexcept
    : nbasis(std::exchange(other.nbasis, 0)), nocc_up(std::exchange(other.nocc_up, 0)),
      nocc_dn(std::exchange(other.nocc_dn, 0)), nparam(std::exchange(other.nparam, 0)),
      nequation(std::exchange(other.nequation, 0)), nproj(std::exchange(other.nproj, 0)),
      nconn(std::exchange(other.nconn, 0)), op(std::move(other.op)) {
    return;
}

FanCI::FanCI(const Ham &ham, const DOCIWfn &wfn, const long nparam_, const long nequation_,
             const long nproj_)
    : nbasis(wfn.nbasis), nocc_up(wfn.nocc_up), nocc_dn(wfn.nocc_dn), nparam(nparam_),
      nequation(nequation_), nproj(nproj_), nconn(wfn.ndet), op(0, 0, false) {
    if (nequation_ < nparam_ || wfn.ndet < nproj_)
        throw std::runtime_error("Not enough determinants");
    op.update(ham, wfn, nproj, nconn, 0);
    op.squeeze();
}

FanCI::FanCI(const Ham &ham, const FullCIWfn &wfn, const long nparam_, const long nequation_,
             const long nproj_)
    : nbasis(wfn.nbasis), nocc_up(wfn.nocc_up), nocc_dn(wfn.nocc_dn), nparam(nparam_),
      nequation(nequation_), nproj(nproj_), nconn(wfn.ndet), op(0, 0, false) {
    if (nequation_ < nparam_ || wfn.ndet < nproj_)
        throw std::runtime_error("Not enough determinants");
    op.update(ham, wfn, nproj, nconn, 0);
    op.squeeze();
}

} // namespace pyci
