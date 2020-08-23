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

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pyci.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef PYCI_VERSION
#define PYCI_VERSION 0.0.0
#endif
#define LITERAL(S) #S
#define STRINGIZE(S) LITERAL(S)

namespace py = pybind11;

using namespace pyci;

/* Pybind11 typedefs. */

typedef typename py::array_t<int_t, py::array::c_style | py::array::forcecast> i_array_t;

typedef typename py::array_t<uint_t, py::array::c_style | py::array::forcecast> u_array_t;

typedef typename py::array_t<double, py::array::c_style | py::array::forcecast> d_array_t;

struct PyHam final : public Ham {
public:
    d_array_t one_mo_array, two_mo_array, h_array, v_array, w_array;

    PyHam(const PyHam &ham)
        : Ham(ham), one_mo_array(ham.one_mo_array), two_mo_array(ham.two_mo_array),
          h_array(ham.h_array), v_array(ham.v_array), w_array(ham.w_array) {
    }

    PyHam(PyHam &&ham) noexcept
        : Ham(ham), one_mo_array(std::move(ham.one_mo_array)),
          two_mo_array(std::move(ham.two_mo_array)), h_array(std::move(ham.h_array)),
          v_array(std::move(ham.v_array)), w_array(std::move(ham.w_array)) {
    }

    PyHam(const std::string &filename) : Ham() {
        py::tuple args = py::module::import("pyci.fcidump").attr("_load_ham")(filename);
        init(args);
    }

    PyHam(const double e, const d_array_t &mo1, const d_array_t &mo2) : Ham() {
        py::tuple args = py::module::import("pyci.fcidump").attr("_load_ham")(mo1, mo2);
        init(args);
    }

    void to_file(const std::string &filename, const int_t nelec, const int_t ms2) const {
        py::module::import("pyci.fcidump")
            .attr("write_fcidump")(filename, Ham::ecore, one_mo_array, two_mo_array, nelec, ms2);
    }

private:
    void init(const py::tuple &args) {
        one_mo_array = args[1].cast<d_array_t>();
        two_mo_array = args[2].cast<d_array_t>();
        h_array = args[3].cast<d_array_t>();
        v_array = args[4].cast<d_array_t>();
        w_array = args[5].cast<d_array_t>();
        Ham::nbasis = one_mo_array.request().shape[0];
        Ham::ecore = args[0].cast<double>();
        Ham::one_mo = reinterpret_cast<double *>(one_mo_array.request().ptr);
        Ham::two_mo = reinterpret_cast<double *>(two_mo_array.request().ptr);
        Ham::h = reinterpret_cast<double *>(h_array.request().ptr);
        Ham::v = reinterpret_cast<double *>(v_array.request().ptr);
        Ham::w = reinterpret_cast<double *>(w_array.request().ptr);
    }
};

int_t wfn_length(const Wfn &wfn) {
    return wfn.ndet;
}

u_array_t onespinwfn_getitem(const OneSpinWfn &wfn, int_t index) {
    u_array_t array(wfn.nword);
    wfn.copy_det(index, reinterpret_cast<uint_t *>(array.request().ptr));
    return array;
}

u_array_t twospinwfn_getitem(const TwoSpinWfn &wfn, int_t index) {
    u_array_t array({static_cast<int_t>(2), wfn.nword});
    wfn.copy_det(index, reinterpret_cast<uint_t *>(array.request().ptr));
    return array;
}

u_array_t onespinwfn_to_det_array(const OneSpinWfn &wfn, int_t start, int_t end) {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = wfn.ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    u_array_t array({end - start, wfn.nword});
    wfn.to_occ_array(start, end, reinterpret_cast<int_t *>(array.request().ptr));
    return array;
}

i_array_t onespinwfn_to_occ_array(const OneSpinWfn &wfn, int_t start, int_t end) {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = wfn.ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    i_array_t array({end - start, wfn.nocc_up});
    wfn.to_occ_array(start, end, reinterpret_cast<int_t *>(array.request().ptr));
    return array;
}

u_array_t twospinwfn_to_det_array(const TwoSpinWfn &wfn, int_t start, int_t end) {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = wfn.ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    u_array_t array({end - start, static_cast<int_t>(2), wfn.nword});
    wfn.to_occ_array(start, end, reinterpret_cast<int_t *>(array.request().ptr));
    return array;
}

i_array_t twospinwfn_to_occ_array(const TwoSpinWfn &wfn, int_t start, int_t end) {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = wfn.ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    i_array_t array({end - start, static_cast<int_t>(2), wfn.nocc_up});
    wfn.to_occ_array(start, end, reinterpret_cast<int_t *>(array.request().ptr));
    return array;
}

int_t onespinwfn_index_det(const OneSpinWfn &wfn, const u_array_t det) {
    return wfn.index_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

int_t twospinwfn_index_det(const TwoSpinWfn &wfn, const u_array_t det) {
    return wfn.index_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

uint_t onespinwfn_rank_det(const OneSpinWfn &wfn, const u_array_t det) {
    return wfn.rank_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

uint_t twospinwfn_rank_det(const TwoSpinWfn &wfn, const u_array_t det) {
    return wfn.rank_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

int_t onespinwfn_add_det(OneSpinWfn &wfn, const u_array_t det) {
    return wfn.add_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

int_t twospinwfn_add_det(TwoSpinWfn &wfn, const u_array_t det) {
    return wfn.add_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

void onespinwfn_add_excited_dets(OneSpinWfn &wfn, const int_t exc, const u_array_t det) {
    wfn.add_excited_dets(reinterpret_cast<const uint_t *>(det.request().ptr), exc);
}

void twospinwfn_add_excited_dets(TwoSpinWfn &wfn, const int_t exc, const u_array_t det) {
    wfn.add_excited_dets(reinterpret_cast<const uint_t *>(det.request().ptr), exc, exc);
}

int_t onespinwfn_add_occs(OneSpinWfn &wfn, const i_array_t occs) {
    return wfn.add_det_from_occs(reinterpret_cast<const int_t *>(occs.request().ptr));
}

int_t twospinwfn_add_occs(TwoSpinWfn &wfn, const i_array_t occs) {
    return wfn.add_det_from_occs(reinterpret_cast<const int_t *>(occs.request().ptr));
}

/* Python C extension. */

PYBIND11_MODULE(pyci, m) {

    /*
    Section: Initialization
    */

    m.doc() = "PyCI C extension module.";

    m.attr("__version__") = STRINGIZE(PYCI_VERSION);
    m.attr("c_int") = py::dtype::of<int_t>();
    m.attr("c_uint") = py::dtype::of<uint_t>();
    m.attr("c_double") = py::dtype::of<double>();

    if (std::getenv("OMP_NUM_THREADS") == nullptr)
        omp_set_num_threads(1);

    /*
    Section: Hamiltonian class
    */

    py::class_<PyHam> hamiltonian(m, "hamiltonian");
    hamiltonian.doc() = "Hamiltonian class.";
    hamiltonian.def_readonly("nbasis", &Ham::nbasis);
    hamiltonian.def_readonly("ecore", &Ham::ecore);
    hamiltonian.def_readonly("one_mo", &PyHam::one_mo_array);
    hamiltonian.def_readonly("two_mo", &PyHam::two_mo_array);
    hamiltonian.def_readonly("h", &PyHam::h_array);
    hamiltonian.def_readonly("v", &PyHam::v_array);
    hamiltonian.def_readonly("w", &PyHam::w_array);

    hamiltonian.def(py::init<const std::string &>(), py::arg("filename"));

    hamiltonian.def(py::init<const double, const d_array_t &, const d_array_t &>(),
                    py::arg("ecore"), py::arg("one_mo"), py::arg("two_mo"));

    hamiltonian.def("to_file", &PyHam::to_file, py::arg("filename"), py::arg("nelec") = 0,
                    py::arg("ms2") = 0);

    /*
    Section: Wavefunction class
    */

    py::class_<Wfn> wavefunction(m, "wavefunction");
    wavefunction.doc() = "Wave function class.";

    wavefunction.def_readonly("nbasis", &Wfn::nbasis);
    wavefunction.def_readonly("nocc", &Wfn::nocc);
    wavefunction.def_readonly("nocc_up", &Wfn::nocc_up);
    wavefunction.def_readonly("nocc_dn", &Wfn::nocc_dn);
    wavefunction.def_readonly("nvir", &Wfn::nvir);
    wavefunction.def_readonly("nvir_up", &Wfn::nvir_up);
    wavefunction.def_readonly("nvir_dn", &Wfn::nvir_dn);

    wavefunction.def("__len__", &wfn_length);

    wavefunction.def("squeeze", &Wfn::squeeze);

    /*
    Section: One-spin wavefunction class
    */

    py::class_<OneSpinWfn, Wfn> one_spin_wfn(m, "one_spin_wfn");
    one_spin_wfn.doc() = "Single-spin wave function class.";

    wavefunction.def("__getitem__", &onespinwfn_getitem, py::arg("index"));

    one_spin_wfn.def("to_file", &OneSpinWfn::to_file, py::arg("filename"));

    one_spin_wfn.def("to_det_array", &onespinwfn_to_det_array, py::arg("low") = -1,
                     py::arg("high") = -1);

    one_spin_wfn.def("to_occ_array", &onespinwfn_to_occ_array, py::arg("low") = -1,
                     py::arg("high") = -1);

    one_spin_wfn.def("index_det", &onespinwfn_index_det, py::arg("det"));

    one_spin_wfn.def("index_det_from_rank", &OneSpinWfn::index_det_from_rank, py::arg("rank"));

    one_spin_wfn.def("rank_det", &onespinwfn_rank_det, py::arg("det"));

    one_spin_wfn.def("add_det", &onespinwfn_add_det, py::arg("det"));

    one_spin_wfn.def("add_occs", &onespinwfn_add_occs, py::arg("occs"));

    one_spin_wfn.def("add_hartreefock_det", &OneSpinWfn::add_hartreefock_det);

    one_spin_wfn.def("add_all_dets", &OneSpinWfn::add_all_dets);

    one_spin_wfn.def("add_excited_dets", &onespinwfn_add_excited_dets, py::arg("ref"),
                     py::arg("exc"));

    one_spin_wfn.def("add_dets_from_wfn", &OneSpinWfn::add_dets_from_wfn, py::arg("wfn"));

    one_spin_wfn.def("reserve", &OneSpinWfn::reserve, py::arg("n"));

} // PYBIND_MODULE
