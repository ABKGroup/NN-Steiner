#include "rmst.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(rmst, m) {
    m.doc() =
        "The python interface for rectilinear MST"; // optional
                                                    // docstring
    m.def("find_RMST", &find_RMST,
          "gets terminals and return steiner points and edges");
}