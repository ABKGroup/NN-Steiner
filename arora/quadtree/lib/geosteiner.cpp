
#include <tuple>
#include <vector>

#include <iostream>

extern "C" {
#include "geosteiner-5.3/geosteiner.h"
}
#define CPLEX 40
#define IL_STD 1


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

std::tuple<std::vector<std::tuple<int, int>>, std::vector<std::tuple<int, int>>>
find_RSMT_GEO(const std::vector<std::tuple<int, int>> &terminals,
              int fst) {
    if (gst_open_geosteiner()) {
        std::cout << "Failed to open geosteiner 5.3 " << std::endl;
        abort();
    }

    // int status;
    // CPXENVptr env;
    // env = CPXopenCPLEX(&status);
    // gst_attach_cplex(env);

    // inputs
    // number of terminals
    int nterms = terminals.size();
    // input points
    double *terms = new double[2 * nterms];
    int cnt = 0;
    for (const auto &pt : terminals) {
        terms[cnt++] = std::get<0>(pt);
        terms[cnt++] = std::get<1>(pt);
    }

    // outputs
    double length;
    // number of steiner points
    int nsps;
    // steiner points ... but how many?
    double *sps = new double[nterms*2];
    // number of edges
    int nedges;
    // 2 * nedges
    int *edges = new int[4 * nterms];

    gst_param_ptr myParam = gst_create_param(NULL);
    if (fst > 2) {
        gst_set_int_param(myParam, GST_PARAM_MAX_FST_SIZE, fst);
    }
    gst_rsmt(nterms, terms, &length, &nsps, sps, &nedges, edges, NULL, myParam);

    // generate steiner pts
    std::vector<std::tuple<int, int>> steiners;
    for (int i = 0; i < nsps; i++) {
        double x = sps[2 * i], y = sps[2 * i + 1];
        steiners.push_back(std::make_tuple(x, y));
    }

    // point idx >= nedges is steiner point
    std::vector<std::tuple<int, int>> edge_list;
    for (int i = 0; i < nedges; i++) {
        int src_idx = edges[2 * i], tgt_idx = edges[2 * i + 1];
        edge_list.push_back(std::make_tuple(src_idx, tgt_idx));
    }

    // clean
    gst_close_geosteiner();
    // have to free all memories
    // delete[] terms;
    // delete[] sps;
    // delete[] edges;

    return std::make_tuple(steiners, edge_list);
}

PYBIND11_MODULE(geosteiner, m) {
    m.doc() = "The python interface for geosteiner"; // optional module docstring
    // auto package = pybind11::module::import("goesteiner");
    // auto module = package.attr("module");
    // m.add_object("module", module);
    m.def("find_RSMT_GEO", &find_RSMT_GEO, "gets terminals and return steiner points and edges");
}