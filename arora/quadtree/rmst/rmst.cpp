#include "rmst.hpp"
#include <assert.h>
#include <iostream>

extern "C" {
#include "global.h"
#include "mst2.h"
}

// return list of point connections
std::vector<std::tuple<size_t, size_t>>
find_RMST(const std::vector<std::tuple<size_t, size_t>> &points) {
    std::vector<std::tuple<size_t, size_t>> ret;
    Point *pt = new Point[points.size()];
    long *parent = new long[points.size()];
    for (size_t i = 0; i < points.size(); ++i) {
        auto point = points[i];
        pt[i].x = std::get<0>(point);
        pt[i].y = std::get<1>(point);
    }

    mst2_package_init(points.size());
    mst2(points.size(), pt, parent);
    mst2_package_done();
    for (size_t i = 0; i < points.size(); ++i) {
        long p = parent[i];
        if (i != p) {
            ret.push_back(std::make_tuple(i, p));
        }
    }
    assert(ret.size() == points.size() - 1);
    // delete[] pt;
    // delete[] parent;
    return ret;
}