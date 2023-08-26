#include "rmst.hpp"
#include <iostream>
#include <fstream>

std::vector<std::tuple<size_t, size_t>> read_points(std::fstream& in) {
    std::vector<std::tuple<size_t, size_t>> ret;
    size_t num_points;
    in >> num_points;
    size_t x, y;
    for(size_t i = 0; i < num_points; ++i) {
        in >> x;
        in >> y;
        ret.push_back(std::make_tuple(x, y));
    }
    return ret;
}

bool hasDuplicates(const std::vector<std::tuple<size_t, size_t>>& points) {
    for (size_t i = 0; i < points.size(); i++) {
        for (size_t j = i + 1; j < points.size(); j++) {
            auto p_i = points[i];
            auto p_j = points[j];
            if ((std::get<0>(p_i) == std::get<0>(p_j)) && (std::get<1>(p_i) == std::get<1>(p_j))) {
                std::cerr << "Duplicated points (" << i << "," << j << "), with (x,y) = (" << std::get<0>(p_i) << "," << std::get<1>(p_i) << ")" << std::endl;
                return true;
            }
        }
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: ./dbg <filename>" << std::endl;
        return 1;
    }

    // read file
    std::fstream in;
    in.open(argv[1]);
    if(!in) {
        std::cerr << "file \"" << argv[1] << "\" does not exist." <<std::endl;
        return 1;
    }
    auto points = read_points(in);
    if(hasDuplicates(points)) {
        return 1;
    }
    auto edges = find_RMST(points);
    for(auto edge: edges) {
        std::cout << "(" << std::get<0>(edge) << "," << std::get<1>(edge) << ")";
    }
}