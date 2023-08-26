#pragma once
#include <vector>
#include <tuple>

std::vector<std::tuple<size_t, size_t>>
find_RMST(const std::vector<std::tuple<size_t, size_t>> &points);