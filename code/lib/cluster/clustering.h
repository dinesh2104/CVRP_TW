#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>

#include "../vrp.h"

std::vector<std::vector<int>> clustering_sweep(const VRP &vrp);
std::vector<std::vector<int>> clustering_angle_sweep(const VRP &vrp,
                                                     double angle_range);

std::vector<std::vector<int>> clustering_angle_sweep_parallel(
    const VRP &vrp, double angle_range, int num_trials = 100);

#endif
