#include "clustering.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <random>
#include <omp.h>
#include <iostream>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct PolarCustomer {
  int id;
  double angle;
  double demand;
};

vector<vector<int>> clustering_sweep(const VRP &vrp) {
  int n = vrp.getSize();
  vector<vector<int>> clusters;
  if (n <= 1) {
    return clusters;
  }

  double depot_x = vrp.node[0].x;
  double depot_y = vrp.node[0].y;
  double max_capacity = vrp.getCapacity();

  vector<PolarCustomer> sweep_list;
  sweep_list.reserve(n - 1);

  for (int i = 1; i < n; i++) {
    double dx = vrp.node[i].x - depot_x;
    double dy = vrp.node[i].y - depot_y;
    double angle = atan2(dy, dx);
    if (angle < 0) {
      angle += 2.0 * M_PI;
    }
    sweep_list.push_back({i, angle, vrp.node[i].demand});
  }

  sort(sweep_list.begin(), sweep_list.end(),
       [](const PolarCustomer &a, const PolarCustomer &b) {
         return a.angle < b.angle;
       });

  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dist(0, static_cast<int>(sweep_list.size()) - 1);
  int start_index = dist(gen);

  vector<int> current_cluster;
  double current_load = 0.0;
  int num_customers = static_cast<int>(sweep_list.size());

  for (int step = 0; step < num_customers; step++) {
    int actual_index = (start_index + step) % num_customers;
    const auto &cust = sweep_list[actual_index];

    if (current_load + cust.demand > max_capacity) {
      if (!current_cluster.empty()) {
        clusters.push_back(current_cluster);
      }
      current_cluster.clear();
      current_cluster.push_back(cust.id);
      current_load = cust.demand;
    } else {
      current_cluster.push_back(cust.id);
      current_load += cust.demand;
    }
  }

  if (!current_cluster.empty()) {
    clusters.push_back(current_cluster);
  }

  return clusters;
}

vector<vector<int>> clustering_angle_sweep(const VRP &vrp, double angle_range) {
  int n = vrp.getSize();
  vector<vector<int>> clusters;
  if (n <= 1) {
    return clusters;
  }

  double depot_x = vrp.node[0].x;
  double depot_y = vrp.node[0].y;

  vector<PolarCustomer> sweep_list;
  sweep_list.reserve(n - 1);

  for (int i = 1; i < n; i++) {
    double dx = vrp.node[i].x - depot_x;
    double dy = vrp.node[i].y - depot_y;
    double angle_rad = atan2(dy, dx);
    double angle_deg = angle_rad * (180.0 / M_PI);
    if (angle_deg < 0) {
      angle_deg += 360.0;
    }
    sweep_list.push_back({i, angle_deg, vrp.node[i].demand});
  }

  sort(sweep_list.begin(), sweep_list.end(),
       [](const PolarCustomer &a, const PolarCustomer &b) {
         return a.angle < b.angle;
       });

  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dist(0, static_cast<int>(sweep_list.size()) - 1);
  int start_index = dist(gen);

  vector<int> current_cluster;
  double current_sector_start_angle = -1.0;
  int num_customers = static_cast<int>(sweep_list.size());

  for (int step = 0; step < num_customers; step++) {
    int actual_index = (start_index + step) % num_customers;
    const auto &cust = sweep_list[actual_index];

    if (current_cluster.empty()) {
      current_cluster.push_back(cust.id);
      current_sector_start_angle = cust.angle;
      continue;
    }

    double angular_diff = cust.angle - current_sector_start_angle;
    if (angular_diff < 0) {
      angular_diff += 360.0;
    }

    if (angular_diff > angle_range) {
      clusters.push_back(current_cluster);
      current_cluster.clear();
      current_cluster.push_back(cust.id);
      current_sector_start_angle = cust.angle;
    } else {
      current_cluster.push_back(cust.id);
    }
  }

  if (!current_cluster.empty()) {
    clusters.push_back(current_cluster);
  }

  return clusters;
}

vector<vector<int>> clustering_angle_sweep_parallel(const VRP &vrp,
                                                    double angle_range,
                                                    int num_trials) {
    int n = vrp.getSize();
    if (n <= 1) return {};

    double depot_x = vrp.node[0].x;
    double depot_y = vrp.node[0].y;

    // 1. Pre-calculate and sort (Sequential, but very fast O(N log N))
    struct PolarCustomer { int id; double angle; double demand; };
    vector<PolarCustomer> sweep_list;
    sweep_list.reserve(n - 1);

    for (int i = 1; i < n; i++) {
        double dx = vrp.node[i].x - depot_x;
        double dy = vrp.node[i].y - depot_y;
        double angle_deg = atan2(dy, dx) * (180.0 / M_PI);
        if (angle_deg < 0) angle_deg += 360.0;
        sweep_list.push_back({i, angle_deg, vrp.node[i].demand});
    }

    sort(sweep_list.begin(), sweep_list.end(), [](const PolarCustomer &a, const PolarCustomer &b) {
        return a.angle < b.angle;
    });

    int num_customers = static_cast<int>(sweep_list.size());
    vector<vector<vector<int>>> trials(num_trials);

    #pragma omp parallel
    {
        mt19937 gen(omp_get_thread_num() + (unsigned int)time(NULL));
        uniform_int_distribution<> dist(0, num_customers - 1);

        #pragma omp for
        for (int t = 0; t < num_trials; t++) {
            int start_index = dist(gen);
            vector<vector<int>> local_clusters;
            vector<int> current_cluster;
            double current_sector_start_angle = -1.0;

            for (int step = 0; step < num_customers; step++) {
                int idx = (start_index + step) % num_customers;
                const auto &cust = sweep_list[idx];

                if (current_cluster.empty()) {
                    current_cluster.push_back(cust.id);
                    current_sector_start_angle = cust.angle;
                } else {
                    double diff = cust.angle - current_sector_start_angle;
                    if (diff < 0) diff += 360.0;

                    if (diff > angle_range) {
                        local_clusters.push_back(current_cluster);
                        current_cluster = {cust.id};
                        current_sector_start_angle = cust.angle;
                    } else {
                        current_cluster.push_back(cust.id);
                    }
                }
            }
            if (!current_cluster.empty()) local_clusters.push_back(current_cluster);
            trials[t] = std::move(local_clusters);
        }
    }

    auto get_size_variance = [](const vector<vector<int>>& clusters) {
        if (clusters.empty()) return 0.0;
        
        double mean = 0.0;
        for (const auto& c : clusters) mean += c.size();
        mean /= clusters.size();
        
        double variance = 0.0;
        for (const auto& c : clusters) {
            double diff = c.size() - mean;
            variance += (diff * diff);
        }
        return variance;
    };

    int best_idx = 0;
    for (int i = 1; i < num_trials; i++) {
        if (trials[i].size() < trials[best_idx].size()) {
            best_idx = i;
        } 
        else if (trials[i].size() == trials[best_idx].size()) {
            double current_variance = get_size_variance(trials[i]);
            double best_variance = get_size_variance(trials[best_idx]);
            
            if (current_variance > best_variance) {
                best_idx = i; 
                // cout << "Tie-breaker: Trial " << i << " has better balance (variance: " 
                //      << current_variance << ") than trial " << best_idx 
                //      << " (variance: " << best_variance << ")" << endl;
            }
        }
    }

    return trials[best_idx];
}
