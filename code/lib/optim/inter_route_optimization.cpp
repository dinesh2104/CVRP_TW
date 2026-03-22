#include "inter_route_optimization.h"

#include <vector>

#include "../route_utils.h"

using namespace std;

void inter_route_relocate(const VRP &vrp, vector<vector<node_t>> &routes) {
  bool improvement = true;

  while (improvement) {
    improvement = false;

    for (size_t r1 = 0; r1 < routes.size(); r1++) {
      for (size_t r2 = 0; r2 < routes.size(); r2++) {
        if (r1 == r2) {
          continue;
        }

        auto &routeA = routes[r1];
        auto &routeB = routes[r2];
        if (routeA.size() <= 2) {
          continue;
        }

        for (size_t i = 1; i < routeA.size() - 1; i++) {
          node_t u = routeA[i];
          node_t t = routeA[i - 1];
          node_t w = routeA[i + 1];

          double savings_A =
              vrp.get_dist(t, u) + vrp.get_dist(u, w) - vrp.get_dist(t, w);

          for (size_t j = 1; j < routeB.size(); j++) {
            node_t x = routeB[j - 1];
            node_t y = routeB[j];

            double cost_B =
                vrp.get_dist(x, u) + vrp.get_dist(u, y) - vrp.get_dist(x, y);
            double total_gain = savings_A - cost_B;

            if (total_gain > 1e-6) {
              vector<node_t> new_routeA = routeA;
              vector<node_t> new_routeB = routeB;

              new_routeA.erase(new_routeA.begin() + i);
              new_routeB.insert(new_routeB.begin() + j, u);

              if (verify_single_route(vrp, new_routeA) &&
                  verify_single_route(vrp, new_routeB)) {
                routeA = new_routeA;
                routeB = new_routeB;
                improvement = true;
                goto end_of_search;
              }
            }
          }
        }
      }
    }
  end_of_search:
    for (auto it = routes.begin(); it != routes.end();) {
      if (it->size() <= 2) {
        it = routes.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void inter_route_swap(const VRP &vrp, vector<vector<node_t>> &routes) {
  bool improvement = true;

  while (improvement) {
    improvement = false;

    for (size_t r1 = 0; r1 < routes.size(); r1++) {
      for (size_t r2 = r1 + 1; r2 < routes.size(); r2++) {
        auto &routeA = routes[r1];
        auto &routeB = routes[r2];

        if (routeA.size() <= 2 || routeB.size() <= 2) {
          continue;
        }

        for (size_t i = 1; i < routeA.size() - 1; i++) {
          node_t u = routeA[i];
          node_t t = routeA[i - 1];
          node_t w = routeA[i + 1];

          for (size_t j = 1; j < routeB.size() - 1; j++) {
            node_t v = routeB[j];
            node_t x = routeB[j - 1];
            node_t y = routeB[j + 1];

            double cost_before = vrp.get_dist(t, u) + vrp.get_dist(u, w) +
                                 vrp.get_dist(x, v) + vrp.get_dist(v, y);
            double cost_after = vrp.get_dist(t, v) + vrp.get_dist(v, w) +
                                vrp.get_dist(x, u) + vrp.get_dist(u, y);
            double total_gain = cost_before - cost_after;

            if (total_gain > 1e-6) {
              double current_load_A = vrp.get_route_load(routeA);
              double current_load_B = vrp.get_route_load(routeB);

              double new_load_A =
                  current_load_A - vrp.node[u].demand + vrp.node[v].demand;
              double new_load_B =
                  current_load_B - vrp.node[v].demand + vrp.node[u].demand;

              if (new_load_A <= vrp.getCapacity() &&
                  new_load_B <= vrp.getCapacity()) {
                vector<node_t> new_routeA = routeA;
                vector<node_t> new_routeB = routeB;

                new_routeA[i] = v;
                new_routeB[j] = u;

                if (verify_single_route(vrp, new_routeA) &&
                    verify_single_route(vrp, new_routeB)) {
                  routeA = new_routeA;
                  routeB = new_routeB;
                  improvement = true;
                  goto end_of_search;
                }
              }
            }
          }
        }
      }
    }
  end_of_search:;
  }
}

void inter_route_2opt_star(const VRP &vrp, vector<vector<node_t>> &routes) {
  bool improvement = true;

  while (improvement) {
    improvement = false;

    for (size_t r1 = 0; r1 < routes.size(); r1++) {
      for (size_t r2 = r1 + 1; r2 < routes.size(); r2++) {
        auto &routeA = routes[r1];
        auto &routeB = routes[r2];

        if (routeA.size() <= 2 || routeB.size() <= 2) {
          continue;
        }

        for (size_t i = 0; i < routeA.size() - 1; i++) {
          node_t t = routeA[i];
          node_t u = routeA[i + 1];

          for (size_t j = 0; j < routeB.size() - 1; j++) {
            node_t x = routeB[j];
            node_t v = routeB[j + 1];

            double cost_before = vrp.get_dist(t, u) + vrp.get_dist(x, v);
            double cost_after = vrp.get_dist(t, v) + vrp.get_dist(x, u);
            double total_gain = cost_before - cost_after;

            if (total_gain > 1e-6) {
              vector<node_t> new_routeA;
              vector<node_t> new_routeB;

              new_routeA.reserve(routeA.size() + routeB.size());
              new_routeB.reserve(routeA.size() + routeB.size());

              new_routeA.insert(new_routeA.end(), routeA.begin(), routeA.begin() + i + 1);
              new_routeA.insert(new_routeA.end(), routeB.begin() + j + 1, routeB.end());

              new_routeB.insert(new_routeB.end(), routeB.begin(), routeB.begin() + j + 1);
              new_routeB.insert(new_routeB.end(), routeA.begin() + i + 1, routeA.end());

              double new_load_A = vrp.get_route_load(new_routeA);
              double new_load_B = vrp.get_route_load(new_routeB);

              if (new_load_A <= vrp.getCapacity() &&
                  new_load_B <= vrp.getCapacity()) {
                if (verify_single_route(vrp, new_routeA) &&
                    verify_single_route(vrp, new_routeB)) {
                  routeA = new_routeA;
                  routeB = new_routeB;
                  improvement = true;
                  goto end_of_search;
                }
              }
            }
          }
        }
      }
    }
  end_of_search:
    for (auto it = routes.begin(); it != routes.end();) {
      if (it->size() <= 2) {
        it = routes.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void updated_relocate(const VRP &vrp, vector<vector<node_t>> &routes) {
  bool improvement = true;

  while (improvement) {
    improvement = false;

    for (size_t r1 = 0; r1 < routes.size(); r1++) {
      for (size_t r2 = 0; r2 < routes.size(); r2++) {
        if (r1 == r2) {
          continue;
        }

        auto &routeA = routes[r1];
        auto &routeB = routes[r2];

        if (routeA.size() <= 2) {
          continue;
        }

        for (size_t i = 1; i < routeA.size() - 1; i++) {
          node_t u = routeA[i];
          node_t t = routeA[i - 1];
          node_t w = routeA[i + 1];

          double savings_A =
              vrp.get_dist(t, u) + vrp.get_dist(u, w) - vrp.get_dist(t, w);

          vector<node_t> new_routeA = routeA;
          new_routeA.erase(new_routeA.begin() + i);
          if (!verify_single_route(vrp, new_routeA)) {
            continue;
          }

          vector<node_t> best_routeB;
          double best_gain = 0.0;

          for (size_t j = 1; j < routeB.size(); j++) {
            node_t x = routeB[j - 1];
            node_t y = routeB[j];

            double cost_B =
                vrp.get_dist(x, u) + vrp.get_dist(u, y) - vrp.get_dist(x, y);
            double total_gain = savings_A - cost_B;

            if (total_gain > 1e-6 && total_gain > best_gain) {
              vector<node_t> new_routeB = routeB;
              new_routeB.insert(new_routeB.begin() + j, u);
              if (verify_single_route(vrp, new_routeB)) {
                best_gain = total_gain;
                best_routeB = new_routeB;
              }
            }
          }

          if (best_gain > 1e-6) {
            routeA = new_routeA;
            routeB = best_routeB;
            improvement = true;
            goto end_of_search;
          }
        }
      }
    }
  end_of_search:
    for (auto it = routes.begin(); it != routes.end();) {
      if (it->size() <= 2) {
        it = routes.erase(it);
      } else {
        ++it;
      }
    }
  }
}
