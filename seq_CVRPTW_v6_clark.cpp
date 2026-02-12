#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>  //stringstream

#include <random>
#include <chrono>  //timing CPU


using namespace std;

//~ Define types
using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int; 

using tw_t=unsigned int; // time window.

const node_t DEPOT = 0;  // CVRP depot is always assumed to be zero.

class Edge {
  public:
  node_t to;
  weight_t length;

  Edge() {}
  ~Edge() {}
  Edge(node_t t, weight_t l) {
    to = t;
    length = l;
  }
  bool operator<(const Edge &e) {
    return length < e.length;
  }
};

class Point {
  public:
  //~ int id; // may be needed later for SS.
  point_t x;
  point_t y;
  demand_t demand;

  //adding the parameter for the time window
  tw_t earlyTime;
  tw_t latestTime;   // earliest time to start service
  tw_t serviceTime;  // service time required
  Point() {}
};

// To Hold the contents input.vrp
class VRP {
  size_t size;
  demand_t capacity;
  string type;
  

  public:
  VRP() {}
  ~VRP() {}
  unsigned read(string filename);
  void print();

  void print_dist();

  std::vector<std::vector<Edge>> cal_graph_dist();
  //~ std::vector<std::vector<Edge>> PrimsAlgo();
  weight_t get_dist(node_t i, node_t j) const {
    if (i == j)
      return 0.0;
    node_t temp;
    if (i > j) {
      temp = i;
      i = j;
      j = temp;
    }

    size_t myoffset = ((2 * i * size) - (i * i) + i) / 2;
    size_t correction = 2 * i + 1;
    return dist[myoffset + j - correction];
  }

  public:
  vector<Point> node;
  vector<weight_t> dist;

  size_t getSize() const {
    return size;
  }
  demand_t getCapacity() const {
    return capacity;
  }
};

std::vector<std::vector<Edge>>
VRP::cal_graph_dist() {
  //std::cout<< "size:" << (size*(size-1))/2 << '\n';

  dist.resize((size * (size - 1)) / 2);  //n \choose 2. i.e n(n-1)/2

  std::vector<std::vector<Edge>> nG(size);

  size_t k = 0;
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = i + 1; j < size; ++j) {
      // if(i < j){
      //~ printf("%zd %zd: (%lf-%lf)^2 - (%lf-%lf)^2\n",i,j,node[i].x, node[j].x,node[i].y, node[j].y);
      weight_t w = sqrt(((node[i].x - node[j].x) * (node[i].x - node[j].x)) + ((node[i].y - node[j].y) * (node[i].y - node[j].y)));

      dist[k] = w;  //not to round.

      nG[i].push_back(Edge(j, w));
      nG[j].push_back(Edge(i, w));
      //~ printf("k=%zd d[%zd][%zd]=%lf\n",k,i,j,w);
      k++;
      //   }
    }
  }
  //~ cout << "k = " << k << endl;
  return nG;
}

void VRP::print_dist() {
  for (size_t i = 0; i < size; ++i) {
    std::cout << i << ":";
    for (size_t j = 0; j < size; ++j) {
      cout << setw(10) << get_dist(i, j) << ' ';
    }
    std::cout << std::endl;
  }
}

unsigned VRP::read(string filename) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Could not open the file \"" << filename << "\"" << endl;
        exit(1);
    }

    string line;

    // --- Read file name (like C101) ---
    getline(in, line);
    string file_name = line;
    cout << "filename: " << file_name << endl;
    getline(in, line); // "COMMENT :  ..."
    // --- Skip VEHICLE line ---
    getline(in, line); // "VEHICLE"

    // --- Read NUMBER CAPACITY header ---
    getline(in, line); // "NUMBER     CAPACITY"
    // --- Read NUMBER and CAPACITY values ---
    int numVehicles;
    in >> numVehicles >> capacity;
    cout << "Vehicles: " << numVehicles << ", Capacity: " << capacity << endl;

    // --- Skip CUSTOMER header lines ---
    getline(in, line); // finish previous line
    getline(in, line); // "CUSTOMER"
    getline(in, line); // "CUST NO.  XCOORD.  ..."
    getline(in, line); // "----------  ------  ..."
    // --- Read customers until EOF ---

    int id;
    double x, y, demand, ready, due, service;
    while (in >> id >> x >> y >> demand >> ready >> due >> service) {
        Point p; 
        p.x = x;
        p.y = y;
        p.demand = demand;
        p.earlyTime = ready;
        p.latestTime = due;
        p.serviceTime = service;
        node.push_back(p);
    }

    size = node.size();
    cout << "Total customers : " << size << endl;
    in.close();
    return capacity;
}


// To print and check if read it okay.
void VRP::print() {
  std::cout << "DIMENSION:" << size << '\n';
  std::cout << "CAPACITY:" << capacity << '\n';
  for (auto i = 0u; i < size; ++i) {
    std::cout << i << ':'
              << setw(6) << node[i].x << ' '
              << setw(6) << node[i].y << ' '
              << setw(6) << node[i].demand << std::endl;
  }
}

// Helper functions for route calculations

double calculate_route_distance(const VRP &vrp,const std::vector<node_t> &route) {
  double total_distance=0.0;
  if(route.empty()) return total_distance;

  // From depot to first customer
  total_distance+=vrp.get_dist(DEPOT,route[0]);

  // Between customers
  for(size_t i=1;i<route.size();i++){
    total_distance+=vrp.get_dist(route[i-1],route[i]);
  }

  // From last customer back to depot
  total_distance+=vrp.get_dist(route[route.size()-1],DEPOT);

  return total_distance;
}

double calculate_total_cost(const VRP &vrp,const std::vector<std::vector<node_t>> &routes) {
  double total_cost=0.0;
  for(auto route:routes){
    total_cost+=calculate_route_distance(vrp,route);
  }
  return total_cost;
}

bool verify_route(const VRP &vrp,const std::vector<std::vector<node_t>> &routes) {
  demand_t vCapacity = vrp.getCapacity();
  for(auto route:routes){
    demand_t residueCap = vCapacity;
    tw_t process_time=0;
    node_t prev=0;
    for(auto v:route){
      process_time+=(vrp.get_dist(prev,v)); // from prev to v
      if(residueCap - vrp.node[v].demand >= 0 && process_time<=vrp.node[v].latestTime){  // can add to current route
        residueCap = residueCap - vrp.node[v].demand;
        process_time=max(process_time,vrp.node[v].earlyTime) + vrp.node[v].serviceTime;
        prev=v;
      }else{
        return false;
      }
    }
  }
  return true;
}

bool verify_single_route(const VRP &vrp,const std::vector<node_t> &route) {
  demand_t vCapacity = vrp.getCapacity();
  demand_t residueCap = vCapacity;
  tw_t process_time=0;
  node_t prev=0;
  for(auto v:route){
    process_time+=(vrp.get_dist(prev,v)); // from prev to v
    if(residueCap - vrp.node[v].demand >= 0 && process_time<=vrp.node[v].latestTime){  // can add to current route
      residueCap = residueCap - vrp.node[v].demand;
      process_time=max(process_time,vrp.node[v].earlyTime) + vrp.node[v].serviceTime;
      prev=v;
    }else{
      return false;
    }
  }
  return true;
}

// Print the output routes.

void print_routes(const std::vector<std::vector<node_t>> &routes) {
  cout << "Final Routes:" << endl;
  for (size_t i = 0; i < routes.size(); ++i) {
    cout << "Route #" << i + 1 << ": ";
    for (size_t j = 0; j < routes[i].size(); ++j) {
      cout << routes[i][j] << " ";
    }
    cout << endl;
  }

}

void save_routes_snapshot(const vector<vector<node_t>> &routes,
                          const string &filename)
{
    ofstream file(filename);

    for (const auto &route : routes)
    {
        if (route.empty()) continue;

        file << "0 ";   // depot
        for (auto node : route)
            file << node << " ";
        file << "0\n";  // return to depot
    }

    file.close();
}


// Main functions

struct Saving {
  node_t i, j;
  double value;
};

vector<vector<node_t>> clarke_wright_cvrptw(const VRP &vrp)
{
    size_t N = vrp.getSize();

    vector<vector<node_t>> routes;
    vector<demand_t> route_demand;
    vector<int> node_to_route(N, -1);

    // ---- Initial routes ----
    for (node_t i = 1; i < N; i++)
    {
        routes.push_back({i});
        route_demand.push_back(vrp.node[i].demand);
        node_to_route[i] = routes.size() - 1;
    }

    // ---- Compute savings ----
    vector<Saving> savings;

    for (node_t i = 1; i < N; i++)
    {
        for (node_t j = i + 1; j < N; j++)
        {
            weight_t s =
                vrp.get_dist(DEPOT, i) +
                vrp.get_dist(DEPOT, j) -
                vrp.get_dist(i, j);

            savings.push_back({i, j, s});
        }
    }

    sort(savings.begin(), savings.end(),
         [](const Saving &a, const Saving &b)
         { return a.value > b.value; });

    int step = 0; // for debugging and visualization

    // ---- Merge ----
    for (const auto &s : savings)
    {
        node_t i = s.i;
        node_t j = s.j;

        int r_i = node_to_route[i];
        int r_j = node_to_route[j];

        if (r_i == r_j)
            continue;

        // Capacity constraint
        if (route_demand[r_i] + route_demand[r_j] > vrp.getCapacity())
            continue;

        auto route_i = routes[r_i];  // copy
        auto route_j = routes[r_j];  // copy

        bool i_start = (route_i.front() == i);
        bool i_end   = (route_i.back() == i);

        bool j_start = (route_j.front() == j);
        bool j_end   = (route_j.back() == j);

        if (!(i_start || i_end)) continue;
        if (!(j_start || j_end)) continue;

        vector<node_t> merged;

        if (i_end && j_start)
        {
            merged = route_i;
            merged.insert(merged.end(), route_j.begin(), route_j.end());
        }
        else if (i_start && j_end)
        {
            merged = route_j;
            merged.insert(merged.end(), route_i.begin(), route_i.end());
        }
        else if (i_end && j_end)
        {
            reverse(route_j.begin(), route_j.end());
            merged = route_i;
            merged.insert(merged.end(), route_j.begin(), route_j.end());
        }
        else if (i_start && j_start)
        {
            reverse(route_i.begin(), route_i.end());
            merged = route_i;
            merged.insert(merged.end(), route_j.begin(), route_j.end());
        }
        else
            continue;

        if (!verify_single_route(vrp, merged))
            continue;

        
        // ---- Accept merge ----
        routes[r_i] = merged;
        route_demand[r_i] += route_demand[r_j];

        for (node_t node : routes[r_j])
            node_to_route[node] = r_i;

        routes[r_j].clear();
        route_demand[r_j] = 0;
        
        save_routes_snapshot(routes, "snap/step_" + to_string(step++) + ".csv");

        // print the routes after each merge for debugging
        // cout << "Merged Route: " << i << " and " << j << " -> ";
        // print_routes(routes);  
    }

    // ---- Collect final routes ----
    vector<vector<node_t>> final_routes;

    for (size_t i = 0; i < routes.size(); i++)
        if (!routes[i].empty())
            final_routes.push_back(routes[i]);

    return final_routes;
}

// Post-Optimization function ..............

bool verify_tour_t(const VRP &vrp,const std::vector<node_t> &tour, node_t ncities) {
  tw_t process_time=0;
  for(int i=1;i<ncities;i++){
    process_time+=vrp.get_dist(tour[i-1],tour[i]); // in minutes
    if(process_time>vrp.node[tour[i]].latestTime){
      return false;
    }
    process_time=max(process_time,vrp.node[tour[i]].earlyTime) + vrp.node[tour[i]].serviceTime;
  }
  return true;
}

bool verify_route_t(const VRP &vrp,const std::vector<std::vector<node_t>> &routes) {
  demand_t vCapacity = vrp.getCapacity();
  for(auto route:routes){
    demand_t residueCap = vCapacity;
    tw_t process_time=0;
    node_t prev=0;
    for(auto v:route){
      process_time+=(vrp.get_dist(prev,v)); // from prev to v
      if(residueCap - vrp.node[v].demand >= 0 && process_time<=vrp.node[v].latestTime){  // can add to current route
        residueCap = residueCap - vrp.node[v].demand;
        process_time=max(process_time,vrp.node[v].earlyTime) + vrp.node[v].serviceTime;
        prev=v;
      }else{
        return false;
      }
    }
  }
  return true;
}

double calculate_tour_distance_t(const VRP &vrp,const std::vector<node_t> &tour, node_t ncities) {
  double total_distance=0.0;
  for(int i=1;i<ncities;i++){
    total_distance+=vrp.get_dist(tour[i-1],tour[i]);
  }
  total_distance+=vrp.get_dist(tour[ncities-1],tour[0]);
  return total_distance;
}


void tsp_approx(const VRP &vrp, std::vector<node_t> &cities, std::vector<node_t> &tour, node_t ncities) {
  node_t i, j;
  node_t ClosePt = 0;
  weight_t CloseDist;
  //~ node_t endtour=0;

  for (i = 1; i < ncities; i++)
    tour[i] = cities[i - 1];

  tour[0] = cities[ncities - 1];
  
  double bestDistance=calculate_tour_distance_t(vrp,tour,ncities);
  
  for (i = 1; i < ncities; i++) {
    //~ double ThisX = points.x_coords[tour[i-1]];
    //~ double ThisY = points.y_coords[tour[i-1]];
    weight_t ThisX = vrp.node[tour[i - 1]].x;
    weight_t ThisY = vrp.node[tour[i - 1]].y;
    CloseDist = DBL_MAX;
    for (j = ncities - 1;; j--) {
      weight_t ThisDist = (vrp.node[tour[j]].x - ThisX) * (vrp.node[tour[j]].x - ThisX);
      if (ThisDist <= CloseDist) {
        ThisDist += (vrp.node[tour[j]].y - ThisY) * (vrp.node[tour[j]].y - ThisY);
        if (ThisDist <= CloseDist) {
          if (j < i)
            break;
          CloseDist = ThisDist;
          ClosePt = j;
        }
      }
    }
    /*swapping tour[i] and tour[ClosePt]*/
    unsigned temp = tour[i];
    tour[i] = tour[ClosePt];
    tour[ClosePt] = temp;

    double newDistance=calculate_tour_distance_t(vrp,tour,ncities);
    if(newDistance<bestDistance && verify_tour_t(vrp,tour,ncities)==true){
      cout<<"TSP Approx Improvement: "<<bestDistance<<" to "<<newDistance<<endl;
      bestDistance=newDistance;
    }else{
      //revert the swap
      temp = tour[i];
      tour[i] = tour[ClosePt];
      tour[ClosePt] = temp;
    }

  }
  // // verify if the tour is valid with respect to time windows and if invalid tour then revert the changes...
  // if(verify_tour(vrp,tour,ncities)==false){
  //   //cout<<"Reverting TSP Approximation as tour invalid"<<endl;
  //   for(int i=1;i<ncities;i++){
  //     tour[i]=cities[i-1];
  //   }
  //   tour[0]=cities[ncities-1];
  // }
}

std::vector<std::vector<node_t>>
postprocess_tsp_approx(const VRP &vrp, std::vector<std::vector<node_t>> &solRoutes) {
  std::vector<std::vector<node_t>> modifiedRoutes;

  unsigned nroutes = solRoutes.size();
  for (unsigned i = 0; i < nroutes; ++i) {
    // postprocessing solRoutes[i]
    unsigned sz = solRoutes[i].size();

    std::vector<node_t> cities(sz + 1);
    std::vector<node_t> tour(sz + 1);

    for (unsigned j = 0; j < sz; ++j)
      cities[j] = solRoutes[i][j];

    cities[sz] = 0;  // the last node is the depot.

    tsp_approx(vrp, cities, tour, sz + 1);

    // the first element of the tour is now the depot. So, ignore tour[0] and insert the rest into the vector.

    vector<node_t> curr_route;
    for (unsigned kk = 1; kk < sz + 1; ++kk) {
      curr_route.push_back(tour[kk]);
    }
    modifiedRoutes.push_back(curr_route);
  }
  return modifiedRoutes;
}

void tsp_2opt(const VRP &vrp, std::vector<node_t> &cities, std::vector<node_t> &tour, unsigned ncities) {
  // cities: customer-only vector of length ncities
  // tour: aux array length ncities

  unsigned improve = 0;

  while (improve < 2) {
    double best_distance = 0.0;

    best_distance += vrp.get_dist(DEPOT, cities[0]);
    for (unsigned jj = 1; jj < ncities; ++jj)
      best_distance += vrp.get_dist(cities[jj - 1], cities[jj]);
    best_distance += vrp.get_dist(cities[ncities - 1], DEPOT);

    for (unsigned i = 0; i < ncities - 1; ++i) {
      for (unsigned k = i + 1; k < ncities; ++k) {

        // prefix [0..i-1]
        for (unsigned c = 0; c < i; ++c)
          tour[c] = cities[c];

        // reversed segment [i..k]
        unsigned dec = 0;
        for (unsigned c = i; c <= k; ++c) {
          tour[c] = cities[k - dec];
          ++dec;
        }

        // suffix [k+1..ncities-1]
        for (unsigned c = k + 1; c < ncities; ++c)
          tour[c] = cities[c];

        // compute new distance (with depot legs)
        double new_distance = 0.0;
        new_distance += vrp.get_dist(DEPOT, tour[0]);
        for (unsigned jj = 1; jj < ncities; ++jj)
          new_distance += vrp.get_dist(tour[jj - 1], tour[jj]);
        new_distance += vrp.get_dist(tour[ncities - 1], DEPOT);

        // Build a temp tour WITH depot for verification (if verifier expects depot at index 0)
        std::vector<node_t> tmp_tour_with_depot;
        tmp_tour_with_depot.reserve(ncities + 1);
        tmp_tour_with_depot.push_back(DEPOT);            // depot at pos 0
        for (unsigned t = 0; t < ncities; ++t) tmp_tour_with_depot.push_back(tour[t]);

        // Call verify_tour with correct size (ncities + 1) and appropriate flag
        if (new_distance < best_distance && verify_tour_t(vrp, tmp_tour_with_depot, ncities + 1)) {
          std::cout << "2OPT Improvement: " << best_distance << " to " << new_distance << std::endl;
          improve = 0;
          for (unsigned jj = 0; jj < ncities; ++jj)
            cities[jj] = tour[jj];
          best_distance = new_distance;
        }
      }
    }
    ++improve;
  }
}


std::vector<std::vector<node_t>>
postprocess_2OPT(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes) {
  std::vector<std::vector<node_t>> postprocessed_final_routes;

  unsigned nroutes = final_routes.size();
  for (unsigned i = 0; i < nroutes; ++i) {
    // postprocessing final_routes[i]
    unsigned sz = final_routes[i].size();

    std::vector<node_t> cities(sz);
    std::vector<node_t> tour(sz);

    for (unsigned j = 0; j < sz; ++j)
      cities[j] = final_routes[i][j];

    vector<node_t> curr_route;

    if (sz > 2)                         // for sz <= 1, the cost of the path cannot change. So no point running this.
      tsp_2opt(vrp, cities, tour, sz);  //MAIN

    for (unsigned kk = 0; kk < sz; ++kk) {
      curr_route.push_back(cities[kk]);
    }

    postprocessed_final_routes.push_back(curr_route);
  }
  return postprocessed_final_routes;
}


std::vector<std::vector<node_t>>
postProcessIt(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes, weight_t& minCost) {
  std::vector<std::vector<node_t>> postprocessed_final_routes;

  auto postprocessed_final_routes1 = postprocess_tsp_approx(vrp, final_routes);
  if(verify_route_t(vrp,postprocessed_final_routes1)){
    cout<<"\nPostprocess 1 route valid"<<endl;
  }else{
    cout<<"\nPostprocess 1 route invalid"<<endl;
  }
  auto postprocessed_final_routes2 = postprocess_2OPT(vrp, postprocessed_final_routes1);
  if(verify_route_t(vrp,postprocessed_final_routes2)){
    cout<<"Postprocess 2 route valid"<<endl;
  }else{
    cout<<"Postprocess 2 route invalid"<<endl;
  }
  auto postprocessed_final_routes3 = postprocess_2OPT(vrp, final_routes);

  weight_t postprocessed_final_routes_cost = 0;
  
  if(verify_route_t(vrp,postprocessed_final_routes3)){
    cout<<"Postprocess 3 route valid"<<endl;
  }else{
    cout<<"Postprocess 3 route invalid"<<endl;
  }

  for (unsigned zzz = 0; zzz < final_routes.size(); ++zzz) {
    // include the better route between postprocessed_final_routes2[zzz] and postprocessed_final_routes3[zzz] in the final solution.

    vector<node_t> postprocessed_route2 = postprocessed_final_routes2[zzz];
    vector<node_t> postprocessed_route3 = postprocessed_final_routes3[zzz];

    unsigned sz2 = postprocessed_route2.size();
    unsigned sz3 = postprocessed_route3.size();

    // finding the cost of postprocessed_route2

    weight_t postprocessed_route2_cost = 0.0;

    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[0]);  // computing distance of the first point in the route with the depot.
    for (unsigned jj = 1; jj < sz2; ++jj) {
      postprocessed_route2_cost += vrp.get_dist(postprocessed_route2[jj - 1], postprocessed_route2[jj]);
    }

    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[sz2 - 1]);

    // finding the cost of postprocessed_route3

    weight_t postprocessed_route3_cost = 0.0;

    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[0]);
    for (unsigned jj = 1; jj < sz3; ++jj) {
      postprocessed_route3_cost += vrp.get_dist(postprocessed_route3[jj - 1], postprocessed_route3[jj]);
    }

    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[sz3 - 1]);

    // postprocessed_route2_cost is lower
    if (postprocessed_route3_cost > postprocessed_route2_cost) {
      postprocessed_final_routes_cost += postprocessed_route2_cost;
      postprocessed_final_routes.push_back(postprocessed_route2);
    }
    // postprocessed_route3_cost is lower
    else {
      postprocessed_final_routes_cost += postprocessed_route3_cost;
      postprocessed_final_routes.push_back(postprocessed_route3);
    }
  }

  minCost = postprocessed_final_routes_cost;
  return postprocessed_final_routes;
}


// print the maximum length of the route in the solution. This is useful for understanding the longest route and can be helpful for debugging or analysis.
int max_length_of_route(const std::vector<std::vector<node_t>> &routes) {
  size_t max_length = 0;
  for (const auto &route : routes) {
    if (route.size() > max_length) {
      max_length = route.size();
    }
  }
  return max_length;
}





int main(int argc, char *argv[]) {
VRP vrp;
  if (argc < 2) {
    std::cout << "seqCVRPTW version 3" << '\n';
    std::cout << "Usage: " << argv[0] << " toy.vrp" << '\n';
    exit(1);
  }

  vrp.read(argv[1]);
  vrp.cal_graph_dist();

  chrono::steady_clock::time_point mid_start = chrono::steady_clock::now();
  chrono::steady_clock::time_point total_start = chrono::steady_clock::now();
  
  // Implementing the Clark and Wright Savings Algorithm for CVRPTW
  auto routes = clarke_wright_cvrptw(vrp);
  
  chrono::steady_clock::time_point mid_end = chrono::steady_clock::now();

  weight_t min_cost = calculate_total_cost(vrp, routes);
  weight_t min_cost1=min_cost;
  auto best_routes = routes;
  
  chrono::steady_clock::time_point post_start = chrono::steady_clock::now();

  weight_t post_optimized_cost=min_cost;
  best_routes=postProcessIt(vrp,best_routes,post_optimized_cost);

  chrono::steady_clock::time_point post_end = chrono::steady_clock::now();
  chrono::steady_clock::time_point total_end = chrono::steady_clock::now();

  min_cost=calculate_total_cost(vrp,best_routes);
  print_routes(best_routes);
  if(verify_route(vrp,best_routes)){
    cerr<<"File: "<<argv[1]<<" ";
    cerr<<"Route_Construction_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(mid_end - mid_start).count()*1.E-9)<<" s ";
    cerr<<"Post_Optimization_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(post_end - post_start).count()*1.E-9)<<" s ";
    cerr<<"Total_Cost-1: "<<min_cost1<<" ";
    cerr<<"Total_Cost-2: "<<min_cost<<" ";
    cerr<<"Total_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(total_end - total_start).count()*1.E-9)<<" s ";
    cerr<<"Vehicle_Used: "<<best_routes.size()<<" ";
    cerr<<"route_length: "<<max_length_of_route(best_routes)<<" ";
    cerr<<"VALID"<<endl;
  }
  return 0;
}