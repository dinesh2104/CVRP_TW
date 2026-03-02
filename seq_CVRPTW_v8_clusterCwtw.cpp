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

  demand_t get_route_load(const vector<node_t> &route) const {
    demand_t load = 0.0;
    for (auto node : route) {
      load += this->node[node].demand;
    }
    return load;
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

double compute_waiting_time(const VRP &vrp,
                            const vector<node_t> &route)
{
    double time = 0.0;
    double total_wait = 0.0;

    node_t prev = 0; // depot

    for (node_t node : route)
    {
        time += vrp.get_dist(prev, node);

        if (time < vrp.node[node].earlyTime)
        {
            total_wait +=
                vrp.node[node].earlyTime - time;
            time = vrp.node[node].earlyTime;
        }

        time += vrp.node[node].serviceTime;
        prev = node;
    }

    return total_wait;
}


vector<vector<int>> clustering_kmedoid(VRP vrp,int k){
  int n=vrp.getSize()-1;
  vector<int> medoids_id;
  // Randomly select k medoids
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dis(1, n); // assuming customer IDs are from 1 to n

  while(medoids_id.size()<k){
    int m_id=dis(gen);
    if(find(medoids_id.begin(),medoids_id.end(),m_id)==medoids_id.end()){
      medoids_id.push_back(m_id);
    }
  }

  cout<<"Initial Medoids: ";
  for(auto m:medoids_id){
    cout<<m<<" ";
  }
  cout<<endl;

  bool changed=true;
  vector<vector<int>> clusters(k);
  while(changed){
    changed=false;
    // Assignment Step
    clusters.clear();
    clusters.resize(k);
    for(int i=1;i<=n;i++){
      weight_t min_dist=DBL_MAX;
      int assigned_cluster=-1;
      for(int j=0;j<k;j++){
        weight_t dist=vrp.get_dist(i,medoids_id[j]);
        if(dist<min_dist){
          min_dist=dist;
          assigned_cluster=j;
        }
      }
      clusters[assigned_cluster].push_back(i);
    }

    // Update Step
    for(int j=0;j<k;j++){
      weight_t min_total_dist=DBL_MAX;
      int new_medoid=-1;
      for(auto candidate:clusters[j]){
        weight_t total_dist=0.0;
        for(auto point:clusters[j]){
          total_dist+=vrp.get_dist(candidate,point);
        }
        if(total_dist<min_total_dist){
          min_total_dist=total_dist;
          new_medoid=candidate;
        }
      }
      if(new_medoid!=medoids_id[j]){
        medoids_id[j]=new_medoid;
        changed=true;
      }
    }
  }

  // Output final clusters
  for(int j=0;j<k;j++){
    cout<<"Cluster "<<j+1<<" (Medoid: "<<medoids_id[j]<<"): ";
    for(auto customer:clusters[j]){
      cout<<customer<<" ";
    }
    cout<<endl;
  }
  return clusters;
}

vector<vector<node_t>>
clarke_wright_cvrptw(const VRP &vrp,
                     const vector<vector<int>> &clusters)
{
    double alpha = 0.7;
    double beta  = 0.3;

    vector<vector<node_t>> final_routes;

    // ======================================================
    // Helper functions
    // ======================================================
    auto compute_arrival_time = [&](const vector<node_t> &route)
    {
        double time = 0.0;
        node_t prev = 0;

        for (node_t node : route)
        {
            time += vrp.get_dist(prev, node);

            if (time < vrp.node[node].earlyTime)
                time = vrp.node[node].earlyTime;

            if (time > vrp.node[node].latestTime)
                return -1.0;

            time += vrp.node[node].serviceTime;
            prev = node;
        }

        return time;
    };

    auto verify_route = [&](const vector<node_t> &route)
    {
        return compute_arrival_time(route) >= 0;
    };

    // ======================================================
    // Process each cluster independently
    // ======================================================
    int step=0;
    for (const auto &cluster : clusters)
    {
        vector<vector<node_t>> routes;
        vector<double> route_demand;

        // ---- Initialize single-customer routes ----
        for (auto node : cluster)
        {
            routes.push_back({node});
            route_demand.push_back(vrp.node[node].demand);
        }

        // ==================================================
        // Clarke–Wright inside this cluster only
        // ==================================================
        while (true)
        {
            double best_saving = -1e18;
            int best_i = -1, best_j = -1;
            vector<node_t> best_merge;
            int from_;
            int to_;

            for (size_t r_i = 0; r_i < routes.size(); r_i++)
            {
                if (routes[r_i].empty()) continue;

                for (size_t r_j = r_i + 1; r_j < routes.size(); r_j++)
                {
                    if (routes[r_j].empty()) continue;

                    if (route_demand[r_i] + route_demand[r_j] >
                        vrp.getCapacity())
                        continue;

                    auto &Ri = routes[r_i];
                    auto &Rj = routes[r_j];

                    node_t i1 = Ri.front();
                    node_t i2 = Ri.back();
                    node_t j1 = Rj.front();
                    node_t j2 = Rj.back();

                    double arrival_i_end = compute_arrival_time(Ri);
                    double arrival_j_end = compute_arrival_time(Rj);

                    if (arrival_i_end < 0 || arrival_j_end < 0)
                        continue;

                    struct Candidate {
                        node_t from;
                        node_t to;
                        vector<node_t> merged;
                    };

                    vector<Candidate> candidates;

                    // i2 -> j1
                    {
                        vector<node_t> merged = Ri;
                        merged.insert(merged.end(),
                                      Rj.begin(), Rj.end());
                        candidates.push_back({i2, j1, merged});
                    }

                    // j2 -> i1
                    {
                        vector<node_t> merged = Rj;
                        merged.insert(merged.end(),
                                      Ri.begin(), Ri.end());
                        candidates.push_back({j2, i1, merged});
                    }

                    // i1 -> j1
                    {
                        vector<node_t> Ri_rev = Ri;
                        reverse(Ri_rev.begin(), Ri_rev.end());
                        vector<node_t> merged = Ri_rev;
                        merged.insert(merged.end(),
                                      Rj.begin(), Rj.end());
                        candidates.push_back({i1, j1, merged});
                    }

                    // i2 -> j2
                    {
                        vector<node_t> Rj_rev = Rj;
                        reverse(Rj_rev.begin(), Rj_rev.end());
                        vector<node_t> merged = Ri;
                        merged.insert(merged.end(),
                                      Rj_rev.begin(), Rj_rev.end());
                        candidates.push_back({i2, j2, merged});
                    }

                    for (auto &cand : candidates)
                    {
                        node_t from = cand.from;
                        node_t to   = cand.to;

                        double arrival_from =
                            compute_arrival_time(
                                (from == i2 || from == i1) ? Ri : Rj
                            );

                        if (arrival_from < 0) continue;

                        double dist_saving =
                            vrp.get_dist(0, from) +
                            vrp.get_dist(0, to) -
                            vrp.get_dist(from, to);

                        double arrival_to =
                            arrival_from +
                            vrp.get_dist(from, to);

                        double waiting = 0.0;
                        if (arrival_to < vrp.node[to].earlyTime)
                            waiting =
                                vrp.node[to].earlyTime -
                                arrival_to;

                        double total_saving =
                            alpha * dist_saving
                            - beta * waiting;

                        if (!verify_route(cand.merged))
                            continue;

                        if (total_saving > best_saving)
                        {
                            best_saving = total_saving;
                            best_i = r_i;
                            best_j = r_j;
                            from_=from;
                            to_=to;
                            best_merge = cand.merged;
                        }
                    }
                }
            }

            if (best_saving <= 0)
                break;

            routes[best_i] = best_merge;
            route_demand[best_i] += route_demand[best_j];

            routes[best_j].clear();
            route_demand[best_j] = 0;
            // cout<<"Merged "<<from_<<" to "<<to_<<endl;
            save_routes_snapshot(routes, "snap/step_" + to_string(step++) + ".csv");

        }

        // Add cluster routes to final result
        for (auto &r : routes)
            if (!r.empty())
                final_routes.push_back(r);
    }

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

// Post optimization - Inter Route Relocate

// --- The Relocate Algorithm ---
void inter_route_relocate(const VRP &vrp, vector<vector<node_t>> &routes) {
    bool improvement = true;

    // Keep running until we reach a Local Optimum (no more improvements possible)
    while (improvement) {
        improvement = false;

        // Iterate through all possible pairs of routes
        for (size_t r1 = 0; r1 < routes.size(); r1++) {
            for (size_t r2 = 0; r2 < routes.size(); r2++) {
                
                if (r1 == r2) continue; // Don't relocate within the same route here

                auto &routeA = routes[r1]; // Source Route
                auto &routeB = routes[r2]; // Destination Route

                // Skip if Route A is practically empty (e.g., just [Depot, Depot])
                if (routeA.size() <= 2) continue; 

                // Iterate through every customer 'u' in Route A (skip depots at ends)
                for (size_t i = 1; i < routeA.size() - 1; i++) {
                    node_t u = routeA[i];
                    node_t t = routeA[i - 1]; // Predecessor in A
                    node_t w = routeA[i + 1]; // Successor in A

                    // 1. Calculate the distance saved by removing 'u' from Route A
                    // Distance saved = (t->u) + (u->w) - (t->w)
                    double savings_A = vrp.get_dist(t, u) + vrp.get_dist(u, w) - vrp.get_dist(t, w);

                    // Iterate through every possible insertion point 'j' in Route B
                    for (size_t j = 1; j < routeB.size(); j++) {
                        node_t x = routeB[j - 1]; // Predecessor in B
                        node_t y = routeB[j];     // Successor in B

                        // 2. Calculate the distance cost of inserting 'u' between x and y
                        // Distance added = (x->u) + (u->y) - (x->y)
                        double cost_B = vrp.get_dist(x, u) + vrp.get_dist(u, y) - vrp.get_dist(x, y);

                        // Net change in total distance
                        double total_gain = savings_A - cost_B;

                        // 3. Fast Pruning (Only proceed if the move actually saves distance)
                        if (total_gain > 1e-6) {
                            
                            // Create copies of the routes to test the move
                            vector<node_t> new_routeA = routeA;
                            vector<node_t> new_routeB = routeB;

                            // Perform the physical relocation on the copies
                            new_routeA.erase(new_routeA.begin() + i);
                            new_routeB.insert(new_routeB.begin() + j, u);

                            // 4. Strict Constraint Verification (Capacity & Time Windows)
                            // This is the O(N) step, which is why we only do it if total_gain > 0
                            if (verify_single_route(vrp, new_routeA) && verify_single_route(vrp, new_routeB)) {
                                
                                // The move is valid and improves the cost! Apply it permanently.
                                cout<<"Relocating customer "<<u<<" from route "<<r1<<" to route "<<r2<<" at position "<<j<<endl;
                                routeA = new_routeA;
                                routeB = new_routeB;
                                
                                improvement = true;
                                
                                // First-Improvement strategy: restart the search immediately 
                                // because route structures have changed.
                                goto end_of_search; 
                            }
                        }
                    }
                }
            }
        }
        end_of_search:;
        
        // --- Cleanup Phase ---
        // If a route was completely emptied by relocations, remove it from the list
        for (auto it = routes.begin(); it != routes.end(); ) {
            if (it->size() <= 2) { 
                it = routes.erase(it);
            } else {
                ++it;
            }
        }
    }
}

// --- The Swap Algorithm ---

void inter_route_swap(const VRP &vrp, vector<vector<node_t>> &routes) {
    bool improvement = true;

    // Continue until Local Optimum is reached
    while (improvement) {
        improvement = false;

        // Iterate through unique pairs of routes
        // r2 starts at r1 + 1 to avoid duplicate checks and self-swaps
        for (size_t r1 = 0; r1 < routes.size(); r1++) {
            for (size_t r2 = r1 + 1; r2 < routes.size(); r2++) {
                
                auto &routeA = routes[r1];
                auto &routeB = routes[r2];

                // Skip routes that only contain the Depot
                if (routeA.size() <= 2 || routeB.size() <= 2) continue;

                // Iterate through every customer 'u' in Route A (skip depots)
                for (size_t i = 1; i < routeA.size() - 1; i++) {
                    node_t u = routeA[i];
                    node_t t = routeA[i - 1]; // Predecessor of u
                    node_t w = routeA[i + 1]; // Successor of u

                    // Iterate through every customer 'v' in Route B (skip depots)
                    for (size_t j = 1; j < routeB.size() - 1; j++) {
                        node_t v = routeB[j];
                        node_t x = routeB[j - 1]; // Predecessor of v
                        node_t y = routeB[j + 1]; // Successor of v

                        // 1. Fast O(1) Distance Delta Calculation
                        double cost_before = vrp.get_dist(t, u) + vrp.get_dist(u, w) + 
                                             vrp.get_dist(x, v) + vrp.get_dist(v, y);
                        
                        double cost_after = vrp.get_dist(t, v) + vrp.get_dist(v, w) + 
                                            vrp.get_dist(x, u) + vrp.get_dist(u, y);

                        double total_gain = cost_before - cost_after;

                        // 2. Fast Pruning: Only proceed if distance actually improves
                        if (total_gain > 1e-6) {
                            
                            // 3. Quick Capacity Check
                            // Route A gains 'v' and loses 'u'. Route B gains 'u' and loses 'v'.
                            double current_load_A = vrp.get_route_load(routeA);
                            double current_load_B = vrp.get_route_load(routeB);
                            
                            double new_load_A = current_load_A - vrp.node[u].demand + vrp.node[v].demand;
                            double new_load_B = current_load_B - vrp.node[v].demand + vrp.node[u].demand;

                            if (new_load_A <= vrp.getCapacity() && new_load_B <= vrp.getCapacity()) {
                                
                                // 4. Strict Time Window Verification (O(N) Step)
                                vector<node_t> new_routeA = routeA;
                                vector<node_t> new_routeB = routeB;

                                // Perform the physical swap on the candidate copies
                                new_routeA[i] = v;
                                new_routeB[j] = u;

                                if (verify_single_route(vrp, new_routeA) && verify_single_route(vrp, new_routeB)) {
                                    
                                    // Apply the valid and improving move
                                    cout<<"Swapping customer "<<u<<" in route "<<r1<<" with customer "<<v<<" in route "<<r2<<endl;
                                    routeA = new_routeA;
                                    routeB = new_routeB;
                                    
                                    improvement = true;
                                    
                                    // First-Improvement strategy: break out and restart
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

// --Inter Route 2-Opt* Algorithm (Cross-Exchange)--

void inter_route_2opt_star(const VRP &vrp, vector<vector<node_t>> &routes) {
    bool improvement = true;

    // Run until Local Optimum is reached
    while (improvement) {
        improvement = false;

        // Iterate through unique pairs of routes
        for (size_t r1 = 0; r1 < routes.size(); r1++) {
            for (size_t r2 = r1 + 1; r2 < routes.size(); r2++) {
                
                auto &routeA = routes[r1];
                auto &routeB = routes[r2];

                // Skip routes that only contain the Depot
                if (routeA.size() <= 2 || routeB.size() <= 2) continue;

                // Iterate through edges in Route A. 
                // Edge is (t, u) where t is at index i, u is at index i+1
                for (size_t i = 0; i < routeA.size() - 1; i++) {
                    node_t t = routeA[i];
                    node_t u = routeA[i + 1];

                    // Iterate through edges in Route B.
                    // Edge is (x, v) where x is at index j, v is at index j+1
                    for (size_t j = 0; j < routeB.size() - 1; j++) {
                        node_t x = routeB[j];
                        node_t v = routeB[j + 1];

                        // 1. Fast O(1) Distance Delta Calculation
                        double cost_before = vrp.get_dist(t, u) + vrp.get_dist(x, v);
                        double cost_after  = vrp.get_dist(t, v) + vrp.get_dist(x, u);

                        double total_gain = cost_before - cost_after;

                        // 2. Fast Pruning: Only proceed if distance improves
                        if (total_gain > 1e-6) {
                            
                            // 3. Construct the Candidate Routes
                            // New Route A: Depot -> ... -> t -> v -> ... -> Depot
                            // New Route B: Depot -> ... -> x -> u -> ... -> Depot
                            
                            vector<node_t> new_routeA;
                            vector<node_t> new_routeB;
                            
                            // Reserve space to avoid reallocation overhead
                            new_routeA.reserve(routeA.size() + routeB.size());
                            new_routeB.reserve(routeA.size() + routeB.size());

                            // Build New Route A: [Start of A up to t] + [Tail of B from v]
                            new_routeA.insert(new_routeA.end(), routeA.begin(), routeA.begin() + i + 1);
                            new_routeA.insert(new_routeA.end(), routeB.begin() + j + 1, routeB.end());

                            // Build New Route B: [Start of B up to x] + [Tail of A from u]
                            new_routeB.insert(new_routeB.end(), routeB.begin(), routeB.begin() + j + 1);
                            new_routeB.insert(new_routeB.end(), routeA.begin() + i + 1, routeA.end());

                            // 4. Quick Capacity Check (Optional but recommended)
                            // Skip the heavy O(N) time window check if the load clearly exceeds capacity
                            double new_load_A = vrp.get_route_load(new_routeA);
                            double new_load_B = vrp.get_route_load(new_routeB);

                            if (new_load_A <= vrp.getCapacity() && new_load_B <= vrp.getCapacity()) {
                                
                                // 5. Strict Time Window Verification (O(N) Step)
                                if (verify_single_route(vrp, new_routeA) && verify_single_route(vrp, new_routeB)) {
                                    
                                    // Move is valid and improves the cost! Apply it.
                                    cout<<"2-opt-Exchange"<<t<<"->"<<u<<"between"<<x<<"->"<<v<<endl;
                                    routeA = new_routeA;
                                    routeB = new_routeB;
                                    
                                    improvement = true;
                                    
                                    // First-Improvement strategy: break out and restart search
                                    goto end_of_search; 
                                }
                            }
                        }
                    }
                }
            }
        }
        end_of_search:;
        
        // --- Cleanup Phase ---
        // Clean up empty routes dynamically if tails were completely swapped 
        // in a way that left one route as just [Depot, Depot]
        for (auto it = routes.begin(); it != routes.end(); ) {
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
                
                if (r1 == r2) continue;

                auto &routeA = routes[r1];
                auto &routeB = routes[r2];

                if (routeA.size() <= 2) continue; 

                for (size_t i = 1; i < routeA.size() - 1; i++) {
                    node_t u = routeA[i];
                    node_t t = routeA[i - 1]; 
                    node_t w = routeA[i + 1]; 

                    // 1. Calculate savings from removing 'u'
                    double savings_A = vrp.get_dist(t, u) + vrp.get_dist(u, w) - vrp.get_dist(t, w);

                    // OPTIMIZATION: Create modified Route A ONCE
                    vector<node_t> new_routeA = routeA;
                    new_routeA.erase(new_routeA.begin() + i);
                    
                    // OPTIMIZATION: Verify Route A once. Removing a node usually 
                    // keeps it valid, but it's safe to check.
                    if (!verify_single_route(vrp, new_routeA)) continue;

                    vector<node_t> best_routeB;
                    double best_gain = 0.0;
                    
                    // Iterate through every possible insertion point 'j' in Route B
                    for (size_t j = 1; j < routeB.size(); j++) {
                        node_t x = routeB[j - 1]; 
                        node_t y = routeB[j];     

                        double cost_B = vrp.get_dist(x, u) + vrp.get_dist(u, y) - vrp.get_dist(x, y);
                        double total_gain = savings_A - cost_B;

                        // OPTIMIZATION: Check total_gain FIRST to short-circuit the heavy operations
                        if (total_gain > 1e-6 && total_gain > best_gain) {
                            
                            vector<node_t> new_routeB = routeB;
                            new_routeB.insert(new_routeB.begin() + j, u);

                            // Only verify Route B if it's the best distance so far
                            if (verify_single_route(vrp, new_routeB)) {
                                best_gain = total_gain;
                                best_routeB = new_routeB;
                            }
                        }
                    }
                    
                    // If a valid, improving move was found for 'u' in Route B
                    if (best_gain > 1e-6) {
                        cout << "Relocating customer " << u << " from route " << r1 << " to route " << r2 << endl;
                        
                        routeA = new_routeA;
                        routeB = best_routeB;
                        
                        improvement = true;
                        goto end_of_search;
                    }
                }
            }
        }
        end_of_search:;
        
        // --- Cleanup Phase ---
        for (auto it = routes.begin(); it != routes.end(); ) {
            if (it->size() <= 2) { 
                it = routes.erase(it);
            } else {
                ++it;
            }
        }
    }
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

  
  chrono::steady_clock::time_point total_start = chrono::steady_clock::now();
  chrono::steady_clock::time_point pre_start = chrono::steady_clock::now();
  //Pre-processing - CLustering.
  int n_clusters;
  int sum_demand=0;
  for(int i=1;i<vrp.getSize();i++){
    sum_demand+=vrp.node[i].demand;
  }
  n_clusters=sum_demand/vrp.getCapacity();


  vector<vector<node_t>> clusters;
  
  clusters=clustering_kmedoid(vrp,n_clusters); // you can change the number of clusters here. I have set it to 10 for now. You can experiment with different values and see how it affects the solution quality and runtime.
  chrono::steady_clock::time_point pre_end = chrono::steady_clock::now();

  chrono::steady_clock::time_point mid_start = chrono::steady_clock::now();
  // Implementing the Clark and Wright Savings Algorithm for CVRPTW
  auto routes = clarke_wright_cvrptw(vrp,clusters);
  
  chrono::steady_clock::time_point mid_end = chrono::steady_clock::now();

  
  print_routes(routes);

  //Adding 0....0 to the route.
  for(auto &route:routes){
    route.insert(route.begin(),DEPOT);
    route.push_back(DEPOT);
  }
  weight_t min_cost = calculate_total_cost(vrp, routes);
  weight_t min_cost1=min_cost;

  chrono::steady_clock::time_point post_start = chrono::steady_clock::now();
  cout<<"Total Distance: "<<calculate_total_cost(vrp,routes)<<endl;
  inter_route_relocate(vrp,routes);
  print_routes(routes);
  cout<<"Total Distance after inter_route_relocate: "<<calculate_total_cost(vrp, routes)<<endl;
  
  inter_route_swap(vrp,routes);
  print_routes(routes);

  cout<<"Total Distance after inter_route_swap: "<<calculate_total_cost(vrp, routes)<<endl;

  inter_route_2opt_star(vrp,routes);
  print_routes(routes);
  cout<<"Total Distance after inter_route_2opt_star: "<<calculate_total_cost(vrp, routes)<<endl;
  auto best_routes = routes;
  
  


  weight_t post_optimized_cost=min_cost;
  //best_routes=postProcessIt(vrp,best_routes,post_optimized_cost);

  chrono::steady_clock::time_point post_end = chrono::steady_clock::now();
  chrono::steady_clock::time_point total_end = chrono::steady_clock::now();

  min_cost=calculate_total_cost(vrp,best_routes);
  print_routes(best_routes);
  if(verify_route(vrp,best_routes)){
    cerr<<"File: "<<argv[1]<<" ";
    cerr<<"Preprocessing_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(pre_end - pre_start).count()*1.E-9)<<" s ";
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