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

// Main functions

struct Saving {
  node_t i, j;
  double value;
};

vector<vector<node_t>> clarke_wright_cvrptw(const VRP &vrp) {

  size_t N = vrp.getSize();

  // --- Step 1: initial routes ---
  vector<vector<node_t>> routes;
  for(node_t i = 1; i < N; i++) {   // skip depot
    routes.push_back({i});
  }

  bool merged = true;

  // --- Step 2: greedy merge loop ---
  while(merged) {
    merged = false;

    double best_saving = -1e9;
    int best_ri = -1, best_rj = -1;
    vector<node_t> best_route;

    // --- recompute savings ---
    for(size_t ri = 0; ri < routes.size(); ri++) {
      if(routes[ri].empty()) continue;

      for(size_t rj = ri + 1; rj < routes.size(); rj++) {
        if(routes[rj].empty()) continue;

        auto &A = routes[ri];
        auto &B = routes[rj];

        node_t A_start = A.front();
        node_t A_end   = A.back();
        node_t B_start = B.front();
        node_t B_end   = B.back();

        // ----- Try A -> B -----
        {
          vector<node_t> merged_route = A;
          merged_route.insert(merged_route.end(), B.begin(), B.end());

          if(verify_single_route(vrp, merged_route)) {
            double saving =
              vrp.get_dist(DEPOT, A_end) +
              vrp.get_dist(DEPOT, B_start) -
              vrp.get_dist(A_end, B_start);

            if(saving > best_saving) {
              best_saving = saving;
              best_ri = ri;
              best_rj = rj;
              best_route = merged_route;
            }
          }
        }

        // ----- Try B -> A -----
        {
          vector<node_t> merged_route = B;
          merged_route.insert(merged_route.end(), A.begin(), A.end());

          if(verify_single_route(vrp, merged_route)) {
            double saving =
              vrp.get_dist(DEPOT, B_end) +
              vrp.get_dist(DEPOT, A_start) -
              vrp.get_dist(B_end, A_start);

            if(saving > best_saving) {
              best_saving = saving;
              best_ri = ri;
              best_rj = rj;
              best_route = merged_route;
            }
          }
        }
      }
    }

    // --- Apply best merge ---
    if(best_saving > 0 && best_ri != -1) {
      routes[best_ri] = best_route;
      routes[best_rj].clear();
      merged = true;
    }
    //print intermediate routes after each merge
    cout << "Intermediate Routes after merge:" << endl;
    for (size_t i = 0; i < routes.size(); ++i)
    {
      cout << "Route #" << i + 1 << ": ";
      for (size_t j = 0; j < routes[i].size(); ++j)
      {
        cout << routes[i][j] << " ";
      }
      cout << endl;
    }
  }

  // --- Step 3: remove empty routes ---
  vector<vector<node_t>> final_routes;
  for(auto &r : routes) {
    if(!r.empty())
      final_routes.push_back(r);
  }

  return final_routes;
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
  
  // Implementing the Clark and Wright Savings Algorithm for CVRPTW

  auto routes = clarke_wright_cvrptw(vrp);

  if(verify_route(vrp, routes)) {
    print_routes(routes);
    cout << "Total Cost: "
        << calculate_total_cost(vrp, routes)
        << endl;
  } else {
    cout << "INVALID SOLUTION" << endl;
  }





  // if(verify_route(vrp,best_routes)){
  //   cerr<<"File: "<<argv[1]<<" ";
  //   cerr<<"Pre-processing_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(pre_end - pre_start).count()*1.E-9)<<" s ";
  //   cerr<<"Route_Construction_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(mid_end - mid_start).count()*1.E-9)<<" s ";
  //   cerr<<"Total_Cost: "<<min_cost<<" ";
  //   cerr<<"Total_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(total_end - total_start).count()*1.E-9)<<" s ";
  //   cerr<<"Vehicle_Used: "<<best_routes.size()<<" ";
  //   cerr<<"VALID"<<endl;
  // }

  return 0;
}