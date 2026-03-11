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

unsigned DEBUGCODE = 0;
#define DEBUG if (DEBUGCODE)
int flag=1;

using namespace std;

//~ Define types
using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int;  // let's keep as int than unsigned. -1 is init. nodes ids 0 to n-1

using tw_t=unsigned int; // time window not used in this code.

const node_t DEPOT = 0;  // CVRP depot is always assumed to be zero.

// Cmdline params
class Params {
  public:
  Params() {
    toRound = 0;  // DEFAULT is round
    //~ nThreads = 20; // DEFAULT is 20 OMP threads
  }
  ~Params() {}

  bool toRound;
  //~ short nThreads;
};



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
  Params params;

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

//~ One time computation to compute distances between every pair of nodes.
//~ Decision to round or not round is actioned here
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

      dist[k] = (params.toRound ? round(w) : w);  //TO round or not to.

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

// Prints distance of every pair of nodes
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

void printOutput(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes) {
  weight_t total_cost = 0.0;

  cout<<"No of routes: "<<final_routes.size()<<endl;

  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    std::cout << "Route #" << ii + 1 << ":";
    for (unsigned jj = 0; jj < final_routes[ii].size(); ++jj) {
      std::cout << " " << final_routes[ii][jj];
    }
    std::cout << '\n';
  }

  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    weight_t curr_route_cost = 0;

    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
    //cout<< "From 0 to " << final_routes[ii][0] << ": " << vrp.get_dist(DEPOT, final_routes[ii][0]) << endl;

    for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj) {
      curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);

      //cout<< "From " << final_routes[ii][jj - 1] << " to " << final_routes[ii][jj] << ": " << vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]) << endl;

    }
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][final_routes[ii].size() - 1]);
    //cout<< "From " << final_routes[ii][final_routes[ii].size() - 1] << " to 0: " << vrp.get_dist(DEPOT, final_routes[ii][final_routes[ii].size() - 1]) << endl;

    total_cost += curr_route_cost;
  }

  std::cout << "Cost " << total_cost << std::endl;
}

// Prims's MST using STL set
std::vector<std::vector<Edge>>
PrimsAlgo(const VRP &vrp, std::vector<std::vector<Edge>> &graph) {
  auto N = graph.size();
  const node_t INIT = -1;
  //! std::cout<< "N "<< N << '\n';

  std::vector<weight_t> key(N, INT_MAX);
  std::vector<weight_t> toEdges(N, -1);
  std::vector<bool> visited(N, false);

  std::set<std::pair<weight_t, node_t>> active;  // holds value and vertex
  std::vector<std::vector<Edge>> nG(N);

  //! key[0] = INT_MAX;
  //! visited[0] = true; // incorrect to set here!
  node_t src = 0;
  key[src] = 0.0;
  active.insert({0.0, src});

  long long edge_cost = 0;

  while (active.size() > 0) {
    auto where = active.begin()->second;
    int cost= active.begin()->first;

    //! DEBUG std::cout << "picked " << where <<"\tsize"<< active.size()<< std::endl;
    active.erase(active.begin());
    if (visited[where]) {
      continue;
    }
    edge_cost += cost;
    visited[where] = true;
    for (Edge E : graph[where]) {
      if (!visited[E.to] && E.length < key[E.to]) {  //W[{where,E.to}]
        key[E.to] = E.length;                        //W[{where,E.to}]
        active.insert({key[E.to], E.to});
        //! DEBUG std::cout << key[E.to] <<" ~ " <<  E.to << std::endl;
        toEdges[E.to] = where;
      }
    }
  }

  //print the edge
  // for (node_t v = 0; v < N; ++v) {
  //   cout<< toEdges[v] << " - " << v << endl;
  // }

  cerr << "edge_cost: " << edge_cost << " ";

  //! std::vector < std::pair<int,int>> edges; // not used
  node_t u = 0;
  for (auto v : toEdges) {  // nice parallel code or made to parallel
    if (v != INIT) {
      //! int w = W[{u,v}];
      weight_t w = vrp.get_dist(u, v);

      nG[u].push_back(Edge(v, w));
      nG[v].push_back(Edge(u, w));
      //! edges.push_back(std::make_pair(u,v));
      DEBUG std::cout << u << " -- " << v << '\n';
    }
    u++;
  }
  return nG;
}

// Graph's Adjacency information.
void printAdjList(const std::vector<std::vector<Edge>> &graph) {
  int i = 0;
  for (auto vec : graph) {
    std::cout << i << ": ";
    for (auto e : vec) {
      std::cout << e.to << " ";
    }
    i++;
    std::cout << std::endl;
  }
}

// DFS Recursive.
void ShortCircutTour(std::vector<std::vector<Edge>> &g, std::vector<bool> &visited, node_t u, std::vector<node_t> &out) {
  visited[u] = true;
  DEBUG std::cout << u << ' ';
  //! cvrpInOut.addRouteVertex(u);
  out.push_back(u);
  for (auto e : g[u]) {
    node_t v = e.to;
    if (!visited[v]) {
      ShortCircutTour(g, visited, v, out);
    }
  }
}

// Converts a permutation to set of routes
std::vector<std::vector<node_t>>
convertToVrpRoutes(const VRP &vrp, const std::vector<node_t> &singleRoute) {
  std::vector<std::vector<node_t>> routes;

  demand_t vCapacity = vrp.getCapacity();
  demand_t residueCap = vCapacity;
  std::vector<node_t> aRoute;
  tw_t process_time=0;

  
  // for(auto v:singleRoute){
  //   cout<<v<<" ";
  // }
  // cout<<endl;

  int size_singleRoute=singleRoute.size();
  vector<bool> visited(size_singleRoute,false);
  
  int flag=1;
  while(flag==1){
    node_t prev=0;
    flag=0;
    aRoute.clear();
    residueCap = vCapacity;
    process_time=0;
    for (auto v : singleRoute) {
      if (v == 0)
        continue;
      if(visited[v]==true){
        continue;
      }
      if(residueCap - vrp.node[v].demand >= 0 && process_time+vrp.get_dist(prev,v) <= vrp.node[v].latestTime) {  // can add to current route
        aRoute.push_back(v);
        residueCap = residueCap - vrp.node[v].demand;
        process_time+=vrp.get_dist(prev,v);
        process_time=max(process_time,vrp.node[v].earlyTime) + vrp.node[v].serviceTime;
        prev=v;
        visited[v]=true;
        flag=1;
      }
    }
    if(aRoute.size()>0){
      routes.push_back(aRoute);
    }

  }
  
  //printOutput(vrp, routes);
  // Checking whether customer is served or not
  int served_count=0;
  for(int i=1;i<size_singleRoute;i++){
    if(visited[i]==true){
      served_count++;
    }
  }
  if(served_count!=size_singleRoute-1){
    std::cerr<<"Some customers are not served in the solution!"<<std::endl;
  }

  return routes;

}

// Cost of a CVRP Solution!.
weight_t calRouteValue(const VRP &vrp, const std::vector<node_t> &aRoute, node_t depot = 1) {  //return cost of "a" route
  weight_t routeVal = 0;
  node_t prevPoint = 0;  //First point in a route is depot

  for (auto aPoint : aRoute) {
    routeVal += vrp.get_dist(prevPoint, aPoint);
    prevPoint = aPoint;
  }
  routeVal += vrp.get_dist(prevPoint, 0);  //Last point in a route is depot

  return routeVal;
}

// Print in DIMACS output format http://dimacs.rutgers.edu/programs/challenge/vrp/cvrp/
// Depot is 0
// Route #1: 1 2 3
// Route #2: 4 5
// ...
// Route #k: n-1 n
//


/* Verify tour require tour starting from depot eg 0 1 2 3 */
bool verify_tour(const VRP &vrp,const std::vector<node_t> &tour, node_t ncities) {
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

double calculate_tour_distance(const VRP &vrp,const std::vector<node_t> &tour, node_t ncities) {
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
  
  double bestDistance=calculate_tour_distance(vrp,tour,ncities);
  
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

    double newDistance=calculate_tour_distance(vrp,tour,ncities);
    if(newDistance<bestDistance && verify_tour(vrp,tour,ncities)==true){
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
        if (new_distance < best_distance && verify_tour(vrp, tmp_tour_with_depot, ncities + 1)) {
          //std::cout << "2OPT Improvement: " << best_distance << " to " << new_distance << std::endl;
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

weight_t get_total_cost_of_routes(const VRP &vrp, vector<vector<node_t>> &final_routes) {
  weight_t total_cost = 0.0;
  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    weight_t curr_route_cost = 0;

    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
    for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj) {
      curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
    }

    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][final_routes[ii].size() - 1]);

    total_cost += curr_route_cost;
  }

  return total_cost;
}

//
// MAIN POST PROCESS ROUTINE
//
std::vector<std::vector<node_t>>
postProcessIt(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes, weight_t& minCost) {
  std::vector<std::vector<node_t>> postprocessed_final_routes;

  auto postprocessed_final_routes1 = postprocess_tsp_approx(vrp, final_routes);
  if(verify_route(vrp,postprocessed_final_routes1)){
    cout<<"\nPostprocess 1 route valid"<<endl;
  }else{
    cout<<"\nPostprocess 1 route invalid"<<endl;
  }
  auto postprocessed_final_routes2 = postprocess_2OPT(vrp, postprocessed_final_routes1);
  if(verify_route(vrp,postprocessed_final_routes2)){
    cout<<"Postprocess 2 route valid"<<endl;
  }else{
    cout<<"Postprocess 2 route invalid"<<endl;
  }
  auto postprocessed_final_routes3 = postprocess_2OPT(vrp, final_routes);

  weight_t postprocessed_final_routes_cost = 0;
  
  if(verify_route(vrp,postprocessed_final_routes3)){
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

std::pair<weight_t, std::vector<std::vector<node_t>>>
calCost(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes) {
  weight_t total_cost = 0.0;
  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    weight_t curr_route_cost = 0;
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
    for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj) {
      curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
    }
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][final_routes[ii].size() - 1]);
    total_cost += curr_route_cost;
  }
  return {total_cost, final_routes};
}

bool verify_sol(const VRP &vrp, vector<vector<node_t>> final_routes, unsigned capacity) {
  /* verifies if the solution is valid or not */
  /**
   * 1. All vertices appear in the solution exactly once.
   * 2. For every route, the capacity constraint is respected.
   **/

  unsigned *hist = (unsigned *)malloc(sizeof(unsigned) * vrp.getSize());
  memset(hist, 0, sizeof(unsigned) * vrp.getSize());

  for (unsigned i = 0; i < final_routes.size(); ++i) {
    unsigned route_sum_of_demands = 0;
    for (unsigned j = 0; j < final_routes[i].size(); ++j) {
      //~ route_sum_of_demands += points.demands[final_routes[i][j]];
      route_sum_of_demands += vrp.node[final_routes[i][j]].demand;
      hist[final_routes[i][j]] += 1;
    }
    if (route_sum_of_demands > capacity) {
      return false;
    }
  }

  for (unsigned i = 1; i < vrp.getSize(); ++i) {
    if (hist[i] > 1) {
      return false;
    }
    if (hist[i] == 0) {
      return false;
    }
  }
  return true;
}

//------------------------------post optim code----------------------

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
                                // cout<<"Relocating customer "<<u<<" from route "<<r1<<" to route "<<r2<<" at position "<<j<<endl;
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
                                    // cout<<"Swapping customer "<<u<<" in route "<<r1<<" with customer "<<v<<" in route "<<r2<<endl;
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
                                    // cout<<"2-opt-Exchange"<<t<<"->"<<u<<"between"<<x<<"->"<<v<<endl;
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
//-------------------------------------------


int main(int argc, char *argv[]) {
  VRP vrp;
  if (argc < 2) {
    std::cout << "seqMDS version 1.1" << '\n';
    std::cout << "Usage: " << argv[0] << " toy.vrp [-round 0 or 1 DEFAULT:1 means round it!]" << '\n';
    exit(1);
  }

  for (int ii = 2; ii < argc; ii += 2) {
    if (std::string(argv[ii]) == "-round")
      vrp.params.toRound = atoi(argv[ii + 1]);
    else {
      std::cerr << "INVALID Arguments!" << '\n';
      std::cerr << "Usage:" << argv[0] << " toy.vrp -round 1" << '\n';
      exit(1);
    }
  }

  //~ std::cout<< "Round:" << (vrp.params.toRound?"True":"False") << '\n';

  vrp.read(argv[1]);
  chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

  auto cG = vrp.cal_graph_dist();  // complete graph.

  //vrp.print_dist();
  clock_t pre_st=clock();

  auto mstG = PrimsAlgo(vrp, cG);

  clock_t pre_end=clock();
  cerr<< "MST pre-proc Time: " << (double)(pre_end-pre_st)/CLOCKS_PER_SEC;
  
  clock_t mid_st=clock();

  std::vector<bool> visited(mstG.size(), false);
  visited[0] = true;
  std::vector<int> singleRoute;

  weight_t minCost = INT_MAX * 1.0f;
  std::vector<std::vector<node_t>> minRoute;

  for (int i = 0; i < 1; ++i) {
    // RANDOMIZE THE ADJ LIST OF MST
    for (auto &list : mstG) {
      std::shuffle(list.begin(), list.end(), std::default_random_engine(rand()));
    }

    //reset
    singleRoute.clear();

    std::vector<bool> visited(mstG.size(), false);
    visited[0] = true;

    ShortCircutTour(mstG, visited, 0, singleRoute);  //a DFS
    
    DEBUG std::cout << '\n';

    auto aRoutes = convertToVrpRoutes(vrp, singleRoute);

    auto aCostRoute = calCost(vrp, aRoutes);

    if (aCostRoute.first < minCost) {
      minCost = aCostRoute.first;
      minRoute = aCostRoute.second;
    }
  }

  weight_t min_cost_after_one_iteration = minCost;
  auto time_till_one_iteration = (double)((chrono::high_resolution_clock::now() - start).count() * 1.E-9);

  for (int i = 1; i < 1000; ++i) {
    // RANDOMIZE THE ADJ LIST OF MST
    for (auto &list : mstG) {
      std::shuffle(list.begin(), list.end(), std::default_random_engine(rand()));
    }   

    //reset
    singleRoute.clear();

    std::vector<bool> visited(mstG.size(), false);
    visited[0] = true;

    ShortCircutTour(mstG, visited, 0, singleRoute);  //a DFS
    DEBUG std::cout << '\n';

    auto aRoutes = convertToVrpRoutes(vrp, singleRoute);

    auto aCostRoute = calCost(vrp, aRoutes);

    if (aCostRoute.first < minCost) {
      minCost = aCostRoute.first;
      minRoute = aCostRoute.second;
    }   
  }

  weight_t min_cost_after_super_loop = minCost;

  auto time_till_super_loop = (double)((chrono::high_resolution_clock::now() - start).count() * 1.E-9);

  clock_t mid_end=clock();
  cerr<< " MST Main loop time: " << (double)(mid_end-mid_st)/CLOCKS_PER_SEC<<" ";

  clock_t post_st=clock();

  auto postRoutes = postProcessIt(vrp, minRoute, minCost);
  //auto postRoutes = minRoute;

  // -------------------Inter Route Postprocessing-------------------

  for(auto &route:postRoutes){
    route.insert(route.begin(),DEPOT);
    route.push_back(DEPOT);
  }

  inter_route_relocate(vrp,postRoutes);
  weight_t post_relocate_cost=calculate_total_cost(vrp, postRoutes);

  inter_route_swap(vrp,postRoutes);
  weight_t post_swap_cost=calculate_total_cost(vrp, postRoutes);
  inter_route_2opt_star(vrp,postRoutes);
  weight_t post_2opt_star_cost=calculate_total_cost(vrp, postRoutes);

  minCost=post_2opt_star_cost;

  //-----------------------------------------------------------------

  chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
  uint64_t elapsed = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  double total_time = (double)(elapsed * 1.E-9);
  std::cerr << argv[1];

  /// VALIDATION
  bool verified = false;
  verified = verify_sol(vrp, postRoutes, vrp.getCapacity()) && verify_route(vrp,postRoutes);

  clock_t post_end=clock();
  cerr<< " Post-process Time: " << (double)(post_end-post_st)/CLOCKS_PER_SEC;

  if (verified)
  {
    cerr << " Cost " << min_cost_after_one_iteration << " "
                     << min_cost_after_super_loop    << " "
                     << minCost;
    cerr << " Time(seconds) " << time_till_one_iteration << " "
                              << time_till_super_loop    << " "
                              << total_time;
    cerr << " Vehicle_Used"<<" " << postRoutes.size();
    cerr << " VALID\n";
  }else
  {
    cerr << " Cost " << min_cost_after_one_iteration << " " 
                     << min_cost_after_super_loop    << " " 
                     << minCost;
    cerr << " Time(seconds) " << time_till_one_iteration << " " 
                              << time_till_super_loop    << " " 
                              << total_time;
    cerr << " INVALID\n";
  }
  
  // PRINT ANS
  printOutput(vrp, postRoutes);

  return 0;
}
