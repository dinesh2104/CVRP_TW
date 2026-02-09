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

//k-medoid clustering............

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

struct RouteNode {
  node_t customer_id;
  tw_t processing_time;
};

// Construction function........
vector<vector<node_t>> constructRoutes(VRP &vrp,vector<vector<int>> &clusters,int rcl){
  vector<vector<node_t>> final_routes;
  vector<node_t> current_route;

  int current_capacity = vrp.getCapacity();
  tw_t current_process_time = 0;
  int k = 0;

  while(k < clusters.size()){
    // If no customers left in this cluster → move to next
    if(clusters[k].empty()){
      if(!current_route.empty()){
        final_routes.push_back(current_route);
        current_route.clear();
        current_capacity = vrp.getCapacity();
        current_process_time = 0;
      }
      k++;
      continue;
    }

    // Build candidate list with processing times
    vector<RouteNode> route_nodes;
    for(node_t cust : clusters[k]){
      tw_t processing_time;

      if(current_route.empty()){
        processing_time = max((tw_t)vrp.get_dist(DEPOT, cust),vrp.node[cust].earlyTime);
      }
      else{
        node_t prev = current_route.back();
        processing_time = current_process_time + vrp.get_dist(prev, cust);
        processing_time = max(processing_time,vrp.node[cust].earlyTime);
      }
      // Add only the route nodes that are feasible in terms of time windows
      if(processing_time <= vrp.node[cust].latestTime){
        route_nodes.push_back({cust, processing_time});
      }
    }
        // Safety check 
    // Note: Previous version we are assuming that from depot all the customers are feasible to serve......
    if(route_nodes.empty()){
      if(!current_route.empty()){
        final_routes.push_back(current_route);
        current_route.clear();
        current_capacity = vrp.getCapacity();
        current_process_time = 0;
      }else{
        cout<<" No valid route can be formed for the customers in the cluster"<<endl;
        exit(1);
      }
      continue;
    }

        // Greedy ordering
    sort(route_nodes.begin(), route_nodes.end(),
            [](const RouteNode &a, const RouteNode &b){
            return a.processing_time < b.processing_time;
            });

        // Build RCL
    int flag=0;
    for(auto &node : route_nodes){
      int selected_customer=node.customer_id;
      int prev_node=current_route.empty() ? DEPOT : current_route.back();
      tw_t selected_time=current_process_time + vrp.get_dist(prev_node, selected_customer);
      selected_time = max(selected_time, vrp.node[selected_customer].earlyTime);
      if(current_capacity >= vrp.node[selected_customer].demand &&
        selected_time <= vrp.node[selected_customer].latestTime){
            // Assign customer
        flag=1;
        current_route.push_back(selected_customer);
        current_capacity -= vrp.node[selected_customer].demand;
        current_process_time =selected_time + vrp.node[selected_customer].serviceTime;
              // Remove customer from current cluster
        clusters[k].erase(remove(clusters[k].begin(), clusters[k].end(), selected_customer),clusters[k].end());
      }
    }
    if(flag==0){ // No customer could be added to current route, start a new route
      if(!current_route.empty()){
        final_routes.push_back(current_route);
        current_route.clear();
        current_capacity = vrp.getCapacity();
        current_process_time = 0;
      }
    }
    route_nodes.clear();
    //cout<<"Node added to route: "<<selected_customer<<" k="<<k<<endl;
    //break; 
  }
  return final_routes;
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





// Print the output routes.

int max_length_of_route(const std::vector<std::vector<node_t>> &routes) {
  size_t max_length = 0;
  for (const auto &route : routes) {
    if (route.size() > max_length) {
      max_length = route.size();
    }
  }
  return max_length;
}

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


int main(int argc, char *argv[]) {
VRP vrp;
  if (argc < 2) {
    std::cout << "seqCVRPTW version 3" << '\n';
    std::cout << "Usage: " << argv[0] << " toy.vrp" << '\n';
    exit(1);
  }

  vrp.read(argv[1]);
  vrp.cal_graph_dist();
  //vrp.print();
  // Todo: Clustering the customers based on k-medoid
  chrono::high_resolution_clock::time_point pre_start = chrono::high_resolution_clock::now();
  chrono::high_resolution_clock::time_point total_start = chrono::high_resolution_clock::now();

  vector<vector<int>> clusters = clustering_kmedoid(vrp, 5); // 5 clusters

  chrono::high_resolution_clock::time_point pre_end = chrono::high_resolution_clock::now();

  // Todo: Construct routes within each cluster.
  
  chrono::high_resolution_clock::time_point mid_start = chrono::high_resolution_clock::now();

  vector<vector<int>> clustor_cpy=clusters; // copy of clusters for route construction
  vector<vector<node_t>> final_routes=constructRoutes(vrp,clustor_cpy,3);
  int route_cost=calculate_total_cost(vrp,final_routes);

  int min_cost=route_cost;
  int min_cost1=route_cost;
  vector<vector<node_t>> best_routes=final_routes;
  

  // for(int i=0;i<1000;i++){
  //   clustor_cpy=clusters; // reset clusters
  //   vector<vector<node_t>> new_routes=constructRoutes(vrp,clustor_cpy,3);
  //   int new_cost=calculate_total_cost(vrp,new_routes);
  //   if(new_cost<min_cost && verify_route(vrp,new_routes)){
  //     min_cost=new_cost;
  //     best_routes=new_routes;
  //   }
  // }

  // int min_cost2=calculate_total_cost(vrp,best_routes);

  //printing the best routes after construction
  // cout<<"Route before post-processing: "<<endl;
  // for(int i=0;i<best_routes.size();i++){
  //   for(int j=0;j<best_routes[i].size();j++){
  //     cout<<best_routes[i][j]<<" ";
  //   }
  //   cout<<endl;
  // }


  chrono::high_resolution_clock::time_point mid_end = chrono::high_resolution_clock::now();
  chrono::high_resolution_clock::time_point total_end = chrono::high_resolution_clock::now();
  
  // TODO: Post-Optimization of routes.
  weight_t post_optimized_cost=min_cost;
  best_routes=postProcessIt(vrp,best_routes,post_optimized_cost);

  min_cost=calculate_total_cost(vrp,best_routes);


  //printing final routes
  cout<<"Best Cost: "<<min_cost<<endl;
  cout<<"Verifying best routes: "<<(verify_route(vrp,best_routes)?"Valid":"Invalid")<<endl;
  cout<<"Number of routes: "<<best_routes.size()<<endl;
  print_routes(best_routes);

  if(verify_route(vrp,best_routes)){
    cerr<<"File: "<<argv[1]<<" ";
    cerr<<"Pre-processing_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(pre_end - pre_start).count()*1.E-9)<<" s ";
    cerr<<"Route_Construction_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(mid_end - mid_start).count()*1.E-9)<<" s ";
    cerr<<"Total_Cost1: "<<min_cost1<<" ";
    cerr<<"Total_Cost2: "<<min_cost<<" ";
    cerr<<"Total_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(total_end - total_start).count()*1.E-9)<<" s ";
    cerr<<"Vehicle_Used: "<<best_routes.size()<<" ";
    cerr<<"route_length: "<<max_length_of_route(best_routes)<<" ";
    cerr<<"VALID"<<endl;
  }

  return 0;
}