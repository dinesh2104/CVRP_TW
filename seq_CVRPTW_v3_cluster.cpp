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

      route_nodes.push_back({cust, processing_time});
    }
        // Safety check
    if(route_nodes.empty()){
      k++;
      continue;
    }

        // Greedy ordering
    sort(route_nodes.begin(), route_nodes.end(),
            [](const RouteNode &a, const RouteNode &b){
            return a.processing_time < b.processing_time;
            });

        // Build RCL
    int rcl_size = min(rcl, (int)route_nodes.size());
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, rcl_size - 1);
    int selected_index = dis(gen);

    node_t selected_customer = route_nodes[selected_index].customer_id;
    tw_t selected_time = route_nodes[selected_index].processing_time;

        // Feasibility check
    if(current_capacity >= vrp.node[selected_customer].demand &&
      selected_time <= vrp.node[selected_customer].latestTime){

            // Assign customer
      current_route.push_back(selected_customer);
      current_capacity -= vrp.node[selected_customer].demand;
      current_process_time =selected_time + vrp.node[selected_customer].serviceTime;
            // Remove customer from current cluster
      clusters[k].erase(remove(clusters[k].begin(), clusters[k].end(), selected_customer),clusters[k].end());
    }
    else{
            // Close current route and start a new one
      if(!current_route.empty()){
        final_routes.push_back(current_route);
      }
      current_route.clear();
      current_capacity = vrp.getCapacity();
      current_process_time = 0;
    }
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
  vector<vector<node_t>> best_routes=final_routes;

  for(int i=0;i<100000;i++){
    clustor_cpy=clusters; // reset clusters
    vector<vector<node_t>> new_routes=constructRoutes(vrp,clustor_cpy,3);
    int new_cost=calculate_total_cost(vrp,new_routes);
    if(new_cost<min_cost && verify_route(vrp,new_routes)){
      min_cost=new_cost;
      best_routes=new_routes;
    }
  }

  chrono::high_resolution_clock::time_point mid_end = chrono::high_resolution_clock::now();
  chrono::high_resolution_clock::time_point total_end = chrono::high_resolution_clock::now();
  
  // TODO: Post-Optimization of routes.

  //printing final routes
  cout<<"Best Cost: "<<min_cost<<endl;
  cout<<"Verifying best routes: "<<(verify_route(vrp,best_routes)?"Valid":"Invalid")<<endl;
  cout<<"Number of routes: "<<best_routes.size()<<endl;
  print_routes(best_routes);

  if(verify_route(vrp,best_routes)){
    cerr<<"File: "<<argv[1]<<" ";
    cerr<<"Pre-processing_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(pre_end - pre_start).count()*1.E-9)<<" s ";
    cerr<<"Route_Construction_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(mid_end - mid_start).count()*1.E-9)<<" s ";
    cerr<<"Total_Cost: "<<min_cost<<" ";
    cerr<<"Total_Time: "<<(double)(chrono::duration_cast<chrono::nanoseconds>(total_end - total_start).count()*1.E-9)<<" s ";
    cerr<<"Vehicle_Used: "<<best_routes.size()<<" ";
    cerr<<"VALID"<<endl;
  }

  return 0;
}