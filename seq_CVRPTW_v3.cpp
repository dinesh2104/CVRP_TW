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
  int n=vrp.getSize();
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
  int k=0;
  int vehicle_count=0;
  vector<node_t> current_route;
  int current_capacity=vrp.getCapacity();
  int current_process_time=0;

  while(k<clusters.size()){
    // Finding the customer with earliest time window in the cluster
    vector<RouteNode> route_nodes;
    for(int j=0;j<clusters[k].size();j++){
      if(current_route.size()==0){
        tw_t processing_time=max((unsigned int) vrp.get_dist(DEPOT,clusters[k][j]),vrp.node[clusters[k][j]].earlyTime);
        route_nodes.push_back({clusters[k][j],processing_time});
      }
      else{
          int prev_customer=current_route.back();
          tw_t processing_time=current_process_time + (unsigned int)vrp.get_dist(prev_customer,clusters[k][j]);
          processing_time=max(processing_time,vrp.node[clusters[k][j]].earlyTime);
          route_nodes.push_back({clusters[k][j],processing_time});
      }
    }
    sort(route_nodes.begin(),route_nodes.end(),[](const RouteNode &a,const RouteNode &b){
      return a.processing_time<b.processing_time;
    });

    int random_index=min(rcl-1,(int)route_nodes.size()-1);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, random_index);
    int selected_index=dis(gen);
    node_t selected_customer=route_nodes[selected_index].customer_id;
    
    if(current_capacity - vrp.node[selected_customer].demand >=0 && route_nodes[selected_index].processing_time <= vrp.node[selected_customer].latestTime){
      current_route.push_back(selected_customer);
      current_capacity -= vrp.node[selected_customer].demand;
      current_process_time=route_nodes[selected_index].processing_time + vrp.node[selected_customer].serviceTime;

      // Remove selected customer from cluster
      clusters[k].erase(remove(clusters[k].begin(),clusters[k].end(),selected_customer),clusters[k].end());
    }
    else{
      final_routes.push_back(current_route);
      vehicle_count++;
      current_route.clear();
      current_capacity=vrp.getCapacity();
      current_process_time=0;
    }

    // Todo: This condition may change....
    if(clusters[k].size()==0){
      if(current_route.size()>0){
        final_routes.push_back(current_route);
        vehicle_count++;
        current_route.clear();
        current_capacity=vrp.getCapacity();
        current_process_time=0;
      }
      k++;
    }

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
  // Todo: Clustering the customers based on k-medoid
  vector<vector<int>> clusters = clustering_kmedoid(vrp, 5); // 5 clusters

  // Todo: Construct routes within each cluster.
  
  vector<vector<node_t>> final_routes=constructRoutes(vrp,clusters);



  return 0;
}