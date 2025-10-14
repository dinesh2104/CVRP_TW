#include<iostream>
#include<cuda_runtime.h>
#include<string.h>
#include<fstream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/device_free.h>
#include <thrust/device_free.h>
#include <ctime>
#include <math.h>
#include <random>
#include <chrono>

using namespace std;

struct Point {
    int x, y;
    double demand;

    //adding the parameter for the time window
    double earlyTime;
    double latestTime;   // earliest time to start service
    double serviceTime;  // service time required
  Point() {}
};

int read(string filename, int *h_x, int *h_y, double *h_demand, double *h_earlyTime, double *h_latestTime, double *h_serviceTime) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error opening file: " << filename << endl;
        return -1;
    }
    string line;
    for(int i=0;i<4;i++){
        getline(infile,line);
    }
    int vechiceleCapacity,nvechicles;
    infile >>nvechicles>> vechiceleCapacity;

    for(int i=0;i<4;i++){
        getline(infile,line);
    }

    int idx=0;
    while (getline(infile, line)) {
        if (line.empty()) continue; // Skip empty lines
        int no,x, y;
        double demand, earlyTime, latestTime, serviceTime;
        if (!(infile >> no >> x >> y >> demand >> earlyTime >> latestTime >> serviceTime)) {
            //cerr << "Error reading line: " << line << endl;
            continue; // Skip lines that don't match the expected format
        }
        h_x[idx]=x;
        h_y[idx]=y;
        h_demand[idx]=demand;
        h_earlyTime[idx]=earlyTime;
        h_latestTime[idx]=latestTime;
        h_serviceTime[idx]=serviceTime;
        idx++;
    }

    infile.close();
    return idx;
}

__global__ void weightUpdate(int *d_x, int *d_y, float *d_weights, bool *d_inMST, int *d_parent, int current, int nodes) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id==current){
        d_weights[id]=INT_MAX;
        return;
    }
    if(id >= nodes)
        return;
    // if(id==nodes-1){
    //     printf("Current: %d\n",current);
    //     printf("Weights[%d]: %d\n",id,d_weights[id]);
    //     printf("Parent[%d]: %d\n",id,d_parent[id]);
    //     printf("d_x[%d]: %d, d_y[%d]: %d\n",id,d_x[id],id,d_y[id]);
    // }

    if(d_inMST[id])
        return;
    int dx = d_x[current] - d_x[id];
    int dy = d_y[current] - d_y[id];
    float distance = sqrtf(dx * dx + dy * dy); // Squared distance to avoid sqrt for efficiency
    if(!d_inMST[id] && d_weights[id] > distance) {
        d_weights[id] = distance;
        d_parent[id] = current;
    }
    // if(id==20){
    //     printf("Current: %d, Weights[id]: %d\n",current,d_weights[id]);
    // }
}

__host__ __device__ float calculateDistance(int x1, int y1, int x2, int y2) {
    int dx = x1 - x2;
    int dy = y1 - y2;
    return sqrtf(dx * dx + dy * dy); // Squared distance
}

__device__ double calculate_cost(int *tour, int tour_length, int *d_x, int *d_y, double *d_demand) {
    double total_cost = 0.0;
    for (int i = 0; i < tour_length - 1; i++) {
        int from = tour[i];
        int to = tour[i + 1];
        total_cost += calculateDistance(d_x[from], d_y[from], d_x[to], d_y[to]);
    }
    // Add cost to return to depot
    total_cost += calculateDistance(d_x[tour[tour_length - 1]], d_y[tour[tour_length - 1]], d_x[0], d_y[0]);
    return total_cost;
}

// Iterative DFS inside device
__device__ void dfs_iterative(int start, bool *visited, int *d_route, int &route_idx, int *d_u, int *d_v) {
    // Manual stack (big enough for graph size, adjust as needed)
    int stack[1024];  
    int top = -1;

    // Push start node
    stack[++top] = start;

    while (top >= 0) {
        int node = stack[top--];  // pop
        //printf("%d, %d",node,visited[node]);
        if (!visited[node]) {
            //printf("%d \n", node);

            visited[node] = true;
            d_route[route_idx++] = node;

            int idx = d_u[node];
            int idx_end = d_u[node + 1];
            //printf("%d %d\n", idx, idx_end);

            // Push neighbors (reverse order if you want same traversal as recursive)
            for (int i = idx_end - 1; i >= idx; i--) {
                int neighbor = d_v[i];
                if (!visited[neighbor]) {
                    stack[++top] = neighbor;
                }
            }
        }
    }
}

__global__ void createRoute(int *d_u,int *d_v,int *d_x,int *d_y,double *d_demand,int capacity,double *d_earlyTime,double *d_latestTime,double *d_serviceTime,int *d_route,int nodes, double *min_route_cost,int *opt_final_route,int *route_len) {
    //printf("Creating Route\n");
    bool *visited = new bool[nodes];
    int route_idx = 0;

    for(int i=0;i<nodes;i++){
        visited[i]=false;
    }

    dfs_iterative(0, visited, d_route, route_idx, d_u, d_v);
    
    // printf("DFS interation: ");
    // for(int i=0;i<route_idx;i++){
    //     printf("%d ",d_route[i]);
    // }
    // printf("Finished\n");
    // printf("\n");
    // int *final_route=new int[route_idx*2];

    int idx=0;
    int residual_capacity=capacity;
    double current_time=0;
    int* final_route=new int[route_idx*2];
    idx++;
    int prev=0;
    for(int i=1;i<route_idx;i++){
        int node=d_route[i];
        double travel_time=calculateDistance(d_x[prev],d_y[prev],d_x[node],d_y[node])/50*60;
        current_time+=travel_time;
        if(current_time<d_earlyTime[node]){
            current_time=d_earlyTime[node];
        }
        if(residual_capacity>=d_demand[node] && current_time<=d_latestTime[node]){
            final_route[idx]=node;
            idx++;
            residual_capacity-=d_demand[node];
            current_time+=d_serviceTime[node];
            prev=node;
        }
        else{
            final_route[idx]=0;
            idx++;
            
            //go to the current node from depot
            travel_time=calculateDistance(d_x[0],d_y[0],d_x[node],d_y[node])/50*60;
            current_time=travel_time;
            if(current_time<d_earlyTime[node]){
                current_time=d_earlyTime[node];
            }
            if(current_time>d_latestTime[node]){
                printf("Node %d cannot be serviced due to time window constraints.\n", node);
                return;
            }
            final_route[idx]=node;
            idx++;
            residual_capacity=capacity-d_demand[node];
            current_time+=d_serviceTime[node];
            prev=node;
        }
    }

    // printf("Printing the Final Route\n");
    // for(int i=0;i<idx;i++){
    //     printf("%d ",final_route[i]);
    // }
    // printf("\n");

    double cost=calculate_cost(final_route,idx,d_x,d_y,d_demand);

    //printf("\ncost: %lf\n",cost);
    if(*min_route_cost==-1 || cost<*min_route_cost ){
        for(int i=0;i<*route_len;i++){
            opt_final_route[i]=0;
        }
        *min_route_cost=cost;
        *route_len=idx;
        for(int i=0;i<idx;i++){
            opt_final_route[i]=final_route[i];
        }
        // printf("Printing the optimal route:\n");
        // for(int i=0;i<idx;i++){
        //     if(opt_final_route[i]==-1)
        //         break;
        //     printf("%d ",opt_final_route[i]);
        // }
        // printf("\n");
    }


    delete[] visited;
}

__device__ bool verify_route(int *tour, int tour_length, int *d_x, int *d_y, double *d_demand, int capacity, double *d_earlyTime, double *d_latestTime, double *d_serviceTime) {
    double current_time = 0.0;
    int current_load = 0;
    int prev_node = 0; // Start from depot

    for (int i = 0; i < tour_length; i++) {
        int node = tour[i];
        if (node == 0) { // Depot
            current_time = 0;
            current_load = 0; // Unload at depot
            prev_node = 0;
            continue;
        }

        // Travel to next node
        current_time += calculateDistance(d_x[prev_node], d_y[prev_node], d_x[node], d_y[node]) / 50.0 * 60.0;

        // Check time window
        if (current_time < d_earlyTime[node]) {
            current_time = d_earlyTime[node]; // Wait until early time
        }
        if (current_time > d_latestTime[node]) {
            //printf("Tour invalid: Arrived at node %d after latest time.\n", node);
            return false; // Violates latest time
        }

        // Service the node
        current_time += d_serviceTime[node];
        current_load += d_demand[node];

        // Check capacity
        if (current_load > capacity) {
            //printf("Tour invalid: Capacity exceeded at node %d.\n", node);
            return false; // Exceeds vehicle capacity
        }

        prev_node = node;
    }

    // Return to depot at end of tour
    current_time += calculateDistance(d_x[prev_node], d_y[prev_node], d_x[0], d_y[0]) / 50.0 * 60.0;

    //printf("Tour valid: Completed with total time %.2f minutes.\n", current_time);
    return true;
}


__device__ bool verify_tour(int *tour,int start, int end, int *d_x, int *d_y, double *d_demand, int capacity, double *d_earlyTime, double *d_latestTime, double *d_serviceTime) {
    double current_time = 0.0;
    int current_load = 0;
    int prev_node = 0; // Start from depot
    // printf("start=%d end=%d\n", start, end);
    // printf("%d %d\n", tour[start], tour[end]);
    for (int i = start; i < end; i++) {
        int node = tour[i];
        if (node == 0) { // Depot
            current_time = 0;
            current_load = 0; // Unload at depot
            prev_node = 0;
            continue;
        }

        // Travel to next node
        current_time += calculateDistance(d_x[prev_node], d_y[prev_node], d_x[node], d_y[node]) / 50.0 * 60.0;

        // Check time window
        if (current_time < d_earlyTime[node]) {
            current_time = d_earlyTime[node]; // Wait until early time
        }
        if (current_time > d_latestTime[node]) {
            //printf("Tour invalid: Arrived at node %d after latest time.\n", node);
            return false; // Violates latest time
        }

        // Service the node
        current_time += d_serviceTime[node];
        current_load += d_demand[node];

        // Check capacity
        if (current_load > capacity) {
            //printf("Tour invalid: Capacity exceeded at node %d.\n", node);
            return false; // Exceeds vehicle capacity
        }

        prev_node = node;
    }

    // Return to depot at end of tour
    current_time += calculateDistance(d_x[prev_node], d_y[prev_node], d_x[0], d_y[0]) / 50.0 * 60.0;

    //printf("Tour valid: Completed with total time %.2f minutes.\n", current_time);
    return true;
}

__global__ void postprocess_tsp_approx(int *final_route,int route_length,int *d_x,int *d_y,double *d_demand,int capacity,double *d_earlyTime,double *d_latestTime,double *d_serviceTime,int *optimized_route) {
    // Implement 2-opt or other local search heuristics here
    // Try to swap the nodes with lesser distance
    printf("Post Processing Route\n");

    for(int i=0;i<route_length;i++){
        optimized_route[i]=final_route[i];
    }
    int optimized_length=route_length;
    int start=1;
    int end=1;
    for(int i=0;i<route_length-1;i++){
        if(optimized_route[i]==0){
            start=i+1;
            int cnt=0;
            for(int k=start;k<route_length;k++){
                if(optimized_route[k]!=0)
                    cnt++;
                else
                    break;
            }
            end=start+cnt;
            continue;
        }
        int cnt=0;
        for(int k=i;k<route_length;k++){
            if(optimized_route[k]!=0)
                cnt++;
            else
                break;
        }
        
        
        //If only two element then skip
        if(cnt<=2){
            continue;
        }


        /* Finding the minimum distance point with respect to i-1*/
        double min_dist=INT_MAX;
        int min_index=-1;
        // printf("Inside post- %d, %d\n",i,cnt);
        for(int j=i+cnt-1;j>i;j--){
            //printf("%d %d\n",optimized_route[j],optimized_route[i-1]);
            double dist=calculateDistance(d_x[optimized_route[i-1]],d_y[optimized_route[i-1]],d_x[optimized_route[j]],d_y[optimized_route[j]]);
            if(dist<min_dist){
                min_dist=dist;
                min_index=j;
            }
        }
        if(min_index!=-1 && min_index!=i){
            //printf("Swapping %d and %d\n",optimized_route[i],optimized_route[min_index]);
            //swap
            int temp=optimized_route[i];
            optimized_route[i]=optimized_route[min_index];
            optimized_route[min_index]=temp;
        }

        if(!verify_tour(optimized_route,start,end,d_x,d_y,d_demand,capacity,d_earlyTime,d_latestTime,d_serviceTime)){
            //printf("Tour invalid after swaping\n");
            //undo the swap
            int temp=optimized_route[i];
            optimized_route[i]=optimized_route[min_index];
            optimized_route[min_index]=temp;
        }

    }
    // printf("Optimized Route: ");
    // for(int i=0;i<optimized_length;i++){
    //     printf("%d ",optimized_route[i]);
    // }
    // printf("\n");

    /* Final check to verify that whole route is valid. */
    bool isValid=verify_route(optimized_route, optimized_length, d_x, d_y, d_demand, capacity, d_earlyTime, d_latestTime, d_serviceTime);

    if(isValid){
        final_route=optimized_route;
    }
}

__global__ void postprocess_2_opt(int *final_route,int route_length,int *d_x,int *d_y,double *d_demand,int capacity,double *d_earlyTime,double *d_latestTime,double *d_serviceTime,int *optimized_route) {
    // Implement 2-opt or other local search heuristics here
    // Try to reverse the path
    printf("Post 2 opt Processing Route\n");
    int optimized_length=route_length;

    for(int i=0;i<route_length-1;i++){
        int cnt=1;
        if(optimized_route[i]==0)
            continue;
        for(int k=i+1;k<route_length;k++){
            if(optimized_route[k]!=0)
                cnt++;
            else
                break;
        }
        if(cnt<=2){
            continue;
        }
        for(int j=0;j<cnt;j++){
            for(int p=0;p<route_length;p++){
                optimized_route[p]=final_route[p];
            }
            double min_dist=calculate_cost(optimized_route, optimized_length, d_x, d_y, d_demand);
            // reversing the segment between i and i+1+j
            for(int k=0;k<=j/2;k++){
                //printf("Reversing %d and %d\n",optimized_route[i+k],optimized_route[i+j-k]);
                int temp=optimized_route[i+k];
                optimized_route[i+k]=optimized_route[i+j-k];
                optimized_route[i+j-k]=temp;
            }
            //printf("Trying 2-opt between %d and %d\n", i+1, i+1+j);
            // for(int k=0;k<cnt;k++){
            //     printf("%d ",optimized_route[k+i]);
            // } 
            // printf("\n");
            double new_dist=calculate_cost(optimized_route, optimized_length, d_x, d_y, d_demand);
            if(new_dist<min_dist){
                if(verify_route(optimized_route, optimized_length, d_x, d_y, d_demand, capacity, d_earlyTime, d_latestTime, d_serviceTime)){
                    printf("2-opt between %d and %d improved cost from %.2f to %.2f\n", i+1, i+1+j, min_dist, new_dist);
                    final_route=optimized_route;
                    break;
                }
            }
        }
        
    }
    // printf("2-opt Route: ");
    // for(int i=0;i<route_length;i++){
    //     printf("%d ",final_route[i]);
    // }
    // printf("\n");
}

void printRoute(int *route, int length,int *h_x,int *h_y) {
    // calculate total distance
    long double total_distance = 0;
    std::cout << "Final Route: \n";
    
    int k=0;
    int prev=0;
    for(int i = 1; i < length; i++) {
        cout<<"Route #"<<k+1<<": ";
        int flag=0;
        while(i<length && route[i] != 0) {
            total_distance+=calculateDistance(h_x[prev],h_y[prev],h_x[route[i]],h_y[route[i]]);
            std::cout << route[i] << " ";
            prev=route[i];
            i++;
            flag=1;
            
        }
        if(flag==0){
            break;
        }
        cout<<endl;
        k++;
        total_distance+=calculateDistance(h_x[prev],h_y[prev],h_x[0],h_y[0]);
        prev=0;
    }
    cout<<"Total Distance: "<<total_distance<<endl;
}

bool validateRoute(int *route, int length, int *h_x, int *h_y, double *h_demand, int capacity, double *h_earlyTime, double *h_latestTime, double *h_serviceTime) {
    // Implement route validation logic here
    // Check for capacity constraints, time window violations, etc.
    double current_time = 0.0;
    int current_load = 0;
    int prev_node = 0; // Start from depot
    for (int i = 0; i < length; i++) {
        int node = route[i];
        if (node == 0) { // Depot
            current_time = 0;
            current_load = 0; // Unload at depot
            prev_node = 0;
            continue;
        }

        // Travel to next node
        current_time += calculateDistance(h_x[prev_node], h_y[prev_node], h_x[node], h_y[node]) / 50.0 * 60.0;

        // Check time window
        if (current_time < h_earlyTime[node]) {
            current_time = h_earlyTime[node]; // Wait until early time
        }
        if (current_time > h_latestTime[node]) {
            printf("Route invalid: Arrived at node %d after latest time.\n", node);
            return false; // Violates latest time
        }

        // Service the node
        current_time += h_serviceTime[node];
        current_load += h_demand[node];

        // Check capacity
        if (current_load > capacity) {
            printf("Route invalid: Capacity exceeded at node %d.\n", node);
            return false; // Exceeds vehicle capacity
        }

        prev_node = node;
    }
    return true;

}

int main(int argc, char** argv) {
    // if(argc<1) {
    //     std::cout<<"Please provide the input file name"<<std::endl;
    //     return -1;
    // }
    string filename=argv[1];
    int *h_x,*h_y;
    double *h_demand,*h_earlyTime,*h_latestTime,*h_serviceTime;

    int capacity=200;


    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << argv[1] << "\n";
        return 1;
    }

    string line;
    int count = 0;
    while (getline(file, line)) {
        count++;
    }
    h_x=new int[count];
    h_y=new int[count];
    h_demand=new double[count];
    h_earlyTime=new double[count];
    h_latestTime=new double[count];
    h_serviceTime=new double[count];
    cout<<"Total number of lines in the file: "<<count<<endl;

    int nodes=read(filename,h_x,h_y,h_demand,h_earlyTime,h_latestTime,h_serviceTime);
    cout<<"Total number of nodes including depot: "<<nodes<<endl;
    for(int i=0;i<5;i++){
        cout<<"Customer "<<i+1<<": ("<<h_x[i]<<", "<<h_y[i]<<"), Demand: "<<h_demand[i]<<", Time Window: ["<<h_earlyTime[i]<<", "<<h_latestTime[i]<<"], Service Time: "<<h_serviceTime[i]<<endl;
    }
    file.close();
    // Code for MST.

    //Time measurement start
    auto start = std::chrono::high_resolution_clock::now();

    int *d_x,*d_y;
    double *d_demand,*d_earlyTime,*d_latestTime,*d_serviceTime;
    cudaMalloc((void**)&d_x,nodes*sizeof(int));
    cudaMalloc((void**)&d_y,nodes*sizeof(int));
    cudaMalloc((void**)&d_demand,nodes*sizeof(double));
    cudaMalloc((void**)&d_earlyTime,nodes*sizeof(double));
    cudaMalloc((void**)&d_latestTime,nodes*sizeof(double));
    cudaMalloc((void**)&d_serviceTime,nodes*sizeof(double));
    cudaMemcpy(d_x, h_x, nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_demand, h_demand, nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_earlyTime, h_earlyTime, nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_latestTime, h_latestTime, nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_serviceTime, h_serviceTime, nodes * sizeof(double), cudaMemcpyHostToDevice);
    
    long long int edge_sum=0;
    int current=0;
    int cnt=0;

    // int *parent=new int[nodes];
    // parent[0]=-1;
    bool *inMST=new bool[nodes];
    vector<float> weights(nodes);
    for(int i=0;i<nodes;i++){
        weights[i]=INT_MAX;
        inMST[i]=false;
    }
    
    weights[0]=0.0f;

    thrust::device_vector<float> d_weights(weights.begin(), weights.end());
    thrust::device_ptr<float> ptr=d_weights.data();
    thrust::device_vector<bool> inMST_d(inMST,inMST+nodes);
    thrust::device_vector<int> d_parent(nodes);
    d_parent[0]=-1;

    cout<<"calling MST kernel"<<endl;
    while(cnt<nodes-1){
        cnt++;
        inMST_d[current]=true;
        weightUpdate<<<1,nodes>>>(d_x, d_y, thrust::raw_pointer_cast(d_weights.data()), thrust::raw_pointer_cast(inMST_d.data()), thrust::raw_pointer_cast(d_parent.data()), current, nodes);
        cudaDeviceSynchronize();
        int min_index=thrust::min_element(ptr, ptr + nodes) - ptr;
        //cout<<"Current Node: "<<current<<", Next Node: "<<min_index<<", Weight: "<<d_weights[min_index]<<endl;
        edge_sum+=d_weights[min_index];
        current=min_index;
        d_weights[min_index]=INT_MAX;
    }
    // for(int i=0;i<nodes;i++){
    //     cout<<"Node: "<<i<<", Parent: "<<d_parent[i]<<endl;
    // }

    // Create a new adjacency list to represent the MST
    vector<vector<int>> mst_adj_list(nodes);
    for (int i = 1; i < nodes; ++i) {
        int parent = d_parent[i];
        mst_adj_list[parent].push_back(i);
        mst_adj_list[i].push_back(parent); // Since the MST is undirected
    }

    
    // for(int i=0;i<nodes;i++){
    //     cout<<i<<": ";
    //     for(int neighbor : mst_adj_list[i]) {
    //         cout << neighbor << " ";
    //     }
    //     cout<<endl;
    // }

    //TODO: Working fine.........

    // Generate Route using Preorder DFS
    int step=1;
    thrust::device_vector<int> final_route(nodes*2);
    thrust::device_vector<int> opt_final_route(nodes*2);
    int* d_route_len;
    cudaMalloc(&d_route_len, sizeof(int));
    cudaMemset(d_route_len, 0, sizeof(int));

    double* d_min_cost_route;
    cudaMalloc((void**)&d_min_cost_route,nodes*2*sizeof(double));
    double min_cost_route_length=-1;
    cudaMemcpy(d_min_cost_route, &min_cost_route_length, sizeof(double), cudaMemcpyHostToDevice);

    while(step<1000){

        //shuffle the adj list
        for (auto &list : mst_adj_list) {
            std::shuffle(list.begin(), list.end(), std::default_random_engine(rand()));
        }   
    
        vector<int> h_u(nodes+1);
        vector<int> h_v;
        int edge_count=0;
        for(int i=0;i<nodes;i++){
            h_u[i]=edge_count;
            for(int neighbor : mst_adj_list[i]) {
                h_v.push_back(neighbor);
            }
            edge_count+=mst_adj_list[i].size();
        }
        h_u[nodes]=edge_count;



        thrust::device_vector<int> d_route(nodes);
        // Convert adjacency list to CSR format for GPU processing
        thrust::device_vector<int> d_u(h_u.begin(), h_u.end());
        thrust::device_vector<int> d_v(h_v.begin(), h_v.end());
        
        
        //printing the CSR representation
        // for(int i=0;i<=nodes;i++){
        //     printf("%d ",h_u[i]);
        // }
        // cout<<endl;
        // for(int i=0;i<edge_count;i++){
        //     printf("%d ",h_v[i]);
        // }

        //printing the node and neighbour
        // cout<<"Printing the CSR:"<<endl;
        // for(int i=0;i<5;i++){
        //     printf("%d: ",i);
        //     for(int j=h_u[i];j<h_u[i+1];j++){
        //         printf("%d ",h_v[j]);
        //     }
        //     printf("\n");
        // }

        //TODO: checkpoint 2
        createRoute<<<1,1>>>(thrust::raw_pointer_cast(d_u.data()), thrust::raw_pointer_cast(d_v.data()), d_x, d_y, d_demand, capacity, d_earlyTime, d_latestTime, d_serviceTime, thrust::raw_pointer_cast(d_route.data()), nodes,d_min_cost_route,thrust::raw_pointer_cast(opt_final_route.data()),d_route_len);
        cudaDeviceSynchronize();
        // for(int i=0;i<final_route.size();i++){
        //     cout<<final_route[i]<<" ";
        // }
        // cout<<endl;

        //TODO: Implement shuffling
        step++;
    }
    // cout<<"print the min cost route:\n";
    // for(int i=0;i<opt_final_route.size();i++){
    //     if(opt_final_route[i]==-1)
    //         break;
    //     cout<<opt_final_route[i]<<" ";
    // }


    thrust::device_vector<int> d_optimized_route(nodes*2);

    postprocess_tsp_approx<<<1,1>>>(thrust::raw_pointer_cast(opt_final_route.data()), opt_final_route.size(), d_x, d_y, d_demand, capacity, d_earlyTime, d_latestTime, d_serviceTime, thrust::raw_pointer_cast(d_optimized_route.data()));
    cudaDeviceSynchronize();

    thrust::device_vector<int> d_final_route(opt_final_route);
    postprocess_2_opt<<<1,1>>>(thrust::raw_pointer_cast(d_final_route.data()), d_final_route.size(), d_x, d_y, d_demand, capacity, d_earlyTime, d_latestTime, d_serviceTime, thrust::raw_pointer_cast(d_optimized_route.data()));
    cudaDeviceSynchronize();

    //Time measurement end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    int *h_final_route=new int[d_final_route.size()];
    cudaMemcpy(h_final_route, thrust::raw_pointer_cast(d_final_route.data()), d_final_route.size()*sizeof(int),cudaMemcpyDeviceToHost);

    printRoute(h_final_route, d_final_route.size(),h_x,h_y);

    //Validate the final route in cpu side
    int *h_final_route_=new int[d_final_route.size()];
    cudaMemcpy(h_final_route_, thrust::raw_pointer_cast(d_final_route.data()), d_final_route.size()*sizeof(int),cudaMemcpyDeviceToHost);
    bool isValid=validateRoute(h_final_route_, d_final_route.size(), h_x, h_y, h_demand, capacity, h_earlyTime, h_latestTime, h_serviceTime);
    
    // for(int i=0;i<d_final_route.size();i++){
    //     cout<<h_final_route_[i]<<" ";
    // }
    if(isValid){
        cout<<"Final route is valid"<<endl;
    }
    else{
        cout<<"Final route is invalid"<<endl;
    }



    delete[] h_x;
    delete[] h_y;
    delete[] h_demand;
    delete[] h_earlyTime;
    delete[] h_latestTime;
    delete[] h_serviceTime;
    return 0;

}