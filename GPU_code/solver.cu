#include <iostream>
#include <vector>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/device_free.h>
#include <ctime>
#include <fstream>
#include <climits>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
//#define MAX_ROUTE_LEN 256

using namespace std;

#define CUDA_CHECK(call)                                         \
    do                                                           \
    {                                                            \
        cudaError_t e = (call);                                  \
        if (e != cudaSuccess)                                    \
        {                                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(e) \
                      << " at " << __FILE__ << ":" << __LINE__   \
                      << " (err=" << e << ")\n";                 \
            std::abort();                                        \
        }                                                        \
    } while (0)

/* Device and Host function to cal Distance using (x1,y1) and (x2,y2)*/
__host__ __device__ float calculateDistance(int x1, int y1, int x2, int y2)
{
    double dx = double(x1) - double(x2);
    double dy = double(y1) - double(y2);
    return sqrtf(dx * dx + dy * dy); // Squared distance
}


__device__ double calculate_cost(int *tour, int tour_length, int *d_x, int *d_y, double *d_demand)
{
    double total_cost = 0.0;
    for (int i = 0; i < tour_length - 1; i++)
    {
        int from = tour[i];
        int to = tour[i + 1];
        total_cost += calculateDistance(d_x[from], d_y[from], d_x[to], d_y[to]);
    }
    // Add cost to return to depot
    total_cost += calculateDistance(d_x[tour[tour_length - 1]], d_y[tour[tour_length - 1]], d_x[0], d_y[0]);
    return total_cost;
}

/* Device function to calculate cost of a given tour/route is the route is 0 x y z 0 */
__device__ double calculate_local_cost(int *tour, int tour_length, int *d_x, int *d_y)
{
    // printf("Calculating local cost for tour: ");
    // for(int i=0;i<tour_length;i++){
    //     printf("%d ",tour[i]);
    // }
    // printf("\n");
    double total_cost = 0.0;
    for (int i = 0; i < tour_length - 1; i++)
    {
        int from = tour[i];
        int to = tour[i + 1];
        total_cost += calculateDistance(d_x[from], d_y[from], d_x[to], d_y[to]);
    }
    return total_cost;
}

__host__ double calculate_local_cost_host(int *tour, int tour_length, int *d_x, int *d_y)
{
    double total_cost = 0.0;
    for (int i = 0; i < tour_length - 1; i++)
    {
        int from = tour[i];
        int to = tour[i + 1];
        total_cost += calculateDistance(d_x[from], d_y[from], d_x[to], d_y[to]);
    }
    return total_cost;
}
// --------------------------------------------------------------------------
/* Function that read the data*/
pair<int,int> read(const string &filename, int *h_x, int *h_y, double *h_demand,
         double *h_earlyTime, double *h_latestTime, double *h_serviceTime)
{
    ifstream infile(filename);
    if (!infile)
    {
        cerr << "Error opening file: " << filename << endl;
        return {-1,-1};
    }
    cerr << filename << ", ";
    string line;

    // Skip the first 4 lines
    for (int i = 0; i < 4 && getline(infile, line); i++)
        ;

    int vehicleCapacity, nvehicles;
    {
        // Read vehicle info line safely
        getline(infile, line);
        stringstream ss(line);
        ss >> nvehicles >> vehicleCapacity;
    }

    // Skip the next 4 lines
    for (int i = 0; i < 4 && getline(infile, line); i++)
        ;

    int idx = 0;

    while (getline(infile, line))
    {
        if (line.empty())
            continue; // Skip blank lines

        stringstream ss(line);
        int no, x, y;
        double demand, earlyTime, latestTime, serviceTime;

        if (!(ss >> no >> x >> y >> demand >> earlyTime >> latestTime >> serviceTime))
        {
            // Optional: log malformed line
            // cerr << "Skipping malformed line: " << line << endl;
            continue;
        }

        h_x[idx] = x;
        h_y[idx] = y;
        h_demand[idx] = demand;
        h_earlyTime[idx] = earlyTime;
        h_latestTime[idx] = latestTime;
        h_serviceTime[idx] = serviceTime;
        idx++;
    }

    infile.close();
    return {idx,vehicleCapacity};
}

__global__ void weightUpdate(int *d_x, int *d_y, float *d_weights, bool *d_inMST, int *d_parent, int current, int nodes)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id == current)
    {
        d_weights[id] = INT_MAX;
        return;
    }
    if (id >= nodes)
        return;
    // if(id==nodes-1){
    //     printf("Current: %d\n",current);
    //     printf("Weights[%d]: %d\n",id,d_weights[id]);
    //     printf("Parent[%d]: %d\n",id,d_parent[id]);
    //     printf("d_x[%d]: %d, d_y[%d]: %d\n",id,d_x[id],id,d_y[id]);
    // }

    if (d_inMST[id])
        return;
    int dx = d_x[current] - d_x[id];
    int dy = d_y[current] - d_y[id];
    float distance = sqrtf(dx * dx + dy * dy); // Squared distance to avoid sqrt for efficiency
    if (!d_inMST[id] && d_weights[id] > distance)
    {
        d_weights[id] = distance;
        d_parent[id] = current;
    }
    // if(id==20){
    //     printf("Current: %d, Weights[id]: %d\n",current,d_weights[id]);
    // }
}

/* Kernels to create the route*/
__device__ void dfs_iterative(int start, bool *visited, int *d_route, int &route_idx, int *d_u, int *d_v)
{
    // Manual stack (big enough for graph size, adjust as needed)
    int stack[1024];
    int top = -1;

    // Push start node
    stack[++top] = start;

    while (top >= 0)
    {
        int node = stack[top--]; // pop
        // printf("%d, %d",node,visited[node]);
        if (!visited[node])
        {
            // printf("%d \n", node);

            visited[node] = true;
            d_route[route_idx++] = node;

            int idx = d_u[node];
            int idx_end = d_u[node + 1];
            // printf("%d %d\n", idx, idx_end);

            // Push neighbors (reverse order if you want same traversal as recursive)
            for (int i = idx_end - 1; i >= idx; i--)
            {
                int neighbor = d_v[i];
                if (!visited[neighbor])
                {
                    stack[++top] = neighbor;
                }
            }
        }
    }
}

__global__ void createRoute(int *d_u, int *d_v, int *d_x, int *d_y, double *d_demand, int capacity, double *d_earlyTime,
                            double *d_latestTime, double *d_serviceTime, int *d_route, int nodes, double *min_route_cost,
                            int *opt_final_route, int *route_len, bool *visited, int *final_route)
{
    // printf("Creating Route\n");
    int route_idx = 0;
    // bool visited=new bool[nodes];
    for (int i = 0; i < nodes; i++)
    {
        visited[i] = false;
    }

    // dfs_iterative(0, visited, d_route, route_idx, d_u, d_v);

    /* Code for iterative DFS*/
    int stack[1024];
    int top = -1;

    // Push start node
    stack[++top] = 0;

    while (top >= 0)
    {
        int node = stack[top--]; // pop
        // printf("%d, %d",node,visited[node]);
        if (!visited[node])
        {
            // printf("%d \n", node);

            visited[node] = true;
            d_route[route_idx++] = node;

            int idx = d_u[node];
            int idx_end = d_u[node + 1];
            // printf("%d %d\n", idx, idx_end);

            // Push neighbors (reverse order if you want same traversal as recursive)
            for (int i = idx_end - 1; i >= idx; i--)
            {
                int neighbor = d_v[i];
                if (!visited[neighbor])
                {
                    stack[++top] = neighbor;
                }
            }
        }
    }
    /*-----------------------*/

    // printf("DFS interation: ");
    // for(int i=0;i<route_idx;i++){
    //     printf("%d ",d_route[i]);
    // }
    // printf("\n");
    // printf("Finished\n");
    // printf("\n");
    // int *final_route=new int[route_idx*2];

    int idx = 0;
    int residual_capacity = capacity;
    double current_time = 0;
    // int* final_route=new int[route_idx*2];
    idx++;
    int prev = 0;
    for (int i = 1; i < route_idx; i++)
    {
        int node = d_route[i];
        double travel_time = calculateDistance(d_x[prev], d_y[prev], d_x[node], d_y[node]);
        current_time += travel_time;
        if (current_time < d_earlyTime[node])
        {
            current_time = d_earlyTime[node];
        }
        if (residual_capacity >= d_demand[node] && current_time <= d_latestTime[node])
        {
            final_route[idx] = node;
            idx++;
            residual_capacity -= d_demand[node];
            current_time += d_serviceTime[node];
            prev = node;
        }
        else
        {
            final_route[idx] = 0;
            idx++;

            // go to the current node from depot
            travel_time = calculateDistance(d_x[0], d_y[0], d_x[node], d_y[node]);
            current_time = travel_time;
            if (current_time < d_earlyTime[node])
            {
                current_time = d_earlyTime[node];
            }
            if (current_time > d_latestTime[node])
            {
                printf("Node %d cannot be serviced due to time window constraints.\n", node);
                return;
            }
            final_route[idx] = node;
            idx++;
            residual_capacity = capacity - d_demand[node];
            current_time += d_serviceTime[node];
            prev = node;
        }
    }

    // printf("Printing the Final Route\n");
    // for(int i=0;i<idx;i++){
    //     printf("%d ",final_route[i]);
    // }
    // printf("\n");

    /* idx --> has the length of the final_route */
    double cost = calculate_cost(final_route, idx, d_x, d_y, d_demand);

    // printf("\ncost: %lf\n",cost);
    if (*min_route_cost == -1 || cost < *min_route_cost)
    {
        // printf("Found a better route with cost: %lf\n",cost);
        for (int i = 0; i < *route_len; i++)
        {
            opt_final_route[i] = 0;
        }
        *min_route_cost = cost;
        *route_len = idx;
        for (int i = 0; i < idx; i++)
        {
            opt_final_route[i] = final_route[i];
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

/*Kernels for post processing*/

__device__ bool verify_route(int *tour, int tour_length, int *d_x, int *d_y, double *d_demand, int capacity, double *d_earlyTime, double *d_latestTime, double *d_serviceTime)
{
    double current_time = 0.0;
    int current_load = 0;
    int prev_node = 0; // Start from depot

    for (int i = 0; i < tour_length; i++)
    {
        int node = tour[i];
        if (node == 0)
        { // Depot
            current_time = 0;
            current_load = 0; // Unload at depot
            prev_node = 0;
            continue;
        }

        current_time += calculateDistance(d_x[prev_node], d_y[prev_node], d_x[node], d_y[node]);

        if (current_time < d_earlyTime[node])
        {
            current_time = d_earlyTime[node];
        }
        if (current_time > d_latestTime[node])
        {
            // printf("Tour invalid: Arrived at node %d after latest time.\n", node);
            return false; // Violates latest time
        }

        current_time += d_serviceTime[node];
        current_load += d_demand[node];

        // Check capacity
        if (current_load > capacity)
        {
            // printf("Tour invalid: Capacity exceeded at node %d.\n", node);
            return false; // Exceeds vehicle capacity
        }

        prev_node = node;
    }

    // Return to depot at end of tour
    current_time += calculateDistance(d_x[prev_node], d_y[prev_node], d_x[0], d_y[0]);

    // printf("Tour valid: Completed with total time %.2f minutes.\n", current_time);
    return true;
}

/*Input tour 0 1 2 3 0 format*/
__device__ bool verify_local_route(int *tour, int tour_length, int *d_x, int *d_y, double *d_demand, int capacity, double *d_earlyTime, double *d_latestTime, double *d_serviceTime)
{
    // printf("Verifying local tour: ");
    // for(int i=0;i<tour_length;i++){
    //     printf("%d ",tour[i]);
    // }
    double current_time = 0.0;
    int current_load = 0;
    int prev_node = 0; // Start from depot

    for (int i = 1; i < tour_length - 1; i++)
    {
        int node = tour[i];
        current_time += calculateDistance(d_x[prev_node], d_y[prev_node], d_x[node], d_y[node]);

        if (current_time < d_earlyTime[node])
        {
            current_time = d_earlyTime[node];
        }
        if (current_time > d_latestTime[node])
        {
            return false;
        }

        current_time += d_serviceTime[node];
        current_load += d_demand[node];

        if (current_load > capacity)
        {
            return false;
        }

        prev_node = node;
    }

    // Return to depot at end of tour
    current_time += calculateDistance(d_x[prev_node], d_y[prev_node], d_x[0], d_y[0]);
    return true;
}

__device__ bool verify_tour(int *tour, int start, int end, int *d_x, int *d_y, double *d_demand, int capacity, double *d_earlyTime, double *d_latestTime, double *d_serviceTime)
{
    double current_time = 0.0;
    int current_load = 0;
    int prev_node = 0; // Start from depot
    // printf("start=%d end=%d\n", start, end);
    // printf("%d %d\n", tour[start], tour[end]);
    for (int i = start; i < end; i++)
    {
        int node = tour[i];
        if (node == 0)
        { // Depot
            current_time = 0;
            current_load = 0; // Unload at depot
            prev_node = 0;
            continue;
        }

        // Travel to next node
        current_time += calculateDistance(d_x[prev_node], d_y[prev_node], d_x[node], d_y[node]);

        // Check time window
        if (current_time < d_earlyTime[node])
        {
            current_time = d_earlyTime[node]; // Wait until early time
        }
        if (current_time > d_latestTime[node])
        {
            // printf("Tour invalid: Arrived at node %d after latest time.\n", node);
            return false; // Violates latest time
        }

        // Service the node
        current_time += d_serviceTime[node];
        current_load += d_demand[node];

        // Check capacity
        if (current_load > capacity)
        {
            // printf("Tour invalid: Capacity exceeded at node %d.\n", node);
            return false; // Exceeds vehicle capacity
        }

        prev_node = node;
    }

    // Return to depot at end of tour
    current_time += calculateDistance(d_x[prev_node], d_y[prev_node], d_x[0], d_y[0]);

    // printf("Tour valid: Completed with total time %.2f minutes.\n", current_time);
    return true;
}

__global__ void postprocess_tsp_approx(int *final_route, int route_length, int *d_x, int *d_y, double *d_demand, int capacity, double *d_earlyTime, double *d_latestTime, double *d_serviceTime, int *optimized_route_buf){
    for (int i = 0; i < route_length; i++)
    { 
        optimized_route_buf[i] = final_route[i];
    }

    // printf("Final route before TSP approx postprocessing: \n");
    // for (int i = 0; i < route_length; i++)
    // {
    //     printf("%d ", optimized_route_buf[i]);
    // }
    // printf("\n");

    int start=-1;
    int end=-1;

    for(int i=0;i<route_length;i++){
        if(start==-1 && optimized_route_buf[i]==0){
            start=i;
        }
        else if(start!=-1 && optimized_route_buf[i]==0){
            end=i;

            int local_route_len=end-start+1;
            if(local_route_len-2>2){
                int local_route[256];
                int local_opt[256];

                for(int j=0;j<local_route_len;j++){
                    local_route[j]=optimized_route_buf[start+j];
                    local_opt[j]=optimized_route_buf[start+j];
                }
                double best_distance=calculate_local_cost(local_route,local_route_len,d_x,d_y);

                for(int m=0;m<local_route_len-2;m++){
                    int min_index=-1;
                    double min_dist=INT_MAX;
                    for(int n=m+1;n<local_route_len-1;n++){
                        double dist=calculateDistance(d_x[local_opt[m]],d_y[local_opt[m]],d_x[local_opt[n]],d_y[local_opt[n]]);
                        if(dist<min_dist){
                            min_dist=dist;
                            min_index=n;
                        }
                    }
                    int temp=local_opt[m+1];
                    local_opt[m+1]=local_opt[min_index];
                    local_opt[min_index]=temp;
                    double new_distance=calculate_local_cost(local_opt,local_route_len,d_x,d_y);
                    if(new_distance<best_distance &&
                       verify_local_route(local_opt,local_route_len,d_x,d_y,d_demand,
                                    capacity,d_earlyTime,d_latestTime,d_serviceTime)){
                        // printf("Found better local route: ");
                        // for(int i=0;i<local_route_len;i++){
                        //     printf("%d ",local_opt[i]);
                        // }
                        // printf("\n");
                        best_distance=new_distance;
                    }else{
                        temp=local_opt[m+1];
                        local_opt[m+1]=local_opt[min_index];
                        local_opt[min_index]=temp;
                    }
                }

                // Copy optimized route back to global final_route
                for(int j=0;j<local_route_len;j++)
                    optimized_route_buf[start+j]=local_opt[j];
            }

            // Prepare for next route
            start=end; // next route will start here again
        }
    }

    //check whether optimized_route_buf is valid or not..
    if(verify_route(optimized_route_buf,route_length,d_x,d_y,d_demand,capacity,d_earlyTime,d_latestTime,d_serviceTime)){
        printf("optimized route is valid\n");
    }else{
        printf("optimized route is invalid\n");
    }
}

__global__ void postprocess_2_opt(
    int *final_route, int total_length,
    int *d_x, int *d_y, double *d_demand,
    int capacity, double *d_earlyTime, double *d_latestTime, double *d_serviceTime,
    int *optimized_route_buf)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;

    int start = -1;
    int end = -1;

    for(int i=0;i<total_length;i++){
        optimized_route_buf[i] = final_route[i];
    }

    for (int idx = 0; idx < total_length; ++idx)
    {
        if (optimized_route_buf[idx] == 0)
        {
            if (start == -1)
            {
                start = idx;
            }
            else
            {
                end = idx;

                int route_length = end - start + 1;

                if (route_length - 2 > 2)
                {
                    int local_route[256];
                    int local_opt[256];

                    for (int i = 0; i < route_length; ++i)
                    {
                        local_route[i] = optimized_route_buf[start + i];
                        local_opt[i] = optimized_route_buf[start + i];
                    }


                    double best_distance = calculate_local_cost(local_route, route_length, d_x, d_y);

                    bool improvement = true;
                    int iteration = 0;

                    while (improvement && iteration < 10)
                    {
                        improvement = false;

                        for (int i = 1; i < route_length - 2; ++i)
                        {
                            for (int k = i + 1; k < route_length - 1; ++k)
                            {
                                int temp[256];
                                for (int c = 0; c < i; ++c)
                                    temp[c] = local_opt[c];
                                int dec = 0;
                                for (int c = i; c <= k; ++c)
                                    temp[c] = local_opt[k - dec++];
                                for (int c = k + 1; c < route_length; ++c)
                                    temp[c] = local_opt[c];

                                double new_distance = calculate_local_cost(temp, route_length, d_x, d_y);

                                if (new_distance < best_distance &&
                                    verify_local_route(temp, route_length, d_x, d_y, d_demand,
                                                 capacity, d_earlyTime, d_latestTime, d_serviceTime))
                                {
                                    // printf("Improved route [%d-%d]: cost %.2f -> %.2f (swap %d,%d)\n",
                                    //        start, end, best_distance, new_distance, i, k);
                                    for (int c = 0; c < route_length; ++c)
                                        local_opt[c] = temp[c];

                                    best_distance = new_distance;
                                    improvement = true;

                                    
                                }
                            }
                        }
                        iteration++;
                    }

                    // Copy optimized route back to global optimized_route_buf
                    for (int i = 0; i < route_length; ++i)
                        optimized_route_buf[start + i] = local_opt[i];
                }

                // Prepare for next route
                start = end;  // next route will start here again
            }
            
        }
    }
    if(verify_route(optimized_route_buf,total_length,d_x,d_y,d_demand,capacity,d_earlyTime,d_latestTime,d_serviceTime)){
        printf("2-OPT optimized route is valid\n");
        // printf("Optimized Route: \n");
        // for(int i=0;i<total_length;i++){
        //     printf("%d ",optimized_route_buf[i]);
        // }
        // printf("\n");
        // printf("final route:\n");
        // for(int i=0;i<total_length;i++){
        //     printf("%d ",final_route[i]);
        // }
        // printf("\n");
    }else{
        printf("2-OPT optimized route is invalid\n");
    }

}



void printRoute(int *route, int length, int *h_x, int *h_y)
{
    // calculate total distance
    long double total_distance = 0;
    std::cout << "Final Route: \n";

    int k = 0;
    int prev = 0;
    for (int i = 1; i < length; i++)
    {
        cout << "Route #" << k + 1 << ": ";
        int flag = 0;
        while (i < length && route[i] != 0)
        {
            total_distance += calculateDistance(h_x[prev], h_y[prev], h_x[route[i]], h_y[route[i]]);
            std::cout << route[i] << " ";
            prev = route[i];
            i++;
            flag = 1;
        }
        if (flag == 0)
        {
            break;
        }
        cout << endl;
        k++;
        total_distance += calculateDistance(h_x[prev], h_y[prev], h_x[0], h_y[0]);
        prev = 0;
    }
    cout << "Total Distance: " << total_distance << endl;
    cerr << total_distance << ", " << k <<", ";
}

bool validateRoute(int *route, int length, int *h_x, int *h_y, double *h_demand, int capacity, double *h_earlyTime, double *h_latestTime, double *h_serviceTime)
{
    // Implement route validation logic here
    // Check for capacity constraints, time window violations, etc.
    double current_time = 0.0;
    int current_load = 0;
    int prev_node = 0; // Start from depot
    for (int i = 0; i < length; i++)
    {
        int node = route[i];
        if (node == 0)
        { // Depot
            current_time = 0;
            current_load = 0; // Unload at depot
            prev_node = 0;
            continue;
        }

        // Travel to next node
        current_time += calculateDistance(h_x[prev_node], h_y[prev_node], h_x[node], h_y[node]);

        // Check time window
        if (current_time < h_earlyTime[node])
        {
            current_time = h_earlyTime[node]; // Wait until early time
        }
        if (current_time > h_latestTime[node])
        {
            printf("Route invalid: Arrived at node %d after latest time.\n", node);
            return false; // Violates latest time
        }

        // Service the node
        current_time += h_serviceTime[node];
        current_load += h_demand[node];

        // Check capacity
        if (current_load > capacity)
        {
            printf("Route invalid: Capacity exceeded at node %d.\n", node);
            return false; // Exceeds vehicle capacity
        }

        prev_node = node;
    }
    return true;
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        std::cout << "Please provide the input file name" << std::endl;
        return -1;
    }
    string filename = argv[1];
    int *h_x, *h_y;
    double *h_demand, *h_earlyTime, *h_latestTime, *h_serviceTime;

    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << filename << "\n";
        return 1;
    }

    string line;
    int count = 0;
    while (getline(file, line))
    {
        count++;
    }
    file.close();

    h_x = new int[count];
    h_y = new int[count];
    h_demand = new double[count];
    h_earlyTime = new double[count];
    h_latestTime = new double[count];
    h_serviceTime = new double[count];
    // cout<<"Total number of lines in the file: "<<count<<endl;

    auto [nodes,capacity] = read(filename, h_x, h_y, h_demand, h_earlyTime, h_latestTime, h_serviceTime);
    cout<<"Vehicle Capacity: "<<capacity<<endl;
    cout << "Total number of nodes including depot: " << nodes << endl;
    for (int i = 0; i < 5; i++)
    {
        cout << "Customer " << i + 1 << ": (" << h_x[i] << ", " << h_y[i] << "), Demand: " << h_demand[i] << ", Time Window: [" << h_earlyTime[i] << ", " << h_latestTime[i] << "], Service Time: " << h_serviceTime[i] << endl;
    }

    /*Allocate device memory */
    int *d_x, *d_y;
    double *d_demand, *d_earlyTime, *d_latestTime, *d_serviceTime;
    cudaMalloc((void **)&d_x, nodes * sizeof(int));
    cudaMalloc((void **)&d_y, nodes * sizeof(int));
    cudaMalloc((void **)&d_demand, nodes * sizeof(double));
    cudaMalloc((void **)&d_earlyTime, nodes * sizeof(double));
    cudaMalloc((void **)&d_latestTime, nodes * sizeof(double));
    cudaMalloc((void **)&d_serviceTime, nodes * sizeof(double));
    cudaMemcpy(d_x, h_x, nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_demand, h_demand, nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_earlyTime, h_earlyTime, nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_latestTime, h_latestTime, nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_serviceTime, h_serviceTime, nodes * sizeof(double), cudaMemcpyHostToDevice);

    long long int edge_sum = 0;
    int current = 0;
    int cnt = 0;

    // int *parent=new int[nodes];
    // parent[0]=-1;
    // bool *inMST=new bool[nodes];
    // vector<float> weights(nodes);
    // for(int i=0;i<nodes;i++){
    //     weights[i]=INT_MAX;
    //     inMST[i]=false;
    // }

    // weights[0]=0.0f;

    thrust::device_vector<float> d_weights(nodes, INT_MAX);
    d_weights[0] = 0.0f;
    thrust::device_ptr<float> ptr = d_weights.data();
    thrust::device_vector<bool> d_inMST(nodes, false);
    thrust::device_vector<int> d_parent(nodes);
    d_parent[0] = -1;

    // ======================== Main code ====================================
    clock_t begin = clock();

    clock_t pre_start=clock();
    /* Pre-processing includes initial MST construction */

    // cout<<"calling MST kernel"<<endl;
    while (cnt < nodes - 1)
    {
        cnt++;
        d_inMST[current] = true;
        weightUpdate<<<1, nodes>>>(d_x, d_y, thrust::raw_pointer_cast(d_weights.data()), thrust::raw_pointer_cast(d_inMST.data()), thrust::raw_pointer_cast(d_parent.data()), current, nodes);
        cudaDeviceSynchronize();
        int min_index = thrust::min_element(ptr, ptr + nodes) - ptr;
        // cout<<"Current Node: "<<current<<", Next Node: "<<min_index<<", Weight: "<<d_weights[min_index]<<endl;
        // cout<<"Adding edge cost: "<<d_weights[min_index]<<endl;
        // cout<<"Added edge: "<<min_index<<endl;

        edge_sum += d_weights[min_index];
        current = min_index;
        d_weights[min_index] = INT_MAX;
    }

    // for(int i=0;i<nodes;i++){
    //     cout<<"Node: "<<i<<", Parent: "<<d_parent[i]<<endl;
    // }

    cerr<<"MST Cost:, "<<edge_sum<<", ";

    cout << "MST cost: " << edge_sum << endl;
    // TODO: MST working for 1000 nodes.

    /* Create a new adjacency list to represent the MST*/
    vector<vector<int>> mst_adj_list(nodes);
    for (int i = 1; i < nodes; ++i)
    {
        int parent = d_parent[i];
        mst_adj_list[parent].push_back(i);
        mst_adj_list[i].push_back(parent); // Since the MST is undirected
    }

    clock_t pre_end=clock();
    cerr << "MST Pre-processing time:, "<<double(pre_end - pre_start) / CLOCKS_PER_SEC<<", ";

    clock_t mid_start=clock();
    /* It includes shuffling + DFS kernel + Route Construction*/

    /* Generate Route using Preorder DFS*/
    int step = 1;
    thrust::device_vector<int> d_final_route_buf(nodes * 2);
    thrust::device_vector<int> d_opt_final_route(nodes * 2);
    int *d_route_len;
    cudaMalloc(&d_route_len, sizeof(int));
    cudaMemset(d_route_len, 0, sizeof(int));

    double *d_min_cost_route;
    cudaMalloc((void **)&d_min_cost_route, sizeof(double));
    double h_min_cost_route = -1;
    cudaMemcpy(d_min_cost_route, &h_min_cost_route, sizeof(double), cudaMemcpyHostToDevice);
    bool *d_visited;
    cudaMalloc((void **)&d_visited, nodes * sizeof(bool));

    while (step < 100000)
    {
        // shuffle the adj list
        for (auto &list : mst_adj_list)
        {
            std::shuffle(list.begin(), list.end(), std::default_random_engine(rand()));
        }

        vector<int> h_u(nodes + 1);
        vector<int> h_v;
        int edge_count = 0;
        for (int i = 0; i < nodes; i++)
        {
            h_u[i] = edge_count;
            for (int neighbor : mst_adj_list[i])
            {
                h_v.push_back(neighbor);
            }
            edge_count += mst_adj_list[i].size();
        }
        h_u[nodes] = edge_count;

        thrust::device_vector<int> d_route(nodes);
        // Convert adjacency list to CSR format for GPU processing
        thrust::device_vector<int> d_u(h_u.begin(), h_u.end());
        thrust::device_vector<int> d_v(h_v.begin(), h_v.end());

        // printing the CSR representation
        //  for(int i=0;i<=nodes;i++){
        //      printf("%d ",h_u[i]);
        //  }
        //  cout<<endl;
        //  for(int i=0;i<edge_count;i++){
        //      printf("%d ",h_v[i]);
        //  }

        // printing the node and neighbour
        //  cout<<"Printing the CSR:"<<endl;
        //  for(int i=0;i<5;i++){
        //      printf("%d: ",i);
        //      for(int j=h_u[i];j<h_u[i+1];j++){
        //          printf("%d ",h_v[j]);
        //      }
        //      printf("\n");
        //  }

        createRoute<<<1, 1>>>(thrust::raw_pointer_cast(d_u.data()), thrust::raw_pointer_cast(d_v.data()), d_x, d_y, d_demand, capacity, d_earlyTime, d_latestTime, d_serviceTime,
                              thrust::raw_pointer_cast(d_route.data()), nodes, d_min_cost_route, thrust::raw_pointer_cast(d_opt_final_route.data()),
                              d_route_len, d_visited, thrust::raw_pointer_cast(d_final_route_buf.data()));
        CUDA_CHECK(cudaDeviceSynchronize());
        step++;
    }

    // for(int i=0;i<d_opt_final_route.size();i++){
    //     cout<<d_opt_final_route[i]<<" ";
    // }
    // cout<<endl;

    clock_t mid_end=clock();
    cerr << "Main Loop time:, "<<double(mid_end - mid_start) / CLOCKS_PER_SEC<<", ";

    // TODO: Working fine for 1000 nodes

    // TODO: Need to Optimise and refactor the post processing...

    clock_t post_start=clock();
    /* Post-process includes tsp approx + 2-opt + final gpu to cpu memcpy + checking the final route validity*/

    thrust::device_vector<int> d_optimized_route1(nodes * 2);
    thrust::device_vector<int> d_optimized_route2(nodes * 2);
    thrust::device_vector<int> d_optimized_route3(nodes * 2);

    postprocess_tsp_approx<<<1, 1>>>(thrust::raw_pointer_cast(d_opt_final_route.data()), d_opt_final_route.size(), d_x, d_y, d_demand, capacity, d_earlyTime, d_latestTime, d_serviceTime, thrust::raw_pointer_cast(d_optimized_route1.data()));
    cudaDeviceSynchronize();

    postprocess_2_opt<<<1,1>>>(thrust::raw_pointer_cast(d_optimized_route1.data()), d_opt_final_route.size(), d_x, d_y, d_demand, capacity, d_earlyTime, d_latestTime, d_serviceTime, thrust::raw_pointer_cast(d_optimized_route2.data()));
    cudaDeviceSynchronize();


    postprocess_2_opt<<<1,1>>>(thrust::raw_pointer_cast(d_opt_final_route.data()), d_opt_final_route.size(), d_x, d_y, d_demand, capacity, d_earlyTime, d_latestTime, d_serviceTime, thrust::raw_pointer_cast(d_optimized_route3.data()));
    cudaDeviceSynchronize();


    clock_t end = clock();

    cout << "Time taken: " << double(end - begin) / CLOCKS_PER_SEC << " seconds" << endl;
    cerr << double(end - begin) / CLOCKS_PER_SEC <<", ";

    int *h_optimized_route1 = new int[d_optimized_route3.size()];
    int *h_optimized_route2 = new int[d_optimized_route3.size()];
    int *h_optimized_route3 = new int[d_optimized_route3.size()];
    int *h_final_route = new int[d_opt_final_route.size()];
    cudaMemcpy(h_optimized_route1, thrust::raw_pointer_cast(d_optimized_route1.data()), d_optimized_route1.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_optimized_route2, thrust::raw_pointer_cast(d_optimized_route2.data()), d_optimized_route2.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_optimized_route3, thrust::raw_pointer_cast(d_optimized_route3.data()), d_optimized_route3.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_route, thrust::raw_pointer_cast(d_opt_final_route.data()), d_opt_final_route.size() * sizeof(int), cudaMemcpyDeviceToHost);
    /* Check whether all these route are valid */
    bool isValid1 = validateRoute(h_optimized_route1, d_optimized_route1.size(), h_x, h_y, h_demand, capacity, h_earlyTime, h_latestTime, h_serviceTime);
    bool isValid2 = validateRoute(h_optimized_route2, d_optimized_route2.size(), h_x, h_y, h_demand, capacity, h_earlyTime, h_latestTime, h_serviceTime);
    bool isValid3 = validateRoute(h_optimized_route3, d_optimized_route3.size(), h_x, h_y, h_demand, capacity, h_earlyTime, h_latestTime, h_serviceTime);

    if(isValid1){
        cout<<"Optimized Route 1 is valid"<<endl;
    }
    if(isValid2){
        cout<<"Optimized Route 2 is valid"<<endl;
    }
    if(isValid3){
        cout<<"Optimized Route 3 is valid"<<endl;
    }
    

    /* Final updated value is stored in the d_final_route_*/
    int *h_final_route_ = new int[d_opt_final_route.size()];
    
    //Calculate costs of all three optimized routes and choose the best one
    int start_t=-1;
    int end_t=-1;
    for(int i=0;i<d_opt_final_route.size();i++){
        if(start_t==-1 && h_final_route[i]==0){
            start_t=i;
        }
        else if(start_t!=-1 && h_final_route[i]==0){
            end_t=i;

            int local_route_len=end_t-start_t+1;
            double cost1=calculate_local_cost_host(&h_optimized_route1[start_t],local_route_len,h_x,h_y);
            double cost2=calculate_local_cost_host(&h_optimized_route2[start_t],local_route_len,h_x,h_y);
            double cost3=calculate_local_cost_host(&h_optimized_route3[start_t],local_route_len,h_x,h_y);

            if(cost1<=cost2 && cost1<=cost3){
                for(int j=0;j<local_route_len;j++){
                    h_final_route_[start_t+j]=h_optimized_route1[start_t+j];
                }
            }else if(cost2<=cost1 && cost2<=cost3){
                for(int j=0;j<local_route_len;j++){
                    h_final_route_[start_t+j]=h_optimized_route2[start_t+j];
                }
            }else if(cost3<=cost1 && cost3<=cost2){
                for(int j=0;j<local_route_len;j++){
                    h_final_route_[start_t+j]=h_optimized_route3[start_t+j];
                }
            }else{
                for(int j=0;j<local_route_len;j++){
                    h_final_route_[start_t+j]=h_final_route_[start_t+j];
                }
            }

            // Prepare for next route
            start_t=end_t; // next route will start here again
        }
    }
    int prev_node=0;
    long double final_cost=0;
    for(int i=0;i<d_opt_final_route.size();i++){
        final_cost+=calculateDistance(h_x[prev_node],h_y[prev_node],h_x[h_final_route[i]],h_y[h_final_route[i]]);
        prev_node=h_final_route[i];
    }
    cerr<<"Final Route Cost:, "<<final_cost<<", ";

    printRoute(h_final_route_, d_opt_final_route.size(), h_x, h_y);

    bool isValid = validateRoute(h_final_route_, d_opt_final_route.size(), h_x, h_y, h_demand, capacity, h_earlyTime, h_latestTime, h_serviceTime);

    clock_t post_end=clock();
    cerr << "Post-processing time:, "<<double(post_end - post_start) / CLOCKS_PER_SEC<<endl;

    
    if (isValid)
    {
        cout << "Final route is valid" << endl;
    }
    else
    {
        cout << "Final route is invalid" << endl;
    }

    
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_demand);
    cudaFree(d_earlyTime);
    cudaFree(d_latestTime);
    cudaFree(d_serviceTime);
    delete[] h_x;
    delete[] h_y;
    delete[] h_demand;
    delete[] h_earlyTime;
    delete[] h_latestTime;
    delete[] h_serviceTime;
    // delete[] inMST;

    return 0;
}
