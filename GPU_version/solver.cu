//1. First get the input(Done)
//2. Code for MST
//3. Code for different DFS traversals
//4. Code for different optimization techniques

#include<iostream>
#include<cuda_runtime.h>
#include<string.h>
#include<fstream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/device_free.h>
#include <thrust/device_free.h>
#include <ctime>

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
            cerr << "Error reading line: " << line << endl;
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

__global__ void weightUpdate(int *d_x, int *d_y, int *d_weights, bool *d_inMST, int *d_parent, int current, int nodes) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= nodes)
        return;
    if(d_inMST[id])
        return;
    int dx = d_x[current] - d_x[id];
    int dy = d_y[current] - d_y[id];
    int distance = dx * dx + dy * dy; // Squared distance to avoid sqrt for efficiency
    if(!d_inMST[id] && d_weights[id] > distance) {
        d_weights[id] = distance;
        d_parent[id] = current;
    }
    if(id==1){
        printf("Current: %d, Weights[1]: %d\n",current,d_weights[1]);
    }
}

int main(char* argv[], int argc) {
    // if(argc<1) {
    //     std::cout<<"Please provide the input file name"<<std::endl;
    //     return -1;
    // }
    string filename="c101.txt";
    int *h_x,*h_y;
    double *h_demand,*h_earlyTime,*h_latestTime,*h_serviceTime;


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

    int *parent=new int[nodes];
    parent[0]=-1;
    bool *inMST=new bool[nodes];
    vector<int> weights(nodes);
    for(int i=0;i<nodes;i++){
        weights[i]=INT_MAX;
        inMST[i]=false;
    }
    //welcome
    weights[0]=0;

    thrust::device_vector<int> d_weights(weights.begin(), weights.end());
    thrust::device_ptr<int> ptr=d_weights.data();
    thrust::device_vector<bool> inMST_d(inMST,inMST+nodes);

    while(cnt<nodes-1){
        cnt++;
        inMST[current]=true;

        weightUpdate<<<1,nodes>>>(d_x, d_y, thrust::raw_pointer_cast(d_weights.data()), thrust::raw_pointer_cast(inMST_d.data()), parent, current, nodes);
        cudaDeviceSynchronize();
        break;
    }


    delete[] h_x;
    delete[] h_y;
    delete[] h_demand;
    delete[] h_earlyTime;
    delete[] h_latestTime;
    delete[] h_serviceTime;
    return 0;

}