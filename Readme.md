# CVRPTW

## Overview
This repository includes both sequential and parallel code, along with test cases for the  **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)**.

## Folder Layout
- `inputs/` — Contains input files for different instances.  
- `run.sh` — Script to run the sequential CVRPTW code.  
- `outputs/` *(created by script)* — Solver results and logs.
- `seq_CVRPTW.cpp` — C++ source code for the sequential CVRPTW solver.
- `GPU_code/` — Contains GPU-accelerated code for CVRPTW using CUDA.
- `solver.cu` — CUDA source code for GPU-based CVRPTW solver.
- `cuda_run.sh` — Script to run the GPU-based CVRPTW code.

## How to Run `run.sh` for Sequential CVRPTW
1. Make the script executable (if not already):  
   ```bash
   chmod +x run.sh
    ./run.sh
    ```

## How to Run `cuda_run.sh` for GPU CVRPTW
1. Make the script executable (if not already):
    ```bash
    cd GPU_code
    chmod +x cuda_run.sh
    ./cuda_run.sh
    ```

## Output
- Results will be written to `outputs/` and may include solution files and summaries.

## Observations
- Please refer to the sheet for the observations made from the experiments conducted using both sequential and parallel implementations.
[Link](https://docs.google.com/spreadsheets/d/1V28KGXpMsJk--x5vN1gmyf1QgQrvdzqU/edit?gid=2004541567#gid=2004541567)