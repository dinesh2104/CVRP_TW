#define MAX_ROUTE_LEN 256  // adjust as needed

__global__ void postprocess_2_opt(
    int *final_route, int total_length,
    int *d_x, int *d_y, double *d_demand,
    int capacity, double *d_earlyTime, double *d_latestTime, double *d_serviceTime,
    int *optimized_route)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;  // sequential kernel

    printf("Running 2-OPT on concatenated routes (total length = %d)\n", total_length);

    int start = -1;  // start index of current route
    int end = -1;    // end index (inclusive)

    // Iterate through the concatenated routes
    for (int idx = 0; idx < total_length; ++idx)
    {
        if (final_route[idx] == 0)
        {
            if (start == -1)
            {
                // Found the start of a new route
                start = idx;
            }
            else
            {
                // Found the end of the current route
                end = idx;

                int route_length = end - start + 1;

                // Skip very short routes (0 X 0 or 0 0)
                if (route_length > 2)
                {
                    // Copy this route segment into temp buffers
                    int local_route[MAX_ROUTE_LEN];
                    int local_opt[MAX_ROUTE_LEN];

                    for (int i = 0; i < route_length; ++i)
                    {
                        local_route[i] = final_route[start + i];
                        local_opt[i] = final_route[start + i];
                    }

                    double best_distance = calculate_cost(local_route, route_length, d_x, d_y, d_demand);

                    bool improvement = true;
                    int iteration = 0;

                    // Perform 2-opt until no improvement or max iterations
                    while (improvement && iteration < 10)
                    {
                        improvement = false;

                        for (int i = 1; i < route_length - 2; ++i)
                        {
                            for (int k = i + 1; k < route_length - 1; ++k)
                            {
                                // Reverse segment [i, k]
                                int temp[MAX_ROUTE_LEN];
                                for (int c = 0; c < i; ++c)
                                    temp[c] = local_opt[c];
                                int dec = 0;
                                for (int c = i; c <= k; ++c)
                                    temp[c] = local_opt[k - dec++];
                                for (int c = k + 1; c < route_length; ++c)
                                    temp[c] = local_opt[c];

                                double new_distance = calculate_cost(temp, route_length, d_x, d_y, d_demand);

                                if (new_distance + 1e-6 < best_distance &&
                                    verify_route(temp, route_length, d_x, d_y, d_demand,
                                                 capacity, d_earlyTime, d_latestTime, d_serviceTime))
                                {
                                    for (int c = 0; c < route_length; ++c)
                                        local_opt[c] = temp[c];

                                    best_distance = new_distance;
                                    improvement = true;

                                    printf("Improved route [%d-%d]: cost %.2f -> %.2f (swap %d,%d)\n",
                                           start, end, best_distance, new_distance, i, k);
                                }
                            }
                        }
                        iteration++;
                    }

                    // Copy optimized route back to global final_route
                    for (int i = 0; i < route_length; ++i)
                        final_route[start + i] = local_opt[i];

                    printf("Route [%d-%d] optimized: final cost = %.2f\n", start, end, best_distance);
                }

                // Prepare for next route
                start = end;  // next route will start here again
            }
        }
    }

    printf("All 2-OPT route optimizations complete.\n");
}
