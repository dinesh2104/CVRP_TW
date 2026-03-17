#!/bin/bash

val=9

# Create output directory and build
mkdir -p "outputs9.1"
make build$val

result_file="outputs9.1/result.csv"


echo "Processing test cases (Running 5 angles each, saving the best)..."

for infile in testcase/*; do
    # Skip if it is not a file (e.g., a subdirectory)
    [ -f "$infile" ] || continue

    filename=$(basename "$infile")
    outfile="outputs9.1/${filename}.out"
    
    # Initialize best cost to a ridiculously high number
    best_cost=999999999.0
    
    # Create temporary files to hold the best outputs
    best_out_file=$(mktemp)
    best_err_file=$(mktemp)
    
    echo -n "Running $filename "
    
    # Correct Bash array syntax (spaces, no commas)
    angles=(30 45 60 90 180)

    # Inner loop: Iterate directly through the array values
    for current_angle in "${angles[@]}"; do
        # Temp files for this specific run
        temp_out=$(mktemp)
        temp_err=$(mktemp)

        # Run the solver passing the current angle
        ./seq.out "$infile" "$current_angle" > "$temp_out" 2> "$temp_err"
        
        # Extract the Final_Cost from the summary line
        cost=$(cat "$temp_err" "$temp_out" | awk '{
            for(i=1; i<=NF; i++) {
                if ($i == "Final_Cost:") {
                    print $(i+1)
                    exit
                }
            }
        }')
        
        # If the solver failed and didn't output a cost, set to infinity
        if [ -z "$cost" ]; then
            cost=999999999.0
        fi
        
        # Compare floating point numbers using awk
        is_better=$(awk -v c1="$cost" -v c2="$best_cost" 'BEGIN {print (c1 < c2) ? 1 : 0}')
        
        # If this run is strictly better, save it as the new best
        if [ "$is_better" -eq 1 ]; then
            best_cost="$cost"
            cat "$temp_out" > "$best_out_file"
            cat "$temp_err" > "$best_err_file"
        fi
        
        # Print a dot to show progress
        echo -n "."
        
        # Delete temp files for this run
        rm -f "$temp_out" "$temp_err"
    done
    
    # After all 5 angles, save the absolute best one to the final destination
    if [ $(awk -v bc="$best_cost" 'BEGIN{print (bc < 999999999.0) ? 1 : 0}') -eq 1 ]; then
        cat "$best_out_file" > "$outfile"
        cat "$best_err_file" >> "$result_file"
        echo " Best Cost: $best_cost -> Saved"
    else
        echo " Failed to find a valid solution in 5 runs."
    fi
    
    # Clean up the best tracking files
    rm -f "$best_out_file" "$best_err_file"
done

echo "All files processed. Best results are in outputs9.1/result.csv"