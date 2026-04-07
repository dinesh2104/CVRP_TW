#!/bin/bash
mkdir -p outputs
make

result_file="outputs/result.csv"


# Define the array of angles to test
angles=(30 45 60 90 180)

for infile in testcase/*; do
    filename=$(basename "$infile")
    best_outfile="outputs/${filename}.out"
    
    # Initialize variables to track the best solution
    best_angle=""
    best_cost=999999999 # Start with an arbitrarily large number
    best_err_output=""

    echo "Processing $infile..."

    for angle in "${angles[@]}"; do
        temp_outfile="outputs/${filename}_${angle}.tmp"
        temp_errfile="outputs/${filename}_${angle}.err"

        # Run the executable, redirecting stdout and stderr to temporary files
        ./solve_cvrptw "$infile" "$angle" > "$temp_outfile" 2> "$temp_errfile"

        # ======================================================================
        # EXTRACTION LOGIC: Update the grep/awk command below to match your 
        # C++ program's exact output format. 
        # For example, if your program prints "Total Distance: 1234.56", 
        # this will grab the "1234.56".
        # ======================================================================
        current_cost=$(grep -i "cost\|distance" "$temp_outfile" | awk '{print $NF}')

        # Skip if the solver failed or didn't output a valid cost
        if [ -z "$current_cost" ]; then
            echo "  [Angle $angle] Failed to extract a valid cost/distance."
            rm -f "$temp_outfile" "$temp_errfile"
            continue
        fi

        echo "  [Angle $angle] Cost = $current_cost"

        # Compare costs using bc (handles floating point numbers)
        is_better=$(echo "$current_cost < $best_cost" | bc -l)

        if [ "$is_better" -eq 1 ]; then
            best_cost=$current_cost
            best_angle=$angle
            
            # Overwrite the best outfile with this current best
            mv "$temp_outfile" "$best_outfile"
            # Read the stderr output to append to the CSV later
            best_err_output=$(cat "$temp_errfile")
        else
            # Not the best, discard the temp output file
            rm -f "$temp_outfile"
        fi
        
        # Always clean up the temp error file
        rm -f "$temp_errfile"
    done

    if [ -n "$best_angle" ]; then
        echo "-> Best result for $filename is Angle: $best_angle (Cost: $best_cost)"
        # Append the best run's stderr to the result csv
        echo "$best_err_output" >> "$result_file"
    else
        echo "-> Could not find a valid solution for $filename across any angle."
    fi
    echo "---------------------------------------------------"
done

echo "All files processed. Best results are in outputs/ and $result_file"