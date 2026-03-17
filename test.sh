#!/bin/bash

# Define the output file
OUTPUT_FILE="result.sol"


echo "Starting 4 iterations of 'make build-9'..."

for i in {1..5}
do
    echo "Running iteration $i/4..."
    
    # Write a header for this run to the file
    echo "=== Run $i ===" >> "$OUTPUT_FILE"
    
    # 1. The '2>&1' ensures we capture BOTH standard output and standard error
    FULL_OUTPUT=$(make test9 2>&1)
    
    # 2. Extract only the lines starting with "Cluster"
    echo "$FULL_OUTPUT" | grep "^Cluster " >> "$OUTPUT_FILE"
    
    # 3. Bulletproof extraction: Find "Final_Cost:" and print it along with the number right after it
    echo "$FULL_OUTPUT" | awk '{
        for(i=1; i<=NF; i++) {
            if ($i == "Final_Cost:") {
                print $i " " $(i+1)
            }
        }
    }' >> "$OUTPUT_FILE"
    
    # Add a blank line for readability between runs
    echo "" >> "$OUTPUT_FILE"
done

echo "Done! Results have been saved to $OUTPUT_FILE"