#!/bin/bash

# Create output folder if not exists
mkdir -p outputs

make main

result_file="outputs/result.sol"

# Loop through all files in inputs directory
for infile in inputs/*; do
    # Get the base filename (without directory)
    filename=$(basename "$infile")

    # Define output file path
    outfile="outputs/${filename}.out"

    # Run the program with input redirection and save output
    ./seq.out "$infile" > "$outfile" 2>> "$result_file"

    echo "Processed $infile -> $outfile"
done
