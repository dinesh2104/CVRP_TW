#!/bin/bash

# Input and output directories
INPUT_DIR="inputs"
OUTPUT_DIR="outputs"
SOLVER="./solver"

make build

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all .txt files in input folder
for file in "$INPUT_DIR"/*.txt; do
    # Extract base filename (e.g., c101.txt → c101)
    base=$(basename "$file" .txt)
    echo "Processing $file ..."
    
    # Run solver and save output
    "$SOLVER" "$file" > "$OUTPUT_DIR/${base}.out"
    
    echo "Saved output to $OUTPUT_DIR/${base}.out"
done

echo "✅ All files processed!"