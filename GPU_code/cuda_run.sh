#!/bin/bash

INPUT_DIR="inputs"
OUTPUT_DIR="outputs"
SOLVER="./solver"

make build

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.txt; do
    base=$(basename "$file" .txt)
    echo "Processing $file ..."
    
    "$SOLVER" "$file" > "$OUTPUT_DIR/${base}.out" 2>>"$OUTPUT_DIR/results.sol"
    
    echo "Saved output to $OUTPUT_DIR/${base}.out"
done

echo "All files processed!"