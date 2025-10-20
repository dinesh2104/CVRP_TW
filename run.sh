#!/bin/bash

mkdir -p outputs

make build

result_file="outputs/result.sol"

for infile in inputs/*; do
    filename=$(basename "$infile")
    outfile="outputs/${filename}.out"
    ./seq.out "$infile" > "$outfile" 2>> "$result_file"
    echo "Processed $infile -> $outfile"
done
