#!/bin/bash

mkdir -p outputs
mkdir -p outputs2

make build

result_file="outputs1/result.sol"

for infile in inputs/*; do
    filename=$(basename "$infile")
    outfile="outputs1/${filename}.out"
    ./seq.out "$infile" > "$outfile" 2>> "$result_file"
    echo "Processed $infile -> $outfile"
done

make build2

result_file="outputs2/result.sol"

for infile in inputs/*; do
    filename=$(basename "$infile")
    outfile="outputs2/${filename}.out"
    ./seq.out "$infile" > "$outfile" 2>> "$result_file"
    echo "Processed $infile -> $outfile"
done