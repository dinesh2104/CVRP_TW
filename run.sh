#!/bin/bash

val=3

mkdir -p outputs$val
make build$val

result_file="outputs$val/result.sol"

for infile in testcase/*; do
    filename=$(basename "$infile")
    outfile="outputs$val/${filename}.out"
    ./seq.out "$infile" > "$outfile" 2>> "$result_file"
    echo "Processed $infile -> $outfile"
done
echo "All files processed. Results are in outputs$val/result.sol"