

input_file = "result.sol"
output_file = "solution_CVRPTW.csv"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Split by whitespace and join with commas
        parts = line.split()
        outfile.write(",".join(parts) + "\n")
