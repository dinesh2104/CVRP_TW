input_file = "data.txt"       # Input file with raw data
output_file = "latex_table.txt"  # Output file with LaTeX-ready rows

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue  # skip empty lines

        # Split columns (space or tab)
        parts = line.split()

        # --- Clean and escape filename ---
        if parts[0].startswith("inputs/"):
            filename = parts[0].replace("inputs/", "")
        else:
            filename = parts[0]
        filename = filename.replace("_", r"\_")  # escape underscores

        # Replace first part with formatted filename
        parts[0] = filename

        # Join with LaTeX separators
        latex_line = " & ".join(parts) + r" \\"
        outfile.write(latex_line + "\n")

print("✅ Conversion complete! Output written to:", output_file)
