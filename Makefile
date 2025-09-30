# Compiler and flags
CXX = g++

# Default build
test:
	$(CXX) test.cpp -o test.out
	./test.out < c101.txt

main:
	$(CXX) seq_CVRPTW.cpp -o seq.out
	./seq.out c101.txt

# Clean build
clean:
	rm -f *.out
