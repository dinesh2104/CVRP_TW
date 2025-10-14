# Compiler and flags
CXX = g++

# Default build
test:
	$(CXX) seq_CVRPTW.cpp -o seq.out
	./seq.out toy.txt

test2:
	$(CXX) seq_CVRPTW.cpp -o seq.out
	./seq.out toy2.txt

test3:
	$(CXX) seq_CVRPTW.cpp -o seq.out
	./seq.out c100.txt

main:
	$(CXX) seq_CVRPTW.cpp -o seq.out
	./seq.out c101.txt

build:
	$(CXX) seq_CVRPTW.cpp -o seq.out

# Clean build
clean:
	rm -f *.out 
	rm -rf outputs
