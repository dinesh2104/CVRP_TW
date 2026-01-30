CXX = g++

# Testing
test1:
	$(CXX) seq_CVRPTW.cpp -o seq.out
	./seq.out c101.txt

test2:
	$(CXX) seq_CVRPTW.cpp -o seq.out
	./seq.out r102.txt

test3:
	$(CXX) seq_CVRPTW_v2.cpp -o seq.out
	./seq.out r102.txt


build:
	$(CXX) seq_CVRPTW.cpp -o seq.out

build2:
	$(CXX) seq_CVRPTW_v2.cpp -o seq.out

# Clean
clean:
	rm -f *.out 
	rm -rf outputs
