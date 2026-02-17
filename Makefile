CXX = g++

# Testing
test0:
	$(CXX) seq_CVRPTW.cpp -o seq.out
	./seq.out c101.txt

test1:
	$(CXX) seq_CVRPTW.cpp -o seq.out
	./seq.out r102.txt

test2:
	$(CXX) seq_CVRPTW_v2.cpp -o seq.out
	./seq.out r102.txt

test2.1:
	$(CXX) seq_CVRPTW_v2.1.cpp -o seq.out
	./seq.out r102.txt

test3:
	$(CXX) seq_CVRPTW_v3_cluster.cpp -o seq.out
	./seq.out RC2_10_2.txt

test4:
	$(CXX) seq_CVRPTW_v4.cpp -o seq.out
	./seq.out r102.txt

test5:
	$(CXX) seq_CVRPTW_v5.cpp -o seq.out
	./seq.out r102.txt

test6:
	$(CXX) seq_CVRPTW_v6_clark.cpp -o seq.out
	./seq.out c_50.txt

test7:
	$(CXX) seq_CVRPTW_v7_clarkwithTW.cpp -o seq.out
	./seq.out c_50.txt

build1:
	$(CXX) seq_CVRPTW.cpp -o seq.out

build2:
	$(CXX) seq_CVRPTW_v2_addFeasibleCust.cpp -o seq.out

build2.1:
	$(CXX) seq_CVRPTW_v2.1.cpp -o seq.out

build3:
	$(CXX) seq_CVRPTW_v3_cluster.cpp -o seq.out

build3.1:
	$(CXX) seq_CVRPTW_v3.1_cluster.cpp -o seq.out

build4:
	$(CXX) seq_CVRPTW_v4.cpp -o seq.out

build5:
	$(CXX) seq_CVRPTW_v5.cpp -o seq.out

build6:
	$(CXX) seq_CVRPTW_v6_clark.cpp -o seq.out

build7:
	$(CXX) seq_CVRPTW_v7_clarkwithTW.cpp -o seq.out

# Clean
clean:
	rm -f *.out 
	rm -rf outputs*
