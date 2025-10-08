all:
	nvcc -arch=sm_80 solver.cu -o solver
	./solver c101.txt

test:
	nvcc -arch=sm_80 mst.cu -o mst
	./mst c101.txt

build:
	nvcc -arch=sm_80 solver.cu -o solver

clean:
	rm -f solver mst
	