all:
	nvcc -arch=sm_80 solver.cu -o solver
	./solver c101.txt

test1:
	nvcc -arch=sm_80 solver.cu -o solver
	./solver c101.txt

test2:
	nvcc -arch=sm_80 solver.cu -o solver
	./solver r102.txt

build:
	nvcc -arch=sm_80 solver.cu -o solver

clean:
	rm -f solver mst
	rm -rf outputs
	