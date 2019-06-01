kuda:
	nvcc src.cu -O3 -o perm  -Xcompiler -fopenmp -Xcompiler -O3 -Xcompiler -std=c++11

cpu:
	g++ -fopenmp permanent_hw1.cpp -mavx -mavx2 -o cpu_perm
