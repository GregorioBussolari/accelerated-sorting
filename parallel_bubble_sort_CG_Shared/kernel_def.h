#include <cuda_runtime.h>

__device__ void swap(int* a, int* b);
__global__ void sum(int* a, int*b, int*c, int n);

__global__ void parallel_bubble_sort(int * array, int colonne, int righe, int dim);
