#include <cuda_runtime.h>

__device__ void swap(int* a, int* b);
__global__ void sum(int* a, int*b, int*c, int n);

__global__ void ordina_riga(int * array, int colonne, int righe, int dim);
__global__ void check_confini(int *array, int colonne, int righe, int dim);
__global__ void kernel_padre(int *array, int colonne, int righe, int dim, dim3 block, dim3 grid);
