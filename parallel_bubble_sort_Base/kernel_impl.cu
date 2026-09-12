#include "kernel_def.h"
#include <stdio.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        return; \
    } \
}


__device__ int counter = 0;

__device__ void swap(int* a, int* b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

/* __global__ void kernel_padre(int *array, int colonne, int righe, int dim, dim3 block, dim3 grid) {
    
    for(int i=0; i<dim;i++){
        ordina_riga <<<grid, block>>>(array, colonne, righe, dim);
        cudaDeviceSynchronize();
        check_confini<<<grid, block>>>(array, colonne, righe, dim);
        cudaDeviceSynchronize();
    }
    
} */

__global__ void ordina_riga(int * array, int colonne,int righe, int dim){

    int x = blockIdx.x * blockDim.x + threadIdx.x;   // colonna globale
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // riga globale
    int tid = y * (gridDim.x * blockDim.x) + x;

    int start = tid * colonne; //qua calcolo lo start, ovvero l'elemento da cui partire

    //start minore di dim perchè, nel caso peggiore, start = 

    if(start < dim){
        for(int i = 0; i < colonne - 1; i++){
            if(array[start + i ] > array[ start + i + 1]){
                swap(&array[start + i], &array[start + i + 1]);
            }
        }
    }
    
}

__global__ void check_confini(int *array, int colonne, int righe, int dim){
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // colonna globale
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // riga globale
    int tid = y * (gridDim.x * blockDim.x) + x;

    int rigaIndex = tid * colonne; // abbiamo indice del primo el di 
    //di una riga
    
    //l'ultimo check lo facciamo alla penultima riga 
    //in questo calcolo, si considera anche se l'array ha un numero dispari di elementi
    int penultimaRigaIndex = (dim - (colonne * 2)) + (colonne - dim%colonne)*dim%colonne ;
        
    //sopra calcolo l'indice del primo elemento della penultima riga
    if(rigaIndex <= penultimaRigaIndex){
        int lastElRiga = rigaIndex + (colonne - 1);
        if(array[lastElRiga] > array[lastElRiga + 1]){
            //printf("Swap %d %d", array[start], array[start+1]);
            swap(&array[lastElRiga], &array[lastElRiga + 1]);
        }
    }
}