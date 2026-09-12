#include "kernel_def.h"
#include <stdio.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ int counter = 0;

__device__ void swap(int* a, int* b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

__global__ void ordina_riga(int * array, int colonne,int righe, int dim){
    
    cg::grid_group grid = cg::this_grid();
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // colonna globale
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // riga globale
    int tid = y * (gridDim.x * blockDim.x) + x;

    int start = tid * colonne; //qua calcolo lo start, ovvero l'elemento da cui partire

    //start minore di dim perchè, nel caso peggiore, start = 
    for(int j = 0; j < dim; j++){
        //ordina_riga
        if(start < dim){
            for(int i = 0; i < colonne - 1 && (start + i + 1)<dim; i++){
                    if(array[start + i ] > array[ start + i + 1]){
                        swap(&array[start + i], &array[start + i + 1]);
                    } 
            }
        }
        grid.sync();   
        //l'ultimo check lo facciamo alla penultima riga 
        //in questo calcolo, si considera anche se l'array ha un numero dispari di elementi
        int penultimaRigaIndex = (dim - (colonne * 2)) + (colonne - dim%colonne)*dim%colonne ;
        //sopra calcolo l'indice del primo elemento della penultima riga
        if(start <= penultimaRigaIndex){
            int lastElRiga = start + (colonne - 1);
            if(array[lastElRiga] > array[lastElRiga + 1]){
                swap(&array[lastElRiga], &array[lastElRiga + 1]);
            }
        }
        grid.sync();
    }
    
}