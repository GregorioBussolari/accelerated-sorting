#include "kernel_def.h"
#include <stdio.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ int counter = 0;

__device__ void inline swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

__global__ void parallel_bubble_sort(int *array, int colonne, int righe, int dim)
{
    cg::grid_group grid = cg::this_grid();
    // shared memory
    extern __shared__ int sh_array[];
    
    //////////////////////////////////////////////////////////////////////////
    // Coordinate globale
    int threadsPerBlock  = blockDim.x * blockDim.y;
    //coordiante thread nel blocco
    int block_tid = threadIdx.x + blockDim.x * threadIdx.y;
    //coordinate blocco nella griglia
    int blockNumInGrid = blockIdx.x  + gridDim.x  * blockIdx.y;

    int  global_tid = blockNumInGrid * threadsPerBlock + block_tid;

    //////////////////////////////////////////////////////////////////
    // Start = indice GLOBALE primo elemento della riga
    int global_start = global_tid * colonne;
    // NOTE: "first" and "last" is referenced in the context OF A SINGLE BLOCK
    int is_global_last = (global_tid == righe-1);
    //last row except for the last global row
    int is_last_row = ((threadIdx.x == blockDim.x - 1) && (threadIdx.y == blockDim.y - 1) && !is_global_last);
    //first row except for the first global row
    int is_first_row = (block_tid == 0 && global_tid != 0);


    if (global_start < dim)
    {   
        //for check confini
        int phase = colonne - 1;

        // POPOLO SHARED MEMORY
        for (int i = 0; i < colonne; i++)
        {
            sh_array[block_tid * colonne + i] = array[global_start + i];
        }
        // sync non necessario, righe per thread indipendenti
        //__syncthreads();

        for (int j = 0; j < dim; j++)
        {
            //bubble sort on a single row
            for (int i = 0; i < colonne - 1; i++)
            {
                if (sh_array[block_tid * colonne + i] > sh_array[block_tid * colonne + i + 1])
                {
                    swap(&sh_array[block_tid * colonne + i], &sh_array[block_tid * colonne + i + 1]);
                }
            }

            ////////////////////////////////////////////////////////////////
            // update global memory with last element of the block
            if (is_last_row)
            {
                array[global_start + colonne - 1] = sh_array[block_tid * colonne + (colonne - 1)];
            }

            // update global memory with first element of the block
            if (is_first_row)
            {
                array[global_start] = sh_array[0]; // always zero
            }

            // sync grid-wide
            grid.sync();

            
            // check dei confini
            // Se non è l'ultima riga dell'array (altrimenti OOB)
            if(!is_global_last){
                //Posso usare la shared memory sempre tranne se eccedo il blocco
                if (!is_last_row){
                        if (sh_array[block_tid * colonne + phase] > sh_array[block_tid * colonne + 1 + phase])
                        {
                            swap(&sh_array[block_tid * colonne + phase], &sh_array[block_tid * colonne + 1 + phase]);
                        }
                } else { 
                        //Caso nel quale siamo nell'ultima riga del blocco, 
                        //Devo usare memoria globale
                        if (array[global_start + phase] > array[global_start + phase + 1])
                        {
                            swap(&array[global_start + phase], &array[global_start + phase + 1]);
                            // SOLO IN QUESTO CASO FACCIO LA STORE DELL'ultimo ELEMENTO
                            sh_array[block_tid * colonne + phase] = array[global_start + phase];
                        }
                }
            }
            //Qua eseguo la sync in quanto devo leggere i nuovi
            //elementi aggiornati DELLA PRIMA RIGA
            grid.sync();
            if(is_first_row){
                sh_array[0] = array[global_start];
            }
        }
        
        // store shared global
        for (int i = 0; i < colonne; i++){
            array[global_start + i] = sh_array[block_tid * colonne + i];
        }

    }
}
