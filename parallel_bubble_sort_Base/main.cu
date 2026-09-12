#include "kernel_def.h"
#include "host_def.h"
#include "stdio.h"
#include "math.h"


#define COLONNE 4
#define DATA_SIZE 4
// Error checking macro
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}


int main(int argc, char **argv){
    if(argc < 3){
        printf("Usage: <block_dim.x> <block_dim.y> <array_dim>");
        printf("Example: 32 32 1024");
        return 1;
    }
    int blockSize_x = atoi(argv[1]);
    int blockSize_y = atoi(argv[2]);
    int dim_array = atoi(argv[3]);
    srand(time(NULL));

    int righe = (dim_array+COLONNE-1)/COLONNE;
    //per test GPU
    int *h_matrix = (int *)malloc(dim_array* sizeof(int));
    riempiArray(h_matrix, righe * COLONNE);
    //per test CPU
    int *h_cpu_matrix = (int *)malloc(dim_array* sizeof(int));
    memcpy(h_cpu_matrix, h_matrix, dim_array* sizeof(int));

    printf("Stampo array prima dell'ordinamento\n");
    printf("Matrice prima di ordinamento\n");
    //stampaMatrice(h_matrix, righe, COLONNE);

    int * d_matrix;

    //alloco memoria su GPU
    CHECK(cudaMalloc((void **) &d_matrix, dim_array * DATA_SIZE));
    //copio memoria su GPU
    CHECK(cudaMemcpy(d_matrix, h_matrix, dim_array * DATA_SIZE, cudaMemcpyHostToDevice));

    //dimensionamento kernel orizzontale
    printf("Num righe %d\n\n", righe);
    dim3 block(blockSize_x, blockSize_y);
    //calcolo del numero di blocchi per griglia
    int num_blocchi = (righe + block.x*block.y - 1) / (block.x*block.y);
    int gridDimX = ceil(sqrt((double)num_blocchi)); // se non è un quadrato perfetto approssimo per eccesso
    dim3 grid( gridDimX, (num_blocchi+gridDimX-1) / gridDimX );    
    printf("Grid x: %d, Grid y : %d \n", grid.x, grid.y);
    
    printf("Num blocchi attesi: %d \n Num blocchi calcolati %d\n\n", num_blocchi, grid.x*grid.y);
    printf("Num righe %d\n", righe);

    //i cicli devono eseguire un numero deterministico di volte
    printf("\nEsecuzione GPU...\n");
    
    clock_t gpu_start = clock();
    for(int i=0; i<(dim_array);i++){
        ordina_riga <<<grid, block>>>(d_matrix, COLONNE, righe, dim_array);
        CHECK(cudaDeviceSynchronize());
        check_confini<<<grid, block>>>(d_matrix, COLONNE, righe, dim_array);
        CHECK(cudaDeviceSynchronize());
    } 
    clock_t gpu_end = clock();
   
    
    /* Implementazione con kernel padre (deprecata e inefficiente)
    clock_t gpu_start = clock();
    kernel_padre<<<1, 1>>>(d_matrix, COLONNE, righe, dim_array, block, grid);
    CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    printf("Launch error: %s\n", cudaGetErrorString(err));
    clock_t gpu_end = clock(); */

    double gpu_time = ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Tempo GPU: %.3f ms\n", gpu_time);

    //copio da GPU a host
    CHECK(cudaMemcpy(h_matrix,d_matrix,dim_array * DATA_SIZE,cudaMemcpyDeviceToHost));

    printf("\n\n Matrice ordinata:\n");
    //stampaMatrice(h_matrix, righe, COLONNE);

     //esecuzione CPU 
    printf("\nEsecuzione CPU...\n");
    clock_t cpu_start = clock();
    bubbleSort_cpu(h_cpu_matrix, dim_array);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Tempo CPU: %.3f ms\n", cpu_time);

    printf("Array Ordinato da CPU:\n");
    //stampaMatrice(h_cpu_matrix, righe, COLONNE);

    printf("\nVerifica correttezza...\n");
    bool correct = verifyResults(h_cpu_matrix, h_matrix, dim_array);
    printf(correct == 0 ? "✓ Test PASSATO\n" : "✗ Test FALLITO\n"); 
    float speed_up = cpu_time/gpu_time;
    printf("SPEED UP NGL: %.3f \n\n\n", speed_up);
    free(h_matrix);
    free(h_cpu_matrix);
    CHECK(cudaFree(d_matrix));
    return 0;
}