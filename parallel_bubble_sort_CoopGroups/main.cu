#include "kernel_def.h"
#include "host_def.h"
#include "stdio.h"
#include "math.h"


#define STANDARD_COLUMN 2
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
    srand(time(NULL));
    int blockSize_x = atoi(argv[1]);
    int blockSize_y = atoi(argv[2]);
    int threadPerBlock = blockSize_x*blockSize_y;
    int dim_array = atoi(argv[3]);

    //---------------------------------------------------------------------
    //ensure that the device supports cooperative launches
    int dev = 0; //default device, see docs
    int supportsCoopLaunch = 0;
    CHECK(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));
    if(!supportsCoopLaunch){
        printf("Cooperative Groups are not supported in your\n \
            default device, terminating...");
        return -1;
    } else {
        printf("Your dev supports coop launches \n");
    }

    //---------------------------------------------------------------------
    // maximize the exposed parallelism by calculating how many 
    //blocks can fit simultaneously per-SM using the occupancy;
    // numThreads must be decided prior to this calculation
    int numBlocksPerSm = -1;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));


    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, ordina_riga, threadPerBlock, 0));
    int maxBlocksPerKernel = numBlocksPerSm * deviceProp.multiProcessorCount;
    printf("Given %d threadPerBlock\n  \
    you can launch max %d (#SM) x %d (#MaxActiveBlocksPerSM), hence %d blocks\n\n", \ 
    threadPerBlock,deviceProp.multiProcessorCount,\
    numBlocksPerSm, maxBlocksPerKernel);

    //modulating column number to permit the execution,
    //dimensionamento kernel orizzontale
    //the number of threads is fixed by user
    dim3 block(blockSize_x, blockSize_y);

    //we can modulate the number of columns, by takeing the smallest
    int colonne = STANDARD_COLUMN;
    int righe = (dim_array+colonne-1)/colonne;
    int num_blocchi = (righe + block.x*block.y - 1) / (block.x*block.y);
    while(num_blocchi>maxBlocksPerKernel){
        //doule the number of columns and try again
        colonne *= 2;
        righe = (dim_array+colonne-1)/colonne;
        num_blocchi = (righe + block.x*block.y - 1) / (block.x*block.y);
    }
    printf("Num righe %d\n Num Colonne %d\n\n", righe, colonne);
    int gridDimX = ceil(sqrt((double)num_blocchi)); // se non è un quadrato perfetto approssimo per eccesso
    dim3 grid( gridDimX, (num_blocchi+gridDimX-1) / gridDimX );    
    printf("Grid x: %d, Grid y : %d \n", grid.x, grid.y);
    
    printf("Num blocchi attesi: %d \n Num blocchi calcolati %d\n\n", num_blocchi, grid.x*grid.y);
    printf("Num righe %d\n", righe);
    
    //per test GPU
    int *h_matrix = (int *)malloc(dim_array* sizeof(int));
    riempiArray(h_matrix, dim_array);
    //per test CPU
    int *h_cpu_matrix = (int *)malloc(dim_array* sizeof(int));
    memcpy(h_cpu_matrix, h_matrix, dim_array* sizeof(int));

    //printf("Stampo array prima dell'ordinamento\n");
    //printf("Matrice prima di ordinamento\n");
    //stampaMatrice(h_matrix, righe, colonne, dim_array);

    int * d_matrix;

    //alloco memoria su GPU
    CHECK(cudaMalloc((void **) &d_matrix, dim_array * sizeof(int)));

    //copio memoria su GPU
    CHECK(cudaMemcpy(d_matrix, h_matrix, dim_array * sizeof(int), cudaMemcpyHostToDevice));


    //i cicli devono eseguire un numero deterministico di volte
    printf("\nEsecuzione GPU...\n");

    void* args[] = {
        &d_matrix,
        &colonne,
        &righe,
        &dim_array
    };

    clock_t gpu_start = clock();
    CHECK(cudaLaunchCooperativeKernel(
        (void*)ordina_riga,   // puntatore al kernel
        grid,                 // dim3 grid
        block,                // dim3 block
        args                  // array di parametri
    ));

    //ordina_riga <<<grid, block>>>(d_matrix, STANDARD_COLUMN, righe, dim_array, num_blocchi);

    CHECK(cudaDeviceSynchronize());

    clock_t gpu_end = clock();
    double gpu_time = ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Tempo GPU: %.3f ms\n", gpu_time);

    //copio da GPU a host
    CHECK(cudaMemcpy(h_matrix,d_matrix,dim_array * DATA_SIZE,cudaMemcpyDeviceToHost));

    //printf("\n\n Matrice ordinata:\n");
    //stampaMatrice(h_matrix, righe, colonne, dim_array);

     //esecuzione CPU 
    printf("\nEsecuzione CPU...\n");
    clock_t cpu_start = clock();
    bubbleSort_cpu(h_cpu_matrix, dim_array);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Tempo CPU: %.3f ms\n", cpu_time);

    //printf("Array Ordinato da CPU:\n");
    //stampaMatrice(h_cpu_matrix, righe, colonne, dim_array);

    printf("\nVerifica correttezza...\n");
    bool correct = verifyResults(h_cpu_matrix, h_matrix, dim_array);
    printf(correct == 0 ? "✓ Test PASSATO\n" : "✗ Test FALLITO\n"); 
    float speed_up = cpu_time/gpu_time;
    printf("SPEED UP: %.3f \n\n\n", speed_up);

    
    free(h_matrix);
    free(h_cpu_matrix);
    CHECK(cudaFree(d_matrix));
    return 0;
}