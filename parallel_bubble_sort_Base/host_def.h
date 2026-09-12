#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void riempiArray(int* arr, int n);
void stampaArray(int* arr, int n);
void riempiMatirce(int ** matrix, int righe, int colonne);
void stampaMatrice(int * matrix, int righe, int colonne);
void quickSort_cpu(int array[], int begin, int end);
int verifyResults(int* cpu_result, int* gpu_result, int size);
void bubbleSort_cpu(int array[], int dim);

struct dim_kernel {
    int threadPerBlock;
    int blocksPerGrid;
};
#ifdef __cplusplus
}
#endif
