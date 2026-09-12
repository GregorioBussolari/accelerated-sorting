#include "host_def.h"

void riempiArray(int*a, int dim){
    for(int i = 0; i < dim; i++){
        a[i] = rand() % 256;
    }
}

void stampaArray(int*a, int dim){
    for(int i = 0; i < dim; i++){
        printf("A[%d]=%d\n", i,a[i]);
    }
}

void riempiMatirce(int ** matrix, int righe, int colonne){
    for(int r = 0; r < righe; r++){
        for(int c = 0; c < colonne; c++){
            matrix[r][c] = rand() % 65536;
        }
    }
}

void stampaMatrice(int * matrix, int righe, int colonne, int dim_array){
    for(int r = 0; r < righe; r++){
        for(int c = 0; c < colonne && r*colonne +c < dim_array; c++){
            printf("  M[%d][%d] = %d  ", r, c, matrix[(r * colonne) + c]);
        }
        printf("\n");
    }
}

int verifyResults(int* cpu_result, int* gpu_result, 
                   int size)
{
    int errors = 0;
    for (int i = 0; i < size; i++) {
        // Tolleriamo differenze di ±1 dovute ad arrotondamenti
        int diff = abs((int)cpu_result[i] - (int)gpu_result[i]);
        if (diff > 1) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at index %d: CPU=%d, GPU=%d (diff=%d)\n", 
                       i, cpu_result[i], gpu_result[i], diff);
            }
        }
    }
    
    if (errors > 0) {
        printf("Total errors: %d / %d (%.2f%%)\n", 
               errors, size, 100.0f * errors / size);
    }
    
    return errors;
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void quickSort_cpu(int array[], int begin, int end) {
    int pivot, l, r; 
    if (end > begin) {
       pivot = array[begin];
       l = begin + 1;
       r = end+1;
       while(l < r)
          if (array[l] < pivot) 
             l++;
          else {
             r--;
             swap(&array[l], &array[r]); 
          }
       l--;
       swap(&array[begin], &array[l]);
       quickSort_cpu(array, begin, l);
       quickSort_cpu(array, r, end);
    }
 }

 void bubbleSort_cpu(int array[], int dim){
    int swapped; 

    for (int i = 0; i < dim - 1; i++) {
        swapped = 0;

        for (int j = 0; j < dim - i - 1; j++) {
            if (array[j] > array[j + 1]) {
                swap(&array[j], &array[j+1]);
                swapped = 1;
            }
        }

        if (!swapped)
            break;
    }
 }

