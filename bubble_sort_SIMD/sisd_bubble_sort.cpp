#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>
#include <inttypes.h>

void bubble_sort(char *A, int N) {
    int swapped = 1;
    while(swapped) {
        swapped = 0;
        for(int i = 0; i < N - 1; i++) {
            if (A[i] > A[i + 1]) {
                char tmp = A[i];
                A[i] = A[i + 1];
                A[i + 1] = tmp;
                swapped = 1;
            }
        }
    }
}



int main(int argc, char** argv) {

    if (argc != 2)
    {
        printf("Usage: ./main NUM_ELEMENTI\n");
        return -1;
    }


    u_int64_t clock_counter_scalar_start, clock_counter_scalar_end;

    int N = atoi(argv[1]);
    //uint32_t* A = (uint32_t*) malloc(sizeof(uint32_t) * N);
    char* A = (char*)malloc(N * sizeof(char));
    // riempimento random
    
    srand(42); // per test
    for (int i = 0; i < N; i++){
        A[i] = rand() % 127;
    }
        

    // misura
    clock_counter_scalar_start = __rdtsc(); 
    bubble_sort(A, N);
    clock_counter_scalar_end = __rdtsc();
    free(A); 
    u_int64_t cicli = clock_counter_scalar_end -clock_counter_scalar_start;
    printf("Cicli con %d elementi: %" PRIu64 "\n", N, cicli);
    return 0;
}
