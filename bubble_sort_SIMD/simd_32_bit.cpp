#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>
#include "simd_utils.h"
#include <inttypes.h>
#define FILE_PATH "./performance/simd_perf_stack.txt"

#define SSE_DATA_LANE 16
#define DATA_SIZE 4 
//Dato che non esite una funzione che fa nativamente il 
//compare unsigned, il bit MSB (bit di segno) è sempre posto a 0,
//che equivale a dire che usaimo solo numeri positivi.
#define MAX 1073741824 //(1<<30)

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: ./main NUM_ELEMENTI\n");
        return -1;
    }

    int DIM_ARRAY = atoi(argv[1]);
    if (DIM_ARRAY % DATA_SIZE != 0)
    {
        int to_add = DATA_SIZE - (DIM_ARRAY % DATA_SIZE);
        printf("Dimensione non normalizzata, aggiungo %d\n", to_add);
        DIM_ARRAY += to_add;
    }

    u_int64_t clock_counter_SIMD_start, clock_counter_SIMD_end;

    //int32_t A[DIM_ARRAY] __attribute__((aligned(SSE_DATA_LANE)));
    int32_t* A = (int32_t*) _mm_malloc(DIM_ARRAY * DATA_SIZE, SSE_DATA_LANE);
    /****************************************************************/

    // Registro per i minimi dell'array
    int32_t MIN_ARRAY[4] __attribute__((aligned(SSE_DATA_LANE)));

    // Registro per i relativi indici
    int32_t MIN_INDEX_ARRAY[4] __attribute__((aligned(SSE_DATA_LANE)));
    /****************************************************************/
    srand(42); //per test
    //srand(time(NULL));

    

    for (int i = 0; i < DIM_ARRAY; i++)
        A[i] = rand() % MAX;

    __m128i *p_A = (__m128i *)A;

    int count = 0;
    int32_t temp_array[4];
    //print_output(A, DIM_ARRAY);
    clock_counter_SIMD_start = __rdtsc();

    for (int k = 0; k < DIM_ARRAY; k++)
    {

        
        int start = k >> 2; // k/4
        __m128i reg_min = _mm_load_si128(p_A + start);

        int base = start * 4; // CALCOLO INDICI CORRETTO
        __m128i index_reg = _mm_set_epi32(
            base + 3,
            base + 2,
            base + 1,
            base + 0);
        
        //Ogni 4 cicli di k, abbiamo una load in meno da fare
        //quindi si inizia da start
        //Il + 1 c'è perchè ignoriamo il primo registro, in quanto
        //è gia stato letto ed è dentro reg_min e index_reg
        for (int i = start + 1; i < DIM_ARRAY / (SSE_DATA_LANE / DATA_SIZE); i++)
        {
            int global_idx = i << 2; // i * 4

            __m128i this_indexes = _mm_set_epi32(
                global_idx + 3,
                global_idx + 2,
                global_idx + 1,
                global_idx + 0);

            __m128i vec = _mm_load_si128(p_A + i);

            __m128i mask = _mm_cmplt_epi32(vec, reg_min);

            //vedi logicaBinaria.txt
            reg_min = _mm_or_si128(
                _mm_and_si128(mask, vec),
                _mm_andnot_si128(mask, reg_min));

            index_reg = _mm_or_si128(
                _mm_and_si128(mask, this_indexes),
                _mm_andnot_si128(mask, index_reg));
        }


        //Faccio la store in array normali così da iterare dentro essi
        _mm_store_si128((__m128i *)MIN_ARRAY, reg_min);
        _mm_store_si128((__m128i *)MIN_INDEX_ARRAY, index_reg);

        /*Semplice logica di trovare minimo e indice relativo
        da array di 4 elementi*/
        int32_t min = MIN_ARRAY[0];
        int offset_min_index = 0;
        for (int j = 1; j < 4; j++)
        {
            if (MIN_ARRAY[j] < min)
            {
                min = MIN_ARRAY[j];
                offset_min_index = j;
            }
        }
        /***************************************************** */
        // Indice globale del valore minimo
        int global_min_index = MIN_INDEX_ARRAY[offset_min_index]; 

        //Indice interno, ovvero quale indice da 0 a 3 
        //del registro temporaneo con i minimi stiamo trattando

        int internal_index = k % 4; //k & 3
        temp_array[internal_index] = min;

        // scambio elemento minimo con elemento k-esimo
        // e elimino elemento minimo dall'array (temporaneamente)
        //[5,4,3,2,1] -> [1,4,3,2,5]
        //[MAX,4,3,2,5]
        A[global_min_index] = A[k];
        A[k] = MAX;

        // store ogni 4 elementi
        if (internal_index == 3)
        {
            __m128i to_store = _mm_set_epi32(
                temp_array[3],
                temp_array[2],
                temp_array[1],
                temp_array[0]);
            _mm_store_si128(p_A + (k >> 2), to_store);
        }
    }
    clock_counter_SIMD_end = __rdtsc();
    //print_output(A, DIM_ARRAY);
    _mm_free(A);
    u_int64_t cicli = clock_counter_SIMD_end - clock_counter_SIMD_start;
    printf("Cicli con %d elementi: %" PRIu64 "\n", DIM_ARRAY, cicli);

    return 0;
}
