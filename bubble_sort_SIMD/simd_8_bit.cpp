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
#define DATA_SIZE 1
#define MAX 127

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: ./main NUM_ELEMENTI\n");
        return -1;
    }

    int DIM_ARRAY = atoi(argv[1]);

    int data_in_a_lane = SSE_DATA_LANE / DATA_SIZE; // con dato char, allora 16 dati
    if (DIM_ARRAY % data_in_a_lane != 0)
    {
        int to_add = data_in_a_lane - (DIM_ARRAY % data_in_a_lane);
        printf("Dimensione non normalizzata, aggiungo %d\n", to_add);
        DIM_ARRAY += to_add;
    }

    u_int64_t clock_counter_SIMD_start, clock_counter_SIMD_end;

    char *A = (char *)_mm_malloc(DIM_ARRAY * DATA_SIZE, SSE_DATA_LANE);
    /****************************************************************/

    // Registro per i minimi dell'array
    char MIN_ARRAY[data_in_a_lane] __attribute__((aligned(SSE_DATA_LANE)));

    // Registro per i relativi indici
    int MIN_INDEX_ARRAY[data_in_a_lane] __attribute__((aligned(SSE_DATA_LANE)));
    /****************************************************************/
    srand(42); // per test
    // srand(time(NULL));

    /*ATTENZIONE, NON ESISTE IL COMPARE UNSIGNED QUINDI*/
    /*NON VIENE MAI USATO L'MSB (bit segno sempre a zero) CAUSEREBBE PROBLEMI*/

    for (int i = 0; i < DIM_ARRAY; i++)
        A[i] = rand() % MAX;

    __m128i *p_A = (__m128i *)A;

    int count = 0;
    char temp_array[data_in_a_lane];
    //print_output_char(A, DIM_ARRAY);
    clock_counter_SIMD_start = __rdtsc();

    for (int k = 0; k < DIM_ARRAY; k++)
    {

        int start = k / data_in_a_lane; // k>>4
        __m128i reg_min = _mm_load_si128(p_A + start);

        int base = start * data_in_a_lane; // CALCOLO INDICI CORRETTO

        __m128i index_reg_0_3 = _mm_set_epi32(base + 3, base + 2, base + 1, base + 0);
        __m128i index_reg_4_7 = _mm_set_epi32(base + 7, base + 6, base + 5, base + 4);
        __m128i index_reg_8_11 = _mm_set_epi32(base + 11, base + 10, base + 9, base + 8);
        __m128i index_reg_12_15 = _mm_set_epi32(base + 15, base + 14, base + 13, base + 12);

        // Ogni "data_in_a_lane" cicli di k, abbiamo una load in meno da fare
        // quindi si inizia da start
        // Il + 1 c'è perchè ignoriamo il primo registro, in quanto
        // è gia stato letto ed è dentro reg_min e index_reg
        for (int i = start + 1; i < DIM_ARRAY / data_in_a_lane; i++)
        {
            int global_idx = i * data_in_a_lane; // i * 16

            __m128i this_idx_0_3 = _mm_set_epi32(global_idx + 3, global_idx + 2, global_idx + 1, global_idx + 0);
            __m128i this_idx_4_7 = _mm_set_epi32(global_idx + 7, global_idx + 6, global_idx + 5, global_idx + 4);
            __m128i this_idx_8_11 = _mm_set_epi32(global_idx + 11, global_idx + 10, global_idx + 9, global_idx + 8);
            __m128i this_idx_12_15 = _mm_set_epi32(global_idx + 15, global_idx + 14, global_idx + 13, global_idx + 12);

            __m128i vec = _mm_load_si128(p_A + i);

            __m128i mask = _mm_cmplt_epi8(vec, reg_min);

            // trasformo la maschera da 8 bit a 32 bit per
            // lavorare con gli indici, i quali sono a
            // 32 bit

            /*Logica per trasformare la maschera***************/
            __m128i mask_lo16 = _mm_unpacklo_epi8(mask, mask); 
            __m128i mask_hi16 = _mm_unpackhi_epi8(mask, mask); 


            __m128i mask_0_3 = _mm_unpacklo_epi16(mask_lo16, mask_lo16);  // lanes 0..3  -> 4 x 0x00000000 o 0xFFFFFFFF
            __m128i mask_4_7 = _mm_unpackhi_epi16(mask_lo16, mask_lo16);  // lanes 4..7
            __m128i mask_8_11 = _mm_unpacklo_epi16(mask_hi16, mask_hi16); // lanes 8..11
            __m128i mask_12_15 = _mm_unpackhi_epi16(mask_hi16, mask_hi16);

            /******************************************************************* */
            // vedi logicaBinaria.txt
            reg_min = _mm_or_si128(
                _mm_and_si128(mask, vec),
                _mm_andnot_si128(mask, reg_min));

            index_reg_0_3 = _mm_or_si128(
                _mm_and_si128(mask_0_3, this_idx_0_3),
                _mm_andnot_si128(mask_0_3, index_reg_0_3));

            index_reg_4_7 = _mm_or_si128(
                _mm_and_si128(mask_4_7, this_idx_4_7),
                _mm_andnot_si128(mask_4_7, index_reg_4_7));

            index_reg_8_11 = _mm_or_si128(
                _mm_and_si128(mask_8_11, this_idx_8_11),
                _mm_andnot_si128(mask_8_11, index_reg_8_11));

            index_reg_12_15 = _mm_or_si128(
                _mm_and_si128(mask_12_15, this_idx_12_15),
                _mm_andnot_si128(mask_12_15, index_reg_12_15));
        }

        // Faccio la store in array normali così da iterare dentro essi
        _mm_store_si128((__m128i *)MIN_ARRAY, reg_min);

        //Abbiamo MIN_INDEX_ARRAY di 4bytex16 interi, quindi è grande 64 byte.
        //Servono quindi 4 store diverse
        _mm_store_si128((__m128i *)&MIN_INDEX_ARRAY[0], index_reg_0_3);
        _mm_store_si128((__m128i *)&MIN_INDEX_ARRAY[4], index_reg_4_7);
        _mm_store_si128((__m128i *)&MIN_INDEX_ARRAY[8], index_reg_8_11);
        _mm_store_si128((__m128i *)&MIN_INDEX_ARRAY[12], index_reg_12_15);

        
        /*Semplice logica di trovare minimo e indice relativo
        da array di 4 elementi*/
        char min = MIN_ARRAY[0];
        int offset_min_index = 0;
        for (int j = 1; j < data_in_a_lane; j++)
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

        // Indice interno, ovvero quale indice da 0 a 15
        // del registro temporaneo con i minimi stiamo trattando

        int internal_index = k % data_in_a_lane;
        temp_array[internal_index] = min;

        // scambio elemento minimo con elemento k-esimo
        // e elimino elemento minimo dall'array (temporaneamente)
        //[5,4,3,2,1] -> [1,4,3,2,5]
        //[MAX,4,3,2,5]
        A[global_min_index] = A[k];
        A[k] = MAX;

        // store ogni 15 elementi
        if (internal_index == (data_in_a_lane - 1))
        {
            __m128i to_store = _mm_set_epi8(
                temp_array[15],
                temp_array[14],
                temp_array[13],
                temp_array[12],
                temp_array[11],
                temp_array[10],
                temp_array[9],
                temp_array[8],
                temp_array[7],
                temp_array[6],
                temp_array[5],
                temp_array[4],
                temp_array[3],
                temp_array[2],
                temp_array[1],
                temp_array[0]);
            _mm_store_si128(p_A + (k / (data_in_a_lane)), to_store);
        }
    }
    clock_counter_SIMD_end = __rdtsc();
    //printf("Array ordinato\n");
    //print_output_char(A, DIM_ARRAY);
    _mm_free(A);
    u_int64_t cicli = clock_counter_SIMD_end - clock_counter_SIMD_start;
    printf("Cicli con %d elementi: %" PRIu64 "\n", DIM_ARRAY, cicli);

    return 0;
}
