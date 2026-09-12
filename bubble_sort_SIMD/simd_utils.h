#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

void print_m128i(__m128i reg, const char*name)
{
    // Array temporaneo per estrarre i valori dal registro
    int32_t elems[4];
    _mm_storeu_si128((__m128i*)elems, reg);

    printf("SSE register %s: ",name);
    for (int i = 0; i < 4; i++)
        printf("%d ", elems[i]);
    printf("\n");
}
void print_output(int32_t *A, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        printf("A[%d] = %u\n", i, A[i]);
    }
}

void print_output_char(char *A, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        printf("A[%d] = %u\n", i, A[i]);
    }
}