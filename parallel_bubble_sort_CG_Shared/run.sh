#!/bin/bash

# Verifica argomenti
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dim_array>"
    echo "Example: $0 256"
    exit 1
fi

DIM_ARRAY=$1

# Compila
echo "Compilazione..."
# TODO
nvcc -o my_program main.cu kernel_impl.cu host_impl.c
chmod +x my_program

# Crea directory
mkdir -p ncu_reports ordered_arrays

echo "Sorting Array of ${DIM_ARRAY} Elements"

# Test con i metodi specificati
for bs in 8 16 32; do
        echo "  Block Size ${bs}x${bs}"
        
        # Profiling
        ./my_program $bs $bs $DIM_ARRAY > "output_ker.txt"
        sudo ncu --set full -o test_report ./my_program $bs $bs $DIM_ARRAY
        
        # Rinomina output per distinguere i metodi
        mv test_report.ncu-rep "ncu_reports/test_report_ker${bs}_${DIM_ARRAY}" 2>/dev/null
        mv output_ker.txt "ordered_arrays/output_ker${bs}_${DIM_ARRAY}.txt" 2>/dev/null
done
