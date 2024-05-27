#!/usr/bin/bash
module purge
module load compiler/gcc/9.1.0
module load compiler/intel/2019u5/intelpython3

python3 index.py "${1}"
