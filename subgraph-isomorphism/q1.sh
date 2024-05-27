#!/usr/bin/bash
module purge
module load compiler/gcc/9.1.0
module load compiler/intel/2019u5/intelpython3

python3 preprocessor.py "${1}" fsg
python3 preprocessor.py "${1}" gaston
python3 preprocessor.py "${1}" gspan

python3 q1_run.py "${1}"
