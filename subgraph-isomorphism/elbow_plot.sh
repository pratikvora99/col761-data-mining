module purge
module load compiler/intel/2020u4/intelpython3.7
python3 q3.py "${1}" "${2}" "${3}"
