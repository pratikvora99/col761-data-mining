module purge
module load compiler/intel/2020u4/intelpython3.7
python3 run.py "${1}" "${2}" "${3}"
