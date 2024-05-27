#!/usr/bin/bash
module purge
module load compiler/gcc/9.1.0
module load compiler/intel/2019u5/intelpython3

echo "Unzipping boost library"
unzip -q boost.zip
export CPATH="boost/"
g++ -std=c++11 query.cpp -o query.out
./query.out
