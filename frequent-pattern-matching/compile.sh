#!/usr/bin/bash

. ./common.lib

compile_code() 
{
    case "${1}" in
        "${APRIORI}")
            echo "Compiling source for Apriori algorithm..."
            OUT_FILE="${APRIORI}.out"
            SRC_FILE="apriori_final.cpp"
            ;;
        "${FPTREE}")
            echo "Compiling source for FP-Tree algorithm..."
            OUT_FILE="${FPTREE}.out"
            SRC_FILE="fp_tree_final.cpp"
            ;;
        *)
            echo "Invalid algorithm: ${1}"
            exit 1
    esac

    if [ ! -e "${SRC_FILE}" ]; then
        echo "Source file not found: ${SRC_FILE}"
        echo "Aborting..."
        exit 1
    fi

    echo "Compiling source: ${SRC_FILE}..."
    g++ -o "${OUT_FILE}" "${SRC_FILE}"
    if [ "${?}" -eq "0" ]; then
        echo "Compilation successful"
    else
        echo "Compilation failed"
    fi

}

if [ "${#}" -eq 0 ]; then
    for alg in ${APRIORI} ${FPTREE}
    do
        compile_code "${alg}"
    done
else
    compile_code "${@}"
fi
