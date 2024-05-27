#!/usr/bin/bash

. ./common.lib

compile_binary()
{
    COMPILE_SCRIPT="compile.sh"

    if [ ! -e "${COMPILE_SCRIPT}" ]; then
        echo "Compile script not found: ${COMPILE_SCRIPT}"
        echo "Aborting..."
        exit 1
    fi
    
    echo "Compile script found: ${COMPILE_SCRIPT}"
    echo "Executing compile script..."

    ./"${COMPILE_SCRIPT}" "${@}" # algorithm name apriori or fptree
}

execute_algo()
{
    ALGO="${1}"
    EXC_BIN="${ALGO}".out

    COUNTER=0
    while :
    do
        if [ ! -e "${EXC_BIN}" ]; then
            echo "Compiled binary not found: ${EXC_BIN}"
            compile_binary "${ALGO}"

            COUNTER=`expr ${COUNTER} + 1`
            if [ "${COUNTER}" -gt 3 ]; then
                echo "Excessive failed compilations. Aborting..."
                exit 1
            fi
        else
            echo "Compiled binary exists: ${EXC_BIN}"
            break
        fi
    done

    echo "Running ${ALGO}-algorithm..."
    
    DATASET="${2}"
    echo " - Dataset: ${DATASET}"

    SUPP_LIMIT="${3}"
    echo " - Support-threshold: ${SUPP_LIMIT}%"

    OUT_FILE="${4}"
    echo " - Output-file: ${OUT_FILE}"

    chmod a+rx ./"${EXC_BIN}"

    START="$(date +%s)"
    timeout 1h ./"${EXC_BIN}" "${SUPP_LIMIT}" "${DATASET}" "${OUT_FILE}"
    STATUS_CODE="${?}"
    END="$(date +%s)"

    if [ "${STATUS_CODE}" -eq "0" ]; then
        echo "Execution successful"
        RUN_TIME="$(expr ${END} - ${START})"
    elif [ "${STATUS_CODE}" -eq "124" ]; then
        echo "Execution timed-out!"
        RUN_TIME="-124"
    else
        echo "Execution failed"
        RUN_TIME="-1"
    fi
}

gen_runtime_plot()
{
    echo "Generating runtime plots..."

    DATASET="${1}"
    echo " - Dataset: ${DATASET}"

    PIC_FILE="${2}"
    echo " - Output-[Picture]-file: ${PIC_FILE}"

    OUT_FILE="temp.dat"
    RUN_FILE="runtimes.txt"
    if [ -e "${RUN_FILE}" ]; then
        rm "${RUN_FILE}"
        echo "Existing file removed: ${RUN_FILE}"
    fi

    OUT_LINE=""

    for st in 90 50 25 10 5
    do
        echo "Processing... [Support-Threshold: ${st}]"

        OUT_LINE="${st}"
        for algo in "${APRIORI}" "${FPTREE}"
        do
            execute_algo "${algo}" "${DATASET}" "${st}" "${OUT_FILE}"
            OUT_LINE="${OUT_LINE} ${RUN_TIME}"
        done

        echo "${OUT_LINE}" >> "${RUN_FILE}"
    done

    python plot.py "${RUN_FILE}" "${PIC_FILE}"
}

case "${1}" in
    "-apriori")
        shift
        execute_algo "${APRIORI}" "${@}"
        ;;
    "-fptree")
        shift
        execute_algo "${FPTREE}" "${@}"
        ;;
    "-plot")
        shift
        gen_runtime_plot "${@}"
        ;;
    *)
        echo "CLI arguments: ${@}"
        echo "Invalid CLI option-1: ${1}"
        exit 1
esac
