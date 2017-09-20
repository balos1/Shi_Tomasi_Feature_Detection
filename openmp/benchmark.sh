#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

benchmark_help() {
    echo "usage: ./benchmark.sh <path to executable> <path to arguments file> <path to output csv file> <number of executions>"
    echo "example format of arguments file (csv):"
    echo "  ~/Pictures/lenna.pgm,1.1,5,100"
    echo "  ~/Pictures/other.pgm,1.1,5,100"
    exit
}

xmake() {
    make clean
    make benchmode
}

prebench() {
    IFS=$'\n' read -r args < $argf
    args=$(echo $args | sed 's/,/ /g')
    cmdout=$($exe $args)
    echo -n 'arg_set,args,' > $output
    echo $cmdout | grep -oP '(((?![\d+|ms|px]))[A-z]+)' | paste -sd ',' >> $output  # headers for output csv
}

bench() {
    while IFS=$'\n' read -r args; do
        args=$(echo $args | sed 's/,/ /g')
        for ((it = 1; it <= $nexecs; it++)); do
            echo -en "\rperforming execution $it/$nexecs for arg-set $argset"
            cmdout=$($exe $args)
            echo -n "$argset,$args," >> $output
            echo $cmdout | grep -oP '\d+\w+' | paste -sd ',' >> $output             # data for output csv
        done
        argset=$(($argset+1))
    done < $argf
}

if [[ $# -ne 4 ]]; then
    echo -e "${RED}Error: Illegal number of parameters${NC}"
    benchmark_help
fi;

exe=$1
argf=$2
output=$3
nexecs=$4
argc=$(wc -l < $argf)

xmake
prebench
argset=1
bench
echo -e "\nbenchmark results written to ${output}"



