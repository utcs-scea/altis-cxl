#!/usr/bin/env bash

set -euo pipefail

display_usage() {
    echo "./runall.sh [number of run]"
}

if [[ ( $@ == "--help") ||  $@ == "-h" ]]; then 
    display_usage
    exit 0
elif [[ $# -ne 0 ]]; then
    display_usage
    exit 0
fi 

NVBIT_PATH=/home/tkim/workspace/nvbit_release/tools

pwd="$(dirname $(pwd))"
mkdir -p $pwd/results

benchmarks1=('gups' 'bfs' 'sort' 'pathfinder' 'gemm')
benchmarks2=('nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

# Running these benches
benchmarks=('gups' 'gemm' 'bfs' 'sort' 'pathfinder' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

# Failed on amplification
benchmarks=('gemm' 'nw' 'particlefilter_naive' 'fdtd2d' 'cfd' 'gups' )

# High amplification
benchmarks=('where' 'bfs' 'sort' 'cfd')
# Low amplification
benchmarks=('pathfinder' 'lavamd' 'mandelbrot' 'srad')

benchmarks=('where' 'bfs' 'sort' 'cfd' 'pathfinder' 'lavamd' 'mandelbrot' 'srad')
benchmarks=('where' 'sort')
benchmarks=('particlefilter_naive' 'fdtd2d')

benchmarks=('where' 'sort' 'cfd' 'lavamd' 'srad')

nvbit_results=$pwd/nvbit-results

for bench in "${benchmarks[@]}"
do
    sudo dmesg -C
    rm -rf report1.*
    rm -f dummy.log

    if [[ ${benchmarks1[@]} =~ $bench ]]; then
        level=level1
    elif [[ ${benchmarks2[@]} =~ $bench ]]; then
        level=level2
    else 
        echo "not on listed\n"
        exit 0
    fi

    dummy_var=$(cat $nvbit_results/$bench.csv | cut -d ',' -f1 | tr -d ' ')
    dummy_var=$(($dummy_var*4/1024))

    echo "<<<<< Running $bench without nvbit mem_trace >>>>>"
    LD_PRELOAD=$NVBIT_PATH/mem_trace/mem_trace.so $pwd/build/bin/$level/$bench -s 4 --passes 1 \
        --uvm -o $pwd/results/$bench.csv -b $bench --dummy $dummy_var --oversub-frac 1.2
    mv $pwd/output.csv $pwd/nvbit-results/$bench-oversub.csv
    echo "<<<<< Done nvbit mem_trace >>>>>"

done
