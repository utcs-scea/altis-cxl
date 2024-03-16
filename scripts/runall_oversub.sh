#!/usr/bin/env bash

set -euo pipefail

display_usage() {
    echo "./runall.sh [number of run]"
}

if [[ ( $@ == "--help") ||  $@ == "-h" ]]; then 
    display_usage
    exit 0
elif [[ $# -ne 1 ]]; then
    display_usage
    exit 0
fi 


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

benchmarks=('where' 'bfs' 'sort' 'pathfinder' 'lavamd' 'mandelbrot' 'srad' 'cfd' )

pwd="$(dirname $(pwd))"
mkdir -p $pwd/results
nvbit_results=$pwd/nvbit-results
total_run=$1

for bench in "${benchmarks[@]}"
do
    for i in $(seq 1 $total_run)
    do
        echo 3 | sudo tee /proc/sys/vm/drop_caches
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

        dummy_var=$(cat $nvbit_results/$bench.csv | cut -d ',' -f2 | tr -d ' ')
        dummy_var=$(($dummy_var*32/1024/1024))

        echo "<<<<<Running $bench with dummy memory allocation $dummy_var>>>>>"
        $pwd/build/bin/$level/$bench -s 4 --passes 1 \
            --uvm -o $pwd/results/$bench.csv -b $bench --dummy $dummy_var --oversub-frac 1.2
        sleep 3
    done
done
