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


benchmarks1=('gups' 'bfs' 'sort' 'pathfinder' 'gemm')
benchmarks2=('nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

# Running these benches
benchmarks=('gups' 'gemm' 'bfs' 'sort' 'pathfinder' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

benchmarks=('gemm' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd')

# Failed on amplification
benchmarks=('gemm' 'nw' 'particlefilter_naive' 'fdtd2d' 'cfd' 'gups' )

# High amplification
benchmarks=('where' 'bfs' 'sort' 'cfd')
# Low amplification
benchmarks=('pathfinder' 'lavamd' 'mandelbrot' 'srad')

pwd="$(dirname $(pwd))"
mkdir -p $pwd/results

for bench in "${benchmarks[@]}"
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
    /usr/local/cuda/bin/nsys profile --force-overwrite=true \
        --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --cuda-memory-usage=true \
        $pwd/build/bin/$level/$bench -s 4 --passes 1 --uvm -o $pwd/results/$bench.csv -b $bench 
    echo "<<<<<Finished running $bench with profiling actual memory usage.>>>>>"
    nsys stats -q --report gpumemsizesum --output @"grep \"Unified Memory memcpy HtoD\"" report1.nsys-rep > dummy.log
    dummy_var=`awk -F"," '{print $1}' dummy.log`
    dummy_var=$( printf "%.0f" $dummy_var )
    rm -rf report1.*
    rm -f dummy.log

    echo 3 | sudo tee /proc/sys/vm/drop_caches
    sudo dmesg -C
    sleep 1

    echo "<<<<<Running $bench with dummy memory allocation $dummy_var...>>>>>"
    /usr/local/cuda/bin/nsys profile --force-overwrite=true \
        --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --cuda-memory-usage=true \
        $pwd/build/bin/$level/$bench -s 4 --passes 1 \
            --uvm -o $pwd/results/$bench.csv -b $bench --dummy $dummy_var --oversub-frac 1.2
    nsys stats -q --report gpumemsizesum --output @"grep \"Unified Memory memcpy HtoD\"" report1.nsys-rep > dummy.log
    mv report1.nsys-rep $pwd/nsys-results/amp/$bench-oversub.nsys-rep
    new_var=`awk -F"," '{print $1}' dummy.log`
    new_var=$( printf "%.0f" $new_var )
    printf "$bench, $dummy_var, $new_var\n" >> $pwd/nsys-results/amp/amp-results.log

    sleep 3
done
