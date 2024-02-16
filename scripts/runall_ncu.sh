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
benchmarks=('gups' 'bfs' 'gemm' 'sort' 'pathfinder' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )
#benchmarks=('sort' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )
#benchmarks=('gups')

total_run=$1

pwd="$(dirname $(pwd))"
mkdir -p $pwd/results

for bench in "${benchmarks[@]}"
do
    for i in $(seq 1 $total_run)
    do
        echo 3 | sudo tee /proc/sys/vm/drop_caches
        sudo dmesg -C

        if [[ ${benchmarks1[@]} =~ $bench ]]; then
            level=level1
        elif [[ ${benchmarks2[@]} =~ $bench ]]; then
            level=level2
        else 
            echo "not on listed\n"
            exit 0
        fi
#        ncu --csv -c 1000 --metrics smsp__warps_eligible.avg.per_cycle_elapsed \
#            --log-file $pwd/results/$bench \
        #ncu -c 100 -o $pwd/ncu-results/$bench --section MemoryWorkloadAnalysis_Chart -f \
        ncu -c 30 -o $pwd/ncu-results/$bench --section MemoryWorkloadAnalysis_Tables -f \
            $pwd/build/bin/$level/$bench -s 2 --passes 1 --uvm -o $pwd/results/$bench.csv -b $bench
        echo "Done with $bench ${i} times..."
        sleep 3
    done
done
