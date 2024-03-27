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

benchmarks=('gemm' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd')
benchmarks=('where' 'bfs' 'sort' 'pathfinder' 'lavamd' 'mandelbrot' 'srad' 'cfd' )
benchmarks=('mandelbrot' 'lavamd' 'pathfinder' 'cfd' 'srad' 'sort' 'where')
benchmarks=('where')
configs=('zero-copy')

total_run=$1

pwd="$(dirname $(pwd))"
mkdir -p $pwd/results

for bench in "${benchmarks[@]}"
do
    for config in "${configs[@]}"
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
            #numactl --membind=0 --cpunodebind=0 \
#            /usr/local/cuda/bin/nsys profile --force-overwrite=true \
#                --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --cuda-memory-usage=true \
            $pwd/build/bin/$level/$bench -s 4 --passes 1 --$config -b $bench  --coal
            #$pwd/build/bin/$level/$bench -s 4 --passes 1 --$config -o $pwd/results/$bench-$config.csv -b $bench 
            echo "Done with $bench ${i} times..."
            sleep 3
        done
    done
done
