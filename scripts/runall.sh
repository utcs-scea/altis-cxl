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
benchmarks=('where' 'bfs' 'sort' 'pathfinder' 'lavamd' 'mandelbrot' 'srad' 'cfd' )
benchmarks=('mandelbrot' 'lavamd' 'pathfinder' 'cfd' 'srad' 'sort' 'where')

benchmarks=('where' 'pathfinder')
benchmarks=('fdtd2d' 'particlefilter_naive')
configs=('zero-copy')
configs=('pageable' 'copy')
configs=('pageable' 'copy' 'uvm' 'uvm-prefetch' 'zero-copy')
configs=('pageable' 'copy' 'uvm' 'zero-copy')

configs=('pageable' 'uvm' 'zero-copy')
configs=('zero-copy')
benchmarks=('mandelbrot' 'lavamd' 'pathfinder' 'cfd' 'srad' 'sort' 'where')
configs=('uvm')
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
            #sudo dmesg -C

            if [[ ${benchmarks1[@]} =~ $bench ]]; then
                level=level1
            elif [[ ${benchmarks2[@]} =~ $bench ]]; then
                level=level2
            else 
                echo "not on listed\n"
                exit 0
            fi
#            numactl --membind=1 --cpunodebind=1 \
            $pwd/build/bin/$level/$bench -s 4 --passes 1 --$config -o $pwd/results/gpuddle/$bench-$config.csv -b $bench
            echo "Done with $bench ${i} times..."
            sleep 3
        done
    done
done
