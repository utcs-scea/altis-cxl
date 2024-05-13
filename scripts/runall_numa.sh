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
benchmarks=('mandelbrot')
benchmarks=('gups' 'gemm' 'bfs' 'sort' 'pathfinder' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )
benchmarks=('mandelbrot' 'srad' 'fdtd2d' 'cfd' )
benchmarks=('where')
benchmarks=('gups' 'sort' 'pathfinder' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )
benchmarks=('where' 'sort' 'pathfinder' 'lavamd' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd')
benchmarks=('lavamd' 'srad' 'cfd')
benchmarks=('where' 'sort' 'lavamd' 'srad' 'cfd')
configs=('uvm' 'zero-copy' 'pud')

total_run=$1

pwd="$(dirname $(pwd))"
mkdir -p $pwd/results
nvbit_results=$pwd/nvbit-results

for config in "${configs[@]}"
do
    for bench in "${benchmarks[@]}"
    do
        for numa in $(seq 0 1)
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
                dummy_var=$(cat $nvbit_results/$bench.csv | cut -d ',' -f1 | tr -d ' ')
                dummy_var=$(($dummy_var*4/1024))

                #numactl --membind=$numa --cpunodebind=$numa \
                #$pwd/build/bin/$level/$bench -s 3 --passes 1 --dha -o $pwd/results/numa/$bench-dha-numa$numa.csv -b $bench
                numactl --membind=$numa --cpunodebind=$numa \
                    $pwd/build/bin/$level/$bench -s 4 --passes 1 --$config -o $pwd/results/numa/$bench-numa$numa-$config.csv \
                    -b $bench --dummy $dummy_var --oversub-frac 1.2
                echo "Done with $bench ${i} times..."
                sleep 3
            done
        done
    done
done
