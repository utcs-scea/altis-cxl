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

benchmarks=('pathfinder' 'lavamd' 'mandelbrot' 'srad' 'cfd' )
benchmarks=('where' 'sort' 'lavamd' 'cfd' 'pathfinder' 'srad')
benchmarks=('where' 'pathfinder')
benchmarks=('where' 'mandelbrot' 'lavamd' 'pathfinder' 'sort' 'cfd' 'srad')
benchmarks=('sort')
benchmarks=('cfd' 'srad' 'sort')
benchmarks=('cfd' 'sort')
benchmarks=('where' 'lavamd')
benchmarks=('fdtd2d')
benchmarks=('where')
config=('zero-copy')

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

        dummy_var=$(cat $nvbit_results/$bench.csv | cut -d ',' -f1 | tr -d ' ')
        dummy_var=$(($dummy_var*4/1024))

        echo "<<<<<Running $bench with dummy memory allocation $dummy_var>>>>>"
        numactl --membind=0 --cpunodebind=0 \
        $pwd/build/bin/$level/$bench -s 4 --passes 1 \
            --$config -o $pwd/results/$bench-oversub-$config.csv -b $bench --dummy $dummy_var --oversub-frac 1.2
        sleep 3
    done
done
