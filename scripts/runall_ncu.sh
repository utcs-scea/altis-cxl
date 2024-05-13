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

pwd="$(dirname $(pwd))"
mkdir -p $pwd/results

benchmarks1=('gups' 'bfs' 'sort' 'pathfinder' 'gemm')
benchmarks2=('nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

# Running these benches
benchmarks=('gups' 'bfs' 'gemm' 'sort' 'pathfinder' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )
#benchmarks=('sort' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )
#benchmarks=('sort')

# High amplification
benchmarks=('cfd' 'pathfinder' 'lavamd' 'mandelbrot' 'srad')

benchmarks=('where' 'sort' 'cfd' 'lavamd' 'srad' 'pathfinder')
benchmarks=('where')
configs=('uvm' 'zero-copy' 'pud')

pwd="$(dirname $(pwd))"
nvbit_results=$pwd/nvbit-results
#mkdir -p $pwd/results

for bench in "${benchmarks[@]}"
do
    for config in "${configs[@]}"
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

        ncu --metrics pcie__read_bytes.sum \
            --log-file $pwd/ncu-results/pud/$bench-$config-read-pci.csv --csv \
            $pwd/build/bin/$level/$bench -s 4 --passes 1 --dummy $dummy_var --oversub-frac 1.2\
            --$config -o $pwd/results/gpuddle-pcie/$bench.csv -b $bench 
        sleep 3
    done
done
