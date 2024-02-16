#!/usr/bin/env bash

set -euo pipefail

#benchmarks=('cfd' 'srad' 'lavamd' 'raytracing' 'nw' 'kmeans' 'fdtd2d' )
#benchmarks=('nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )
benchmarks_l1=('gups' 'gemm' 'sort' 'pathfinder')
benchmarks_l2=('nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd')
benchmarks=('gups' 'gemm' 'sort' 'pathfinder')
#configs=('pageable' 'uvm' 'copy')
configs=('uvm')

display_usage() {
    echo "./runall_ncu.sh"
}

if [[ ( $@ == "--help") ||  $@ == "-h" ]]; then 
    display_usage
    exit 0
elif [[ $# -ge 1 ]]; then
    display_usage
    exit 0
fi 

for bench in "${benchmarks_l1[@]}"
do
    for config in "${configs[@]}"
    do
        echo 3 | sudo tee /proc/sys/vm/drop_caches
        sudo dmesg -C
        sleep 6
        echo "Running $bench with $config option..."
        ncu --csv -c 1000 --metrics smsp__warps_eligible.avg.per_cycle_elapsed \
            --log-file ./results/eligible_warp/$bench.csv \
            ./level1/$bench -s 4 --passes 1 --$config
        echo "=========== Done with $bench with $config ==========="
    done
done

for bench in "${benchmarks_l2[@]}"
do
    for config in "${configs[@]}"
    do
        echo 3 | sudo tee /proc/sys/vm/drop_caches
        sudo dmesg -C
        sleep 6
        echo "Running $bench with $config option..."
        ncu --csv -c 1000 --metrics smsp__warps_eligible.avg.per_cycle_elapsed \
            --log-file ./results/eligible_warp/$bench.csv \
            ./level2/$bench -s 4 --passes 1 --$config
        #ncu --csv -c 1000 --metrics smsp__warps_active.avg.per_cycle_active --log-file ./active_warp/$bench.csv ./$bench -s 4 --passes 1 --$config
        echo "=========== Done with $bench with $config ==========="
    done
done
