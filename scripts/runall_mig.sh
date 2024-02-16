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

GPU_UUID=GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b
MIG_UID1=MIG-27fe9e05-cfa5-53c0-aed9-322127a479d6
MIG_UID2=MIG-12ef8e3d-31ef-58c8-8c4b-ff3ca6995caa


benchmarks1=('gups' 'bfs' 'sort' 'pathfinder' 'gemm')
benchmarks2=('nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

# Running these benches
#benchmarks=('gups' 'gemm' 'bfs' 'sort' 'pathfinder' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )
benchmarks=('gemm' 'bfs' 'sort' 'pathfinder' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

#benchmarks=('gemm' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd')
benchmarks=('bfs')

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
        #$pwd/build/bin/$level/$bench -s 4 --passes 1 --uvm -o $pwd/results/$bench.csv -b $bench
        #numactl --membind=0 --cpunodebind=0 \
        CUDA_VISIBLE_DEVICES=$MIG_UID1 \
        $pwd/build/bin/$level/$bench -s 4 --passes 1 --zero-copy -o $pwd/results/$bench.csv -b $bench
        #$pwd/build/bin/$level/$bench -s 4 --passes 1 --emoji -o $pwd/results/$bench-numa.csv -b $bench
        #sudo dmesg > $pwd/results/fault_counts/$bench-faults.log
        echo "Done with $bench ${i} times..."
        sleep 3
    done
done
