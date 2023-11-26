#!/usr/bin/env bash

set -euo pipefail

display_usage() {
    echo "./runall.sh"
}

benchmarks1=('gups' 'sort' 'pathfinder' 'gemm')
benchmarks2=('nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

# Running these benches
benchmarks=('gups' 'sort' 'pathfinder' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

pwd=$(pwd)
mkdir -p $pwd/results

for bench in "${benchmarks[@]}"
do
    # 3 runs
    for i in {1}
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
        $pwd/build/bin/$level/$bench -s 4 --passes 1 --uvm -o $pwd/results/$bench.csv -b $bench
        echo "Done with $bench ${i} times..."
        sleep 3
    done
done
