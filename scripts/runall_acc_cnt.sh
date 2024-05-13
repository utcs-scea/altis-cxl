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

NVBIT_PATH=/home/tkim/workspace/nvbit_release/tools

pwd="$(dirname $(pwd))"
mkdir -p $pwd/results

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

benchmarks=('where' 'sort' 'pathfinder' 'lavamd' 'mandelbrot' 'srad' 'cfd')
benchmarks=('srad')
# Low amplification

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
    echo "<<<<< Running $bench without nvbit mem_trace >>>>>"
    LD_PRELOAD=$NVBIT_PATH/mem_trace/mem_trace.so $pwd/build/bin/$level/$bench -s 4 --passes 1 \
        --uvm -b $bench
    mv /home/tkim/workspace/page_accesses.csv $pwd/nvbit-results/access/$bench-access2.csv
    mv /home/tkim/workspace/alloc_addr.csv $pwd/nvbit-results/access/$bench-access-addr2.csv
#    mv /home/tkim/workspace/page_accesses.csv $pwd/nvbit-results/access/$bench-ord-ro2.csv
#    mv /home/tkim/workspace/alloc_addr.csv $pwd/nvbit-results/access/$bench-alloc-addr2.csv

#    rm /home/tkim/workspace/alloc_addr.csv -f
#    sudo mv /disk/tkim/mem_trace/output.csv /disk/tkim/mem_trace/$bench\_trace.log
#    sudo dmesg > /disk/tkim/mem_trace/$bench\_dmesg.log
    echo "<<<<< Done nvbit mem_trace >>>>>"
done
