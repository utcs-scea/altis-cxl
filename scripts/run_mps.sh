#!/usr/bin/env bash

set -euo pipefail

PWD=$(pwd)
#GPU 0: A100-PCIE-40GB (UUID: GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b)
#  MIG 3g.20gb Device 0: (UUID: MIG-GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b/1/0)
#  MIG 3g.20gb Device 1: (UUID: MIG-GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b/2/0)

benchmarks1=('gups' 'bfs' 'sort' 'pathfinder' 'gemm')
benchmarks2=('nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

# Running these benches
benchmarks=('gups' 'gemm' 'bfs' 'sort' 'pathfinder' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

benchmarks=('gemm' 'nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd')
bench=('gups')

gpuid=GPU-a6f9662c-db8d-8ac3-fa88-261d24609621
MIG_UID=MIG-32e533b7-dc11-5153-9211-8bcf161e4162

if [[ ${benchmarks1[@]} =~ $bench ]]; then
    level=level1
elif [[ ${benchmarks2[@]} =~ $bench ]]; then
    level=level2
else 
    echo "not on listed\n"
    exit 0
fi

if [[ $1 == "quit" ]]; then
   echo quit | nvidia-cuda-mps-control
elif [[ $1 == "init" ]]; then
   echo quit | nvidia-cuda-mps-control
    sudo nvidia-cuda-mps-control -d
    # now launch the job on the specific MIG device 
    # and select the appropriate MPS server on the device
    echo set_default_device_pinned_mem_limit 0 5G | nvidia-cuda-mps-control
    $PWD/build/bin/$level/$bench -s 4 --passes 1 \
        --uvm -o $PWD/results/$bench.csv -b $bench -i $PWD/data/bfs/bfs_16777216
else
   # set the environment variable on each MPS 
   # control daemon and use different socket for each MIG instance
   CUDA_MPS_PIPE_DIRECTORY=/tmp/$gpuid
   mkdir -p $CUDA_MPS_PIPE_DIRECTORY
   CUDA_VISIBLE_DEVICES=$gpuid \
        $PWD/build/bin/$level/$bench -s 4 --passes 1 \
            --uvm -o $PWD/results/$bench.csv -b $bench -i $PWD/data/bfs/bfs_16777216
fi
