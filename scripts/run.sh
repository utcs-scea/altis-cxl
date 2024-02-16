#!/usr/bin/env bash

set -euo pipefail

#GPU 0: A100-PCIE-40GB (UUID: GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b)
#  MIG 3g.20gb Device 0: (UUID: MIG-GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b/1/0)
#  MIG 3g.20gb Device 1: (UUID: MIG-GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b/2/0)

benchmarks1=('gups' 'sort' 'pathfinder' 'gemm')
benchmarks2=('nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )
benchmarks1=('pathfinder')

GPU_UUID=GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b
MIG_UID1=MIG-08dd6e9d-5721-5b55-872d-d8421f4716df
MIG_UID2=MIG-429e08f1-96b3-59b1-ac9f-b8606f0b71ff


display_usage() {
    echo "./run.sh [bench] [0/1] [uvm/zero-copy]"
}

if [[ ( $@ == "--help") ||  $@ == "-h" ]]; then 
    display_usage
    exit 0
elif [[ $# -le 2 ]]; then
    display_usage
    exit 0
fi 

echo 3 | sudo tee /proc/sys/vm/drop_caches
sudo dmesg -C
sleep 1

bench=$1

if [[ ${benchmarks1[@]} =~ $bench ]]; then
    level=level1
elif [[ ${benchmarks2[@]} =~ $bench ]]; then
    level=level2
else 
    echo "no\n"
    exit 0
fi

for i in {1..3}
do
    if [[ $2 == 0 ]]; then
        CUDA_VISIBLE_DEVICES=$MIG_UID1 \
        /usr/local/cuda/bin/nsys profile --force-overwrite=true \
           --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --export=json \
           ./$level/$bench -s 4 --passes 1 --$3 -o $bench
    elif [[ $3 == 1 ]]; then
       ./$level/$bench -s 4 --passes 1 --$3 -o $bench
    fi
    echo 3 | sudo tee /proc/sys/vm/drop_caches
    sudo dmesg -C
    sleep 3
done
