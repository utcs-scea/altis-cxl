#!/usr/bin/env bash

# m:1 UVM m:2 zero
# t:0 Baseline t:1 coalesce 

#pwd="$(pwd)"
pwd="$(dirname $(pwd))"
module_path=/home/tkim/NVIDIA-Linux-x86_64-515.48.07/kernel

benchmarks1=('gups' 'bfs' 'sort' 'pathfinder' 'gemm')
benchmarks2=('nw' 'lavamd' 'where' 'particlefilter_naive' 'mandelbrot' 'srad' 'fdtd2d' 'cfd' )

benchmarks=('mandelbrot' 'lavamd' 'pathfinder' 'cfd' 'srad' 'sort' 'where' 'particlefilter_naive' 'fdtd2d')
latency=('cxl' 'cxl50' 'cxl100' 'cxl200' 'cxl300')
config=('uvm')

#sudo dmesg -C
#sleep 1
#numactl --membind=0 --cpunodebind=0 \
#/usr/local/cuda/bin/nsys profile --force-overwrite=true \
#    --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --cuda-memory-usage=true \

total_run=$1

for lat in "${latency[@]}"
do
    sed -i "s/UVM_BUILD_TYPE = release/UVM_BUILD_TYPE = $lat/g" $module_path/nvidia-uvm/nvidia-uvm.Kbuild
    #sudo $module_path/mod.sh
    make -C $module_path -j 16
    sudo rmmod nvidia_uvm
    sudo insmod $module_path/nvidia-uvm.ko
    sleep 1
    sed -i "s/UVM_BUILD_TYPE = $lat/UVM_BUILD_TYPE = release/g" $module_path/nvidia-uvm/nvidia-uvm.Kbuild
    for bench in "${benchmarks[@]}"
    do
        for i in $(seq 1 $total_run)
        do
            echo 3 | sudo tee /proc/sys/vm/drop_caches
            if [[ ${benchmarks1[@]} =~ $bench ]]; then
                level=level1
            elif [[ ${benchmarks2[@]} =~ $bench ]]; then
                level=level2
            else 
                echo "not on listed\n"
                exit 0
            fi
            $pwd/build/bin/$level/$bench -s 4 --passes 1 --$config -o $pwd/results/latency/$bench-$lat.csv -b $bench
            echo "Done with $bench ${i} times..."
            sleep 3
        done
    done
done
