#!/usr/bin/env bash

set -euo pipefail

#GPU 0: A100-PCIE-40GB (UUID: GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b)
#  MIG 3g.20gb Device 0: (UUID: MIG-GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b/1/0)
#  MIG 3g.20gb Device 1: (UUID: MIG-GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b/2/0)

GPU_UUID=GPU-63feeb45-94c6-b9cb-78ea-98e9b7a5be6b
MIG_UID=MIG-32e533b7-dc11-5153-9211-8bcf161e4162
if [[ $1 == "quit" ]]; then
    for i in $MIG_UID; do
       export CUDA_MPS_PIPE_DIRECTORY=/tmp/$i
       mkdir -p $CUDA_MPS_PIPE_DIRECTORY
       sudo CUDA_VISIBLE_DEVICES=$i \
           CUDA_MPS_PIPE_DIRECTORY=/tmp/$i \
           echo quit | nvidia-cuda-mps-control
   done
elif [[ $1 == "init" ]]; then
    for i in $MIG_UID; do
       export CUDA_MPS_PIPE_DIRECTORY=/tmp/$i
       mkdir -p $CUDA_MPS_PIPE_DIRECTORY
       sudo CUDA_VISIBLE_DEVICES=$i \
           CUDA_MPS_PIPE_DIRECTORY=/tmp/$i \
           nvidia-cuda-mps-control -d

       # now launch the job on the specific MIG device 
       # and select the appropriate MPS server on the device
       CUDA_MPS_PIPE_DIRECTORY=/tmp/$i \
       CUDA_VISIBLE_DEVICES=$i \
       ./mem_access u 5300 1 &

       CUDA_VISIBLE_DEVICES=$i \
       ./mem_access u 4200 1 &
       wait
    done
else
    for i in $MIG_UID; do
   # set the environment variable on each MPS 
   # control daemon and use different socket for each MIG instance
   export CUDA_MPS_PIPE_DIRECTORY=/tmp/$i
   mkdir -p $CUDA_MPS_PIPE_DIRECTORY
   sudo CUDA_VISIBLE_DEVICES=$i \
       CUDA_MPS_PIPE_DIRECTORY=/tmp/$i \
       echo quit | nvidia-cuda-mps-control

   sudo CUDA_VISIBLE_DEVICES=$i \
       CUDA_MPS_PIPE_DIRECTORY=/tmp/$i \
       nvidia-cuda-mps-control -d

   # now launch the job on the specific MIG device 
   # and select the appropriate MPS server on the device
   CUDA_MPS_PIPE_DIRECTORY=/tmp/$i \
   CUDA_VISIBLE_DEVICES=$i \
   ./mem_access u 5300 1 &

   sleep 1
   CUDA_VISIBLE_DEVICES=$i \
   ./mem_access u 4200 1 &
   wait
   done
fi
