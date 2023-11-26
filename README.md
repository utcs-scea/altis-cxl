# Altis Benchmark Suite

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](LICENSE)
![CI](https://github.com/utcs-scea/altis/actions/workflows/action.yml/badge.svg)
[![DOI:10.1109/ISPASS48437.2020.00011](https://zenodo.org/badge/DOI/10.1109/ISPASS48437.2020.00011.svg)](https://doi.org/10.1109/ISPASS48437.2020.00011)

Altis-cxl repository forked from Altis benchmark suite with updates for multiple application running concurrently by synchronizing the demand paging and data transfer.

Altis is a benchmark suite to test the performance and other aspects of systems with Graphics Processing Units (GPUs), developed in [SCEA](https://github.com/utcs-scea) lab at University of Texas at Austin. Altis consists of a collection of GPU applications with differnt performance implications. Altis focuses primarily on [Compute Unified Device Architecture](https://developer.nvidia.com/cuda-toolkit) (CUDA) computing platform.

Documentaion regarding this project can be found at the [Wiki](https://github.com/utcs-scea/altis/wiki) page. The Wiki document contains information regarding Altis setup, installation, usage, and other information.

> We are refactoring Altis codebase for better usability and making it more developer-friendly. We made sure the benchmark still compile properly during refactoring so you can still use it. The refactoring involves changing how each benchmark application is used and adding more benchmarks.

## Setup

Altis relies on the avaialbility of CUDA and CMake (>= 3.8). Please refer to [Environment Setup](https://github.com/utcs-scea/altis/wiki/Environment-Setup) for how to set up Altis.

## Build:

After the environment is setup properly, go to the root directory of Altis, execute:

```bash
./setup.sh
```

For more information regarding building process, please refer to [Build](https://github.com/utcs-scea/altis/wiki/Build) for more information.


## Run

To run all the benchmarks, simply execute ```./runall.sh [number for iteration]``` script. 
This will run all the benchmarks with specified iteration and save the results (kernel execution time and data transfer latency) into ```./results``` directory.
You can modify ```runall.sh``` to change size of dataset (```-s``` options from the script). 


## Cite Us

Bibtex is shown below:  

@INPROCEEDINGS{9238617,  
  author={B. {Hu} and C. J. {Rossbach}},  
  booktitle={2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},  
  title={Altis: Modernizing GPGPU Benchmarks},  
  year={2020},  
  volume={},  
  number={},  
  pages={1-11},  
  doi={10.1109/ISPASS48437.2020.00011}}  

## Publication

B. Hu and C. J. Rossbach, "Altis: Modernizing GPGPU Benchmarks," 2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), Boston, MA, USA, 2020, pp. 1-11, doi: 10.1109/ISPASS48437.2020.00011.
