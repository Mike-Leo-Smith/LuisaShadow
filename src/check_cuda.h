//
// Created by Mike on 11/29/2019.
//

#include <cstdlib>
#include <iostream>
#include <cuda_runtime_api.h>

#define CHECK_CUDA(call)                                                       \
    [&] {                                                                      \
        cudaError_t error = call;                                              \
        if(error != cudaSuccess)                                               \
        {                                                                      \
            std::cerr << "CUDA call [ " << #call << " ] failed with error: '"  \
               << cudaGetErrorString(error)                                    \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            exit(-1);                                                          \
        }                                                                      \
    }()