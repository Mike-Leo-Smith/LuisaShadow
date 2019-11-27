//
// Created by Mike on 11/27/2019.
//

#include <cstdio>
#include <cuda_runtime_api.h>

#include "test.h"

__global__ void do_test() {
    printf("Hello, World!\n");
}

void test() {
    do_test<<<1, 8>>>();
}
