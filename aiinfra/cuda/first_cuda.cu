#include <cstdio>

__global__ void hello_from_gpu(){
    printf("Hello World from GPU!\n From thread %d\n in block %d\n", threadIdx.x, blockIdx.x);
}

int main(){
    printf("Launching kernel from CPU...\n");

    hello_from_gpu<<<2, 4>>>();

    cudaDeviceSynchronize();

    printf("Done!\n");
    return 0;
}

