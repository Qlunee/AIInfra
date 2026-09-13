#include <cstdio>

// 标记函数为 kernel，由 CPU 调用、GPU 执行
__global__ void hello_from_gpu(){
    printf("Hello World from GPU!\n From thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main(){
    printf("Launching kernel from CPU...\n");

    // 启动配置：2 个 Block，每 Block 4 个线程
    hello_from_gpu<<<2, 4>>>();

    cudaDeviceSynchronize();

    printf("Done!\n");
    return 0;
}

