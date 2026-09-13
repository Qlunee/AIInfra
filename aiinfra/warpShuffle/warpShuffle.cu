//////结合 Warp Shuffle 与共享内存实现一个高效的 Block 级求和

#include<cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            printf("CUDA Error at %s:%d — %s\n",                      \
                   __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

__device__ float warp_reduce_sum(float val){
    for(int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void block_reduce_kernel(float* input, float* output, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //每个线程加载一个元素到寄存器中
    float val = (tid < N) ? input[tid] : 0.0f;

    //1.每个warp内部进行求和
    val = warp_reduce_sum(val);

    //2.将每个warp的结果lane0存储到共享内存中
    __shared__ float warp_sums[32]; //假设每个block最多有32个warp
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    if (laneId == 0){
        warp_sums[warpId] = val;
    }

    __syncthreads();

    //3.让第一个warp的线程0对共享内存中的结果进行求和
    int numWarps = blockDim.x / 32;
    val = (threadIdx.x < numWarps) ? warp_sums[threadIdx.x] : 0.0f;

    if (warpId == 0){
        val = warp_reduce_sum(val);
    }

    // lane0写出本block的规约结果
    if (threadIdx.x == 0){
        output[blockIdx.x] = val;
    }
}

// ---------- host 端 ----------
int main() {
    // 1. 配置问题规模
    const int N = 100000;                 // 元素总数（故意不是 32 或 block 的整数倍）
    const int blockSize = 256;            // 每 block 256 线程 = 8 个 warp
    const int gridSize  = (N + blockSize - 1) / blockSize;   // 向上取整

    const size_t bytes_in  = N * sizeof(float);
    const size_t bytes_out = gridSize * sizeof(float);

    printf("N = %d, blockSize = %d, gridSize = %d\n", N, blockSize, gridSize);

    // 2. host 端准备数据
    float* h_in     = (float*)malloc(bytes_in);
    float* h_out    = (float*)malloc(bytes_out);
    float* h_ref    = (float*)malloc(bytes_out);   // CPU 参考结果（逐 block）

    srand(42);
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 100) / 10.0f;   // 0.0 ~ 9.9 之间的随机数
    }

    // 3. CPU 端计算每个 block 的参考和
    double total_cpu = 0.0;
    for (int b = 0; b < gridSize; b++) {
        float s = 0.0f;
        for (int t = 0; t < blockSize; t++) {
            int idx = b * blockSize + t;
            if (idx < N) s += h_in[idx];
        }
        h_ref[b] = s;
        total_cpu += s;
    }

    // 4. device 端分配内存
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  bytes_in));
    CHECK_CUDA(cudaMalloc(&d_out, bytes_out));

    // 5. 数据传输 H2D
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice));

    // 6. 启动 kernel
    block_reduce_kernel<<<gridSize, blockSize>>>(d_in, d_out, N);

    // 7. 结果回传 D2H
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost));

    // 8. 验证：逐 block 比对 + 总和比对
    int errors = 0;
    double total_gpu = 0.0;
    for (int b = 0; b < gridSize; b++) {
        total_gpu += h_out[b];
        float diff = fabsf(h_out[b] - h_ref[b]);
        // 浮点求和顺序不同，允许一定误差
        float tol = 1e-3f * fmaxf(1.0f, fabsf(h_ref[b]));
        if (diff > tol) {
            if (errors < 5) {   // 只打印前几个错误
                printf("Mismatch at block %d: gpu=%.4f, cpu=%.4f, diff=%.4f\n",
                       b, h_out[b], h_ref[b], diff);
            }
            errors++;
        }
    }

    printf("\nTotal sum (CPU) = %.4f\n", total_cpu);
    printf("Total sum (GPU) = %.4f\n", total_gpu);
    printf("Diff            = %.6f\n", fabs(total_gpu - total_cpu));

    if (errors == 0) {
        printf("✅ All %d blocks passed!\n", gridSize);
    } else {
        printf("❌ %d / %d blocks mismatched.\n", errors, gridSize);
    }

    // 9. 释放资源
    free(h_in); free(h_out); free(h_ref);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return errors == 0 ? 0 : 1;
}



