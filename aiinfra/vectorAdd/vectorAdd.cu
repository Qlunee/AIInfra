#include<cstdio>
#include<cstdlib>
#include<cuda_runtime.h>
#include<cmath>
#include<chrono>

// CUDA 错误检查宏  
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// 标记函数为 kernel，由 CPU 调用、GPU 执行
__global__ void vectorADD(const float* A, const float* B, float* C, int N){
    int idx = blockIdx.x  * blockDim.x + threadIdx.x;

    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

// CPU基线
void vectorADD_CPU(const float* A, const float* B, float* C, int N){
    for(int i = 0; i < N; i++){
        C[i] = A[i] + B[i];
    }
}

int main(){
    int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);
    printf("Vector size: %d elements (%.1f MB)\n", N, size / (1024.0 * 1024.0));

    // 1.分配主机内存
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_GPU = (float*)malloc(size);
    float* h_C_CPU = (float*)malloc(size);

    // 初始化输入数据
    for (int i = 0; i < N; i++){
        h_A[i] = sinf(i * 0.001f);
        h_B[i] = cosf(i * 0.001f);
    }

    // CPU计算
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorADD_CPU(h_A, h_B, h_C_CPU, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();
    printf("CPU time: %.3f ms\n", cpu_ms);

    // 2.分配设备内存
    float* d_A, * d_B, * d_C;
    // 告诉我 d_A 这个变量放在哪里，我才能把新申请到的 GPU 地址写回去。
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    //创建cuda计时器
    cudaEvent_t start, stop, Kernelstart, Kernelstop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&Kernelstart));
    CUDA_CHECK(cudaEventCreate(&Kernelstop));

    cudaEventRecord(start); //开始计时

    // 3.将输入数据从主机内存拷贝到设备内存
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 4.配置并执行 kernel 
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEventRecord(Kernelstart); //开始计时kernel
    vectorADD<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(Kernelstop); //结束计时kernel

    CUDA_CHECK(cudaGetLastError()); // 检查 kernel 启动是否成功
    CUDA_CHECK(cudaDeviceSynchronize()); // 等待 kernel 执行完成

    // 5.将结果从设备内存拷贝回主机内存
    CUDA_CHECK(cudaMemcpy(h_C_GPU, d_C, size, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop); 

    float gpu_total_ms = 0, kernel_ms = 0;
    cudaEventElapsedTime(&gpu_total_ms, start, stop);
    cudaEventElapsedTime(&kernel_ms, Kernelstart, Kernelstop);

    printf("GPU kernel time: %.4f ms\n", kernel_ms);
    printf("GPU total time (with memcpy): %.4f ms\n", gpu_total_ms);
    printf("Speedup (kernel only): %.1fx\n", cpu_ms / kernel_ms);
    printf("Speedup (end-to-end): %.1fx\n", cpu_ms / gpu_total_ms);   


    // 6.验证结果
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C_GPU[i] - h_C_CPU[i]) > 1e-5f) {
            if (errors < 5) {
                printf("Mismatch at %d: CPU=%.6f, GPU=%.6f\n",
                       i, h_C_CPU[i], h_C_GPU[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification PASSED!\n");
    } else {
        printf("Verification FAILED: %d mismatches\n", errors);
    }

    // 向量加法读 2 个 float、写 1 个 float = 12 bytes/element
    float bandwidth = (3.0f * size) / (kernel_ms / 1000.0f) / (1024.0f * 1024.0f * 1024.0f); // GB/s
    printf("Effective bandwidth: %.1f GB/s\n", bandwidth);

    // 7.释放设备内存
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(Kernelstart);
    cudaEventDestroy(Kernelstop);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);  
    free(h_C_GPU);
    free(h_C_CPU);

    return 0;
}
