// Simple GPU test to confirm CUDA works
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void simple_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx * 2.0f;
}

int main() {
    printf("=== SIMPLE GPU TEST ===\n");

    // Check for GPU
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("GPU devices found: %d\n", deviceCount);

    if (deviceCount == 0) {
        printf("❌ No GPU found\n");
        return 1;
    }

    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // Test kernel launch
    float *d_data, h_data[256];
    cudaMalloc(&d_data, 256 * sizeof(float));

    simple_kernel<<<1, 256>>>(d_data);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, 256 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nKernel executed successfully!\n");
    printf("Sample results: data[0]=%f, data[1]=%f, data[255]=%f\n",
           h_data[0], h_data[1], h_data[255]);

    cudaFree(d_data);

    printf("\n✅ GPU IS WORKING!\n");
    return 0;
}