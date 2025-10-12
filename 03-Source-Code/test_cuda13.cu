#include <cuda_runtime.h>
#include <stdio.h>

__global__ void testKernel() {
    printf("Hello from CUDA 13 kernel! Thread %d, Block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("CUDA 13 Validation Test\n");
    printf("========================\n\n");

    // Check CUDA version
    int runtimeVersion, driverVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);

    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
    printf("CUDA Driver Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);

    // Get device properties
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("\n❌ Failed to get device count: %s (error code: %d)\n",
               cudaGetErrorString(err), err);
        printf("   This might be because nvidia-smi isn't working\n");
        return 1;
    }

    printf("\nNumber of CUDA devices: %d\n", deviceCount);

    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        printf("\nGPU Device 0 Properties:\n");
        printf("  Name: %s\n", prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);

        // Test kernel launch
        printf("\n🚀 Launching test kernel...\n");
        testKernel<<<2, 4>>>();

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("❌ Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }

        printf("\n✅ CUDA 13 is working correctly!\n");
        return 0;
    }

    printf("❌ No CUDA devices found\n");
    return 1;
}