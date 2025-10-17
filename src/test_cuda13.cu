#include <cuda_runtime.h>
#include <stdio.h>

__global__ void testKernel() {
    printf("Hello from CUDA 13 kernel! Thread %d, Block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("══════════════════════════════════════════════\n");
    printf("      CUDA 13 + Driver 580 Validation\n");
    printf("══════════════════════════════════════════════\n\n");

    int runtimeVersion, driverVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    
    printf("CUDA Runtime: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
    printf("CUDA Driver:  %d.%d\n\n", driverVersion/1000, (driverVersion%100)/10);

    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess) {
        printf("Error: %s (code %d)\n", cudaGetErrorString(err), err);
        return 1;
    }
    
    printf("Number of CUDA devices: %d\n\n", deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        printf("GPU: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Total Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Multiprocessors: %d\n\n", prop.multiProcessorCount);
        
        printf("Launching test kernel...\n");
        testKernel<<<2, 4>>>();
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        
        printf("\n✅ SUCCESS: CUDA 13 + Driver 580 working!\n");
        return 0;
    }
    
    return 1;
}
