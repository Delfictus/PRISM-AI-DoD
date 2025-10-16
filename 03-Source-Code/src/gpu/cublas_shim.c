// CUBLAS Compatibility Shim for CUDA 12.8
// Provides missing symbols that cudarc expects but don't exist in CUDA 12.8

#include <stdint.h>
#include <stddef.h>

// The missing symbol that cudarc tries to load
// This function doesn't exist in CUDA 12.8, but cudarc expects it
// We provide it as a no-op since it's only used for legacy emulation mode
__attribute__((visibility("default")))
int cublasGetEmulationStrategy(void* handle, int* strategy) {
    // Return 0 (no emulation) - this is safe as emulation mode is deprecated
    if (strategy != NULL) {
        *strategy = 0;  // CUBLAS_EMULATION_OFF
    }
    return 0;  // CUBLAS_STATUS_SUCCESS
}

// Additional compatibility functions if needed
__attribute__((visibility("default")))
int cublasSetEmulationStrategy(void* handle, int strategy) {
    // No-op - emulation mode is deprecated in modern CUDA
    return 0;  // CUBLAS_STATUS_SUCCESS
}