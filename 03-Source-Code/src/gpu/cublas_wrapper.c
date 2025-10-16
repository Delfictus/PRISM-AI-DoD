// Comprehensive CUBLAS Wrapper for CUDA 12.8 Compatibility
// This wrapper library provides missing symbols and forwards all other calls to the real CUBLAS

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Handle to the real CUBLAS library
static void* real_cublas_handle = NULL;

// Initialize the real CUBLAS library
__attribute__((constructor))
static void init_real_cublas() {
    // Try to load the real CUBLAS library
    const char* real_cublas_paths[] = {
        "/usr/local/cuda-12.8/lib64/libcublas.so.12",
        "/usr/local/cuda-12.8/lib64/libcublas.so.12.8.0.0",
        "/usr/local/cuda/lib64/libcublas.so.12",
        "/usr/lib/x86_64-linux-gnu/libcublas.so.12",
        NULL
    };

    for (int i = 0; real_cublas_paths[i] != NULL; i++) {
        real_cublas_handle = dlopen(real_cublas_paths[i], RTLD_LAZY | RTLD_GLOBAL);
        if (real_cublas_handle) {
            fprintf(stderr, "[CUBLAS Wrapper] Successfully loaded real CUBLAS from: %s\n", real_cublas_paths[i]);
            break;
        }
    }

    if (!real_cublas_handle) {
        fprintf(stderr, "[CUBLAS Wrapper] WARNING: Could not load real CUBLAS library\n");
    }
}

// Cleanup
__attribute__((destructor))
static void cleanup_real_cublas() {
    if (real_cublas_handle) {
        dlclose(real_cublas_handle);
    }
}

// The missing symbols that cudarc expects but don't exist in CUDA 12.8
__attribute__((visibility("default")))
int cublasGetEmulationStrategy(void* handle, int* strategy) {
    fprintf(stderr, "[CUBLAS Wrapper] cublasGetEmulationStrategy called (providing compatibility stub)\n");
    if (strategy != NULL) {
        *strategy = 0;  // CUBLAS_EMULATION_OFF
    }
    return 0;  // CUBLAS_STATUS_SUCCESS
}

__attribute__((visibility("default")))
int cublasSetEmulationStrategy(void* handle, int strategy) {
    fprintf(stderr, "[CUBLAS Wrapper] cublasSetEmulationStrategy called (providing compatibility stub)\n");
    // No-op - emulation mode is deprecated in modern CUDA
    return 0;  // CUBLAS_STATUS_SUCCESS
}

// Generic symbol forwarding for all other CUBLAS functions
__attribute__((visibility("default")))
void* __dlsym(void* handle, const char* symbol) {
    // First check if this is one of our compatibility symbols
    if (strcmp(symbol, "cublasGetEmulationStrategy") == 0) {
        return (void*)cublasGetEmulationStrategy;
    }
    if (strcmp(symbol, "cublasSetEmulationStrategy") == 0) {
        return (void*)cublasSetEmulationStrategy;
    }

    // Forward to the real dlsym
    typedef void* (*dlsym_fn)(void*, const char*);
    static dlsym_fn real_dlsym = NULL;

    if (!real_dlsym) {
        real_dlsym = (dlsym_fn)dlsym(RTLD_NEXT, "dlsym");
    }

    // If looking for CUBLAS symbols and we have the real library, check there first
    if (real_cublas_handle && handle == RTLD_DEFAULT) {
        void* sym = dlsym(real_cublas_handle, symbol);
        if (sym) return sym;
    }

    // Otherwise use the real dlsym
    return real_dlsym(handle, symbol);
}

// Common CUBLAS functions that we need to forward
// We'll implement a few key ones explicitly for better compatibility

typedef void* cublasHandle_t;
typedef int cublasStatus_t;

// cublasCreate_v2
__attribute__((visibility("default")))
cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
    if (!real_cublas_handle) return 1; // CUBLAS_STATUS_NOT_INITIALIZED

    typedef cublasStatus_t (*cublasCreate_v2_fn)(cublasHandle_t*);
    cublasCreate_v2_fn real_func = (cublasCreate_v2_fn)dlsym(real_cublas_handle, "cublasCreate_v2");
    if (!real_func) {
        fprintf(stderr, "[CUBLAS Wrapper] Failed to find cublasCreate_v2 in real CUBLAS\n");
        return 1;
    }

    return real_func(handle);
}

// cublasDestroy_v2
__attribute__((visibility("default")))
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    if (!real_cublas_handle) return 1; // CUBLAS_STATUS_NOT_INITIALIZED

    typedef cublasStatus_t (*cublasDestroy_v2_fn)(cublasHandle_t);
    cublasDestroy_v2_fn real_func = (cublasDestroy_v2_fn)dlsym(real_cublas_handle, "cublasDestroy_v2");
    if (!real_func) {
        fprintf(stderr, "[CUBLAS Wrapper] Failed to find cublasDestroy_v2 in real CUBLAS\n");
        return 1;
    }

    return real_func(handle);
}

// cublasSetStream_v2
__attribute__((visibility("default")))
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, void* streamId) {
    if (!real_cublas_handle) return 1; // CUBLAS_STATUS_NOT_INITIALIZED

    typedef cublasStatus_t (*cublasSetStream_v2_fn)(cublasHandle_t, void*);
    cublasSetStream_v2_fn real_func = (cublasSetStream_v2_fn)dlsym(real_cublas_handle, "cublasSetStream_v2");
    if (!real_func) {
        fprintf(stderr, "[CUBLAS Wrapper] Failed to find cublasSetStream_v2 in real CUBLAS\n");
        return 1;
    }

    return real_func(handle, streamId);
}

// cublasSgemm_v2 - Single precision matrix multiply
__attribute__((visibility("default")))
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                               int transa, int transb,
                               int m, int n, int k,
                               const float* alpha,
                               const float* A, int lda,
                               const float* B, int ldb,
                               const float* beta,
                               float* C, int ldc) {
    if (!real_cublas_handle) return 1; // CUBLAS_STATUS_NOT_INITIALIZED

    typedef cublasStatus_t (*cublasSgemm_v2_fn)(cublasHandle_t, int, int, int, int, int,
                                                 const float*, const float*, int,
                                                 const float*, int, const float*,
                                                 float*, int);
    cublasSgemm_v2_fn real_func = (cublasSgemm_v2_fn)dlsym(real_cublas_handle, "cublasSgemm_v2");
    if (!real_func) {
        fprintf(stderr, "[CUBLAS Wrapper] Failed to find cublasSgemm_v2 in real CUBLAS\n");
        return 1;
    }

    return real_func(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// cublasGetVersion_v2
__attribute__((visibility("default")))
cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int* version) {
    if (!real_cublas_handle) return 1; // CUBLAS_STATUS_NOT_INITIALIZED

    typedef cublasStatus_t (*cublasGetVersion_v2_fn)(cublasHandle_t, int*);
    cublasGetVersion_v2_fn real_func = (cublasGetVersion_v2_fn)dlsym(real_cublas_handle, "cublasGetVersion_v2");
    if (!real_func) {
        fprintf(stderr, "[CUBLAS Wrapper] Failed to find cublasGetVersion_v2 in real CUBLAS\n");
        return 1;
    }

    return real_func(handle, version);
}

// cublasGetProperty
__attribute__((visibility("default")))
cublasStatus_t cublasGetProperty(int type, int* value) {
    if (!real_cublas_handle) return 1; // CUBLAS_STATUS_NOT_INITIALIZED

    typedef cublasStatus_t (*cublasGetProperty_fn)(int, int*);
    cublasGetProperty_fn real_func = (cublasGetProperty_fn)dlsym(real_cublas_handle, "cublasGetProperty");
    if (!real_func) {
        fprintf(stderr, "[CUBLAS Wrapper] Failed to find cublasGetProperty in real CUBLAS\n");
        return 1;
    }

    return real_func(type, value);
}

// cublasSdot_v2 - Dot product
__attribute__((visibility("default")))
cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n,
                              const float* x, int incx,
                              const float* y, int incy,
                              float* result) {
    if (!real_cublas_handle) return 1; // CUBLAS_STATUS_NOT_INITIALIZED

    typedef cublasStatus_t (*cublasSdot_v2_fn)(cublasHandle_t, int, const float*, int, const float*, int, float*);
    cublasSdot_v2_fn real_func = (cublasSdot_v2_fn)dlsym(real_cublas_handle, "cublasSdot_v2");
    if (!real_func) {
        fprintf(stderr, "[CUBLAS Wrapper] Failed to find cublasSdot_v2 in real CUBLAS\n");
        return 1;
    }

    return real_func(handle, n, x, incx, y, incy, result);
}

// cublasSnrm2_v2 - Euclidean norm
__attribute__((visibility("default")))
cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n,
                               const float* x, int incx,
                               float* result) {
    if (!real_cublas_handle) return 1; // CUBLAS_STATUS_NOT_INITIALIZED

    typedef cublasStatus_t (*cublasSnrm2_v2_fn)(cublasHandle_t, int, const float*, int, float*);
    cublasSnrm2_v2_fn real_func = (cublasSnrm2_v2_fn)dlsym(real_cublas_handle, "cublasSnrm2_v2");
    if (!real_func) {
        fprintf(stderr, "[CUBLAS Wrapper] Failed to find cublasSnrm2_v2 in real CUBLAS\n");
        return 1;
    }

    return real_func(handle, n, x, incx, result);
}