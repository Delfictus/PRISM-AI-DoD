// CUBLAS Dynamic Interposer for CUDA 12.8 Compatibility
// This uses dlsym interposition to forward all symbols dynamically

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
        real_cublas_handle = dlopen(real_cublas_paths[i], RTLD_NOW | RTLD_GLOBAL);
        if (real_cublas_handle) {
            fprintf(stderr, "[CUBLAS Interposer] Loaded real CUBLAS: %s\n", real_cublas_paths[i]);
            break;
        }
    }

    if (!real_cublas_handle) {
        fprintf(stderr, "[CUBLAS Interposer] WARNING: Could not load real CUBLAS\n");
    }
}

// The missing symbols that don't exist in CUDA 12.8
__attribute__((visibility("default")))
int cublasGetEmulationStrategy(void* handle, int* strategy) {
    if (strategy != NULL) {
        *strategy = 0;  // CUBLAS_EMULATION_OFF
    }
    return 0;  // CUBLAS_STATUS_SUCCESS
}

__attribute__((visibility("default")))
int cublasSetEmulationStrategy(void* handle, int strategy) {
    return 0;  // CUBLAS_STATUS_SUCCESS
}

// Override dlsym to intercept symbol lookups
void* dlsym(void* handle, const char* symbol) {
    // Get the real dlsym
    static void* (*real_dlsym)(void*, const char*) = NULL;
    if (!real_dlsym) {
        *(void**)(&real_dlsym) = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym) {
            *(void**)(&real_dlsym) = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.0");
        }
        if (!real_dlsym) {
            fprintf(stderr, "[CUBLAS Interposer] ERROR: Cannot find real dlsym\n");
            return NULL;
        }
    }

    // Handle our compatibility symbols
    if (strcmp(symbol, "cublasGetEmulationStrategy") == 0) {
        return (void*)cublasGetEmulationStrategy;
    }
    if (strcmp(symbol, "cublasSetEmulationStrategy") == 0) {
        return (void*)cublasSetEmulationStrategy;
    }

    // For CUBLAS symbols, try the real library first
    if (real_cublas_handle && strncmp(symbol, "cublas", 6) == 0) {
        void* sym = real_dlsym(real_cublas_handle, symbol);
        if (sym) {
            return sym;
        }
    }

    // Otherwise use the real dlsym
    return real_dlsym(handle, symbol);
}

// Export all CUBLAS symbols that cudarc might look for
// We generate forwarding stubs for the most common ones

#define FORWARD_CUBLAS_SYMBOL(name) \
    __attribute__((visibility("default"))) \
    void* name() { \
        static void* (*real_func)() = NULL; \
        if (!real_func && real_cublas_handle) { \
            real_func = dlsym(real_cublas_handle, #name); \
        } \
        if (real_func) { \
            return real_func(); \
        } \
        return NULL; \
    }

// Generate forwarding functions for common CUBLAS symbols
// Using a macro to create generic forwarding functions
#define CUBLAS_FORWARD(ret_type, name, args, call_args) \
    __attribute__((visibility("default"))) \
    ret_type name args { \
        typedef ret_type (*func_type) args; \
        static func_type real_func = NULL; \
        if (!real_func && real_cublas_handle) { \
            real_func = (func_type)dlsym(real_cublas_handle, #name); \
        } \
        if (real_func) { \
            return real_func call_args; \
        } \
        return (ret_type)1; /* Return error code */ \
    }

// Forward essential CUBLAS functions
typedef void* cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
typedef int cublasDataType_t;
typedef int cublasComputeType_t;

CUBLAS_FORWARD(cublasStatus_t, cublasCreate_v2, (cublasHandle_t* handle), (handle))
CUBLAS_FORWARD(cublasStatus_t, cublasDestroy_v2, (cublasHandle_t handle), (handle))
CUBLAS_FORWARD(cublasStatus_t, cublasSetStream_v2, (cublasHandle_t handle, void* stream), (handle, stream))
CUBLAS_FORWARD(cublasStatus_t, cublasGetVersion_v2, (cublasHandle_t handle, int* version), (handle, version))
CUBLAS_FORWARD(cublasStatus_t, cublasGetProperty, (int type, int* value), (type, value))

// Matrix operations
CUBLAS_FORWARD(cublasStatus_t, cublasSgemm_v2,
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const float* alpha,
     const float* A, int lda, const float* B, int ldb,
     const float* beta, float* C, int ldc),
    (handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc))

CUBLAS_FORWARD(cublasStatus_t, cublasDgemm_v2,
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const double* alpha,
     const double* A, int lda, const double* B, int ldb,
     const double* beta, double* C, int ldc),
    (handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc))

// Vector operations
CUBLAS_FORWARD(cublasStatus_t, cublasSdot_v2,
    (cublasHandle_t handle, int n, const float* x, int incx,
     const float* y, int incy, float* result),
    (handle, n, x, incx, y, incy, result))

CUBLAS_FORWARD(cublasStatus_t, cublasSnrm2_v2,
    (cublasHandle_t handle, int n, const float* x, int incx, float* result),
    (handle, n, x, incx, result))

// Extended precision functions (the ones causing problems)
CUBLAS_FORWARD(cublasStatus_t, cublasAsumEx,
    (cublasHandle_t handle, int n, const void* x, cublasDataType_t xType,
     int incx, void* result, cublasDataType_t resultType, cublasDataType_t executiontype),
    (handle, n, x, xType, incx, result, resultType, executiontype))

CUBLAS_FORWARD(cublasStatus_t, cublasAxpyEx,
    (cublasHandle_t handle, int n, const void* alpha, cublasDataType_t alphaType,
     const void* x, cublasDataType_t xType, int incx,
     void* y, cublasDataType_t yType, int incy, cublasDataType_t executiontype),
    (handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype))

CUBLAS_FORWARD(cublasStatus_t, cublasDotEx,
    (cublasHandle_t handle, int n,
     const void* x, cublasDataType_t xType, int incx,
     const void* y, cublasDataType_t yType, int incy,
     void* result, cublasDataType_t resultType, cublasDataType_t executionType),
    (handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType))

CUBLAS_FORWARD(cublasStatus_t, cublasNrm2Ex,
    (cublasHandle_t handle, int n, const void* x, cublasDataType_t xType,
     int incx, void* result, cublasDataType_t resultType, cublasDataType_t executionType),
    (handle, n, x, xType, incx, result, resultType, executionType))

CUBLAS_FORWARD(cublasStatus_t, cublasScalEx,
    (cublasHandle_t handle, int n, const void* alpha, cublasDataType_t alphaType,
     void* x, cublasDataType_t xType, int incx, cublasDataType_t executionType),
    (handle, n, alpha, alphaType, x, xType, incx, executionType))

// GEMM extended precision
CUBLAS_FORWARD(cublasStatus_t, cublasGemmEx,
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const void* alpha,
     const void* A, cublasDataType_t Atype, int lda,
     const void* B, cublasDataType_t Btype, int ldb,
     const void* beta, void* C, cublasDataType_t Ctype, int ldc,
     cublasComputeType_t computeType, int algo),
    (handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo))

// Math mode functions
CUBLAS_FORWARD(cublasStatus_t, cublasSetMathMode, (cublasHandle_t handle, int mode), (handle, mode))
CUBLAS_FORWARD(cublasStatus_t, cublasGetMathMode, (cublasHandle_t handle, int* mode), (handle, mode))

// Pointer mode
CUBLAS_FORWARD(cublasStatus_t, cublasSetPointerMode_v2, (cublasHandle_t handle, int mode), (handle, mode))
CUBLAS_FORWARD(cublasStatus_t, cublasGetPointerMode_v2, (cublasHandle_t handle, int* mode), (handle, mode))

// Atomics mode
CUBLAS_FORWARD(cublasStatus_t, cublasSetAtomicsMode, (cublasHandle_t handle, int mode), (handle, mode))
CUBLAS_FORWARD(cublasStatus_t, cublasGetAtomicsMode, (cublasHandle_t handle, int* mode), (handle, mode))