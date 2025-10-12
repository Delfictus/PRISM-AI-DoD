// Test GPU library directly
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "=== DIRECT GPU LIBRARY TEST ===\n\n";

    // Load the library
    void* handle = dlopen("./src/libgpu_runtime.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "❌ Cannot load library: " << dlerror() << '\n';
        return 1;
    }
    std::cout << "✅ Library loaded successfully\n";

    // Get function pointers
    typedef int (*gpu_available_t)();
    typedef float (*launch_te_t)(double*, double*, int);
    typedef void (*launch_thermo_t)(double*, double*, int, int);

    gpu_available_t gpu_available = (gpu_available_t)dlsym(handle, "gpu_available");
    launch_te_t launch_te = (launch_te_t)dlsym(handle, "launch_transfer_entropy");
    launch_thermo_t launch_thermo = (launch_thermo_t)dlsym(handle, "launch_thermodynamic");

    if (!gpu_available || !launch_te || !launch_thermo) {
        std::cerr << "❌ Cannot find functions: " << dlerror() << '\n';
        dlclose(handle);
        return 1;
    }
    std::cout << "✅ Functions found\n";

    // Test GPU availability
    int gpu_ready = gpu_available();
    std::cout << "\nGPU available: " << (gpu_ready ? "YES" : "NO") << "\n";

    if (gpu_ready) {
        // Test Transfer Entropy
        std::cout << "\nTesting Transfer Entropy kernel...\n";
        std::vector<double> source(1000), target(1000);
        for (int i = 0; i < 1000; i++) {
            source[i] = sin(i * 0.01);
            target[i] = cos(i * 0.01);
        }

        float te_result = launch_te(source.data(), target.data(), 1000);
        std::cout << "✅ Transfer Entropy computed: " << te_result << "\n";

        // Test Thermodynamic evolution
        std::cout << "\nTesting Thermodynamic kernel...\n";
        std::vector<double> phases(100), velocities(100);
        for (int i = 0; i < 100; i++) {
            phases[i] = sin(i * 0.1);
            velocities[i] = 0.0;
        }

        launch_thermo(phases.data(), velocities.data(), 100, 100);
        std::cout << "✅ Thermodynamic evolution completed\n";
        std::cout << "  Phase[0] after: " << phases[0] << "\n";
        std::cout << "  Velocity[0] after: " << velocities[0] << "\n";

        std::cout << "\n🎉 GPU KERNELS EXECUTED SUCCESSFULLY!\n";
    } else {
        std::cout << "\n⚠️  GPU not detected by library\n";
    }

    dlclose(handle);
    return gpu_ready ? 0 : 1;
}