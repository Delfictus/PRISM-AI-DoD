//! GPU Kernel Executor with actual kernel execution capabilities
//!
//! This module provides the infrastructure to compile, load and execute
//! actual GPU kernels using the correct cudarc API.

use anyhow::{Result, Context as AnyhowContext};
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, CudaModule, CudaFunction},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::collections::HashMap;
use std::sync::Arc;

/// Common GPU kernels used across the system
pub mod kernels {
    pub const VECTOR_ADD: &str = r#"
    extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    "#;

    pub const MATRIX_MUL: &str = r#"
    extern "C" __global__ void matmul(float* a, float* b, float* c, int m, int k, int n) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < n) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    "#;

    pub const RELU: &str = r#"
    extern "C" __global__ void relu(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = fmaxf(0.0f, data[idx]);
        }
    }
    "#;

    pub const SOFTMAX: &str = r#"
    extern "C" __global__ void softmax(float* data, int batch_size, int num_classes) {
        int batch_idx = blockIdx.x;
        if (batch_idx >= batch_size) return;

        float* row = data + batch_idx * num_classes;

        // Find max for numerical stability
        float max_val = row[0];
        for (int i = 1; i < num_classes; i++) {
            max_val = fmaxf(max_val, row[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            row[i] = expf(row[i] - max_val);
            sum += row[i];
        }

        // Normalize
        for (int i = 0; i < num_classes; i++) {
            row[i] /= sum;
        }
    }
    "#;

    pub const SIGMOID: &str = r#"
    extern "C" __global__ void sigmoid(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = 1.0f / (1.0f + expf(-data[idx]));
        }
    }
    "#;

    pub const TANH: &str = r#"
    extern "C" __global__ void tanh_activation(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = tanhf(data[idx]);
        }
    }
    "#;

    pub const BATCH_NORM: &str = r#"
    extern "C" __global__ void batch_norm(
        float* data, float* gamma, float* beta,
        float* mean, float* var,
        int batch_size, int features, float epsilon
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = batch_size * features;

        if (idx < total_elements) {
            int feature_idx = idx % features;
            float normalized = (data[idx] - mean[feature_idx]) /
                              sqrtf(var[feature_idx] + epsilon);
            data[idx] = gamma[feature_idx] * normalized + beta[feature_idx];
        }
    }
    "#;

    // Active Inference Kernels
    pub const KL_DIVERGENCE: &str = r#"
    extern "C" __global__ void kl_divergence(
        float* q, float* p, float* kl_out, int n
    ) {
        int idx = threadIdx.x;

        float local_kl = 0.0f;
        if (idx < n) {
            float q_val = q[idx];
            float p_val = p[idx];
            if (q_val > 1e-10f && p_val > 1e-10f) {
                local_kl = q_val * logf(q_val / p_val);
            }
        }

        // Simple reduction for small arrays (< 256 elements)
        __shared__ float sdata[256];
        sdata[idx] = local_kl;
        __syncthreads();

        // Reduction
        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < n) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        // Write result
        if (idx == 0) {
            kl_out[0] = sdata[0];
        }
    }
    "#;

    pub const ELEMENTWISE_MULTIPLY: &str = r#"
    extern "C" __global__ void elementwise_multiply(
        float* a, float* b, float* c, int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] * b[idx];
        }
    }
    "#;

    pub const NORMALIZE: &str = r#"
    extern "C" __global__ void normalize(float* data, int n) {
        int idx = threadIdx.x;

        // Compute sum using shared memory reduction
        __shared__ float sdata[256];
        sdata[idx] = (idx < n) ? data[idx] : 0.0f;
        __syncthreads();

        // Reduction
        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        float sum = sdata[0];
        __syncthreads();

        // Normalize
        if (idx < n && sum > 0.0f) {
            data[idx] /= sum;
        }
    }
    "#;

    // Neuromorphic Computing Kernels
    pub const LEAKY_INTEGRATE_FIRE: &str = r#"
    extern "C" __global__ void leaky_integrate_fire(
        float* state_current, float* state_previous,
        float* input, float leak_rate, float threshold,
        bool* spikes, int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // Leaky integration
            float new_state = (1.0f - leak_rate) * state_previous[idx] + input[idx];

            // Apply tanh nonlinearity
            new_state = tanhf(new_state);

            // Spike generation
            spikes[idx] = new_state > threshold;

            // Reset if spiked
            if (spikes[idx]) {
                new_state = 0.0f;
            }

            state_current[idx] = new_state;
        }
    }
    "#;

    pub const RESERVOIR_UPDATE: &str = r#"
    extern "C" __global__ void reservoir_update(
        float* state, float* prev_state, float* input,
        float leak_rate, int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // Leaky integration: x(t) = (1-α)x(t-1) + u(t)
            float integrated = (1.0f - leak_rate) * prev_state[idx] + input[idx];
            // Apply tanh nonlinearity
            state[idx] = tanhf(integrated);
        }
    }
    "#;

    pub const STDP_UPDATE: &str = r#"
    extern "C" __global__ void stdp_update(
        float* weights, bool* pre_spikes, bool* post_spikes,
        float* spike_times_pre, float* spike_times_post,
        float learning_rate, float tau_plus, float tau_minus,
        int n_pre, int n_post
    ) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;  // Post neuron
        int j = blockIdx.x * blockDim.x + threadIdx.x;  // Pre neuron

        if (i < n_post && j < n_pre) {
            if (pre_spikes[j] && post_spikes[i]) {
                float dt = spike_times_post[i] - spike_times_pre[j];
                float dw = 0.0f;

                if (dt > 0.0f) {
                    // LTP: post after pre
                    dw = learning_rate * expf(-dt / tau_plus);
                } else if (dt < 0.0f) {
                    // LTD: pre after post
                    dw = -learning_rate * expf(dt / tau_minus);
                }

                int idx = i * n_pre + j;
                weights[idx] += dw;
                // Clamp weights to [-1, 1]
                weights[idx] = fmaxf(-1.0f, fminf(1.0f, weights[idx]));
            }
        }
    }
    "#;

    // Statistical Mechanics / Thermodynamic Kernels
    pub const KURAMOTO_EVOLUTION: &str = r#"
    extern "C" __global__ void kuramoto_evolution(
        float* phases, float* frequencies,
        float* coupling_matrix, float* new_phases,
        int n, float dt, float coupling_strength
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            // Kuramoto model: dθ/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)
            float omega = frequencies[i];
            float coupling_sum = 0.0f;

            for (int j = 0; j < n; j++) {
                if (i != j) {
                    float phase_diff = phases[j] - phases[i];
                    float coupling = coupling_matrix[i * n + j];
                    coupling_sum += coupling * sinf(phase_diff);
                }
            }

            float dphi = omega + (coupling_strength / (float)n) * coupling_sum;
            new_phases[i] = phases[i] + dphi * dt;

            // Wrap to [0, 2π]
            while (new_phases[i] > 6.28318531f) new_phases[i] -= 6.28318531f;
            while (new_phases[i] < 0.0f) new_phases[i] += 6.28318531f;
        }
    }
    "#;

    pub const ENTROPY_PRODUCTION: &str = r#"
    extern "C" __global__ void entropy_production(
        float* velocities, float* entropy_rate,
        float temperature, int n
    ) {
        int idx = threadIdx.x;

        float local_entropy = 0.0f;
        if (idx < n) {
            // Entropy production from velocity dissipation
            float v = velocities[idx];
            local_entropy = v * v / (2.0f * temperature);
        }

        // Reduction
        __shared__ float sdata[256];
        sdata[idx] = local_entropy;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        if (idx == 0) {
            entropy_rate[0] = sdata[0];
        }
    }
    "#;

    pub const ORDER_PARAMETER: &str = r#"
    extern "C" __global__ void order_parameter(
        float* phases, float* order_real, float* order_imag, int n
    ) {
        int idx = threadIdx.x;

        float local_real = 0.0f;
        float local_imag = 0.0f;

        if (idx < n) {
            local_real = cosf(phases[idx]);
            local_imag = sinf(phases[idx]);
        }

        // Reduction for real part
        __shared__ float sdata_real[256];
        __shared__ float sdata_imag[256];

        sdata_real[idx] = local_real;
        sdata_imag[idx] = local_imag;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata_real[idx] += sdata_real[idx + s];
                sdata_imag[idx] += sdata_imag[idx + s];
            }
            __syncthreads();
        }

        if (idx == 0) {
            order_real[0] = sdata_real[0] / (float)n;
            order_imag[0] = sdata_imag[0] / (float)n;
        }
    }
    "#;

    // Transfer Entropy / Information Theory Kernels
    pub const MUTUAL_INFORMATION: &str = r#"
    extern "C" __global__ void mutual_information(
        float* joint_hist, float* marginal_x, float* marginal_y,
        float* mi_out, int n_bins
    ) {
        int idx = threadIdx.x;

        float local_mi = 0.0f;
        if (idx < n_bins * n_bins) {
            int i = idx / n_bins;
            int j = idx % n_bins;

            float p_xy = joint_hist[idx];
            float p_x = marginal_x[i];
            float p_y = marginal_y[j];

            if (p_xy > 1e-10f && p_x > 1e-10f && p_y > 1e-10f) {
                local_mi = p_xy * logf(p_xy / (p_x * p_y));
            }
        }

        // Reduction
        __shared__ float sdata[256];
        sdata[idx] = local_mi;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        if (idx == 0) {
            mi_out[0] = sdata[0];
        }
    }
    "#;

    pub const HISTOGRAM_2D: &str = r#"
    extern "C" __global__ void histogram_2d(
        float* x, float* y, int* hist,
        float min_val, float max_val,
        int n_samples, int n_bins
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n_samples) {
            float x_val = x[idx];
            float y_val = y[idx];

            // Bin calculation
            int bin_x = (int)((x_val - min_val) / (max_val - min_val) * (float)n_bins);
            int bin_y = (int)((y_val - min_val) / (max_val - min_val) * (float)n_bins);

            // Clamp to valid range
            bin_x = max(0, min(n_bins - 1, bin_x));
            bin_y = max(0, min(n_bins - 1, bin_y));

            // Atomic increment
            int hist_idx = bin_y * n_bins + bin_x;
            atomicAdd(&hist[hist_idx], 1);
        }
    }
    "#;

    pub const TIME_DELAYED_EMBEDDING: &str = r#"
    extern "C" __global__ void time_delayed_embedding(
        float* time_series, float* embedded,
        int n_samples, int embedding_dim, int tau
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int n_embedded = n_samples - (embedding_dim - 1) * tau;

        if (idx < n_embedded) {
            for (int d = 0; d < embedding_dim; d++) {
                int ts_idx = idx + d * tau;
                int emb_idx = idx * embedding_dim + d;
                embedded[emb_idx] = time_series[ts_idx];
            }
        }
    }
    "#;

    pub const CONDITIONAL_ENTROPY: &str = r#"
    extern "C" __global__ void conditional_entropy(
        float* joint_xyz, float* joint_xz,
        float* ce_out, int n_bins_xyz, int n_bins_xz
    ) {
        int idx = threadIdx.x;

        float local_ce = 0.0f;
        if (idx < n_bins_xyz) {
            float p_xyz = joint_xyz[idx];
            // Map to corresponding xz index (marginalize over y)
            int xz_idx = idx % n_bins_xz;
            float p_xz = joint_xz[xz_idx];

            if (p_xyz > 1e-10f && p_xz > 1e-10f) {
                local_ce = p_xyz * logf(p_xyz / p_xz);
            }
        }

        // Reduction
        __shared__ float sdata[256];
        sdata[idx] = local_ce;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        if (idx == 0) {
            ce_out[0] = -sdata[0];
        }
    }
    "#;

    pub const FREE_ENERGY: &str = r#"
    extern "C" __global__ void free_energy_kernel(
        float* posterior, float* prior,
        float log_likelihood, float* fe_out, int n
    ) {
        int idx = threadIdx.x;

        float local_kl = 0.0f;
        if (idx < n) {
            float q = posterior[idx];
            float p = prior[idx];
            if (q > 1e-10f && p > 1e-10f) {
                local_kl = q * logf(q / p);
            }
        }

        // Simple reduction for small arrays
        __shared__ float sdata[256];
        sdata[idx] = local_kl;
        __syncthreads();

        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (idx < s && (idx + s) < 256) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }

        // Compute free energy = KL - log_likelihood
        if (idx == 0) {
            fe_out[0] = sdata[0] - log_likelihood;
        }
    }
    "#;
}

/// GPU Kernel Executor that manages kernel compilation and execution
pub struct GpuKernelExecutor {
    context: Arc<CudaContext>,
    modules: HashMap<String, Arc<CudaModule>>,
    kernels: HashMap<String, Arc<CudaFunction>>,
}

impl GpuKernelExecutor {
    /// Create a new kernel executor
    pub fn new(device_id: usize) -> Result<Self> {
        let context = CudaContext::new(device_id)
            .context("Failed to create CUDA context")?;

        println!("✅ GPU Kernel Executor initialized on device {}", device_id);

        Ok(Self {
            context, // Already Arc<CudaContext>
            modules: HashMap::new(),
            kernels: HashMap::new(),
        })
    }

    /// Compile and register a kernel
    pub fn register_kernel(&mut self, name: &str, code: &str) -> Result<()> {
        // Check if already registered
        if self.kernels.contains_key(name) {
            return Ok(());
        }

        println!("  Compiling kernel: {}", name);

        // Compile PTX
        let ptx = compile_ptx_with_opts(code, CompileOptions::default())
            .with_context(|| format!("Failed to compile kernel: {}", name))?;

        // Load module - already returns Arc<CudaModule>
        let module = self.context.load_module(ptx)
            .with_context(|| format!("Failed to load PTX module for: {}", name))?;

        // Get function
        let func = module.load_function(name)
            .with_context(|| format!("Failed to load function: {}", name))?;

        // Store (module is already Arc wrapped)
        self.modules.insert(name.to_string(), module);
        self.kernels.insert(name.to_string(), Arc::new(func));

        println!("    ✅ Kernel '{}' registered", name);
        Ok(())
    }

    /// Register all standard kernels
    pub fn register_standard_kernels(&mut self) -> Result<()> {
        println!("Registering standard GPU kernels...");

        self.register_kernel("vector_add", kernels::VECTOR_ADD)?;
        self.register_kernel("matmul", kernels::MATRIX_MUL)?;
        self.register_kernel("relu", kernels::RELU)?;
        self.register_kernel("softmax", kernels::SOFTMAX)?;
        self.register_kernel("sigmoid", kernels::SIGMOID)?;
        self.register_kernel("tanh_activation", kernels::TANH)?;
        self.register_kernel("batch_norm", kernels::BATCH_NORM)?;

        // Active Inference kernels
        self.register_kernel("kl_divergence", kernels::KL_DIVERGENCE)?;
        self.register_kernel("elementwise_multiply", kernels::ELEMENTWISE_MULTIPLY)?;
        self.register_kernel("normalize", kernels::NORMALIZE)?;
        self.register_kernel("free_energy_kernel", kernels::FREE_ENERGY)?;

        // Neuromorphic kernels
        self.register_kernel("leaky_integrate_fire", kernels::LEAKY_INTEGRATE_FIRE)?;
        self.register_kernel("reservoir_update", kernels::RESERVOIR_UPDATE)?;
        self.register_kernel("stdp_update", kernels::STDP_UPDATE)?;

        // Statistical Mechanics kernels
        self.register_kernel("kuramoto_evolution", kernels::KURAMOTO_EVOLUTION)?;
        self.register_kernel("entropy_production", kernels::ENTROPY_PRODUCTION)?;
        self.register_kernel("order_parameter", kernels::ORDER_PARAMETER)?;

        // Transfer Entropy / Information Theory kernels
        self.register_kernel("mutual_information", kernels::MUTUAL_INFORMATION)?;
        self.register_kernel("histogram_2d", kernels::HISTOGRAM_2D)?;
        self.register_kernel("time_delayed_embedding", kernels::TIME_DELAYED_EMBEDDING)?;
        self.register_kernel("conditional_entropy", kernels::CONDITIONAL_ENTROPY)?;

        println!("✅ All standard kernels registered (24 total)");
        Ok(())
    }

    /// Get a kernel function
    pub fn get_kernel(&self, name: &str) -> Result<&Arc<CudaFunction>> {
        self.kernels.get(name)
            .ok_or_else(|| anyhow::anyhow!("Kernel '{}' not found", name))
    }

    /// Get the CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Execute vector addition
    pub fn vector_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let n = a.len();
        anyhow::ensure!(b.len() == n, "Vector dimensions must match");

        let stream = self.context.default_stream();
        let kernel = self.get_kernel("vector_add")?;

        // Upload data
        let a_dev = stream.memcpy_stod(a)?;
        let b_dev = stream.memcpy_stod(b)?;
        let mut c_dev = stream.alloc_zeros::<f32>(n)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(kernel)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&c_dev)?;
        Ok(result)
    }

    /// Execute matrix multiplication
    pub fn matrix_multiply(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(a.len() == m * k, "Matrix A dimensions incorrect");
        anyhow::ensure!(b.len() == k * n, "Matrix B dimensions incorrect");

        let stream = self.context.default_stream();
        let kernel = self.get_kernel("matmul")?;

        // Upload data
        let a_dev = stream.memcpy_stod(a)?;
        let b_dev = stream.memcpy_stod(b)?;
        let mut c_dev = stream.alloc_zeros::<f32>(m * n)?;

        // Launch with 2D grid
        let block_size = 16;
        let grid_x = (n as u32 + block_size - 1) / block_size;
        let grid_y = (m as u32 + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_size, block_size, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(kernel)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&(m as i32))
                .arg(&(k as i32))
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&c_dev)?;
        Ok(result)
    }

    /// Apply ReLU activation in-place
    pub fn relu_inplace(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        let stream = self.context.default_stream();
        let kernel = self.get_kernel("relu")?;

        // Upload data
        let mut data_dev = stream.memcpy_stod(data)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(kernel)
                .arg(&mut data_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Apply softmax activation
    pub fn softmax(&self, data: &mut [f32], batch_size: usize, num_classes: usize) -> Result<()> {
        anyhow::ensure!(
            data.len() == batch_size * num_classes,
            "Data dimensions must match batch_size * num_classes"
        );

        let stream = self.context.default_stream();
        let kernel = self.get_kernel("softmax")?;

        // Upload data
        let mut data_dev = stream.memcpy_stod(data)?;

        // Launch with one block per batch
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(kernel)
                .arg(&mut data_dev)
                .arg(&(batch_size as i32))
                .arg(&(num_classes as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Apply sigmoid activation
    pub fn sigmoid_inplace(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        let stream = self.context.default_stream();
        let kernel = self.get_kernel("sigmoid")?;

        // Upload data
        let mut data_dev = stream.memcpy_stod(data)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(kernel)
                .arg(&mut data_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Apply tanh activation
    pub fn tanh_inplace(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        let stream = self.context.default_stream();
        let kernel = self.get_kernel("tanh_activation")?;

        // Upload data
        let mut data_dev = stream.memcpy_stod(data)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(kernel)
                .arg(&mut data_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Compute KL divergence on GPU
    pub fn kl_divergence(&self, q: &[f32], p: &[f32]) -> Result<f32> {
        let n = q.len();
        anyhow::ensure!(p.len() == n, "Q and P must have same length");
        anyhow::ensure!(n <= 256, "KL divergence kernel supports max 256 elements");

        let stream = self.context.default_stream();
        let kernel = self.get_kernel("kl_divergence")?;

        // Upload data
        let q_dev = stream.memcpy_stod(q)?;
        let p_dev = stream.memcpy_stod(p)?;
        let mut kl_dev = stream.alloc_zeros::<f32>(1)?;

        // Launch with single block for reduction
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(kernel)
                .arg(&q_dev)
                .arg(&p_dev)
                .arg(&mut kl_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&kl_dev)?;
        Ok(result[0])
    }

    /// Element-wise multiplication on GPU
    pub fn elementwise_multiply(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let n = a.len();
        anyhow::ensure!(b.len() == n, "Vectors must have same length");

        let stream = self.context.default_stream();
        let kernel = self.get_kernel("elementwise_multiply")?;

        // Upload data
        let a_dev = stream.memcpy_stod(a)?;
        let b_dev = stream.memcpy_stod(b)?;
        let mut c_dev = stream.alloc_zeros::<f32>(n)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(kernel)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&c_dev)?;
        Ok(result)
    }

    /// Normalize vector to sum to 1.0 on GPU
    pub fn normalize_inplace(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        let stream = self.context.default_stream();
        let kernel = self.get_kernel("normalize")?;

        // Upload data
        let mut data_dev = stream.memcpy_stod(data)?;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(kernel)
                .arg(&mut data_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Compute free energy on GPU
    pub fn compute_free_energy(&self, posterior: &[f32], prior: &[f32], log_likelihood: f32) -> Result<f32> {
        let n = posterior.len();
        anyhow::ensure!(prior.len() == n, "Posterior and prior must have same length");
        anyhow::ensure!(n <= 256, "Free energy kernel supports max 256 elements");

        let stream = self.context.default_stream();
        let kernel = self.get_kernel("free_energy_kernel")?;

        // Upload data
        let posterior_dev = stream.memcpy_stod(posterior)?;
        let prior_dev = stream.memcpy_stod(prior)?;
        let mut fe_dev = stream.alloc_zeros::<f32>(1)?;

        // Launch with single block for reduction
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(kernel)
                .arg(&posterior_dev)
                .arg(&prior_dev)
                .arg(&log_likelihood)
                .arg(&mut fe_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&fe_dev)?;
        Ok(result[0])
    }

    /// Reservoir state update with leaky integration on GPU
    pub fn reservoir_update(&self, state: &mut [f32], prev_state: &[f32], input: &[f32], leak_rate: f32) -> Result<()> {
        let n = state.len();
        anyhow::ensure!(prev_state.len() == n && input.len() == n, "All arrays must have same length");

        let stream = self.context.default_stream();
        let kernel = self.get_kernel("reservoir_update")?;

        // Upload data
        let prev_state_dev = stream.memcpy_stod(prev_state)?;
        let input_dev = stream.memcpy_stod(input)?;
        let mut state_dev = stream.alloc_zeros::<f32>(n)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(kernel)
                .arg(&mut state_dev)
                .arg(&prev_state_dev)
                .arg(&input_dev)
                .arg(&leak_rate)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&state_dev)?;
        state.copy_from_slice(&result);
        Ok(())
    }
}

/// Global kernel executor instance (lazy initialized)
pub fn get_global_executor() -> Result<&'static std::sync::Mutex<GpuKernelExecutor>> {
    use std::sync::{Mutex, OnceLock};

    static EXECUTOR: OnceLock<Mutex<GpuKernelExecutor>> = OnceLock::new();

    let executor = EXECUTOR.get_or_init(|| {
        let mut exec = GpuKernelExecutor::new(0)
            .expect("Failed to create GPU kernel executor");
        exec.register_standard_kernels()
            .expect("Failed to register standard kernels");
        Mutex::new(exec)
    });

    Ok(executor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_executor() -> Result<()> {
        let mut executor = GpuKernelExecutor::new(0)?;
        executor.register_standard_kernels()?;

        // Test vector addition
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = executor.vector_add(&a, &b)?;

        assert_eq!(c.len(), 4);
        assert!((c[0] - 6.0).abs() < 1e-6);
        assert!((c[3] - 12.0).abs() < 1e-6);

        // Test ReLU
        let mut data = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
        executor.relu_inplace(&mut data)?;

        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_matrix_multiply() -> Result<()> {
        let mut executor = GpuKernelExecutor::new(0)?;
        executor.register_kernel("matmul", kernels::MATRIX_MUL)?;

        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let c = executor.matrix_multiply(&a, &b, 2, 3, 2)?;

        // Expected:
        // [1,2,3] * [[7,8],[9,10],[11,12]] = [58, 64]
        // [4,5,6] * [[7,8],[9,10],[11,12]] = [139, 154]

        assert_eq!(c.len(), 4);
        assert!((c[0] - 58.0).abs() < 1e-5);
        assert!((c[1] - 64.0).abs() < 1e-5);
        assert!((c[2] - 139.0).abs() < 1e-5);
        assert!((c[3] - 154.0).abs() < 1e-5);

        Ok(())
    }
}