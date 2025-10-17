//! Neuromorphic CUDA Kernels
//!
//! Custom CUDA kernels for GPU-accelerated spiking neural network simulation
//! with Izhikevich neurons, spike propagation, and STDP learning.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Izhikevich neuron parameters structure
struct IzhikevichParams {
    float a;  // Time scale of recovery
    float b;  // Sensitivity of recovery
    float c;  // After-spike reset value
    float d;  // After-spike recovery increment
};

// ============================================================================
// KERNEL 1: Batched Izhikevich Neuron Update
// ============================================================================

/**
 * Update all neurons in parallel using Izhikevich model
 *
 * @param v Membrane potentials [n_neurons]
 * @param u Recovery variables [n_neurons]
 * @param I Input currents [n_neurons]
 * @param params Neuron parameters [n_neurons] (a,b,c,d packed)
 * @param spikes Output spike flags [n_neurons]
 * @param dt Time step (ms)
 * @param n_neurons Number of neurons
 */
__global__ void izhikevich_update_kernel(
    float* __restrict__ v,
    float* __restrict__ u,
    float* __restrict__ I,
    const float* __restrict__ params,
    int* __restrict__ spikes,
    float dt,
    int n_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_neurons) {
        // Load neuron state
        float v_curr = v[idx];
        float u_curr = u[idx];
        float I_curr = I[idx];

        // Load parameters (4 floats per neuron)
        int param_idx = idx * 4;
        float a = params[param_idx];
        float b = params[param_idx + 1];
        float c = params[param_idx + 2];
        float d = params[param_idx + 3];

        // Izhikevich equations
        // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        // du/dt = a*(b*v - u)
        float v_new = v_curr + dt * (0.04f * v_curr * v_curr + 5.0f * v_curr + 140.0f - u_curr + I_curr);
        float u_new = u_curr + dt * a * (b * v_curr - u_curr);

        // Check for spike (threshold at 30 mV)
        int spiked = 0;
        if (v_new >= 30.0f) {
            v_new = c;
            u_new = u_new + d;
            spiked = 1;
        }

        // Write results
        v[idx] = v_new;
        u[idx] = u_new;
        spikes[idx] = spiked;

        // Decay input current
        I[idx] = I_curr * 0.9f;
    }
}

// ============================================================================
// KERNEL 2: Sparse Spike Propagation (CSR Format)
// ============================================================================

/**
 * Propagate spikes through synapses using CSR sparse matrix format
 *
 * @param I_post Postsynaptic input currents [n_neurons]
 * @param spikes Presynaptic spike flags [n_neurons]
 * @param csr_row_ptr Row pointers for CSR format [n_neurons+1]
 * @param csr_col_idx Column indices (postsynaptic neurons) [n_synapses]
 * @param csr_weights Synaptic weights [n_synapses]
 * @param n_neurons Number of neurons
 * @param synaptic_strength Synaptic current multiplier
 */
__global__ void spike_propagation_kernel(
    float* __restrict__ I_post,
    const int* __restrict__ spikes,
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_idx,
    const float* __restrict__ csr_weights,
    int n_neurons,
    float synaptic_strength
) {
    int pre_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pre_idx < n_neurons && spikes[pre_idx] == 1) {
        // Get synapse range for this presynaptic neuron
        int syn_start = csr_row_ptr[pre_idx];
        int syn_end = csr_row_ptr[pre_idx + 1];

        // Propagate spike to all postsynaptic neurons
        for (int syn_idx = syn_start; syn_idx < syn_end; syn_idx++) {
            int post_idx = csr_col_idx[syn_idx];
            float weight = csr_weights[syn_idx];

            // Atomic add to avoid race conditions
            atomicAdd(&I_post[post_idx], weight * synaptic_strength);
        }
    }
}

// ============================================================================
// KERNEL 3: STDP Trace Update
// ============================================================================

/**
 * Update STDP traces with exponential decay and spike increments
 *
 * @param x_traces Presynaptic traces [n_neurons]
 * @param y_traces Postsynaptic traces [n_neurons]
 * @param spikes Spike flags [n_neurons]
 * @param dt Time step (ms)
 * @param tau Time constant (ms)
 * @param n_neurons Number of neurons
 */
__global__ void stdp_trace_update_kernel(
    float* __restrict__ x_traces,
    float* __restrict__ y_traces,
    const int* __restrict__ spikes,
    float dt,
    float tau,
    int n_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_neurons) {
        // Exponential decay: trace *= exp(-dt/tau)
        float decay = expf(-dt / tau);

        float x_curr = x_traces[idx] * decay;
        float y_curr = y_traces[idx] * decay;

        // Increment trace if neuron spiked
        if (spikes[idx] == 1) {
            x_curr += 1.0f;
            y_curr += 1.0f;
        }

        x_traces[idx] = x_curr;
        y_traces[idx] = y_curr;
    }
}

// ============================================================================
// KERNEL 4: STDP Weight Update
// ============================================================================

/**
 * Update synaptic weights using STDP learning rule
 *
 * @param weights Synaptic weights [n_synapses]
 * @param pre_indices Presynaptic neuron indices [n_synapses]
 * @param post_indices Postsynaptic neuron indices [n_synapses]
 * @param x_traces Presynaptic traces [n_neurons]
 * @param y_traces Postsynaptic traces [n_neurons]
 * @param spikes Spike flags [n_neurons]
 * @param A_plus LTP amplitude
 * @param A_minus LTD amplitude
 * @param w_min Minimum weight
 * @param w_max Maximum weight
 * @param n_synapses Number of synapses
 */
__global__ void stdp_weight_update_kernel(
    float* __restrict__ weights,
    const int* __restrict__ pre_indices,
    const int* __restrict__ post_indices,
    const float* __restrict__ x_traces,
    const float* __restrict__ y_traces,
    const int* __restrict__ spikes,
    float A_plus,
    float A_minus,
    float w_min,
    float w_max,
    int n_synapses
) {
    int syn_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (syn_idx < n_synapses) {
        int pre = pre_indices[syn_idx];
        int post = post_indices[syn_idx];

        float weight = weights[syn_idx];

        // LTP: Potentiation when postsynaptic neuron spikes
        if (spikes[post] == 1) {
            weight += A_plus * x_traces[pre];
        }

        // LTD: Depression when presynaptic neuron spikes
        if (spikes[pre] == 1) {
            weight -= A_minus * y_traces[post];
        }

        // Clamp weight to bounds
        weight = fminf(fmaxf(weight, w_min), w_max);

        weights[syn_idx] = weight;
    }
}

// ============================================================================
// KERNEL 5: Spike Count and Reduction
// ============================================================================

/**
 * Count total number of spikes using parallel reduction
 *
 * @param spikes Spike flags [n_neurons]
 * @param spike_count Output spike count [1]
 * @param n_neurons Number of neurons
 */
__global__ void spike_count_kernel(
    const int* __restrict__ spikes,
    int* __restrict__ spike_count,
    int n_neurons
) {
    __shared__ int shared_count[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load and accumulate
    int count = 0;
    if (idx < n_neurons) {
        count = spikes[idx];
    }
    shared_count[tid] = count;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_count[tid] += shared_count[tid + stride];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicAdd(spike_count, shared_count[0]);
    }
}

// ============================================================================
// KERNEL 6: Firing Rate Computation
// ============================================================================

/**
 * Compute instantaneous firing rates from spike counts
 *
 * @param rates Output firing rates (Hz) [n_neurons]
 * @param spike_counts Spike counts [n_neurons]
 * @param duration Simulation duration (ms)
 * @param n_neurons Number of neurons
 */
__global__ void firing_rate_kernel(
    float* __restrict__ rates,
    const int* __restrict__ spike_counts,
    float duration,
    int n_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_neurons) {
        // Convert to Hz: (spikes / duration_sec)
        rates[idx] = (spike_counts[idx] / duration) * 1000.0f;
    }
}

// ============================================================================
// Host Functions (C-compatible interface)
// ============================================================================

extern "C" {

// Launch Izhikevich update kernel
void launch_izhikevich_update(
    float* d_v,
    float* d_u,
    float* d_I,
    const float* d_params,
    int* d_spikes,
    float dt,
    int n_neurons,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n_neurons + block_size - 1) / block_size;

    izhikevich_update_kernel<<<grid_size, block_size, 0, stream>>>(
        d_v, d_u, d_I, d_params, d_spikes, dt, n_neurons
    );
}

// Launch spike propagation kernel
void launch_spike_propagation(
    float* d_I_post,
    const int* d_spikes,
    const int* d_csr_row_ptr,
    const int* d_csr_col_idx,
    const float* d_csr_weights,
    int n_neurons,
    float synaptic_strength,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n_neurons + block_size - 1) / block_size;

    spike_propagation_kernel<<<grid_size, block_size, 0, stream>>>(
        d_I_post, d_spikes, d_csr_row_ptr, d_csr_col_idx, d_csr_weights,
        n_neurons, synaptic_strength
    );
}

// Launch STDP trace update kernel
void launch_stdp_trace_update(
    float* d_x_traces,
    float* d_y_traces,
    const int* d_spikes,
    float dt,
    float tau,
    int n_neurons,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n_neurons + block_size - 1) / block_size;

    stdp_trace_update_kernel<<<grid_size, block_size, 0, stream>>>(
        d_x_traces, d_y_traces, d_spikes, dt, tau, n_neurons
    );
}

// Launch STDP weight update kernel
void launch_stdp_weight_update(
    float* d_weights,
    const int* d_pre_indices,
    const int* d_post_indices,
    const float* d_x_traces,
    const float* d_y_traces,
    const int* d_spikes,
    float A_plus,
    float A_minus,
    float w_min,
    float w_max,
    int n_synapses,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n_synapses + block_size - 1) / block_size;

    stdp_weight_update_kernel<<<grid_size, block_size, 0, stream>>>(
        d_weights, d_pre_indices, d_post_indices, d_x_traces, d_y_traces,
        d_spikes, A_plus, A_minus, w_min, w_max, n_synapses
    );
}

// Launch spike count kernel
void launch_spike_count(
    const int* d_spikes,
    int* d_spike_count,
    int n_neurons,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n_neurons + block_size - 1) / block_size;

    spike_count_kernel<<<grid_size, block_size, 0, stream>>>(
        d_spikes, d_spike_count, n_neurons
    );
}

// Launch firing rate kernel
void launch_firing_rate(
    float* d_rates,
    const int* d_spike_counts,
    float duration,
    int n_neurons,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (n_neurons + block_size - 1) / block_size;

    firing_rate_kernel<<<grid_size, block_size, 0, stream>>>(
        d_rates, d_spike_counts, duration, n_neurons
    );
}

} // extern "C"
