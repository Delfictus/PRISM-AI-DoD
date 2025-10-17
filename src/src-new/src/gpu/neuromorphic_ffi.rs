//! FFI bindings for neuromorphic CUDA kernels

use std::os::raw::c_void;

// CUDA stream type
pub type CudaStream = *mut c_void;

extern "C" {
    /// Launch Izhikevich neuron update kernel
    pub fn launch_izhikevich_update(
        d_v: *mut f32,
        d_u: *mut f32,
        d_I: *mut f32,
        d_params: *const f32,
        d_spikes: *mut i32,
        dt: f32,
        n_neurons: i32,
        stream: CudaStream,
    );

    /// Launch spike propagation kernel
    pub fn launch_spike_propagation(
        d_I_post: *mut f32,
        d_spikes: *const i32,
        d_csr_row_ptr: *const i32,
        d_csr_col_idx: *const i32,
        d_csr_weights: *const f32,
        n_neurons: i32,
        synaptic_strength: f32,
        stream: CudaStream,
    );

    /// Launch STDP trace update kernel
    pub fn launch_stdp_trace_update(
        d_x_traces: *mut f32,
        d_y_traces: *mut f32,
        d_spikes: *const i32,
        dt: f32,
        tau: f32,
        n_neurons: i32,
        stream: CudaStream,
    );

    /// Launch STDP weight update kernel
    pub fn launch_stdp_weight_update(
        d_weights: *mut f32,
        d_pre_indices: *const i32,
        d_post_indices: *const i32,
        d_x_traces: *const f32,
        d_y_traces: *const f32,
        d_spikes: *const i32,
        A_plus: f32,
        A_minus: f32,
        w_min: f32,
        w_max: f32,
        n_synapses: i32,
        stream: CudaStream,
    );

    /// Launch spike count kernel
    pub fn launch_spike_count(
        d_spikes: *const i32,
        d_spike_count: *mut i32,
        n_neurons: i32,
        stream: CudaStream,
    );

    /// Launch firing rate kernel
    pub fn launch_firing_rate(
        d_rates: *mut f32,
        d_spike_counts: *const i32,
        duration: f32,
        n_neurons: i32,
        stream: CudaStream,
    );
}
