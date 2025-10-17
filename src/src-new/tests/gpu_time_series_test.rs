//! GPU Time Series Kernel Tests
//! Tests for the 5 new time series forecasting kernels

#[cfg(feature = "cuda")]
use prism_ai::gpu::kernel_executor::GpuKernelExecutor;

#[test]
#[cfg(feature = "cuda")]
fn test_ar_forecast_kernel() {
    // Test AR(2) forecasting
    let executor = GpuKernelExecutor::new(0).expect("Failed to create GPU executor");

    // Historical data: simple linear trend
    let historical = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // AR(2) coefficients: y_t = 0.5*y_{t-1} + 0.3*y_{t-2}
    let coefficients = vec![0.5, 0.3];

    let horizon = 3;

    let forecast = executor.ar_forecast(&historical, &coefficients, horizon)
        .expect("AR forecast failed");

    assert_eq!(forecast.len(), horizon);

    // Basic sanity check: forecast should be non-zero
    for &val in &forecast {
        assert!(val > 0.0, "Forecast value should be positive");
    }

    println!("✅ AR forecast test passed: {:?}", forecast);
}

#[test]
#[cfg(feature = "cuda")]
fn test_lstm_cell_kernel() {
    let executor = GpuKernelExecutor::new(0).expect("Failed to create GPU executor");

    let batch_size = 2;
    let input_dim = 4;
    let hidden_dim = 8;

    // Random input data
    let input = vec![0.1; batch_size * input_dim];
    let hidden_state = vec![0.0; batch_size * hidden_dim];
    let cell_state = vec![0.0; batch_size * hidden_dim];

    // Random weights (would be learned in practice)
    let weights_ih = vec![0.01; 4 * hidden_dim * input_dim];
    let weights_hh = vec![0.01; 4 * hidden_dim * hidden_dim];
    let bias = vec![0.0; 4 * hidden_dim];

    let (output_hidden, output_cell) = executor.lstm_cell_forward(
        &input,
        &hidden_state,
        &cell_state,
        &weights_ih,
        &weights_hh,
        &bias,
        batch_size,
        input_dim,
        hidden_dim,
    ).expect("LSTM cell forward failed");

    assert_eq!(output_hidden.len(), batch_size * hidden_dim);
    assert_eq!(output_cell.len(), batch_size * hidden_dim);

    // LSTM output should be bounded by tanh: [-1, 1]
    for &val in &output_hidden {
        assert!(val >= -1.0 && val <= 1.0, "LSTM hidden output should be in [-1, 1]");
    }

    println!("✅ LSTM cell test passed");
}

#[test]
#[cfg(feature = "cuda")]
fn test_gru_cell_kernel() {
    let executor = GpuKernelExecutor::new(0).expect("Failed to create GPU executor");

    let batch_size = 2;
    let input_dim = 4;
    let hidden_dim = 8;

    // Random input data
    let input = vec![0.1; batch_size * input_dim];
    let hidden_state = vec![0.0; batch_size * hidden_dim];

    // Random weights (would be learned in practice)
    let weights_ih = vec![0.01; 3 * hidden_dim * input_dim];
    let weights_hh = vec![0.01; 3 * hidden_dim * hidden_dim];
    let bias = vec![0.0; 3 * hidden_dim];

    let output_hidden = executor.gru_cell_forward(
        &input,
        &hidden_state,
        &weights_ih,
        &weights_hh,
        &bias,
        batch_size,
        input_dim,
        hidden_dim,
    ).expect("GRU cell forward failed");

    assert_eq!(output_hidden.len(), batch_size * hidden_dim);

    // GRU output should be bounded by tanh: [-1, 1]
    for &val in &output_hidden {
        assert!(val >= -1.0 && val <= 1.0, "GRU hidden output should be in [-1, 1]");
    }

    println!("✅ GRU cell test passed");
}

#[test]
#[cfg(feature = "cuda")]
fn test_kalman_filter_kernel() {
    let executor = GpuKernelExecutor::new(0).expect("Failed to create GPU executor");

    let state_dim = 4;

    // Simple state and covariance
    let state = vec![1.0, 2.0, 3.0, 4.0];
    let covariance = vec![0.1; state_dim * state_dim];
    let measurement = vec![1.1, 2.1, 3.1, 4.1];

    // Identity transition matrix (position holds steady)
    let mut transition_matrix = vec![0.0; state_dim * state_dim];
    for i in 0..state_dim {
        transition_matrix[i * state_dim + i] = 1.0;
    }

    // Identity measurement matrix
    let mut measurement_matrix = vec![0.0; state_dim * state_dim];
    for i in 0..state_dim {
        measurement_matrix[i * state_dim + i] = 1.0;
    }

    let process_noise = vec![0.01; state_dim * state_dim];
    let measurement_noise = vec![0.05; state_dim * state_dim];

    let (output_state, output_cov) = executor.kalman_filter_step(
        &state,
        &covariance,
        &measurement,
        &transition_matrix,
        &measurement_matrix,
        &process_noise,
        &measurement_noise,
        state_dim,
    ).expect("Kalman filter step failed");

    assert_eq!(output_state.len(), state_dim);
    assert_eq!(output_cov.len(), state_dim * state_dim);

    // State should be updated towards measurement
    for i in 0..state_dim {
        assert!(output_state[i] > state[i], "State should move towards measurement");
        assert!(output_state[i] < measurement[i], "State should not overshoot measurement");
    }

    println!("✅ Kalman filter test passed: {:?}", output_state);
}

#[test]
#[cfg(feature = "cuda")]
fn test_uncertainty_propagation_kernel() {
    let executor = GpuKernelExecutor::new(0).expect("Failed to create GPU executor");

    let horizon = 10;

    let forecast_mean = vec![1.0; horizon];
    let model_error_std = vec![0.1; horizon];

    let forecast_variance = executor.uncertainty_propagation(
        &forecast_mean,
        &model_error_std,
        horizon,
    ).expect("Uncertainty propagation failed");

    assert_eq!(forecast_variance.len(), horizon);

    // Variance should increase with forecast horizon
    for i in 1..horizon {
        assert!(
            forecast_variance[i] >= forecast_variance[i - 1],
            "Variance should grow with forecast horizon"
        );
    }

    println!("✅ Uncertainty propagation test passed: {:?}", forecast_variance);
}

#[test]
#[cfg(feature = "cuda")]
fn test_all_time_series_kernels_registered() {
    let executor = GpuKernelExecutor::new(0).expect("Failed to create GPU executor");

    // Verify all 5 kernels are registered
    assert!(executor.get_kernel("ar_forecast").is_ok(), "AR forecast kernel not registered");
    assert!(executor.get_kernel("lstm_cell").is_ok(), "LSTM cell kernel not registered");
    assert!(executor.get_kernel("gru_cell").is_ok(), "GRU cell kernel not registered");
    assert!(executor.get_kernel("kalman_filter_step").is_ok(), "Kalman filter kernel not registered");
    assert!(executor.get_kernel("uncertainty_propagation").is_ok(), "Uncertainty propagation kernel not registered");

    println!("✅ All 5 time series kernels registered successfully");
}
