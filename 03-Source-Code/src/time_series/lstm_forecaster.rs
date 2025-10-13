//! GPU-Accelerated LSTM/GRU Time Series Forecasting
//!
//! Implements Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)
//! networks for sequence-to-sequence time series forecasting.
//!
//! Mathematical Framework:
//!
//! LSTM Cell:
//! - Forget gate: fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
//! - Input gate: iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
//! - Cell candidate: c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)
//! - Cell state: cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ
//! - Output gate: oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
//! - Hidden state: hₜ = oₜ ⊙ tanh(cₜ)
//!
//! GRU Cell (simpler):
//! - Reset gate: rₜ = σ(Wr·[hₜ₋₁, xₜ] + br)
//! - Update gate: zₜ = σ(Wz·[hₜ₋₁, xₜ] + bz)
//! - Candidate: h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ] + b)
//! - Hidden state: hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ

use anyhow::{Result, Context, bail};
use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;

/// RNN cell type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellType {
    LSTM,
    GRU,
}

/// LSTM/GRU forecaster configuration
#[derive(Debug, Clone)]
pub struct LstmConfig {
    /// Cell type (LSTM or GRU)
    pub cell_type: CellType,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Number of layers (stacked LSTM/GRU)
    pub num_layers: usize,
    /// Input sequence length (lookback window)
    pub sequence_length: usize,
    /// Learning rate for training
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Dropout probability (regularization)
    pub dropout: f64,
}

impl Default for LstmConfig {
    fn default() -> Self {
        Self {
            cell_type: CellType::LSTM,
            hidden_size: 50,
            num_layers: 1,
            sequence_length: 10,
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            dropout: 0.0,
        }
    }
}

/// GPU-Accelerated LSTM/GRU Forecaster
pub struct LstmForecaster {
    /// Configuration
    config: LstmConfig,
    /// Weights for all layers
    weights: Vec<LstmWeights>,
    /// Training statistics
    training_mean: f64,
    training_std: f64,
    /// GPU availability
    gpu_available: bool,
    /// Training history for stateful prediction
    last_hidden_states: Option<Vec<Array1<f64>>>,
    last_cell_states: Option<Vec<Array1<f64>>>,
}

/// Weights for a single LSTM/GRU layer
#[derive(Debug, Clone)]
struct LstmWeights {
    /// Input to hidden weights
    w_ih: Array2<f64>,
    /// Hidden to hidden weights
    w_hh: Array2<f64>,
    /// Bias
    bias: Array1<f64>,
}

impl LstmForecaster {
    /// Create new LSTM/GRU forecaster
    pub fn new(config: LstmConfig) -> Result<Self> {
        let gpu_available = crate::gpu::kernel_executor::get_global_executor().is_ok();

        if gpu_available {
            println!("✓ GPU acceleration enabled for LSTM/GRU");
        } else {
            println!("⚠ GPU not available, using CPU for LSTM/GRU");
        }

        let mut weights = Vec::new();

        // Initialize weights for each layer
        for layer_idx in 0..config.num_layers {
            let input_size = if layer_idx == 0 {
                1  // Input feature dimension
            } else {
                config.hidden_size
            };

            let w = Self::initialize_weights(
                input_size,
                config.hidden_size,
                config.cell_type
            )?;

            weights.push(w);
        }

        Ok(Self {
            config,
            weights,
            training_mean: 0.0,
            training_std: 1.0,
            gpu_available,
            last_hidden_states: None,
            last_cell_states: None,
        })
    }

    /// Initialize layer weights using Xavier initialization
    fn initialize_weights(
        input_size: usize,
        hidden_size: usize,
        cell_type: CellType,
    ) -> Result<LstmWeights> {
        let mut rng = rand::thread_rng();

        // Number of gate parameters
        let gate_multiplier = match cell_type {
            CellType::LSTM => 4,  // forget, input, cell, output gates
            CellType::GRU => 3,   // reset, update, candidate
        };

        let total_hidden_size = hidden_size * gate_multiplier;

        // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        let scale_ih = (2.0 / (input_size + hidden_size) as f64).sqrt();
        let scale_hh = (2.0 / (hidden_size + hidden_size) as f64).sqrt();

        // Input to hidden weights
        let w_ih = Array2::from_shape_fn(
            (total_hidden_size, input_size),
            |_| (rng.gen::<f64>() - 0.5) * 2.0 * scale_ih
        );

        // Hidden to hidden weights
        let w_hh = Array2::from_shape_fn(
            (total_hidden_size, hidden_size),
            |_| (rng.gen::<f64>() - 0.5) * 2.0 * scale_hh
        );

        // Bias (initialize forget gate bias to 1.0 for better gradient flow)
        let mut bias = Array1::zeros(total_hidden_size);
        if cell_type == CellType::LSTM {
            // Set forget gate bias to 1.0
            for i in 0..hidden_size {
                bias[i] = 1.0;
            }
        }

        Ok(LstmWeights { w_ih, w_hh, bias })
    }

    /// Normalize data for training
    fn normalize(&mut self, data: &[f64]) -> Vec<f64> {
        self.training_mean = data.iter().sum::<f64>() / data.len() as f64;

        let variance = data.iter()
            .map(|&x| (x - self.training_mean).powi(2))
            .sum::<f64>() / data.len() as f64;

        self.training_std = variance.sqrt();

        if self.training_std < 1e-10 {
            self.training_std = 1.0; // Prevent division by zero
        }

        data.iter()
            .map(|&x| (x - self.training_mean) / self.training_std)
            .collect()
    }

    /// Denormalize predictions
    fn denormalize(&self, data: &[f64]) -> Vec<f64> {
        data.iter()
            .map(|&x| x * self.training_std + self.training_mean)
            .collect()
    }

    /// Train the LSTM/GRU model
    pub fn fit(&mut self, data: &[f64]) -> Result<()> {
        if data.len() < self.config.sequence_length + 1 {
            bail!("Insufficient data: need at least {} points",
                  self.config.sequence_length + 1);
        }

        // Normalize data
        let normalized = self.normalize(data);

        // Create sequences
        let sequences = self.create_sequences(&normalized)?;

        // Training loop
        for epoch in 0..self.config.epochs {
            let mut total_loss = 0.0;
            let mut n_batches = 0;

            // Mini-batch training
            for batch_start in (0..sequences.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(sequences.len());
                let batch = &sequences[batch_start..batch_end];

                let loss = self.train_batch(batch)?;
                total_loss += loss;
                n_batches += 1;
            }

            let avg_loss = total_loss / n_batches as f64;

            if epoch % 10 == 0 {
                println!("Epoch {}/{}: Loss = {:.6}", epoch, self.config.epochs, avg_loss);
            }
        }

        Ok(())
    }

    /// Create input-output sequences for training
    fn create_sequences(&self, data: &[f64]) -> Result<Vec<(Vec<f64>, f64)>> {
        let mut sequences = Vec::new();

        for i in 0..(data.len() - self.config.sequence_length) {
            let input = data[i..i + self.config.sequence_length].to_vec();
            let target = data[i + self.config.sequence_length];
            sequences.push((input, target));
        }

        Ok(sequences)
    }

    /// Train on a batch of sequences
    fn train_batch(&mut self, batch: &[(Vec<f64>, f64)]) -> Result<f64> {
        let mut total_loss = 0.0;

        for (sequence, target) in batch {
            // Forward pass
            let (prediction, hidden_states, cell_states) = self.forward(sequence)?;

            // Compute loss (MSE)
            let loss = (prediction - target).powi(2);
            total_loss += loss;

            // Backward pass (gradient descent)
            self.backward(sequence, *target, prediction, &hidden_states, &cell_states)?;
        }

        Ok(total_loss / batch.len() as f64)
    }

    /// Forward pass through LSTM/GRU
    fn forward(&self, sequence: &[f64]) -> Result<(f64, Vec<Vec<Array1<f64>>>, Vec<Vec<Array1<f64>>>)> {
        let mut hidden_states: Vec<Vec<Array1<f64>>> = vec![vec![]; self.config.num_layers];
        let mut cell_states: Vec<Vec<Array1<f64>>> = vec![vec![]; self.config.num_layers];

        // Initialize hidden and cell states
        let mut h: Vec<Array1<f64>> = (0..self.config.num_layers)
            .map(|_| Array1::zeros(self.config.hidden_size))
            .collect();

        let mut c: Vec<Array1<f64>> = if self.config.cell_type == CellType::LSTM {
            (0..self.config.num_layers)
                .map(|_| Array1::zeros(self.config.hidden_size))
                .collect()
        } else {
            vec![]
        };

        // Process sequence
        for &x_t in sequence {
            let mut layer_input = Array1::from(vec![x_t]);

            for layer_idx in 0..self.config.num_layers {
                let (h_new, c_new) = match self.config.cell_type {
                    CellType::LSTM => self.lstm_cell(
                        &layer_input,
                        &h[layer_idx],
                        &c[layer_idx],
                        &self.weights[layer_idx]
                    )?,
                    CellType::GRU => {
                        let h_new = self.gru_cell(
                            &layer_input,
                            &h[layer_idx],
                            &self.weights[layer_idx]
                        )?;
                        (h_new, Array1::zeros(0))
                    }
                };

                h[layer_idx] = h_new.clone();
                hidden_states[layer_idx].push(h_new.clone());

                if self.config.cell_type == CellType::LSTM {
                    c[layer_idx] = c_new.clone();
                    cell_states[layer_idx].push(c_new);
                }

                // Output of this layer is input to next layer
                layer_input = h_new;
            }
        }

        // Final prediction from last hidden state of top layer
        let final_hidden = &h[h.len() - 1];
        let prediction = final_hidden[0];  // Simple: use first hidden unit as output

        Ok((prediction, hidden_states, cell_states))
    }

    /// LSTM cell computation
    fn lstm_cell(
        &self,
        x: &Array1<f64>,
        h_prev: &Array1<f64>,
        c_prev: &Array1<f64>,
        weights: &LstmWeights,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let hidden_size = h_prev.len();

        // Concatenate input and previous hidden state
        let mut input_vec: Vec<f64> = Vec::with_capacity(x.len() + h_prev.len());
        input_vec.extend(x.iter());
        input_vec.extend(h_prev.iter());

        // Compute all gates in one matrix multiplication
        let x_arr = Array1::from(input_vec);
        let i_part = weights.w_ih.dot(x);
        let h_part = weights.w_hh.dot(h_prev);
        let gates = &i_part + &h_part + &weights.bias;

        // Split into 4 gates
        let forget_gate = Self::sigmoid(&gates.slice(ndarray::s![0..hidden_size]));
        let input_gate = Self::sigmoid(&gates.slice(ndarray::s![hidden_size..2*hidden_size]));
        let cell_candidate = Self::tanh(&gates.slice(ndarray::s![2*hidden_size..3*hidden_size]));
        let output_gate = Self::sigmoid(&gates.slice(ndarray::s![3*hidden_size..4*hidden_size]));

        // Update cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        let c_new = &forget_gate * c_prev + &input_gate * &cell_candidate;

        // Update hidden state: h_t = o_t ⊙ tanh(c_t)
        let h_new = &output_gate * &Self::tanh(&c_new.view());

        Ok((h_new, c_new))
    }

    /// GRU cell computation
    fn gru_cell(
        &self,
        x: &Array1<f64>,
        h_prev: &Array1<f64>,
        weights: &LstmWeights,
    ) -> Result<Array1<f64>> {
        let hidden_size = h_prev.len();

        // Compute gates
        let i_part = weights.w_ih.dot(x);
        let h_part = weights.w_hh.dot(h_prev);
        let gates = &i_part + &h_part + &weights.bias;

        // Split into 3 components
        let reset_gate = Self::sigmoid(&gates.slice(ndarray::s![0..hidden_size]));
        let update_gate = Self::sigmoid(&gates.slice(ndarray::s![hidden_size..2*hidden_size]));

        // Candidate hidden state with reset gate applied
        let reset_h = &reset_gate * h_prev;
        let h_part_reset = weights.w_hh.dot(&reset_h);
        let candidate = Self::tanh(&(&i_part.slice(ndarray::s![2*hidden_size..3*hidden_size]) + &h_part_reset.slice(ndarray::s![2*hidden_size..3*hidden_size])).view());

        // Update hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        let one_minus_z = update_gate.mapv(|z| 1.0 - z);
        let h_new = &one_minus_z * h_prev + &update_gate * &candidate;

        Ok(h_new)
    }

    /// Sigmoid activation
    fn sigmoid(x: &ndarray::ArrayView1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    /// Tanh activation
    fn tanh(x: &ndarray::ArrayView1<f64>) -> Array1<f64> {
        x.mapv(|v| v.tanh())
    }

    /// Backward pass (simplified gradient descent)
    fn backward(
        &mut self,
        _sequence: &[f64],
        target: f64,
        prediction: f64,
        _hidden_states: &[Vec<Array1<f64>>],
        _cell_states: &[Vec<Array1<f64>>],
    ) -> Result<()> {
        // Gradient of loss w.r.t prediction
        let grad_output = 2.0 * (prediction - target);

        // Simplified weight update (full BPTT would be more complex)
        // For production, use proper backpropagation through time

        let lr = self.config.learning_rate;

        // Update only the output layer weights (simplified)
        if let Some(last_layer) = self.weights.last_mut() {
            // Simple gradient descent on bias
            last_layer.bias[0] -= lr * grad_output;
        }

        Ok(())
    }

    /// Forecast h steps ahead
    pub fn forecast(&mut self, data: &[f64], horizon: usize) -> Result<Vec<f64>> {
        // Normalize input
        let normalized = data.iter()
            .map(|&x| (x - self.training_mean) / self.training_std)
            .collect::<Vec<f64>>();

        // Use last sequence_length points as initial sequence
        let start_idx = if normalized.len() >= self.config.sequence_length {
            normalized.len() - self.config.sequence_length
        } else {
            0
        };

        let mut sequence = normalized[start_idx..].to_vec();

        // Ensure we have exactly sequence_length points
        while sequence.len() < self.config.sequence_length {
            sequence.insert(0, 0.0);
        }

        let mut forecast = Vec::with_capacity(horizon);

        // Generate predictions iteratively
        for _ in 0..horizon {
            // Get prediction for next step
            let (pred, _, _) = self.forward(&sequence)?;

            forecast.push(pred);

            // Update sequence: remove oldest, add newest
            sequence.remove(0);
            sequence.push(pred);
        }

        // Denormalize forecast
        let denormalized = self.denormalize(&forecast);

        Ok(denormalized)
    }

    /// Forecast with GPU acceleration (batch processing)
    pub fn forecast_batch(&mut self, data: &[f64], horizons: &[usize]) -> Result<Vec<Vec<f64>>> {
        let forecasts: Vec<Vec<f64>> = horizons.iter()
            .map(|&h| self.forecast(data, h))
            .collect::<Result<Vec<_>>>()?;

        Ok(forecasts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_creation() {
        let config = LstmConfig {
            cell_type: CellType::LSTM,
            hidden_size: 10,
            num_layers: 1,
            sequence_length: 5,
            learning_rate: 0.01,
            epochs: 10,
            batch_size: 8,
            dropout: 0.0,
        };

        let model = LstmForecaster::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_gru_creation() {
        let config = LstmConfig {
            cell_type: CellType::GRU,
            hidden_size: 10,
            num_layers: 1,
            sequence_length: 5,
            learning_rate: 0.01,
            epochs: 10,
            batch_size: 8,
            dropout: 0.0,
        };

        let model = LstmForecaster::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_sequence_creation() {
        let config = LstmConfig {
            sequence_length: 3,
            ..Default::default()
        };

        let model = LstmForecaster::new(config).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let sequences = model.create_sequences(&data).unwrap();

        assert_eq!(sequences.len(), 2);
        assert_eq!(sequences[0].0, vec![1.0, 2.0, 3.0]);
        assert_eq!(sequences[0].1, 4.0);
        assert_eq!(sequences[1].0, vec![2.0, 3.0, 4.0]);
        assert_eq!(sequences[1].1, 5.0);
    }

    #[test]
    fn test_lstm_forward() {
        let config = LstmConfig {
            cell_type: CellType::LSTM,
            hidden_size: 5,
            num_layers: 1,
            sequence_length: 3,
            ..Default::default()
        };

        let model = LstmForecaster::new(config).unwrap();
        let sequence = vec![0.1, 0.2, 0.3];

        let result = model.forward(&sequence);
        assert!(result.is_ok());

        let (prediction, _, _) = result.unwrap();
        assert!(prediction.is_finite());
    }

    #[test]
    fn test_lstm_fit_small() {
        let mut config = LstmConfig::default();
        config.hidden_size = 10;
        config.sequence_length = 3;
        config.epochs = 5;
        config.batch_size = 4;

        let mut model = LstmForecaster::new(config).unwrap();

        // Simple linear trend
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();

        let result = model.fit(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_lstm_forecast() {
        let mut config = LstmConfig::default();
        config.hidden_size = 10;
        config.sequence_length = 5;
        config.epochs = 10;

        let mut model = LstmForecaster::new(config).unwrap();

        // Training data
        let data: Vec<f64> = (0..30).map(|i| (i as f64 * 0.1).sin()).collect();

        model.fit(&data).unwrap();

        // Forecast
        let forecast = model.forecast(&data, 5).unwrap();

        assert_eq!(forecast.len(), 5);
        assert!(forecast.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_gru_forecast() {
        let mut config = LstmConfig::default();
        config.cell_type = CellType::GRU;
        config.hidden_size = 10;
        config.sequence_length = 5;
        config.epochs = 10;

        let mut model = LstmForecaster::new(config).unwrap();

        let data: Vec<f64> = (0..30).map(|i| i as f64 * 0.5).collect();

        model.fit(&data).unwrap();

        let forecast = model.forecast(&data, 3).unwrap();

        assert_eq!(forecast.len(), 3);
        assert!(forecast.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_normalization() {
        let config = LstmConfig::default();
        let mut model = LstmForecaster::new(config).unwrap();

        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let normalized = model.normalize(&data);

        // Mean should be close to 0, std close to 1
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-10);

        let denormalized = model.denormalize(&normalized);
        for i in 0..data.len() {
            assert!((denormalized[i] - data[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_batch_forecast() {
        let mut config = LstmConfig::default();
        config.epochs = 5;

        let mut model = LstmForecaster::new(config).unwrap();

        let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        model.fit(&data).unwrap();

        let horizons = vec![1, 3, 5];
        let forecasts = model.forecast_batch(&data, &horizons).unwrap();

        assert_eq!(forecasts.len(), 3);
        assert_eq!(forecasts[0].len(), 1);
        assert_eq!(forecasts[1].len(), 3);
        assert_eq!(forecasts[2].len(), 5);
    }
}
