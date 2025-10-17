# HIGH PRIORITY ENHANCEMENTS TODO
## Technical Debt Items 1-3 Implementation Plan

**Created:** January 9, 2025
**Target:** Phase II Months 4-6 (Post-SBIR Award)
**Estimated Total Effort:** 5-7 days
**Purpose:** Enhance Mission Bravo from v1.0 to v2.0

**Source:** TECHNICAL-DEBT-INVENTORY.md Items 1-3

---

## STRATEGIC OVERVIEW

### Enhancement Goals

**Current State (v1.0):**
- ✅ Production-ready with heuristics
- ✅ <1ms latency achieved
- ✅ Constitutional compliance verified
- ✅ SBIR demonstration-ready

**Target State (v2.0):**
- ✅ ML-based threat classification (improved accuracy)
- ✅ Real spatial entropy computation (better IR analysis)
- ✅ Multi-frame motion tracking (enhanced temporal awareness)
- ✅ Full Article II & IV compliance (no simplifications)

### Impact Assessment

| Enhancement | Accuracy Gain | Latency Impact | Constitutional Impact |
|-------------|---------------|----------------|----------------------|
| ML Threat Classifier | +10-15% | +50-100μs | Full Article IV |
| Spatial Entropy | +5-8% | +20-30μs | Enhanced Article II |
| Frame Tracking | +8-12% | +30-50μs | Full Article II |
| **COMBINED** | **+20-30%** | **+100-180μs** | **Enhanced** |

**Performance Budget:**
- Current latency: 850μs
- Enhancement overhead: +180μs
- **Target latency: <1.1ms (still excellent)**

---

## ENHANCEMENT 1: ML-BASED THREAT CLASSIFIER

### Strategic Importance: CRITICAL
**Current:** Heuristic rules (satellite_adapters.rs:362-395)
**Target:** Neural network or Bayesian classifier
**Effort:** 2-3 days
**Article:** Article IV (Active Inference) - Full compliance

---

### Task 1.1: Design Active Inference Classifier Architecture
**Objective:** Design neural network that implements variational inference

**Design Specifications:**
```rust
pub struct ActiveInferenceClassifier {
    // Generative model: P(observations | threat_class)
    generative_model: GenerativeNetwork,

    // Recognition model: Q(threat_class | observations)
    recognition_model: RecognitionNetwork,

    // Prior beliefs
    prior_beliefs: Array1<f64>,  // [5] threat classes

    // Free energy tracking (Article IV requirement)
    free_energy_history: VecDeque<f64>,
}

impl ActiveInferenceClassifier {
    /// Classify threat using variational inference
    ///
    /// # Article IV Compliance
    /// Minimizes variational free energy: F = DKL(Q||P) - log P(observations)
    pub fn classify(&mut self, features: &Array1<f64>) -> Result<ThreatClassification> {
        // 1. Recognition pass: Q(class|observations)
        let posterior = self.recognition_model.forward(features)?;

        // 2. Compute free energy
        let free_energy = self.compute_free_energy(&posterior, features)?;

        // 3. Update beliefs (Bayesian)
        let beliefs = self.update_beliefs(&posterior)?;

        // 4. Validate free energy is finite (Article IV requirement)
        assert!(free_energy.is_finite(), "Free energy must be finite");

        Ok(ThreatClassification {
            class_probabilities: beliefs,
            free_energy,
            confidence: self.compute_confidence(&beliefs),
        })
    }

    fn compute_free_energy(&self, posterior: &Array1<f64>, observations: &Array1<f64>) -> Result<f64> {
        // F = DKL(Q||P) - log P(observations)
        let kl_divergence = self.compute_kl_divergence(posterior, &self.prior_beliefs);
        let log_likelihood = self.generative_model.log_likelihood(observations)?;

        Ok(kl_divergence - log_likelihood)
    }
}
```

**Architecture Choices:**
- **Option A:** Small feedforward network (100 → 50 → 25 → 5)
- **Option B:** Bayesian neural network with uncertainty
- **Option C:** Variational autoencoder (generative model)

**Recommendation:** Option C (VAE) - True active inference with generative model

**Files to Create:**
- `src/pwsa/active_inference_classifier.rs` (NEW)

**Estimated Time:** 6-8 hours

---

### Task 1.2: Collect and Prepare Training Data
**Objective:** Gather labeled threat examples for training

**Data Sources:**
1. **Synthetic Data (Immediate):**
   - Generate from existing heuristic (bootstrap)
   - 10,000 samples per class
   - Add noise for robustness

2. **Simulated Scenarios:**
   - Hypersonic: Mach 5-8, high acceleration (>40g)
   - Ballistic: Mach 6-10, low acceleration (<20g)
   - Cruise: Mach 0.8-2, moderate acceleration
   - Aircraft: Mach 0.3-2, sustained flight
   - Background: Low velocity, low thermal

3. **Operational Data (Phase II Month 6+):**
   - Real IR sensor frames from SDA (if available)
   - Labeled by human analysts
   - Fine-tune model with real data

**Data Structure:**
```rust
pub struct ThreatTrainingExample {
    features: Array1<f64>,  // 100-dimensional
    label: ThreatClass,     // Ground truth
    confidence: f64,        // Labeling confidence
    metadata: ExampleMetadata,
}

pub enum ThreatClass {
    NoThreat = 0,
    Aircraft = 1,
    CruiseMissile = 2,
    BallisticMissile = 3,
    Hypersonic = 4,
}
```

**Dataset Size:**
- Training: 40,000 samples (8,000 per class)
- Validation: 10,000 samples (2,000 per class)
- Test: 10,000 samples (2,000 per class)

**Files to Create:**
- `data/training/threat_dataset.rs` (data generator)
- `data/training/generate_training_data.rs` (script)

**Estimated Time:** 4-6 hours

---

### Task 1.3: Implement Neural Network Model
**Objective:** Build and train the classifier network

**Model Architecture:**
```rust
use candle_nn::{Module, VarBuilder, VarMap};
use candle_core::{Tensor, Device};

pub struct ThreatClassifierNetwork {
    fc1: candle_nn::Linear,  // 100 → 64
    fc2: candle_nn::Linear,  // 64 → 32
    fc3: candle_nn::Linear,  // 32 → 16
    fc4: candle_nn::Linear,  // 16 → 5 (output)
    dropout: candle_nn::Dropout,
    device: Device,
}

impl ThreatClassifierNetwork {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(100, 64, vb.pp("fc1"))?,
            fc2: candle_nn::linear(64, 32, vb.pp("fc2"))?,
            fc3: candle_nn::linear(32, 16, vb.pp("fc3"))?,
            fc4: candle_nn::linear(16, 5, vb.pp("fc4"))?,
            dropout: candle_nn::Dropout::new(0.2),
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(features)?;
        let x = x.relu()?;
        let x = self.dropout.forward(&x, true)?;

        let x = self.fc2.forward(&x)?;
        let x = x.relu()?;
        let x = self.dropout.forward(&x, true)?;

        let x = self.fc3.forward(&x)?;
        let x = x.relu()?;

        let x = self.fc4.forward(&x)?;
        x.softmax(1)  // Normalized probabilities
    }
}
```

**Training Configuration:**
```rust
pub struct TrainingConfig {
    learning_rate: f64,      // 0.001 (Adam optimizer)
    batch_size: usize,       // 64
    epochs: usize,           // 100
    early_stopping: usize,   // 10 epochs patience
    validation_split: f64,   // 0.2
}
```

**Files to Create:**
- `src/pwsa/ml_threat_classifier.rs` (NEW)

**Estimated Time:** 8-10 hours

---

### Task 1.4: Train and Validate Model
**Objective:** Train network to convergence

**Training Process:**
```rust
pub fn train_threat_classifier(
    training_data: &[ThreatTrainingExample],
    config: TrainingConfig,
) -> Result<ThreatClassifierNetwork> {
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mut model = ThreatClassifierNetwork::new(vb)?;
    let optimizer = candle_nn::Adam::new(varmap.all_vars(), config.learning_rate)?;

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;

        for batch in training_data.chunks(config.batch_size) {
            // Forward pass
            let features = Tensor::from_vec(/* ... */)?;
            let predictions = model.forward(&features)?;

            // Compute loss (cross-entropy)
            let labels = Tensor::from_vec(/* ... */)?;
            let loss = cross_entropy_loss(&predictions, &labels)?;

            // Backward pass
            optimizer.backward_step(&loss)?;

            epoch_loss += loss.to_scalar::<f32>()?;
        }

        // Validation
        let val_accuracy = validate_model(&model, validation_data)?;
        println!("Epoch {}: Loss={:.4}, Val Acc={:.2}%",
            epoch, epoch_loss, val_accuracy * 100.0);

        // Early stopping
        if should_stop_early(val_accuracy) {
            break;
        }
    }

    Ok(model)
}
```

**Validation Metrics:**
- Accuracy (overall)
- Precision per class
- Recall per class
- F1 score
- Confusion matrix

**Target Performance:**
- Overall accuracy: >90%
- Hypersonic precision: >95% (critical class)
- False positive rate: <5%

**Files to Create:**
- `src/pwsa/train_classifier.rs` (training script)

**Estimated Time:** 6-8 hours (including debugging)

---

### Task 1.5: Integrate Trained Model into Fusion Platform
**Objective:** Replace heuristic classifier with trained model

**Integration:**
```rust
impl TrackingLayerAdapter {
    pub fn new_tranche1_ml(n_dimensions: usize) -> Result<Self> {
        // Load pre-trained model
        let model_path = "models/threat_classifier_v2.safetensors";
        let ml_classifier = ThreatClassifierNetwork::load(model_path)?;

        Ok(Self {
            platform: UnifiedPlatform::new(n_dimensions)?,
            sensor_fov_deg: 120.0,
            frame_rate_hz: 10.0,
            n_dimensions,
            threat_classifier: ClassifierType::ML(ml_classifier),  // NEW
        })
    }

    fn classify_threats(&self, features: &Array1<f64>) -> Result<Array1<f64>> {
        match &self.threat_classifier {
            ClassifierType::Heuristic => {
                // v1.0 fallback
                self.classify_threats_heuristic(features)
            },
            ClassifierType::ML(model) => {
                // v2.0 ML classifier
                let tensor = Tensor::from_slice(features.as_slice().unwrap(), &model.device)?;
                let probs = model.forward(&tensor)?;
                Ok(Array1::from_vec(probs.to_vec1()?))
            }
        }
    }
}
```

**Files to Modify:**
- `src/pwsa/satellite_adapters.rs` (add ML option)

**Estimated Time:** 3-4 hours

---

### Task 1.6: Create ML Classifier Tests
**Objective:** Validate ML classifier performance

**Test Cases:**
```rust
#[test]
fn test_ml_classifier_accuracy() {
    let model = load_trained_model().unwrap();
    let test_data = load_test_dataset().unwrap();

    let mut correct = 0;
    for example in test_data {
        let prediction = model.classify(&example.features).unwrap();
        let predicted_class = prediction.class_probabilities.argmax();

        if predicted_class == example.label {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / test_data.len() as f64;
    assert!(accuracy > 0.90, "Accuracy {:.2}% below 90% target", accuracy * 100.0);
}

#[test]
fn test_ml_classifier_hypersonic_precision() {
    // Critical: Must not miss hypersonic threats
    let hypersonic_examples = load_hypersonic_test_cases().unwrap();

    let mut true_positives = 0;
    for example in hypersonic_examples {
        let prediction = model.classify(&example.features).unwrap();

        if prediction.class_probabilities[4] > 0.5 {  // Class 4 = Hypersonic
            true_positives += 1;
        }
    }

    let precision = true_positives as f64 / hypersonic_examples.len() as f64;
    assert!(precision > 0.95, "Hypersonic precision {:.2}% below 95%", precision * 100.0);
}

#[test]
fn test_ml_classifier_latency() {
    let model = load_trained_model().unwrap();
    let features = generate_test_features();

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = model.classify(&features).unwrap();
    }
    let avg_latency = start.elapsed() / 1000;

    assert!(avg_latency < Duration::from_micros(200),
        "ML classifier latency {}μs exceeds 200μs budget", avg_latency.as_micros());
}

#[test]
fn test_article_iv_free_energy_finite() {
    let model = load_trained_model().unwrap();
    let features = generate_test_features();

    let result = model.classify(&features).unwrap();

    // Article IV: Free energy must be finite
    assert!(result.free_energy.is_finite());
    assert!(result.free_energy >= 0.0);  // Non-negative

    // Probabilities must sum to 1 (normalization)
    let sum: f64 = result.class_probabilities.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}
```

**Files to Create:**
- `tests/pwsa_ml_classifier_test.rs` (NEW)

**Estimated Time:** 3-4 hours

---

### Task 1.7: Benchmark ML vs Heuristic Performance
**Objective:** Validate improvement and latency impact

**Benchmarks:**
```rust
// benches/threat_classifier_comparison.rs
#[bench]
fn bench_heuristic_classifier(b: &mut Bencher) {
    let adapter = TrackingLayerAdapter::new_tranche1_heuristic(900).unwrap();
    let features = generate_test_features();

    b.iter(|| {
        adapter.classify_threats_heuristic(black_box(&features))
    });
}

#[bench]
fn bench_ml_classifier(b: &mut Bencher) {
    let adapter = TrackingLayerAdapter::new_tranche1_ml(900).unwrap();
    let features = generate_test_features();

    b.iter(|| {
        adapter.classify_threats_ml(black_box(&features))
    });
}

#[bench]
fn bench_accuracy_comparison(b: &mut Bencher) {
    // Compare accuracy on test set
    let test_set = load_test_dataset().unwrap();

    let heuristic_accuracy = evaluate_heuristic(&test_set);
    let ml_accuracy = evaluate_ml_classifier(&test_set);

    println!("Heuristic: {:.2}%", heuristic_accuracy * 100.0);
    println!("ML:        {:.2}%", ml_accuracy * 100.0);
    println!("Improvement: +{:.1}%", (ml_accuracy - heuristic_accuracy) * 100.0);
}
```

**Files to Create:**
- `benches/threat_classifier_comparison.rs` (NEW)

**Estimated Time:** 2-3 hours

---

### **Enhancement 1 Total Deliverables:**
- [x] Active inference classifier architecture designed
- [x] Training data generated (50,000+ samples)
- [x] Neural network model implemented
- [x] Model trained to >90% accuracy
- [x] Integrated into TrackingLayerAdapter
- [x] Comprehensive test suite (5+ tests)
- [x] Performance benchmarked

**Total Time for Enhancement 1:** 2-3 days
**Git Commits:** 3-4 (design, implementation, tests, integration)

---

## ENHANCEMENT 2: SPATIAL ENTROPY COMPUTATION

### Strategic Importance: HIGH
**Current:** Fixed value 0.5 (satellite_adapters.rs:289-293)
**Target:** Real Shannon entropy of intensity distribution
**Effort:** 4-6 hours
**Article:** Article II (Neuromorphic) - Enhanced temporal/spatial patterns

---

### Task 2.1: Implement Intensity Histogram Computation
**Objective:** Compute pixel intensity distribution

**Implementation:**
```rust
impl TrackingLayerAdapter {
    fn compute_intensity_histogram(&self, frame: &IrSensorFrame, n_bins: usize) -> Vec<usize> {
        let mut histogram = vec![0; n_bins];

        // Would need actual pixel data in IrSensorFrame
        // For now, use statistical approximation from frame metadata

        let intensity_range = frame.max_intensity - frame.background_level;
        let bin_width = intensity_range / n_bins as f64;

        // Approximate distribution from hotspot data
        // Real implementation would iterate over frame.pixels

        for hotspot_idx in 0..frame.hotspot_count {
            // Hotspots concentrated in high-intensity bins
            let bin = (n_bins * 3 / 4) + (hotspot_idx as usize % (n_bins / 4));
            histogram[bin] += 1;
        }

        // Background pixels in low-intensity bins
        let background_pixels = (frame.width * frame.height) as usize - frame.hotspot_count as usize;
        let pixels_per_bin = background_pixels / (n_bins / 2);
        for i in 0..(n_bins / 2) {
            histogram[i] = pixels_per_bin;
        }

        histogram
    }
}
```

**Note:** Full implementation requires pixel-level data in IrSensorFrame. This version uses statistical approximation.

**Files to Modify:**
- `src/pwsa/satellite_adapters.rs` (add histogram method)

**Estimated Time:** 2-3 hours

---

### Task 2.2: Implement Shannon Entropy Calculation
**Objective:** Compute information-theoretic entropy

**Implementation:**
```rust
fn compute_spatial_entropy(&self, frame: &IrSensorFrame) -> f64 {
    let n_bins = 16;  // 16 bins for intensity histogram
    let histogram = self.compute_intensity_histogram(frame, n_bins);

    // Compute total pixel count
    let total_pixels = histogram.iter().sum::<usize>() as f64;

    if total_pixels == 0.0 {
        return 0.0;
    }

    // Shannon entropy: H = -Σ p(i) log2(p(i))
    let mut entropy = 0.0;
    for &count in &histogram {
        if count > 0 {
            let p = count as f64 / total_pixels;
            entropy -= p * p.log2();
        }
    }

    // Normalize by maximum possible entropy
    let max_entropy = (n_bins as f64).log2();

    if max_entropy > 0.0 {
        entropy / max_entropy  // Normalized [0, 1]
    } else {
        0.0
    }
}
```

**Information-Theoretic Interpretation:**
- **High entropy (→1.0):** Uniform distribution (no focused hotspot)
- **Low entropy (→0.0):** Concentrated distribution (focused threat)

**Relationship to Threat Detection:**
- Single missile launch: Low entropy (concentrated)
- Multiple targets: High entropy (dispersed)
- Background clutter: Medium entropy

**Files to Modify:**
- `src/pwsa/satellite_adapters.rs` (replace placeholder)

**Estimated Time:** 1-2 hours

---

### Task 2.3: Add Spatial Entropy Validation Tests
**Objective:** Validate Shannon entropy properties

**Test Cases:**
```rust
#[test]
fn test_spatial_entropy_uniform_distribution() {
    let adapter = TrackingLayerAdapter::new_tranche1(900).unwrap();

    // Uniform intensity (all bins equal)
    let frame = create_uniform_intensity_frame();

    let entropy = adapter.compute_spatial_entropy(&frame);

    // Should be maximum (1.0 for normalized entropy)
    assert!(entropy > 0.95, "Uniform distribution should have high entropy");
}

#[test]
fn test_spatial_entropy_single_hotspot() {
    let adapter = TrackingLayerAdapter::new_tranche1(900).unwrap();

    // Single concentrated hotspot
    let frame = IrSensorFrame {
        hotspot_count: 1,
        max_intensity: 5000.0,
        background_level: 100.0,
        // ... single focused threat
    };

    let entropy = adapter.compute_spatial_entropy(&frame);

    // Should be low (concentrated)
    assert!(entropy < 0.3, "Single hotspot should have low entropy");
}

#[test]
fn test_spatial_entropy_multiple_dispersed() {
    let adapter = TrackingLayerAdapter::new_tranche1(900).unwrap();

    // Multiple dispersed hotspots
    let frame = IrSensorFrame {
        hotspot_count: 10,
        max_intensity: 2000.0,
        background_level: 100.0,
        // ... dispersed background clutter
    };

    let entropy = adapter.compute_spatial_entropy(&frame);

    // Should be medium-high (dispersed)
    assert!(entropy > 0.5, "Dispersed hotspots should have higher entropy");
}

#[test]
fn test_spatial_entropy_range() {
    let adapter = TrackingLayerAdapter::new_tranche1(900).unwrap();

    for _ in 0..100 {
        let frame = generate_random_frame();
        let entropy = adapter.compute_spatial_entropy(&frame);

        // Must be in [0, 1] range
        assert!(entropy >= 0.0 && entropy <= 1.0,
            "Entropy {} out of range [0,1]", entropy);
    }
}
```

**Files to Create:**
- Add to `tests/pwsa_adapters_test.rs`

**Estimated Time:** 1-2 hours

---

### **Enhancement 2 Total Deliverables:**
- [x] Intensity histogram computation implemented
- [x] Shannon entropy calculation working
- [x] Normalized entropy [0, 1]
- [x] Validation tests passing (4+ tests)
- [x] Information-theoretic correctness verified

**Total Time for Enhancement 2:** 4-6 hours
**Git Commits:** 2 (implementation, tests)

---

## ENHANCEMENT 3: FRAME-TO-FRAME MOTION TRACKING

### Strategic Importance: HIGH
**Current:** Fixed value 0.8 (satellite_adapters.rs:308-312)
**Target:** Multi-frame object tracking with trajectory prediction
**Effort:** 1-2 days
**Article:** Article II (Neuromorphic) - Full temporal pattern compliance

---

### Task 3.1: Design Multi-Frame Tracker Architecture
**Objective:** Track objects across multiple IR frames

**Architecture:**
```rust
pub struct MultiFrameTracker {
    /// Active tracks (object_id → trajectory)
    active_tracks: HashMap<Uuid, ObjectTrajectory>,

    /// Maximum age before track deletion
    max_track_age: Duration,

    /// Association threshold (pixels)
    association_threshold: f64,
}

pub struct ObjectTrajectory {
    object_id: Uuid,
    positions: VecDeque<(f64, f64, SystemTime)>,  // (x, y, timestamp)
    velocities: VecDeque<(f64, f64)>,             // (vx, vy)
    accelerations: VecDeque<(f64, f64)>,          // (ax, ay)
    predicted_next: Option<(f64, f64)>,           // Kalman prediction
    last_update: SystemTime,
}

impl MultiFrameTracker {
    pub fn update(&mut self, frame: &IrSensorFrame) -> Result<Vec<TrackUpdate>> {
        // 1. Predict where existing tracks should be
        let predictions = self.predict_all_tracks(frame.timestamp);

        // 2. Associate observations with tracks
        let associations = self.associate_observations(frame, &predictions)?;

        // 3. Update matched tracks
        let updates = self.update_tracks(&associations)?;

        // 4. Initialize new tracks for unmatched observations
        self.initialize_new_tracks(frame, &associations)?;

        // 5. Delete stale tracks
        self.prune_old_tracks(frame.timestamp);

        Ok(updates)
    }

    fn predict_all_tracks(&self, current_time: SystemTime) -> Vec<(Uuid, (f64, f64))> {
        // Kalman filter prediction for each track
        // ...
    }
}
```

**Files to Create:**
- `src/pwsa/multi_frame_tracker.rs` (NEW)

**Estimated Time:** 4-6 hours

---

### Task 3.2: Implement Kalman Filter for Trajectory Prediction
**Objective:** Predict next object position

**Implementation:**
```rust
pub struct KalmanFilter {
    // State: [x, y, vx, vy, ax, ay]
    state: Array1<f64>,

    // Covariance matrix
    covariance: Array2<f64>,

    // Process noise
    process_noise: Array2<f64>,

    // Measurement noise
    measurement_noise: Array2<f64>,
}

impl KalmanFilter {
    pub fn predict(&mut self, dt: f64) -> (f64, f64) {
        // State transition matrix (constant acceleration model)
        let f = Array2::from_shape_vec((6, 6), vec![
            1.0, 0.0, dt, 0.0, 0.5*dt*dt, 0.0,
            0.0, 1.0, 0.0, dt, 0.0, 0.5*dt*dt,
            0.0, 0.0, 1.0, 0.0, dt, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, dt,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ])?;

        // Predict state
        self.state = f.dot(&self.state);

        // Predict covariance
        self.covariance = f.dot(&self.covariance).dot(&f.t()) + &self.process_noise;

        // Return predicted position
        (self.state[0], self.state[1])
    }

    pub fn update(&mut self, observation: (f64, f64)) {
        // Measurement model (observe position only)
        let h = Array2::from_shape_vec((2, 6), vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ])?;

        // Innovation
        let obs_vec = Array1::from_vec(vec![observation.0, observation.1]);
        let pred_obs = h.dot(&self.state);
        let innovation = obs_vec - pred_obs;

        // Kalman gain
        let s = h.dot(&self.covariance).dot(&h.t()) + &self.measurement_noise;
        let k = self.covariance.dot(&h.t()).dot(&s.inv()?);

        // Update state
        self.state = &self.state + k.dot(&innovation);

        // Update covariance
        let eye = Array2::eye(6);
        self.covariance = (eye - k.dot(&h)).dot(&self.covariance);
    }
}
```

**Files to Create:**
- `src/pwsa/kalman_filter.rs` (NEW)

**Estimated Time:** 4-6 hours

---

### Task 3.3: Implement Data Association Algorithm
**Objective:** Match observations to existing tracks

**Implementation:**
```rust
impl MultiFrameTracker {
    fn associate_observations(
        &self,
        frame: &IrSensorFrame,
        predictions: &[(Uuid, (f64, f64))],
    ) -> Result<Vec<Association>> {
        // Extract hotspot positions (observations)
        let observations = self.extract_hotspot_positions(frame);

        // Hungarian algorithm for optimal assignment
        let cost_matrix = self.compute_cost_matrix(&observations, predictions);

        let assignments = hungarian_algorithm(&cost_matrix)?;

        // Convert to associations
        let mut associations = Vec::new();
        for (obs_idx, track_idx) in assignments {
            if cost_matrix[[obs_idx, track_idx]] < self.association_threshold {
                associations.push(Association {
                    track_id: predictions[track_idx].0,
                    observation: observations[obs_idx],
                    cost: cost_matrix[[obs_idx, track_idx]],
                });
            }
        }

        Ok(associations)
    }

    fn compute_cost_matrix(
        &self,
        observations: &[(f64, f64)],
        predictions: &[(Uuid, (f64, f64))],
    ) -> Array2<f64> {
        let n_obs = observations.len();
        let n_pred = predictions.len();
        let mut cost = Array2::zeros((n_obs, n_pred));

        for (i, obs) in observations.iter().enumerate() {
            for (j, (_, pred)) in predictions.iter().enumerate() {
                // Euclidean distance
                let dx = obs.0 - pred.0;
                let dy = obs.1 - pred.1;
                cost[[i, j]] = (dx * dx + dy * dy).sqrt();
            }
        }

        cost
    }
}

// Hungarian algorithm implementation (or use existing crate)
fn hungarian_algorithm(cost_matrix: &Array2<f64>) -> Result<Vec<(usize, usize)>> {
    // Standard Hungarian/Munkres algorithm
    // ...
}
```

**Dependencies to Add:**
```toml
[dependencies]
pathfinding = "4.0"  # Includes Hungarian algorithm
```

**Files to Modify:**
- `src/pwsa/multi_frame_tracker.rs` (add association)

**Estimated Time:** 3-4 hours

---

### Task 3.4: Implement Motion Consistency Metric
**Objective:** Replace fixed 0.8 with real tracking-based metric

**Implementation:**
```rust
fn compute_motion_consistency(&mut self, frame: &IrSensorFrame) -> f64 {
    // Update tracker with new frame
    let track_updates = self.multi_frame_tracker.update(frame)?;

    if track_updates.is_empty() {
        return 0.5;  // No tracks (first frame or lost track)
    }

    // Compute average prediction error across all tracks
    let mut total_error = 0.0;
    let mut count = 0;

    for update in &track_updates {
        if let Some(predicted) = update.predicted_position {
            let actual = update.observed_position;

            // Normalized prediction error
            let dx = (predicted.0 - actual.0) / frame.width as f64;
            let dy = (predicted.1 - actual.1) / frame.height as f64;
            let error = (dx * dx + dy * dy).sqrt();

            total_error += error;
            count += 1;
        }
    }

    if count == 0 {
        return 0.5;
    }

    let avg_error = total_error / count as f64;

    // Consistency: 1.0 = perfect prediction, 0.0 = random motion
    // Error of 0.1 (10% of frame) = 50% consistency
    (1.0 - avg_error * 5.0).max(0.0).min(1.0)
}
```

**Files to Modify:**
- `src/pwsa/satellite_adapters.rs` (replace placeholder)

**Estimated Time:** 1-2 hours

---

### Task 3.5: Add TrackingLayerAdapter Integration
**Objective:** Wire tracker into adapter

**Modification:**
```rust
pub struct TrackingLayerAdapter {
    platform: UnifiedPlatform,
    sensor_fov_deg: f64,
    frame_rate_hz: f64,
    n_dimensions: usize,
    /// Week 2 Enhancement: Multi-frame tracker
    multi_frame_tracker: Option<MultiFrameTracker>,  // NEW
}

impl TrackingLayerAdapter {
    pub fn new_tranche1_with_tracking(n_dimensions: usize) -> Result<Self> {
        Ok(Self {
            platform: UnifiedPlatform::new(n_dimensions)?,
            sensor_fov_deg: 120.0,
            frame_rate_hz: 10.0,
            n_dimensions,
            multi_frame_tracker: Some(MultiFrameTracker::new(
                max_track_age: Duration::from_secs(30),
                association_threshold: 50.0,  // pixels
            )),
        })
    }

    pub fn ingest_ir_frame_with_tracking(
        &mut self,
        sv_id: u32,
        frame: &IrSensorFrame,
    ) -> Result<ThreatDetection> {
        // Update tracker
        if let Some(tracker) = &mut self.multi_frame_tracker {
            let _track_updates = tracker.update(frame)?;
        }

        // Continue with normal processing
        self.ingest_ir_frame(sv_id, frame)
    }
}
```

**Files to Modify:**
- `src/pwsa/satellite_adapters.rs` (add tracker field)

**Estimated Time:** 2-3 hours

---

### Task 3.6: Create Frame Tracking Tests
**Objective:** Validate multi-frame tracking accuracy

**Test Cases:**
```rust
#[test]
fn test_single_object_tracking() {
    let mut tracker = MultiFrameTracker::new(Duration::from_secs(10), 50.0);

    // Simulate object moving linearly
    for i in 0..10 {
        let frame = IrSensorFrame {
            centroid_x: 100.0 + i as f64 * 50.0,  // Moving right
            centroid_y: 100.0 + i as f64 * 20.0,  // Moving up
            hotspot_count: 1,
            timestamp: SystemTime::now(),
            // ...
        };

        let updates = tracker.update(&frame).unwrap();

        if i >= 3 {
            // After 3 frames, should have good prediction
            assert_eq!(updates.len(), 1);
            let prediction_error = updates[0].compute_error();
            assert!(prediction_error < 10.0, "Prediction error too high");
        }
    }
}

#[test]
fn test_motion_consistency_metric() {
    let mut adapter = TrackingLayerAdapter::new_tranche1_with_tracking(900).unwrap();

    // Track object over 20 frames
    for i in 0..20 {
        let frame = create_moving_object_frame(i);
        let detection = adapter.ingest_ir_frame_with_tracking(1, &frame).unwrap();

        if i >= 5 {
            // After warmup, motion should be consistent
            let features = adapter.extract_ir_features(&frame).unwrap();
            let consistency = features[9];  // Motion consistency feature

            assert!(consistency > 0.7, "Motion consistency too low: {}", consistency);
        }
    }
}

#[test]
fn test_maneuvering_target_detection() {
    let mut adapter = TrackingLayerAdapter::new_tranche1_with_tracking(900).unwrap();

    // Simulate hypersonic glide vehicle with maneuvers
    for i in 0..20 {
        let frame = if i == 10 {
            // Sudden maneuver at frame 10
            create_maneuvering_frame(i, high_acceleration=true)
        } else {
            create_linear_motion_frame(i)
        };

        let detection = adapter.ingest_ir_frame_with_tracking(1, &frame).unwrap();

        if i == 11 {
            // After maneuver, consistency should drop
            let features = adapter.extract_ir_features(&frame).unwrap();
            let consistency = features[9];

            assert!(consistency < 0.5, "Should detect maneuver (low consistency)");
        }
    }
}
```

**Files to Create:**
- `tests/pwsa_frame_tracking_test.rs` (NEW)

**Estimated Time:** 2-3 hours

---

### Task 3.7: Enhance IrSensorFrame Structure (Optional)
**Objective:** Add pixel data support for full tracking

**Enhancement:**
```rust
pub struct IrSensorFrame {
    // Existing fields
    pub sv_id: u32,
    pub timestamp: SystemTime,
    pub width: u32,
    pub height: u32,
    // ...

    /// NEW: Actual pixel data (optional for full tracking)
    pub pixels: Option<Array2<u16>>,  // Raw intensity values

    /// NEW: Detected hotspot positions
    pub hotspot_positions: Vec<(f64, f64)>,  // Pixel coordinates

    /// NEW: Object tracking IDs (if tracker active)
    pub tracked_objects: Vec<TrackedObject>,
}

pub struct TrackedObject {
    pub track_id: Uuid,
    pub position: (f64, f64),
    pub velocity: (f64, f64),
    pub class_hint: Option<ThreatClass>,
}
```

**Note:** This is optional. Basic tracking works with centroid only.

**Files to Modify:**
- `src/pwsa/satellite_adapters.rs` (enhance IrSensorFrame)

**Estimated Time:** 1-2 hours (if needed)

---

### Task 3.8: Benchmark Tracking Performance Impact
**Objective:** Measure latency overhead of tracking

**Benchmarks:**
```rust
#[bench]
fn bench_tracking_adapter_without_tracker(b: &mut Bencher) {
    let mut adapter = TrackingLayerAdapter::new_tranche1(900).unwrap();
    let frame = generate_test_frame();

    b.iter(|| {
        adapter.ingest_ir_frame(1, black_box(&frame))
    });
}

#[bench]
fn bench_tracking_adapter_with_tracker(b: &mut Bencher) {
    let mut adapter = TrackingLayerAdapter::new_tranche1_with_tracking(900).unwrap();
    let frame = generate_test_frame();

    b.iter(|| {
        adapter.ingest_ir_frame_with_tracking(1, black_box(&frame))
    });
}

// Expected result: +30-50μs overhead (acceptable)
```

**Files to Create:**
- Add to `benches/pwsa_benchmarks.rs`

**Estimated Time:** 1 hour

---

### **Enhancement 3 Total Deliverables:**
- [x] MultiFrameTracker architecture designed
- [x] Kalman filter implemented
- [x] Data association working (Hungarian algorithm)
- [x] Motion consistency metric computing from real tracks
- [x] Integrated into TrackingLayerAdapter
- [x] Comprehensive tests (4+ test cases)
- [x] Performance benchmarked (<50μs overhead)

**Total Time for Enhancement 3:** 1-2 days
**Git Commits:** 3-4 (tracker, integration, tests, benchmarks)

---

## OVERALL ENHANCEMENT PLAN

### Phase Approach

#### Phase 2.1: ML Threat Classifier (Days 1-3)
**Priority:** 1 (Most impactful)
**Effort:** 2-3 days
**Dependencies:** Training data generation
**Deliverables:**
- Active inference classifier
- Trained model (>90% accuracy)
- Integration complete
- Tests passing

#### Phase 2.2: Spatial Entropy (Day 4)
**Priority:** 2 (Quick win)
**Effort:** 4-6 hours
**Dependencies:** None
**Deliverables:**
- Shannon entropy computation
- Validation tests
- Integrated and verified

#### Phase 2.3: Frame Tracking (Days 5-7)
**Priority:** 3 (Temporal enhancement)
**Effort:** 1-2 days
**Dependencies:** Kalman filter, data association
**Deliverables:**
- Multi-frame tracker
- Motion consistency from tracking
- Tests and benchmarks

---

## TASK CHECKLIST (ALL 3 ENHANCEMENTS)

### Enhancement 1: ML Threat Classifier (7 tasks)
- [ ] Task 1.1: Design active inference architecture (6-8h)
- [ ] Task 1.2: Collect training data (4-6h)
- [ ] Task 1.3: Implement neural network (8-10h)
- [ ] Task 1.4: Train and validate model (6-8h)
- [ ] Task 1.5: Integrate into fusion platform (3-4h)
- [ ] Task 1.6: Create ML tests (3-4h)
- [ ] Task 1.7: Benchmark performance (2-3h)

**Subtotal:** 32-43 hours (2-3 days)

---

### Enhancement 2: Spatial Entropy (3 tasks)
- [ ] Task 2.1: Implement histogram computation (2-3h)
- [ ] Task 2.2: Implement Shannon entropy (1-2h)
- [ ] Task 2.3: Add validation tests (1-2h)

**Subtotal:** 4-7 hours (0.5-1 day)

---

### Enhancement 3: Frame Tracking (5 tasks)
- [ ] Task 3.1: Design multi-frame tracker (4-6h)
- [ ] Task 3.2: Implement Kalman filter (4-6h)
- [ ] Task 3.3: Implement data association (3-4h)
- [ ] Task 3.4: Implement motion consistency (1-2h)
- [ ] Task 3.5: Integrate into adapter (2-3h)
- [ ] Task 3.6: Create tracking tests (2-3h)
- [ ] Task 3.7: Enhance IrSensorFrame (optional) (1-2h)
- [ ] Task 3.8: Benchmark performance (1h)

**Subtotal:** 17-27 hours (1-2 days)

---

## GRAND TOTAL

**Total Tasks:** 15 tasks (7 + 3 + 5)
**Total Effort:** 53-77 hours
**Calendar Time:** 5-7 days (full-time)
**Or:** 10-14 days (part-time)

---

## SUCCESS CRITERIA

### Performance Targets
- [ ] Overall accuracy improvement: +20-30%
- [ ] Hypersonic detection precision: >95%
- [ ] Total latency impact: <+200μs
- [ ] Final latency: <1.1ms (still excellent)

### Constitutional Compliance
- [ ] Article II: Full temporal pattern compliance
- [ ] Article IV: Full variational inference compliance
- [ ] All other articles: Maintained
- [ ] Zero new violations introduced

### Code Quality
- [ ] Test coverage: Maintain >90%
- [ ] All tests passing
- [ ] Benchmarks show improvements
- [ ] Documentation updated

---

## RISK MITIGATION

### Risk 1: ML Training Fails to Converge
**Mitigation:**
- Keep heuristic as fallback
- Use transfer learning from pre-trained model
- Increase training data size

**Recovery:** Revert to v1.0 heuristic (already working)

### Risk 2: Tracking Adds Too Much Latency
**Mitigation:**
- Profile and optimize critical paths
- Use simpler tracker (centroid-only)
- GPU-accelerate Kalman filter

**Recovery:** Disable tracking, keep fixed 0.8 (current state)

### Risk 3: Integration Breaks Existing Tests
**Mitigation:**
- Maintain backward compatibility
- Add feature flags (enable_ml_classifier, enable_tracking)
- Comprehensive regression testing

**Recovery:** Feature flag to disable enhancements

---

## INTEGRATION STRATEGY

### Backward Compatibility
```rust
pub enum ClassifierMode {
    Heuristic,      // v1.0 (always available)
    ML,             // v2.0 (optional)
    Hybrid,         // ML with heuristic fallback
}

impl TrackingLayerAdapter {
    pub fn with_classifier_mode(mut self, mode: ClassifierMode) -> Self {
        self.classifier_mode = mode;
        self
    }
}
```

### Feature Flags
```toml
[features]
ml_classifier = ["candle-nn", "dep:pathfinding"]
frame_tracking = ["dep:pathfinding"]
enhanced = ["ml_classifier", "frame_tracking"]
```

### Rollback Plan
- v1.0 heuristics always available
- Can disable enhancements via config
- No breaking changes to API

---

## DEPENDENCIES TO ADD

```toml
[dependencies]
# For ML threat classifier
candle-nn = "0.9"  # Already present ✅
candle-core = { version = "0.9", features = ["cuda"] }  # Already present ✅

# For frame tracking
pathfinding = "4.0"  # Hungarian algorithm
nalgebra = "0.32"    # Already present ✅ (matrix operations)

# Optional: For advanced tracking
opencv = { version = "0.88", optional = true }  # If pixel-level tracking needed
```

---

## TESTING STRATEGY

### Unit Tests (Per Enhancement)
- Enhancement 1: 5-7 tests (ML classifier)
- Enhancement 2: 4-5 tests (spatial entropy)
- Enhancement 3: 4-6 tests (frame tracking)

**Total New Tests:** 13-18

### Integration Tests
- [ ] All 3 enhancements working together
- [ ] Latency still <1.1ms
- [ ] Accuracy improvement validated
- [ ] No regression in existing functionality

### Performance Tests
- [ ] Benchmark each enhancement individually
- [ ] Benchmark combined impact
- [ ] Validate <1.1ms target met

---

## DOCUMENTATION REQUIREMENTS

### Code Documentation
- [ ] Rustdoc for all new modules
- [ ] Inline comments for complex algorithms
- [ ] Update existing module docs

### Vault Documentation
- [ ] Update STATUS-DASHBOARD (v2.0 progress)
- [ ] Create ENHANCEMENTS-PROGRESS-TRACKER.md
- [ ] Update Constitutional-Compliance-Matrix.md
- [ ] Update Performance-Benchmarking-Report.md

### SBIR Documentation (If Phase II Continuation)
- [ ] Technical progress report
- [ ] Updated architecture diagrams
- [ ] Performance improvement analysis

---

## GOVERNANCE VALIDATION

### Pre-Enhancement Checklist
- [ ] Review constitutional requirements
- [ ] Verify no violations in design
- [ ] Confirm backward compatibility
- [ ] Plan rollback strategy

### During Enhancement
- [ ] Compile and test frequently
- [ ] Commit after each task
- [ ] Update vault progress
- [ ] Monitor performance impact

### Post-Enhancement Validation
- [ ] Run full test suite
- [ ] Benchmark performance
- [ ] Verify constitutional compliance
- [ ] Update all documentation
- [ ] Commit and push to GitHub

---

## TIMELINE

### Recommended Schedule (Full-Time)

**Day 1:**
- Morning: Task 1.1-1.2 (design + data)
- Afternoon: Task 1.3 (neural network)

**Day 2:**
- Morning: Task 1.4 (training)
- Afternoon: Task 1.5 (integration)

**Day 3:**
- Morning: Task 1.6-1.7 (tests + benchmarks)
- Afternoon: Task 2.1-2.3 (spatial entropy - complete)

**Day 4:**
- Morning: Task 3.1-3.2 (tracker design + Kalman)
- Afternoon: Task 3.3 (data association)

**Day 5:**
- Morning: Task 3.4-3.5 (motion metric + integration)
- Afternoon: Task 3.6-3.8 (tests + benchmarks)

**Day 6-7:**
- Integration testing
- Documentation updates
- Performance validation
- Vault updates

---

## DELIVERABLES CHECKLIST

### Code Deliverables
- [ ] `src/pwsa/active_inference_classifier.rs` (NEW)
- [ ] `src/pwsa/ml_threat_classifier.rs` (NEW)
- [ ] `src/pwsa/multi_frame_tracker.rs` (NEW)
- [ ] `src/pwsa/kalman_filter.rs` (NEW)
- [ ] Updated `src/pwsa/satellite_adapters.rs`
- [ ] Updated `src/pwsa/mod.rs`

### Test Deliverables
- [ ] `tests/pwsa_ml_classifier_test.rs` (NEW)
- [ ] `tests/pwsa_frame_tracking_test.rs` (NEW)
- [ ] Updated `tests/pwsa_adapters_test.rs`

### Benchmark Deliverables
- [ ] `benches/threat_classifier_comparison.rs` (NEW)
- [ ] Updated `benches/pwsa_benchmarks.rs`

### Documentation Deliverables
- [ ] Updated Performance-Benchmarking-Report.md
- [ ] Updated Constitutional-Compliance-Matrix.md
- [ ] Enhancement progress tracker
- [ ] Vault status updates

---

## NEXT STEPS

### Immediate (Post-SBIR Submission)
1. Review this TODO list
2. Prioritize based on Phase II timeline
3. Set up development branch (`feature/v2-enhancements`)
4. Begin with Enhancement 1 (highest impact)

### Sequencing
1. **First:** ML Classifier (biggest accuracy gain)
2. **Second:** Spatial Entropy (quick win)
3. **Third:** Frame Tracking (most complex)

### Coordination
- Keep v1.0 stable (master branch)
- Develop enhancements in feature branches
- Merge only after full validation
- Maintain backward compatibility

---

**Status:** READY TO EXECUTE (Post-SBIR Award)
**Next:** Begin when Phase II funded
**Classification:** UNCLASSIFIED//FOR OFFICIAL USE ONLY
