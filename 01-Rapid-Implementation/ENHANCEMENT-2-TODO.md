# ENHANCEMENT 2: SPATIAL ENTROPY WITH PIXEL PROCESSING
## Complete Implementation TODO

**Created:** January 9, 2025
**Priority:** HIGH (Technical Debt Item #2)
**Estimated Effort:** 6-9 hours (1 day)
**Target:** Operational readiness for real SDA sensor data

---

## STRATEGIC OVERVIEW

### What We're Building

**Current State:**
```rust
fn compute_spatial_entropy(&self, _frame: &IrSensorFrame) -> f64 {
    // Placeholder: compute Shannon entropy of intensity histogram
    0.5  // Fixed value
}
```

**Target State:**
```rust
fn compute_spatial_entropy(&self, frame: &IrSensorFrame) -> f64 {
    // Multi-tier approach:
    // 1. BEST: Compute from raw pixels (operational mode)
    // 2. GOOD: Use pre-computed histogram
    // 3. FALLBACK: Approximate from metadata (current demos)
}

pub struct IrSensorFrame {
    // NEW: Raw pixel support for operational deployment
    pub pixels: Option<Array2<u16>>,  // 1024×1024 IR sensor data
    pub hotspot_positions: Vec<(f64, f64)>,
    pub intensity_histogram: Option<Vec<usize>>,
    pub spatial_entropy: Option<f64>,
    // ... existing metadata fields (backward compatible)
}
```

### Why This Matters

**Operational Readiness:**
- Platform ready for **real SDA sensor data** on Day 1 of Phase II
- No architectural changes needed when switching from demo to operational

**SBIR Proposal:**
- Shows **production-level thinking** (not just prototype)
- Demonstrates **operational awareness** (understands real data format)
- Stronger technical narrative (+7 points estimated)

**Article II Compliance:**
- **Real Shannon entropy** from spatial patterns
- Enhanced neuromorphic spatial analysis
- Information-theoretic correctness

---

## TASK BREAKDOWN (10 Tasks, 6-9 Hours)

### TASK 1: Enhance IrSensorFrame Data Structure
**Effort:** 1 hour
**Priority:** CRITICAL (foundation for all other tasks)
**File:** `src/pwsa/satellite_adapters.rs`

**Implementation:**
```rust
/// IR sensor frame structure
#[derive(Debug, Clone)]
pub struct IrSensorFrame {
    // === SENSOR IDENTIFICATION ===
    pub sv_id: u32,
    pub timestamp: SystemTime,
    pub width: u32,
    pub height: u32,

    // === RAW PIXEL DATA (NEW - Week 2 Enhancement) ===
    /// Raw pixel intensities (width × height)
    /// None = metadata-only mode (current demos)
    /// Some = full pixel processing (operational mode)
    pub pixels: Option<Array2<u16>>,

    // === DERIVED SPATIAL FEATURES (NEW) ===
    /// Detected hotspot positions [(x, y), ...]
    /// Computed from pixels if available, otherwise empty
    pub hotspot_positions: Vec<(f64, f64)>,

    /// Intensity histogram (16 bins)
    /// Computed from pixels, or None if metadata-only
    pub intensity_histogram: Option<Vec<usize>>,

    /// Spatial entropy [0, 1]
    /// Computed from histogram, or None if metadata-only
    pub spatial_entropy: Option<f64>,

    // === COMPUTED METADATA (Backward Compatible) ===
    /// Computed from pixels if available, otherwise provided
    pub max_intensity: f64,
    pub background_level: f64,
    pub hotspot_count: u32,

    // === EXISTING FIELDS (Unchanged) ===
    pub centroid_x: f64,
    pub centroid_y: f64,
    pub velocity_estimate_mps: f64,
    pub acceleration_estimate: f64,
    pub swir_band_ratio: f64,
    pub thermal_signature: f64,
    pub geolocation: (f64, f64),
}
```

**Checklist:**
- [ ] Add `pixels: Option<Array2<u16>>` field
- [ ] Add `hotspot_positions: Vec<(f64, f64)>` field
- [ ] Add `intensity_histogram: Option<Vec<usize>>` field
- [ ] Add `spatial_entropy: Option<f64>` field
- [ ] Update existing constructors to set new fields to None/empty
- [ ] Ensure backward compatibility (existing demos still work)

**Git Checkpoint:** Commit after this task (structure changes only)

---

### TASK 2: Implement Background Level Estimation
**Effort:** 30 minutes
**Priority:** HIGH
**File:** `src/pwsa/satellite_adapters.rs` (add helper function)

**Implementation:**
```rust
/// Compute background intensity level from pixel data
///
/// Uses 25th percentile (robust to hotspots/outliers)
fn compute_background_level(pixels: &Array2<u16>) -> f64 {
    // Collect all pixel values
    let mut pixel_vec: Vec<u16> = pixels.iter().copied().collect();

    // Sort for percentile computation
    pixel_vec.sort_unstable();

    // Use 25th percentile as background
    // (Robust to bright hotspots which would skew mean)
    let idx = pixel_vec.len() / 4;

    pixel_vec.get(idx).copied().unwrap_or(0) as f64
}
```

**Test Case:**
```rust
#[test]
fn test_background_level_estimation() {
    // Create test pixel array
    let mut pixels = Array2::from_elem((100, 100), 100u16);  // 100 = background

    // Add some hotspots (should not affect background estimate)
    pixels[[50, 50]] = 5000;  // Bright hotspot

    let background = compute_background_level(&pixels);

    // Should be ~100 (ignoring hotspot)
    assert!((background - 100.0).abs() < 10.0);
}
```

**Checklist:**
- [ ] Implement `compute_background_level()` function
- [ ] Use 25th percentile (not mean - more robust)
- [ ] Add unit test
- [ ] Validate with synthetic pixel data

---

### TASK 3: Implement Hotspot Detection Algorithm
**Effort:** 1.5 hours
**Priority:** HIGH
**File:** `src/pwsa/satellite_adapters.rs`

**Implementation:**
```rust
/// Detect hotspots from pixel data
///
/// Uses adaptive thresholding (3× background level)
/// Returns list of hotspot positions
fn detect_hotspots(pixels: &Array2<u16>, background_level: f64) -> Vec<(f64, f64)> {
    let threshold = (background_level * 3.0) as u16;  // 3× background = hotspot

    let mut hotspot_pixels = Vec::new();

    // Find all pixels above threshold
    for ((y, x), &intensity) in pixels.indexed_iter() {
        if intensity > threshold {
            hotspot_pixels.push((x as f64, y as f64));
        }
    }

    // Cluster nearby hotspots (simple connected components)
    if hotspot_pixels.is_empty() {
        return Vec::new();
    }

    // Simple clustering: Group pixels within 10-pixel radius
    cluster_hotspots(hotspot_pixels, 10.0)
}

/// Cluster hotspot pixels into centroids
fn cluster_hotspots(pixels: Vec<(f64, f64)>, radius: f64) -> Vec<(f64, f64)> {
    let mut clusters = Vec::new();
    let mut remaining = pixels;

    while !remaining.is_empty() {
        // Start new cluster with first pixel
        let seed = remaining.remove(0);
        let mut cluster_pixels = vec![seed];

        // Find all pixels within radius
        let mut i = 0;
        while i < remaining.len() {
            let pixel = remaining[i];
            let dist = ((pixel.0 - seed.0).powi(2) + (pixel.1 - seed.1).powi(2)).sqrt();

            if dist <= radius {
                cluster_pixels.push(remaining.remove(i));
            } else {
                i += 1;
            }
        }

        // Compute cluster centroid
        let centroid_x = cluster_pixels.iter().map(|(x, _)| x).sum::<f64>() / cluster_pixels.len() as f64;
        let centroid_y = cluster_pixels.iter().map(|(_, y)| y).sum::<f64>() / cluster_pixels.len() as f64;

        clusters.push((centroid_x, centroid_y));
    }

    clusters
}
```

**Test Case:**
```rust
#[test]
fn test_hotspot_detection() {
    // Create pixel array with 2 hotspots
    let mut pixels = Array2::from_elem((1024, 1024), 100u16);

    // Hotspot 1: Centered at (200, 200)
    for y in 195..205 {
        for x in 195..205 {
            pixels[[y, x]] = 4000;
        }
    }

    // Hotspot 2: Centered at (800, 800)
    for y in 795..805 {
        for x in 795..805 {
            pixels[[y, x]] = 3500;
        }
    }

    let hotspots = detect_hotspots(&pixels, 100.0);

    // Should detect 2 hotspots
    assert_eq!(hotspots.len(), 2);

    // Check positions are approximately correct
    let has_hotspot_near_200 = hotspots.iter().any(|(x, y)| {
        (*x - 200.0).abs() < 20.0 && (*y - 200.0).abs() < 20.0
    });
    let has_hotspot_near_800 = hotspots.iter().any(|(x, y)| {
        (*x - 800.0).abs() < 20.0 && (*y - 800.0).abs() < 20.0
    });

    assert!(has_hotspot_near_200);
    assert!(has_hotspot_near_800);
}
```

**Checklist:**
- [ ] Implement `detect_hotspots()` function
- [ ] Implement `cluster_hotspots()` function
- [ ] Use adaptive thresholding (3× background)
- [ ] Add unit tests (2-3 test cases)
- [ ] Test with synthetic pixel arrays

---

### TASK 4: Implement Intensity Histogram Computation
**Effort:** 45 minutes
**Priority:** HIGH
**File:** `src/pwsa/satellite_adapters.rs`

**Implementation:**
```rust
/// Compute intensity histogram from pixel data
///
/// # Arguments
/// * `pixels` - Raw pixel array
/// * `n_bins` - Number of histogram bins (default: 16)
fn compute_intensity_histogram(pixels: &Array2<u16>, n_bins: usize) -> Vec<usize> {
    let min_val = *pixels.iter().min().unwrap_or(&0) as f64;
    let max_val = *pixels.iter().max().unwrap_or(&0) as f64;
    let range = max_val - min_val;

    let mut histogram = vec![0; n_bins];

    if range == 0.0 {
        // All pixels same intensity
        histogram[0] = pixels.len();
        return histogram;
    }

    // Bin each pixel
    for &pixel in pixels.iter() {
        let normalized = (pixel as f64 - min_val) / range;
        let bin = (normalized * (n_bins - 1) as f64) as usize;
        histogram[bin.min(n_bins - 1)] += 1;
    }

    histogram
}
```

**Test Case:**
```rust
#[test]
fn test_intensity_histogram() {
    // Create gradient image (for testing histogram)
    let mut pixels = Array2::zeros((100, 100));

    for ((y, x), pixel) in pixels.indexed_iter_mut() {
        *pixel = (x * 10) as u16;  // Gradient 0-990
    }

    let histogram = compute_intensity_histogram(&pixels, 10);

    // Should have roughly uniform distribution
    assert_eq!(histogram.len(), 10);

    // Each bin should have ~1000 pixels (100×100 / 10 bins)
    for count in histogram {
        assert!(count > 800 && count < 1200, "Bin count {} outside expected range", count);
    }
}
```

**Checklist:**
- [ ] Implement `compute_intensity_histogram()` function
- [ ] Handle edge cases (all pixels same, empty array)
- [ ] Add unit test
- [ ] Validate with gradient test image

---

### TASK 5: Implement Shannon Entropy Calculation
**Effort:** 30 minutes
**Priority:** HIGH
**File:** `src/pwsa/satellite_adapters.rs`

**Implementation:**
```rust
/// Compute Shannon entropy from intensity histogram
///
/// Returns normalized entropy [0, 1]
/// - 0.0 = Completely ordered (single intensity)
/// - 1.0 = Maximum disorder (uniform distribution)
fn compute_shannon_entropy(histogram: &[usize]) -> f64 {
    let total_pixels: usize = histogram.iter().sum();

    if total_pixels == 0 {
        return 0.0;
    }

    // Shannon entropy: H = -Σ p(i) log2(p(i))
    let mut entropy = 0.0;

    for &count in histogram {
        if count > 0 {
            let p = count as f64 / total_pixels as f64;
            entropy -= p * p.log2();
        }
    }

    // Normalize by maximum possible entropy
    let max_entropy = (histogram.len() as f64).log2();

    if max_entropy > 0.0 {
        entropy / max_entropy  // Returns [0, 1]
    } else {
        0.0
    }
}
```

**Test Cases:**
```rust
#[test]
fn test_shannon_entropy_uniform() {
    // Uniform distribution = maximum entropy
    let histogram = vec![1000; 16];  // All bins equal

    let entropy = compute_shannon_entropy(&histogram);

    // Should be close to 1.0 (maximum)
    assert!(entropy > 0.99, "Uniform distribution entropy {} should be ~1.0", entropy);
}

#[test]
fn test_shannon_entropy_concentrated() {
    // Single bin = minimum entropy
    let mut histogram = vec![0; 16];
    histogram[0] = 10000;  // All pixels in one bin

    let entropy = compute_shannon_entropy(&histogram);

    // Should be 0.0 (no disorder)
    assert!(entropy < 0.01, "Concentrated distribution entropy {} should be ~0.0", entropy);
}

#[test]
fn test_shannon_entropy_range() {
    // Random distributions should be in [0, 1]
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let histogram: Vec<usize> = (0..16).map(|_| rng.gen_range(0..1000)).collect();
        let entropy = compute_shannon_entropy(&histogram);

        assert!(entropy >= 0.0 && entropy <= 1.0,
            "Entropy {} out of range [0,1]", entropy);
    }
}
```

**Checklist:**
- [ ] Implement `compute_shannon_entropy()` function
- [ ] Use log2 for bits (information theory standard)
- [ ] Normalize to [0, 1] range
- [ ] Add 3 unit tests (uniform, concentrated, range)
- [ ] Validate information-theoretic correctness

**Git Checkpoint:** Commit after Tasks 2-5 (pixel processing algorithms)

---

### TASK 6: Implement from_pixels() Constructor
**Effort:** 1.5 hours
**Priority:** HIGH
**File:** `src/pwsa/satellite_adapters.rs`

**Implementation:**
```rust
impl IrSensorFrame {
    /// Create IrSensorFrame from raw pixel data (operational mode)
    ///
    /// Automatically computes all metadata from pixels:
    /// - Background level, max intensity
    /// - Hotspot detection and positions
    /// - Intensity histogram
    /// - Spatial entropy
    /// - Centroid position
    ///
    /// # Arguments
    /// * `sv_id` - Satellite vehicle ID
    /// * `pixels` - Raw IR sensor pixel array (width × height)
    /// * `geolocation` - (latitude, longitude) of sensor footprint
    /// * `velocity_estimate` - Target velocity (m/s)
    /// * `acceleration_estimate` - Target acceleration (m/s²)
    pub fn from_pixels(
        sv_id: u32,
        pixels: Array2<u16>,
        geolocation: (f64, f64),
        velocity_estimate_mps: f64,
        acceleration_estimate: f64,
    ) -> Result<Self> {
        let (height, width) = pixels.dim();

        // Compute background level
        let background_level = compute_background_level(&pixels);

        // Detect hotspots
        let hotspot_positions = detect_hotspots(&pixels, background_level);
        let hotspot_count = hotspot_positions.len() as u32;

        // Compute max intensity
        let max_intensity = *pixels.iter().max().unwrap_or(&0) as f64;

        // Compute centroid (weighted by intensity)
        let (centroid_x, centroid_y) = compute_weighted_centroid(&pixels);

        // Compute intensity histogram
        let intensity_histogram = compute_intensity_histogram(&pixels, 16);

        // Compute Shannon entropy
        let spatial_entropy = compute_shannon_entropy(&intensity_histogram);

        // Estimate thermal signature (from intensity distribution)
        let thermal_signature = estimate_thermal_signature(&pixels, background_level);

        // SWIR band ratio (would need multi-band data; use default)
        let swir_band_ratio = 1.0;

        Ok(Self {
            sv_id,
            timestamp: SystemTime::now(),
            width: width as u32,
            height: height as u32,
            pixels: Some(pixels),
            hotspot_positions,
            intensity_histogram: Some(intensity_histogram),
            spatial_entropy: Some(spatial_entropy),
            max_intensity,
            background_level,
            hotspot_count,
            centroid_x,
            centroid_y,
            velocity_estimate_mps,
            acceleration_estimate,
            swir_band_ratio,
            thermal_signature,
            geolocation,
        })
    }

    /// Create from metadata (demo mode - backward compatible)
    pub fn from_metadata(
        sv_id: u32,
        width: u32,
        height: u32,
        max_intensity: f64,
        background_level: f64,
        hotspot_count: u32,
        centroid_x: f64,
        centroid_y: f64,
        velocity_estimate_mps: f64,
        acceleration_estimate: f64,
        swir_band_ratio: f64,
        thermal_signature: f64,
        geolocation: (f64, f64),
    ) -> Self {
        // Original constructor (backward compatible)
        Self {
            sv_id,
            timestamp: SystemTime::now(),
            width,
            height,
            pixels: None,  // No pixel data
            hotspot_positions: Vec::new(),
            intensity_histogram: None,
            spatial_entropy: None,
            max_intensity,
            background_level,
            hotspot_count,
            centroid_x,
            centroid_y,
            velocity_estimate_mps,
            acceleration_estimate,
            swir_band_ratio,
            thermal_signature,
            geolocation,
        }
    }
}
```

**Checklist:**
- [ ] Implement `from_pixels()` constructor
- [ ] Implement `from_metadata()` constructor (backward compat)
- [ ] Call all helper functions (background, hotspots, histogram, entropy)
- [ ] Ensure both constructors work
- [ ] Update existing code to use `from_metadata()`

---

### TASK 7: Implement Weighted Centroid Computation
**Effort:** 30 minutes
**Priority:** MEDIUM
**File:** `src/pwsa/satellite_adapters.rs`

**Implementation:**
```rust
/// Compute intensity-weighted centroid
fn compute_weighted_centroid(pixels: &Array2<u16>) -> (f64, f64) {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_intensity = 0.0;

    for ((y, x), &intensity) in pixels.indexed_iter() {
        let weight = intensity as f64;
        sum_x += x as f64 * weight;
        sum_y += y as f64 * weight;
        sum_intensity += weight;
    }

    if sum_intensity > 0.0 {
        (sum_x / sum_intensity, sum_y / sum_intensity)
    } else {
        (0.0, 0.0)
    }
}
```

**Checklist:**
- [ ] Implement `compute_weighted_centroid()` function
- [ ] Handle zero-intensity case
- [ ] Add simple test

---

### TASK 8: Implement Thermal Signature Estimation
**Effort:** 30 minutes
**Priority:** MEDIUM
**File:** `src/pwsa/satellite_adapters.rs`

**Implementation:**
```rust
/// Estimate thermal signature from pixel intensity
fn estimate_thermal_signature(pixels: &Array2<u16>, background_level: f64) -> f64 {
    // Thermal signature = ratio of bright pixels to background

    let threshold = (background_level * 2.0) as u16;
    let bright_pixels = pixels.iter().filter(|&&p| p > threshold).count();
    let total_pixels = pixels.len();

    if total_pixels > 0 {
        (bright_pixels as f64 / total_pixels as f64).min(1.0)
    } else {
        0.0
    }
}
```

**Checklist:**
- [ ] Implement `estimate_thermal_signature()` function
- [ ] Return [0, 1] normalized value
- [ ] Add simple test

---

### TASK 9: Update compute_spatial_entropy() Method
**Effort:** 30 minutes
**Priority:** CRITICAL (this is the main fix)
**File:** `src/pwsa/satellite_adapters.rs` (line 289-293)

**Implementation:**
```rust
fn compute_spatial_entropy(&self, frame: &IrSensorFrame) -> f64 {
    // Multi-tier approach (best to worst):

    // TIER 1: Compute from raw pixels (operational mode)
    if let Some(ref pixels) = frame.pixels {
        let histogram = compute_intensity_histogram(pixels, 16);
        return compute_shannon_entropy(&histogram);
    }

    // TIER 2: Use pre-computed histogram (if available)
    if let Some(ref histogram) = frame.intensity_histogram {
        return compute_shannon_entropy(histogram);
    }

    // TIER 3: Use pre-computed entropy (if available)
    if let Some(entropy) = frame.spatial_entropy {
        return entropy;
    }

    // TIER 4: Statistical approximation from metadata (fallback for demos)
    // This maintains backward compatibility with current demos
    approximate_entropy_from_metadata(frame)
}

/// Approximate spatial entropy from metadata (fallback)
fn approximate_entropy_from_metadata(frame: &IrSensorFrame) -> f64 {
    // Use hotspot count as proxy for entropy
    // More hotspots = higher entropy (more dispersed)
    // Fewer hotspots = lower entropy (more concentrated)

    if frame.hotspot_count == 0 {
        return 0.5;  // Neutral (no clear signal)
    }

    if frame.hotspot_count == 1 {
        return 0.2;  // Low (concentrated threat)
    }

    // Multiple hotspots: Higher entropy
    let normalized_count = (frame.hotspot_count as f64 / 10.0).min(1.0);
    0.2 + normalized_count * 0.6  // Range: 0.2-0.8
}
```

**Checklist:**
- [ ] Replace placeholder (fixed 0.5) with multi-tier implementation
- [ ] Implement 4-tier fallback logic
- [ ] Implement `approximate_entropy_from_metadata()` (backward compat)
- [ ] Test all 4 tiers work correctly
- [ ] Ensure existing demos still work (use Tier 4)

**Git Checkpoint:** Commit after this task (core enhancement complete)

---

### TASK 10: Add Comprehensive Integration Tests
**Effort:** 1.5 hours
**Priority:** HIGH
**File:** `tests/pwsa_spatial_entropy_test.rs` (NEW)

**Test Cases:**
```rust
//! Spatial Entropy Enhancement Tests
//!
//! Validates real pixel processing and Shannon entropy computation

use prism_ai::pwsa::satellite_adapters::*;
use ndarray::Array2;

#[test]
fn test_from_pixels_constructor() {
    // Create synthetic pixel array (single hotspot)
    let mut pixels = Array2::from_elem((1024, 1024), 100u16);

    // Add hotspot at center
    for y in 500..524 {
        for x in 500..524 {
            pixels[[y, x]] = 4000;
        }
    }

    let frame = IrSensorFrame::from_pixels(
        1,  // sv_id
        pixels,
        (38.0, 127.0),  // geolocation
        1800.0,  // velocity
        45.0,    // acceleration
    ).unwrap();

    // Validate computed metadata
    assert_eq!(frame.width, 1024);
    assert_eq!(frame.height, 1024);
    assert_eq!(frame.max_intensity, 4000.0);
    assert!((frame.background_level - 100.0).abs() < 10.0);
    assert_eq!(frame.hotspot_count, 1);

    // Validate spatial entropy (should be low for single hotspot)
    assert!(frame.spatial_entropy.is_some());
    let entropy = frame.spatial_entropy.unwrap();
    assert!(entropy < 0.3, "Single hotspot should have low entropy, got {}", entropy);
}

#[test]
fn test_spatial_entropy_multiple_hotspots() {
    // Create pixel array with multiple dispersed hotspots
    let mut pixels = Array2::from_elem((1024, 1024), 100u16);

    // Add 5 hotspots at different locations
    let hotspot_locations = [(200, 200), (200, 800), (500, 500), (800, 200), (800, 800)];

    for (cx, cy) in hotspot_locations {
        for y in (cy - 10)..(cy + 10) {
            for x in (cx - 10)..(cx + 10) {
                pixels[[y, x]] = 3000;
            }
        }
    }

    let frame = IrSensorFrame::from_pixels(1, pixels, (38.0, 127.0), 500.0, 10.0).unwrap();

    // Should detect 5 hotspots
    assert!(frame.hotspot_count >= 4 && frame.hotspot_count <= 6,
        "Should detect ~5 hotspots, got {}", frame.hotspot_count);

    // Entropy should be medium-high (dispersed pattern)
    let entropy = frame.spatial_entropy.unwrap();
    assert!(entropy > 0.5, "Dispersed hotspots should have higher entropy, got {}", entropy);
}

#[test]
fn test_backward_compatibility_metadata_mode() {
    // Ensure metadata-only mode still works (existing demos)
    let frame = IrSensorFrame::from_metadata(
        1,      // sv_id
        1024,   // width
        1024,   // height
        3000.0, // max_intensity
        100.0,  // background_level
        2,      // hotspot_count
        512.0,  // centroid_x
        512.0,  // centroid_y
        1800.0, // velocity
        45.0,   // acceleration
        1.0,    // swir_band_ratio
        0.8,    // thermal_signature
        (38.0, 127.0),  // geolocation
    );

    // Should work without pixel data
    assert!(frame.pixels.is_none());
    assert!(frame.spatial_entropy.is_none());

    // Compute spatial entropy (should use Tier 4 approximation)
    let adapter = TrackingLayerAdapter::new_tranche1(900).unwrap();
    let entropy = adapter.compute_spatial_entropy(&frame);

    // Should return reasonable approximation
    assert!(entropy >= 0.0 && entropy <= 1.0);
}

#[test]
fn test_entropy_tiers() {
    let adapter = TrackingLayerAdapter::new_tranche1(900).unwrap();

    // Tier 1: From pixels
    let mut pixels = Array2::from_elem((100, 100), 100u16);
    pixels[[50, 50]] = 5000;  // Single hotspot
    let frame_pixels = IrSensorFrame::from_pixels(1, pixels, (38.0, 127.0), 1800.0, 45.0).unwrap();

    let entropy_tier1 = adapter.compute_spatial_entropy(&frame_pixels);
    assert!(entropy_tier1 < 0.3, "Single hotspot entropy should be low");

    // Tier 4: From metadata (no pixels)
    let frame_metadata = IrSensorFrame::from_metadata(
        1, 100, 100, 5000.0, 100.0, 1, 50.0, 50.0, 1800.0, 45.0, 1.0, 0.8, (38.0, 127.0)
    );

    let entropy_tier4 = adapter.compute_spatial_entropy(&frame_metadata);
    assert!(entropy_tier4 < 0.5, "Single hotspot metadata entropy should be lowish");

    // Both should indicate low entropy (single hotspot)
    println!("Tier 1 (pixels): {:.3}, Tier 4 (metadata): {:.3}", entropy_tier1, entropy_tier4);
}

#[test]
fn test_real_sda_data_format_ready() {
    // Validate we can accept real SDA sensor format
    // SDA IR sensors: 1024×1024, 16-bit unsigned integers

    let pixels = Array2::<u16>::zeros((1024, 1024));  // SDA format

    let frame = IrSensorFrame::from_pixels(
        42,  // SV-42
        pixels,
        (35.0, 125.0),
        0.0,  // No target detected
        0.0,
    );

    assert!(frame.is_ok(), "Should accept SDA pixel format");

    let frame = frame.unwrap();
    assert_eq!(frame.width, 1024);
    assert_eq!(frame.height, 1024);
    assert!(frame.pixels.is_some());

    // Platform is ready for real SDA data ✅
}
```

**Checklist:**
- [ ] Implement `from_pixels()` constructor
- [ ] Implement `from_metadata()` constructor
- [ ] Create 6 comprehensive tests
- [ ] Test backward compatibility
- [ ] Test real SDA format (1024×1024×u16)
- [ ] All tests passing

**File to Create:** `tests/pwsa_spatial_entropy_test.rs`

---

### TASK 11: Update Existing Demo to Support Both Modes
**Effort:** 45 minutes
**Priority:** MEDIUM
**File:** `examples/pwsa_demo.rs`

**Enhancement:**
```rust
// Add flag to generate with pixels (optional)
let use_pixel_mode = std::env::var("PIXEL_MODE").is_ok();

let tracking_frame = if use_pixel_mode {
    // Generate with full pixel data
    let pixels = generate_synthetic_ir_pixels();
    IrSensorFrame::from_pixels(
        tracking_sv,
        pixels,
        (38.0, 127.0),
        1800.0,
        45.0,
    ).unwrap()
} else {
    // Current metadata mode (unchanged)
    telemetry_gen.generate_ir_frame(tracking_sv, inject_threat)
};
```

**Helper Function:**
```rust
fn generate_synthetic_ir_pixels() -> Array2<u16> {
    let mut pixels = Array2::from_elem((1024, 1024), 100u16);

    // Add hotspot if threat
    if inject_threat {
        for y in 500..524 {
            for x in 500..524 {
                pixels[[y, x]] = 4000;
            }
        }
    }

    pixels
}
```

**Checklist:**
- [ ] Add environment variable flag (PIXEL_MODE)
- [ ] Implement `generate_synthetic_ir_pixels()` function
- [ ] Update demo to support both modes
- [ ] Test: `cargo run --example pwsa_demo --features pwsa` (metadata mode)
- [ ] Test: `PIXEL_MODE=1 cargo run --example pwsa_demo --features pwsa` (pixel mode)
- [ ] Verify both work correctly

---

### TASK 12: Test Compilation & Integration
**Effort:** 30 minutes
**Priority:** CRITICAL

**Commands:**
```bash
# Test library compilation
cargo build --features pwsa

# Test all PWSA tests
cargo test --features pwsa --lib

# Test new spatial entropy tests
cargo test --features pwsa --test pwsa_spatial_entropy_test

# Test demos still work
cargo run --example pwsa_demo --features pwsa
cargo run --example pwsa_streaming_demo --features pwsa

# Test pixel mode demo
PIXEL_MODE=1 cargo run --example pwsa_demo --features pwsa
```

**Checklist:**
- [ ] Library compiles without errors
- [ ] All existing tests still pass (no regression)
- [ ] New spatial entropy tests pass
- [ ] Metadata-only demo works (backward compat)
- [ ] Pixel mode demo works (new capability)
- [ ] No performance regression (<1ms fusion latency maintained)

**Git Checkpoint:** Commit after all tests passing

---

### TASK 13: Update Documentation
**Effort:** 1 hour
**Priority:** HIGH
**Files:** Multiple

**Documentation Updates:**

**1. Update TECHNICAL-DEBT-INVENTORY.md:**
- [ ] Mark Item #2 (Spatial Entropy) as "✅ IMPLEMENTED"
- [ ] Move from "Placeholder" to "Real Shannon entropy"

**2. Update Constitutional-Compliance-Matrix.md:**
- [ ] Update Article II row: Spatial entropy now REAL (not placeholder)
- [ ] Add note about pixel processing capability

**3. Create ENHANCEMENT-2-COMPLETION.md:**
- [ ] Implementation details
- [ ] Performance impact
- [ ] Article II compliance enhancement
- [ ] Operational readiness achieved

**4. Update STATUS-DASHBOARD.md:**
- [ ] Note Enhancement 2 complete
- [ ] Update code statistics
- [ ] Update vault file list

**Checklist:**
- [ ] All 4 documentation files updated
- [ ] Changes committed
- [ ] Pushed to GitHub

---

### TASK 14: Final Validation & Benchmarking
**Effort:** 1 hour
**Priority:** HIGH

**Validation Checks:**
```rust
#[test]
fn test_enhancement_2_performance_impact() {
    use std::time::Instant;

    let mut pixels = Array2::from_elem((1024, 1024), 100u16);
    // Add some hotspots
    for i in 0..5 {
        let cx = 200 + i * 150;
        let cy = 200 + i * 150;
        for y in (cy - 10)..(cy + 10) {
            for x in (cx - 10)..(cx + 10) {
                pixels[[y, x]] = 3000;
            }
        }
    }

    let start = Instant::now();

    let frame = IrSensorFrame::from_pixels(1, pixels, (38.0, 127.0), 1800.0, 45.0).unwrap();

    let elapsed = start.elapsed();

    // Should be fast (<50ms for 1024×1024 processing)
    assert!(elapsed.as_millis() < 50,
        "Pixel processing took {}ms, should be <50ms", elapsed.as_millis());

    // Verify spatial entropy was computed
    assert!(frame.spatial_entropy.is_some());
}
```

**Benchmarks to Add:**
```rust
// benches/pwsa_benchmarks.rs
#[bench]
fn bench_spatial_entropy_from_pixels(b: &mut Bencher) {
    let pixels = Array2::from_elem((1024, 1024), 100u16);

    b.iter(|| {
        let histogram = compute_intensity_histogram(black_box(&pixels), 16);
        compute_shannon_entropy(&histogram)
    });
}

// Expected: <10ms for 1024×1024 array
```

**Checklist:**
- [ ] Add performance test
- [ ] Run benchmark suite
- [ ] Validate <50ms overhead (acceptable)
- [ ] Document performance impact

---

## TASK SUMMARY CHECKLIST

### Core Implementation (Tasks 1-9)
- [ ] Task 1: Enhance IrSensorFrame structure (1h)
- [ ] Task 2: Background level estimation (30min)
- [ ] Task 3: Hotspot detection algorithm (1.5h)
- [ ] Task 4: Intensity histogram (45min)
- [ ] Task 5: Shannon entropy calculation (30min)
- [ ] Task 6: from_pixels() constructor (1.5h)
- [ ] Task 7: Weighted centroid (30min)
- [ ] Task 8: Thermal signature estimation (30min)
- [ ] Task 9: Update compute_spatial_entropy() (30min)

**Subtotal:** 7.5 hours

### Testing & Validation (Tasks 10-14)
- [ ] Task 10: Comprehensive integration tests (1.5h)
- [ ] Task 11: Update demo (optional pixel mode) (45min)
- [ ] Task 12: Test compilation & integration (30min)
- [ ] Task 13: Update documentation (1h)
- [ ] Task 14: Final validation & benchmarking (1h)

**Subtotal:** 4.75 hours

**Grand Total:** 12.25 hours (~1.5 days)

---

## EXPECTED OUTCOMES

### Technical Achievements
- ✅ Platform ready for real SDA pixel data (1024×1024×u16)
- ✅ Real Shannon entropy (not placeholder)
- ✅ Article II enhanced (spatial pattern analysis)
- ✅ Backward compatible (existing demos work)
- ✅ Multi-tier fallback (graceful degradation)

### Performance Impact
- Pixel processing: ~20-40ms (for 1024×1024 array)
- Overall fusion: 850μs → 890μs (+40μs, still <1ms)
- Acceptable overhead

### Operational Value
- **Immediate:** Platform accepts real sensor format
- **Phase II Month 1:** No changes needed for real data
- **SBIR Proposal:** "Operationally ready" claim validated

---

## SUCCESS CRITERIA

### Must Achieve (Before Committing)
- [ ] All tests passing (existing + new)
- [ ] Backward compatibility verified (demos work)
- [ ] Real pixel processing validated (1024×1024)
- [ ] Shannon entropy mathematically correct
- [ ] Performance acceptable (<50ms overhead)
- [ ] Documentation updated
- [ ] Code compiles cleanly
- [ ] Committed and pushed to GitHub

### Article II Compliance
- [ ] Spatial pattern analysis: ✅ REAL (not placeholder)
- [ ] Information-theoretic correctness: ✅ Shannon entropy
- [ ] Temporal patterns: ⚠️ Still placeholder (Enhancement 3)

**Overall Article II:** Improved from 9/10 to 9.5/10

---

## GOVERNANCE VALIDATION

### Pre-Implementation
- [x] Constitutional review (Article II enhancement)
- [x] No violations in design
- [x] Backward compatibility planned
- [x] Rollback strategy (multi-tier fallback)

### During Implementation
- [ ] Compile and test after each task
- [ ] Commit after major milestones (Tasks 5, 9, 14)
- [ ] Update vault progress
- [ ] Monitor performance impact

### Post-Implementation
- [ ] Run full test suite
- [ ] Benchmark performance
- [ ] Verify Article II compliance
- [ ] Update all documentation
- [ ] Final commit and push

---

## TIMELINE

### Recommended Schedule (Full-Time)

**Hour 0-2: Data Structure**
- Task 1: Enhance IrSensorFrame (1h)
- Task 2: Background level (30min)
- Task 3: Hotspot detection (start, 30min)

**Hour 2-4: Core Algorithms**
- Task 3: Hotspot detection (finish, 1h)
- Task 4: Histogram computation (45min)
- Task 5: Shannon entropy (30min)
- **Git Commit:** "Pixel processing algorithms complete"

**Hour 4-6: Integration**
- Task 6: from_pixels() constructor (1.5h)
- Task 7: Weighted centroid (30min)

**Hour 6-8: Enhancement Complete**
- Task 8: Thermal signature (30min)
- Task 9: Update compute_spatial_entropy() (30min)
- **Git Commit:** "Enhancement 2 core complete"

**Hour 8-10: Testing**
- Task 10: Integration tests (1.5h)
- Task 11: Update demo (30min)

**Hour 10-12: Finalization**
- Task 12: Test compilation (30min)
- Task 13: Update documentation (1h)
- Task 14: Final validation (1h)
- **Git Commit:** "Enhancement 2 COMPLETE: Real pixel processing"

**Total:** ~12 hours (1.5 days with breaks)

---

## DELIVERABLES CHECKLIST

### Code Deliverables
- [ ] `src/pwsa/satellite_adapters.rs` (MODIFIED, +300 lines)
  - Enhanced IrSensorFrame structure
  - Pixel processing algorithms (7 functions)
  - Updated compute_spatial_entropy()

- [ ] `tests/pwsa_spatial_entropy_test.rs` (NEW, 200+ lines)
  - 6 comprehensive test cases
  - Pixel processing validation
  - Backward compatibility tests

- [ ] `examples/pwsa_demo.rs` (MODIFIED, +30 lines)
  - Optional pixel mode
  - Synthetic pixel generator

### Documentation Deliverables
- [ ] `ENHANCEMENT-2-COMPLETION.md` (NEW)
- [ ] `TECHNICAL-DEBT-INVENTORY.md` (UPDATED)
- [ ] `Constitutional-Compliance-Matrix.md` (UPDATED)
- [ ] `STATUS-DASHBOARD.md` (UPDATED)

### Performance Deliverables
- [ ] Benchmark results (<50ms pixel processing)
- [ ] Fusion latency still <1ms (validated)

---

## RISK MITIGATION

### Risk 1: Takes Longer Than Expected
**Mitigation:** Timebox to 12 hours max
**Fallback:** Skip Task 11 (demo update) if needed
**Impact:** Minimal (demo update is optional)

### Risk 2: Performance Impact Too High
**Mitigation:** Profile and optimize hotspot detection
**Fallback:** Cache pixel processing results
**Impact:** Manageable (optimization straightforward)

### Risk 3: Breaks Existing Demos
**Mitigation:** Comprehensive backward compatibility tests
**Fallback:** Multi-tier approach ensures fallback works
**Impact:** Very low (design prevents this)

---

## NEXT STEPS AFTER ENHANCEMENT 2

**Immediate:**
1. Commit Enhancement 2 completion
2. Push to GitHub
3. Update vault with completion status

**Then:**
- **Option A:** Proceed to Enhancement 3 (frame tracking, 1-2 days)
- **Option B:** Proceed to Week 3 (SBIR proposal writing)

**Recommendation:** Option B (Week 3 proposal writing)
- Enhancement 2 is sufficient enhancement for now
- Focus on SBIR proposal
- Enhancement 3 can wait for Phase II

---

**Status:** COMPREHENSIVE TODO READY
**Total Tasks:** 14 tasks
**Estimated Time:** 12 hours (1.5 days)
**Next:** Begin Task 1 (enhance IrSensorFrame structure)

---

**Classification:** UNCLASSIFIED//FOR OFFICIAL USE ONLY
**Date:** January 9, 2025
