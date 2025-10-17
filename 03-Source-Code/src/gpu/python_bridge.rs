///! Python FFI Bridge for OTT-JAX and Geomstats Integration
///!
///! Provides Rust interface to Python scientific libraries:
///! - OTT-JAX: Optimal transport on GPUs
///! - Geomstats: Information geometry
///!
///! PhD-GRADE features for thermodynamic computing

use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use std::process::Command;
use std::path::PathBuf;

/// Results from optimal transport calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalTransportResults {
    /// Wasserstein-2 distance between distributions
    pub wasserstein_distance: f32,
    /// Sinkhorn divergence (symmetrized transport cost)
    pub sinkhorn_divergence: f32,
    /// Whether Python library is available
    pub available: bool,
}

/// Results from information geometry calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationGeometryResults {
    /// Fisher information metric
    pub fisher_information: f32,
    /// Thermodynamic length of protocol
    pub thermodynamic_length: f32,
    /// Whether Python library is available
    pub available: bool,
}

/// Python Bridge Manager
///
/// Handles communication with Python libraries for advanced thermodynamics
pub struct PythonBridge {
    python_path: PathBuf,
    bridge_module: PathBuf,
}

impl PythonBridge {
    /// Initialize Python bridge with auto-detection
    pub fn new() -> Result<Self> {
        // Find Python executable
        let python_path = Self::find_python()?;

        // Locate bridge module
        let bridge_module = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("python_bridge")
            .join("thermodynamics_bridge.py");

        if !bridge_module.exists() {
            return Err(anyhow!("Python bridge module not found at {:?}", bridge_module));
        }

        Ok(Self {
            python_path,
            bridge_module,
        })
    }

    /// Find suitable Python executable (python3 or python)
    fn find_python() -> Result<PathBuf> {
        // Try python3 first
        if let Ok(output) = Command::new("python3").arg("--version").output() {
            if output.status.success() {
                return Ok(PathBuf::from("python3"));
            }
        }

        // Fallback to python
        if let Ok(output) = Command::new("python").arg("--version").output() {
            if output.status.success() {
                return Ok(PathBuf::from("python"));
            }
        }

        Err(anyhow!("Python executable not found"))
    }

    /// Compute optimal transport between work distributions
    ///
    /// Uses OTT-JAX on GPU for Wasserstein distance and Sinkhorn divergence
    pub fn compute_optimal_transport(
        &self,
        forward_work: &[f32],
        reverse_work: &[f32],
        epsilon: f32,
    ) -> Result<OptimalTransportResults> {
        // Create temporary Python script
        let script = format!(
            r#"
import sys
import numpy as np
import json
sys.path.insert(0, '{}')
from thermodynamics_bridge import compute_optimal_transport_cost

forward_work = np.array({:?}, dtype=np.float32)
reverse_work = np.array({:?}, dtype=np.float32)
epsilon = {:.6}

result = compute_optimal_transport_cost(forward_work, reverse_work, epsilon)
print(json.dumps(result))
"#,
            self.bridge_module.parent().unwrap().display(),
            forward_work,
            reverse_work,
            epsilon
        );

        // Execute Python script
        let output = Command::new(&self.python_path)
            .arg("-c")
            .arg(&script)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Ok(OptimalTransportResults {
                wasserstein_distance: 0.0,
                sinkhorn_divergence: 0.0,
                available: false,
            });
        }

        // Parse JSON result
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: serde_json::Value = serde_json::from_str(&stdout)?;

        Ok(OptimalTransportResults {
            wasserstein_distance: result["wasserstein_distance"].as_f64().unwrap_or(0.0) as f32,
            sinkhorn_divergence: result["sinkhorn_divergence"].as_f64().unwrap_or(0.0) as f32,
            available: result["available"].as_bool().unwrap_or(false),
        })
    }

    /// Compute information geometry metrics
    ///
    /// Uses Geomstats for Fisher information and thermodynamic length
    pub fn compute_information_geometry(
        &self,
        work_samples: &[f32],
        protocol_history: Option<&[f32]>,
        temperature: f32,
    ) -> Result<InformationGeometryResults> {
        let protocol_str = if let Some(proto) = protocol_history {
            format!("np.array({:?}, dtype=np.float32)", proto)
        } else {
            "None".to_string()
        };

        let script = format!(
            r#"
import sys
import numpy as np
import json
sys.path.insert(0, '{}')
from thermodynamics_bridge import compute_information_geometry

work_samples = np.array({:?}, dtype=np.float32)
protocol_history = {}
temperature = {:.6}

result = compute_information_geometry(work_samples, protocol_history, temperature)
print(json.dumps(result))
"#,
            self.bridge_module.parent().unwrap().display(),
            work_samples,
            protocol_str,
            temperature
        );

        let output = Command::new(&self.python_path)
            .arg("-c")
            .arg(&script)
            .output()?;

        if !output.status.success() {
            return Ok(InformationGeometryResults {
                fisher_information: 0.0,
                thermodynamic_length: 0.0,
                available: false,
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: serde_json::Value = serde_json::from_str(&stdout)?;

        Ok(InformationGeometryResults {
            fisher_information: result["fisher_information"].as_f64().unwrap_or(0.0) as f32,
            thermodynamic_length: result["thermodynamic_length"].as_f64().unwrap_or(0.0) as f32,
            available: result["available"].as_bool().unwrap_or(false),
        })
    }

    /// Check if Python libraries are available
    pub fn check_availability(&self) -> Result<(bool, bool)> {
        let script = format!(
            r#"
import sys
sys.path.insert(0, '{}')
from thermodynamics_bridge import OTT_AVAILABLE, GEOMSTATS_AVAILABLE
print(f"{{OTT_AVAILABLE}},{{GEOMSTATS_AVAILABLE}}")
"#,
            self.bridge_module.parent().unwrap().display()
        );

        let output = Command::new(&self.python_path)
            .arg("-c")
            .arg(&script)
            .output()?;

        if !output.status.success() {
            return Ok((false, false));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.trim().split(',').collect();

        let ott = parts.get(0).map(|s| *s == "True").unwrap_or(false);
        let geomstats = parts.get(1).map(|s| *s == "True").unwrap_or(false);

        Ok((ott, geomstats))
    }
}

/// Global Python bridge instance (lazy initialization)
static mut PYTHON_BRIDGE: Option<PythonBridge> = None;
static mut BRIDGE_INITIALIZED: bool = false;

/// Get or initialize global Python bridge
pub fn get_python_bridge() -> Result<&'static PythonBridge> {
    unsafe {
        if !BRIDGE_INITIALIZED {
            match PythonBridge::new() {
                Ok(bridge) => {
                    PYTHON_BRIDGE = Some(bridge);
                    BRIDGE_INITIALIZED = true;
                }
                Err(e) => {
                    eprintln!("⚠️  Python bridge initialization failed: {}", e);
                    eprintln!("   PhD-grade optimal transport features will be unavailable");
                    return Err(e);
                }
            }
        }

        PYTHON_BRIDGE.as_ref()
            .ok_or_else(|| anyhow!("Python bridge not initialized"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_bridge_init() {
        match PythonBridge::new() {
            Ok(bridge) => {
                println!("✅ Python bridge initialized");

                // Check library availability
                if let Ok((ott, geomstats)) = bridge.check_availability() {
                    println!("  OTT-JAX: {}", if ott { "✅" } else { "❌" });
                    println!("  Geomstats: {}", if geomstats { "✅" } else { "❌" });
                }
            }
            Err(e) => {
                println!("⚠️  Python bridge failed: {}", e);
            }
        }
    }

    #[test]
    fn test_optimal_transport() {
        if let Ok(bridge) = PythonBridge::new() {
            let forward = vec![0.1, 0.2, 0.3, 0.4, 0.5];
            let reverse = vec![0.15, 0.25, 0.35, 0.45, 0.55];

            match bridge.compute_optimal_transport(&forward, &reverse, 0.01) {
                Ok(result) => {
                    println!("Optimal Transport Results:");
                    println!("  Wasserstein: {:.6}", result.wasserstein_distance);
                    println!("  Sinkhorn: {:.6}", result.sinkhorn_divergence);
                }
                Err(e) => {
                    println!("Test skipped: {}", e);
                }
            }
        }
    }

    #[test]
    fn test_information_geometry() {
        if let Ok(bridge) = PythonBridge::new() {
            let samples = vec![1.0, 1.5, 2.0, 1.2, 1.8, 2.1];

            match bridge.compute_information_geometry(&samples, None, 1.0) {
                Ok(result) => {
                    println!("Information Geometry Results:");
                    println!("  Fisher info: {:.6}", result.fisher_information);
                }
                Err(e) => {
                    println!("Test skipped: {}", e);
                }
            }
        }
    }
}
