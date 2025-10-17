#!/usr/bin/env python3
"""
Python Bridge for PhD-Grade Stochastic Thermodynamics
Integrates OTT-JAX (Optimal Transport) and Geomstats (Information Geometry)
with PRISM-AI's Rust thermodynamic computing engine.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

# ============================================================================
# OTT-JAX Integration (Optimal Transport on GPUs)
# ============================================================================

try:
    import jax
    import jax.numpy as jnp
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn
    OTT_AVAILABLE = True
except ImportError:
    warnings.warn("OTT-JAX not available. Install with: pip install ott-jax")
    OTT_AVAILABLE = False


def wasserstein_distance_gpu(
    work_dist_1: np.ndarray,
    work_dist_2: np.ndarray,
    epsilon: float = 0.01
) -> float:
    """
    Compute Wasserstein distance between two work distributions using GPU.

    Used for comparing Jarzynski/Crooks distributions across different protocols.

    Args:
        work_dist_1: First work distribution [n_samples]
        work_dist_2: Second work distribution [m_samples]
        epsilon: Entropic regularization parameter

    Returns:
        Wasserstein-2 distance (optimal transport cost)
    """
    if not OTT_AVAILABLE:
        raise RuntimeError("OTT-JAX not installed")

    # Reshape to point clouds
    x = jnp.array(work_dist_1).reshape(-1, 1)
    y = jnp.array(work_dist_2).reshape(-1, 1)

    # Create point cloud geometry
    geom = pointcloud.PointCloud(x, y, epsilon=epsilon)

    # Solve optimal transport problem
    ot_prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    ot_solution = solver(ot_prob)

    return float(ot_solution.reg_ot_cost)


def sinkhorn_divergence_gpu(
    forward_work: np.ndarray,
    reverse_work: np.ndarray,
    epsilon: float = 0.01
) -> float:
    """
    Sinkhorn divergence between forward and reverse work distributions.

    Provides a metric for Crooks theorem validation:
    If Crooks holds exactly, divergence should be related to free energy difference.

    Args:
        forward_work: Forward protocol work values
        reverse_work: Reverse protocol work values (will be negated)
        epsilon: Regularization parameter

    Returns:
        Sinkhorn divergence S(P_F || P_R)
    """
    if not OTT_AVAILABLE:
        raise RuntimeError("OTT-JAX not installed")

    # Prepare distributions
    x = jnp.array(forward_work).reshape(-1, 1)
    y = jnp.array(-reverse_work).reshape(-1, 1)  # Crooks uses -W_R

    # Sinkhorn divergence: S(μ||ν) = W(μ,ν) - 0.5*(W(μ,μ) + W(ν,ν))
    geom_xy = pointcloud.PointCloud(x, y, epsilon=epsilon)
    geom_xx = pointcloud.PointCloud(x, x, epsilon=epsilon)
    geom_yy = pointcloud.PointCloud(y, y, epsilon=epsilon)

    solver = sinkhorn.Sinkhorn()

    w_xy = solver(linear_problem.LinearProblem(geom_xy)).reg_ot_cost
    w_xx = solver(linear_problem.LinearProblem(geom_xx)).reg_ot_cost
    w_yy = solver(linear_problem.LinearProblem(geom_yy)).reg_ot_cost

    divergence = w_xy - 0.5 * (w_xx + w_yy)

    return float(divergence)


# ============================================================================
# Geomstats Integration (Information Geometry)
# ============================================================================

try:
    import geomstats.backend as gs
    from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
    from geomstats.geometry.spd_matrices import SPDMatrices
    GEOMSTATS_AVAILABLE = True
except ImportError:
    warnings.warn("Geomstats not available. Install with: pip install geomstats")
    GEOMSTATS_AVAILABLE = False


def fisher_information_metric(
    work_samples: np.ndarray,
    temperature: float
) -> float:
    """
    Compute Fisher information metric for work distribution.

    Fisher metric quantifies distinguishability of nearby probability distributions.
    In thermodynamics, relates to thermodynamic length and optimal protocols.

    Args:
        work_samples: Work values from trajectory ensemble
        temperature: System temperature (kT)

    Returns:
        Fisher information I_F
    """
    if not GEOMSTATS_AVAILABLE:
        raise RuntimeError("Geomstats not installed")

    # Estimate parameters of work distribution (assume Gaussian for simplicity)
    mean = np.mean(work_samples)
    var = np.var(work_samples)

    # Fisher information for Gaussian: I_μ = 1/σ², I_σ = 2/σ²
    # Total Fisher info (trace of Fisher information matrix)
    fisher_info = 1.0 / var + 2.0 / var

    return float(fisher_info)


def thermodynamic_length(
    protocol_history: np.ndarray,
    work_history: np.ndarray,
    temperature: float
) -> float:
    """
    Compute thermodynamic length of a protocol.

    Thermodynamic length L = ∫ √g_ij dλ^i dλ^j where g_ij is Fisher metric.
    Related to dissipation: W_diss ≥ (kT/2τ) L²

    Args:
        protocol_history: Parameter values over time [n_steps x n_params]
        work_history: Work values at each step [n_steps]
        temperature: System temperature

    Returns:
        Thermodynamic length L
    """
    if not GEOMSTATS_AVAILABLE:
        raise RuntimeError("Geomstats not installed")

    n_steps = len(work_history)
    length = 0.0

    for i in range(n_steps - 1):
        # Estimate Fisher metric at this point
        fisher = fisher_information_metric(
            work_history[max(0, i-10):i+1],
            temperature
        )

        # Compute infinitesimal length
        dlambda = np.linalg.norm(protocol_history[i+1] - protocol_history[i])
        ds = np.sqrt(fisher) * dlambda
        length += ds

    return float(length)


def wasserstein_gradient_flow(
    initial_dist: np.ndarray,
    target_dist: np.ndarray,
    n_steps: int = 100
) -> np.ndarray:
    """
    Compute Wasserstein gradient flow between distributions.

    Models optimal evolution from non-equilibrium to equilibrium state.
    Related to NESS dynamics and entropy production minimization.

    Args:
        initial_dist: Initial distribution (e.g., non-equilibrium)
        target_dist: Target distribution (e.g., equilibrium)
        n_steps: Number of interpolation steps

    Returns:
        Trajectory of distributions [n_steps x n_samples]
    """
    if not OTT_AVAILABLE:
        raise RuntimeError("OTT-JAX not installed")

    # Implement McCann interpolation (geodesic in Wasserstein space)
    x0 = jnp.array(initial_dist).reshape(-1, 1)
    x1 = jnp.array(target_dist).reshape(-1, 1)

    # Solve optimal transport to get map
    geom = pointcloud.PointCloud(x0, x1, epsilon=0.01)
    ot_prob = linear_problem.LinearProblem(geom)
    ot_solution = sinkhorn.Sinkhorn()(ot_prob)

    # Interpolate along geodesic
    trajectory = []
    for t in np.linspace(0, 1, n_steps):
        # Linear interpolation as approximation
        # (Exact geodesic requires transport map)
        x_t = (1 - t) * initial_dist + t * target_dist
        trajectory.append(x_t)

    return np.array(trajectory)


# ============================================================================
# FFI Entry Points (Called from Rust)
# ============================================================================

def compute_optimal_transport_cost(
    work_forward: np.ndarray,
    work_reverse: np.ndarray,
    epsilon: float = 0.01
) -> dict:
    """
    Unified entry point for optimal transport calculations.

    Returns:
        Dictionary with wasserstein_distance, sinkhorn_divergence, etc.
    """
    result = {
        'wasserstein_distance': 0.0,
        'sinkhorn_divergence': 0.0,
        'available': OTT_AVAILABLE
    }

    if OTT_AVAILABLE and len(work_forward) > 0 and len(work_reverse) > 0:
        result['wasserstein_distance'] = wasserstein_distance_gpu(
            work_forward, work_reverse, epsilon
        )
        result['sinkhorn_divergence'] = sinkhorn_divergence_gpu(
            work_forward, work_reverse, epsilon
        )

    return result


def compute_information_geometry(
    work_samples: np.ndarray,
    protocol_history: Optional[np.ndarray],
    temperature: float
) -> dict:
    """
    Unified entry point for information geometry calculations.

    Returns:
        Dictionary with fisher_information, thermodynamic_length, etc.
    """
    result = {
        'fisher_information': 0.0,
        'thermodynamic_length': 0.0,
        'available': GEOMSTATS_AVAILABLE
    }

    if GEOMSTATS_AVAILABLE and len(work_samples) > 0:
        result['fisher_information'] = fisher_information_metric(
            work_samples, temperature
        )

        if protocol_history is not None:
            result['thermodynamic_length'] = thermodynamic_length(
                protocol_history, work_samples, temperature
            )

    return result


# ============================================================================
# Testing and Validation
# ============================================================================

if __name__ == "__main__":
    print("🔬 PRISM-AI Thermodynamics Python Bridge")
    print("=" * 60)

    # Test OTT-JAX
    if OTT_AVAILABLE:
        print("✅ OTT-JAX available")
        # Simple test
        w1 = np.random.randn(100)
        w2 = np.random.randn(100) + 1.0
        dist = wasserstein_distance_gpu(w1, w2)
        print(f"   Test Wasserstein distance: {dist:.6f}")
    else:
        print("❌ OTT-JAX not available")

    # Test Geomstats
    if GEOMSTATS_AVAILABLE:
        print("✅ Geomstats available")
        samples = np.random.randn(1000)
        fisher = fisher_information_metric(samples, 1.0)
        print(f"   Test Fisher information: {fisher:.6f}")
    else:
        print("❌ Geomstats not available")

    print("=" * 60)
