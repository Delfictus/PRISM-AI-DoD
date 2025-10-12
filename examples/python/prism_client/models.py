"""
Data models for PRISM-AI API requests and responses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class ThreatDetection:
    """PWSA threat detection result."""

    threat_id: str
    threat_type: str
    confidence: float
    position: List[float]
    velocity: Optional[List[float]] = None
    time_to_impact: Optional[float] = None
    recommended_action: Optional[str] = None
    timestamp: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreatDetection":
        return cls(
            threat_id=data["threat_id"],
            threat_type=data["threat_type"],
            confidence=data["confidence"],
            position=data["position"],
            velocity=data.get("velocity"),
            time_to_impact=data.get("time_to_impact"),
            recommended_action=data.get("recommended_action"),
            timestamp=data.get("timestamp"),
        )


@dataclass
class PortfolioOptimization:
    """Finance portfolio optimization result."""

    weights: List[Dict[str, Any]]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    optimization_time_ms: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioOptimization":
        return cls(
            weights=data["weights"],
            expected_return=data["expected_return"],
            expected_risk=data["expected_risk"],
            sharpe_ratio=data["sharpe_ratio"],
            optimization_time_ms=data["optimization_time_ms"],
        )


@dataclass
class LLMQuery:
    """LLM query result."""

    text: str
    model_used: str
    tokens_used: int
    cost_usd: float
    latency_ms: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMQuery":
        return cls(
            text=data["text"],
            model_used=data["model_used"],
            tokens_used=data["tokens_used"],
            cost_usd=data["cost_usd"],
            latency_ms=data["latency_ms"],
        )


@dataclass
class LLMConsensus:
    """Multi-model LLM consensus result."""

    consensus_text: str
    confidence: float
    strategy: str
    individual_responses: List[Dict[str, Any]]
    total_cost_usd: float
    total_time_ms: float
    agreement_rate: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConsensus":
        return cls(
            consensus_text=data["consensus_text"],
            confidence=data["confidence"],
            strategy=data["strategy"],
            individual_responses=data["individual_responses"],
            total_cost_usd=data["total_cost_usd"],
            total_time_ms=data["total_time_ms"],
            agreement_rate=data["agreement_rate"],
        )


@dataclass
class TimeSeriesForecast:
    """Time series forecast result."""

    series_id: str
    predictions: List[float]
    timestamps: Optional[List[int]] = None
    confidence_intervals: Optional[List[List[float]]] = None
    metrics: Optional[Dict[str, float]] = None
    computation_time_ms: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSeriesForecast":
        return cls(
            series_id=data["series_id"],
            predictions=data["predictions"],
            timestamps=data.get("timestamps"),
            confidence_intervals=data.get("confidence_intervals"),
            metrics=data.get("metrics"),
            computation_time_ms=data.get("computation_time_ms"),
        )


@dataclass
class PixelProcessing:
    """Pixel processing result."""

    frame_id: int
    processed_pixels: List[int]
    hotspots: List[Dict[str, Any]]
    entropy: Optional[float] = None
    tda_features: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PixelProcessing":
        return cls(
            frame_id=data["frame_id"],
            processed_pixels=data["processed_pixels"],
            hotspots=data["hotspots"],
            entropy=data.get("entropy"),
            tda_features=data.get("tda_features"),
            processing_time_ms=data.get("processing_time_ms"),
        )


@dataclass
class SensorFusion:
    """Multi-sensor fusion result."""

    num_tracks: int
    fusion_quality: float
    tracks: List[Dict[str, Any]]
    processing_time_ms: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorFusion":
        return cls(
            num_tracks=data["num_tracks"],
            fusion_quality=data["fusion_quality"],
            tracks=data["tracks"],
            processing_time_ms=data["processing_time_ms"],
        )


@dataclass
class TrajectoryPrediction:
    """Trajectory prediction result."""

    track_id: str
    model: str
    confidence: float
    predictions: List[Dict[str, Any]]
    time_to_impact: Optional[float] = None
    impact_point: Optional[Dict[str, Any]] = None
    uncertainty: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryPrediction":
        return cls(
            track_id=data["track_id"],
            model=data["model"],
            confidence=data["confidence"],
            predictions=data["predictions"],
            time_to_impact=data.get("time_to_impact"),
            impact_point=data.get("impact_point"),
            uncertainty=data.get("uncertainty"),
        )
