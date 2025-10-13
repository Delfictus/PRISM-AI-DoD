/**
 * Data models for PRISM-AI API requests and responses.
 */

/**
 * PWSA threat detection result.
 */
class ThreatDetection {
  constructor(data) {
    this.threatId = data.threat_id;
    this.threatType = data.threat_type;
    this.confidence = data.confidence;
    this.position = data.position;
    this.velocity = data.velocity || null;
    this.timeToImpact = data.time_to_impact || null;
    this.recommendedAction = data.recommended_action || null;
    this.timestamp = data.timestamp || null;
  }

  static fromJSON(data) {
    return new ThreatDetection(data);
  }
}

/**
 * Finance portfolio optimization result.
 */
class PortfolioOptimization {
  constructor(data) {
    this.weights = data.weights;
    this.expectedReturn = data.expected_return;
    this.expectedRisk = data.expected_risk;
    this.sharpeRatio = data.sharpe_ratio;
    this.optimizationTimeMs = data.optimization_time_ms;
  }

  static fromJSON(data) {
    return new PortfolioOptimization(data);
  }
}

/**
 * LLM query result.
 */
class LLMQuery {
  constructor(data) {
    this.text = data.text;
    this.modelUsed = data.model_used;
    this.tokensUsed = data.tokens_used;
    this.costUsd = data.cost_usd;
    this.latencyMs = data.latency_ms;
  }

  static fromJSON(data) {
    return new LLMQuery(data);
  }
}

/**
 * Multi-model LLM consensus result.
 */
class LLMConsensus {
  constructor(data) {
    this.consensusText = data.consensus_text;
    this.confidence = data.confidence;
    this.strategy = data.strategy;
    this.individualResponses = data.individual_responses;
    this.totalCostUsd = data.total_cost_usd;
    this.totalTimeMs = data.total_time_ms;
    this.agreementRate = data.agreement_rate;
  }

  static fromJSON(data) {
    return new LLMConsensus(data);
  }
}

/**
 * Time series forecast result.
 */
class TimeSeriesForecast {
  constructor(data) {
    this.seriesId = data.series_id;
    this.predictions = data.predictions;
    this.timestamps = data.timestamps || null;
    this.confidenceIntervals = data.confidence_intervals || null;
    this.metrics = data.metrics || null;
    this.computationTimeMs = data.computation_time_ms || null;
  }

  static fromJSON(data) {
    return new TimeSeriesForecast(data);
  }
}

/**
 * Pixel processing result.
 */
class PixelProcessing {
  constructor(data) {
    this.frameId = data.frame_id;
    this.processedPixels = data.processed_pixels;
    this.hotspots = data.hotspots;
    this.entropy = data.entropy || null;
    this.tdaFeatures = data.tda_features || null;
    this.processingTimeMs = data.processing_time_ms || null;
  }

  static fromJSON(data) {
    return new PixelProcessing(data);
  }
}

/**
 * Multi-sensor fusion result.
 */
class SensorFusion {
  constructor(data) {
    this.numTracks = data.num_tracks;
    this.fusionQuality = data.fusion_quality;
    this.tracks = data.tracks;
    this.processingTimeMs = data.processing_time_ms;
  }

  static fromJSON(data) {
    return new SensorFusion(data);
  }
}

/**
 * Trajectory prediction result.
 */
class TrajectoryPrediction {
  constructor(data) {
    this.trackId = data.track_id;
    this.model = data.model;
    this.confidence = data.confidence;
    this.predictions = data.predictions;
    this.timeToImpact = data.time_to_impact || null;
    this.impactPoint = data.impact_point || null;
    this.uncertainty = data.uncertainty || null;
  }

  static fromJSON(data) {
    return new TrajectoryPrediction(data);
  }
}

module.exports = {
  ThreatDetection,
  PortfolioOptimization,
  LLMQuery,
  LLMConsensus,
  TimeSeriesForecast,
  PixelProcessing,
  SensorFusion,
  TrajectoryPrediction,
};
