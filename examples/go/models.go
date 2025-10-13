package prismclient

// ThreatDetection represents a PWSA threat detection result
type ThreatDetection struct {
	ThreatID          string    `json:"threat_id"`
	ThreatType        string    `json:"threat_type"`
	Confidence        float64   `json:"confidence"`
	Position          []float64 `json:"position"`
	Velocity          []float64 `json:"velocity,omitempty"`
	TimeToImpact      *float64  `json:"time_to_impact,omitempty"`
	RecommendedAction *string   `json:"recommended_action,omitempty"`
	Timestamp         *int64    `json:"timestamp,omitempty"`
}

// PortfolioOptimization represents a finance portfolio optimization result
type PortfolioOptimization struct {
	Weights            []WeightAllocation `json:"weights"`
	ExpectedReturn     float64            `json:"expected_return"`
	ExpectedRisk       float64            `json:"expected_risk"`
	SharpeRatio        float64            `json:"sharpe_ratio"`
	OptimizationTimeMs float64            `json:"optimization_time_ms"`
}

// WeightAllocation represents a portfolio weight allocation
type WeightAllocation struct {
	Symbol string  `json:"symbol"`
	Weight float64 `json:"weight"`
}

// LLMQuery represents an LLM query result
type LLMQuery struct {
	Text       string  `json:"text"`
	ModelUsed  string  `json:"model_used"`
	TokensUsed int     `json:"tokens_used"`
	CostUsd    float64 `json:"cost_usd"`
	LatencyMs  float64 `json:"latency_ms"`
}

// LLMConsensus represents a multi-model LLM consensus result
type LLMConsensus struct {
	ConsensusText       string                   `json:"consensus_text"`
	Confidence          float64                  `json:"confidence"`
	Strategy            string                   `json:"strategy"`
	IndividualResponses []map[string]interface{} `json:"individual_responses"`
	TotalCostUsd        float64                  `json:"total_cost_usd"`
	TotalTimeMs         float64                  `json:"total_time_ms"`
	AgreementRate       float64                  `json:"agreement_rate"`
}

// TimeSeriesForecast represents a time series forecast result
type TimeSeriesForecast struct {
	SeriesID           string      `json:"series_id"`
	Predictions        []float64   `json:"predictions"`
	Timestamps         []int64     `json:"timestamps,omitempty"`
	ConfidenceIntervals [][]float64 `json:"confidence_intervals,omitempty"`
	Metrics            map[string]float64 `json:"metrics,omitempty"`
	ComputationTimeMs  *float64    `json:"computation_time_ms,omitempty"`
}

// PixelProcessing represents a pixel processing result
type PixelProcessing struct {
	FrameID           int                      `json:"frame_id"`
	ProcessedPixels   []int                    `json:"processed_pixels"`
	Hotspots          []map[string]interface{} `json:"hotspots"`
	Entropy           *float64                 `json:"entropy,omitempty"`
	TDAFeatures       []map[string]interface{} `json:"tda_features,omitempty"`
	ProcessingTimeMs  *float64                 `json:"processing_time_ms,omitempty"`
}

// SensorFusion represents a multi-sensor fusion result
type SensorFusion struct {
	NumTracks        int                      `json:"num_tracks"`
	FusionQuality    float64                  `json:"fusion_quality"`
	Tracks           []map[string]interface{} `json:"tracks"`
	ProcessingTimeMs float64                  `json:"processing_time_ms"`
}

// TrajectoryPrediction represents a trajectory prediction result
type TrajectoryPrediction struct {
	TrackID       string                   `json:"track_id"`
	Model         string                   `json:"model"`
	Confidence    float64                  `json:"confidence"`
	Predictions   []map[string]interface{} `json:"predictions"`
	TimeToImpact  *float64                 `json:"time_to_impact,omitempty"`
	ImpactPoint   map[string]interface{}   `json:"impact_point,omitempty"`
	Uncertainty   []map[string]interface{} `json:"uncertainty,omitempty"`
}

// APIResponse is a generic API response wrapper
type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data"`
	Error   *string     `json:"error,omitempty"`
}
