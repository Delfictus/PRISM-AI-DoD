package prismclient

import (
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/go-resty/resty/v2"
)

// Client is the PRISM-AI API client
type Client struct {
	apiKey    string
	baseURL   string
	timeout   time.Duration
	verifySsl bool
	client    *resty.Client
}

// ClientConfig holds configuration for the PRISM client
type ClientConfig struct {
	APIKey    string
	BaseURL   string
	Timeout   time.Duration
	VerifySSL bool
}

// NewClient creates a new PRISM-AI API client
func NewClient(config ClientConfig) *Client {
	if config.BaseURL == "" {
		config.BaseURL = "http://localhost:8080"
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	restyClient := resty.New().
		SetBaseURL(config.BaseURL).
		SetTimeout(config.Timeout).
		SetHeader("Authorization", "Bearer "+config.APIKey).
		SetHeader("Content-Type", "application/json").
		SetHeader("User-Agent", "prism-go-client/0.1.0")

	if !config.VerifySSL {
		restyClient.SetTLSClientConfig(&resty.TLSClientConfig{InsecureSkipVerify: true})
	}

	return &Client{
		apiKey:    config.APIKey,
		baseURL:   config.BaseURL,
		timeout:   config.Timeout,
		verifySsl: config.VerifySSL,
		client:    restyClient,
	}
}

// handleError converts HTTP errors to custom exception types
func (c *Client) handleError(resp *resty.Response, err error) error {
	if err != nil {
		return NewNetworkError(fmt.Sprintf("Request failed: %v", err))
	}

	statusCode := resp.StatusCode()

	var errorMsg string
	var apiResp map[string]interface{}
	if err := json.Unmarshal(resp.Body(), &apiResp); err == nil {
		if msg, ok := apiResp["error"].(string); ok {
			errorMsg = msg
		} else if msg, ok := apiResp["message"].(string); ok {
			errorMsg = msg
		}
	}
	if errorMsg == "" {
		errorMsg = "Unknown error"
	}

	switch statusCode {
	case 401:
		return NewAuthenticationError("Authentication failed. Check your API key.", statusCode, resp)
	case 403:
		return NewAuthorizationError("Permission denied.", statusCode, resp)
	case 404:
		return NewNotFoundError("Resource not found.", statusCode, resp)
	case 400:
		return NewValidationError(errorMsg, statusCode, resp)
	case 429:
		retryAfter := 0
		if retryAfterStr := resp.Header().Get("Retry-After"); retryAfterStr != "" {
			if ra, err := strconv.Atoi(retryAfterStr); err == nil {
				retryAfter = ra
			}
		}
		return NewRateLimitError("Rate limit exceeded.", statusCode, resp, retryAfter)
	default:
		if statusCode >= 500 {
			return NewServerError(fmt.Sprintf("Server error: %d", statusCode), statusCode, resp)
		}
	}

	return nil
}

// Health checks API health status
func (c *Client) Health() (map[string]interface{}, error) {
	resp, err := c.client.R().Get("/health")
	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp.Body(), &result); err != nil {
		return nil, err
	}
	return result, nil
}

// Info gets API information
func (c *Client) Info() (map[string]interface{}, error) {
	resp, err := c.client.R().Get("/")
	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp.Body(), &result); err != nil {
		return nil, err
	}
	return result, nil
}

// DetectThreat detects threats from IR sensor data
func (c *Client) DetectThreat(svID int, timestamp int64, irFrame map[string]interface{}) (*ThreatDetection, error) {
	payload := map[string]interface{}{
		"sv_id":     svID,
		"timestamp": timestamp,
		"ir_frame":  irFrame,
	}

	resp, err := c.client.R().
		SetBody(payload).
		Post("/api/v1/pwsa/detect")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data ThreatDetection `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return &apiResp.Data, nil
}

// FuseSensors fuses multi-sensor data
func (c *Client) FuseSensors(svID int, timestamp int64, sensors map[string]interface{}) (*SensorFusion, error) {
	payload := map[string]interface{}{
		"sv_id":     svID,
		"timestamp": timestamp,
		"sensors":   sensors,
	}

	resp, err := c.client.R().
		SetBody(payload).
		Post("/api/v1/pwsa/fuse")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data SensorFusion `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return &apiResp.Data, nil
}

// PredictTrajectory predicts threat trajectory
func (c *Client) PredictTrajectory(trackID string, history []map[string]interface{}, predictionHorizon int, model string) (*TrajectoryPrediction, error) {
	if model == "" {
		model = "kalman_filter"
	}

	payload := map[string]interface{}{
		"track_id":           trackID,
		"history":            history,
		"prediction_horizon": predictionHorizon,
		"model":              model,
	}

	resp, err := c.client.R().
		SetBody(payload).
		Post("/api/v1/pwsa/predict")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data TrajectoryPrediction `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return &apiResp.Data, nil
}

// PrioritizeThreats prioritizes multiple threats
func (c *Client) PrioritizeThreats(threats []map[string]interface{}, strategy string, defensiveAssets map[string]interface{}) (map[string]interface{}, error) {
	if strategy == "" {
		strategy = "time_weighted_risk"
	}

	payload := map[string]interface{}{
		"threats":                 threats,
		"prioritization_strategy": strategy,
	}
	if defensiveAssets != nil {
		payload["defensive_assets"] = defensiveAssets
	}

	resp, err := c.client.R().
		SetBody(payload).
		Post("/api/v1/pwsa/prioritize")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data map[string]interface{} `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return apiResp.Data, nil
}

// OptimizePortfolio optimizes portfolio allocation
func (c *Client) OptimizePortfolio(assets []map[string]interface{}, constraints map[string]interface{}, objective string) (*PortfolioOptimization, error) {
	if objective == "" {
		objective = "maximize_sharpe"
	}

	payload := map[string]interface{}{
		"assets":      assets,
		"constraints": constraints,
		"objective":   objective,
	}

	resp, err := c.client.R().
		SetBody(payload).
		Post("/api/v1/finance/optimize")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data PortfolioOptimization `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return &apiResp.Data, nil
}

// AssessRisk assesses portfolio risk
func (c *Client) AssessRisk(portfolioID string, positions []map[string]interface{}, riskMetrics []string) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"portfolio_id": portfolioID,
		"positions":    positions,
		"risk_metrics": riskMetrics,
	}

	resp, err := c.client.R().
		SetBody(payload).
		Post("/api/v1/finance/risk")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data map[string]interface{} `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return apiResp.Data, nil
}

// BacktestStrategy backtests a trading strategy
func (c *Client) BacktestStrategy(strategyID string, parameters map[string]interface{}, historicalData map[string]interface{}, initialCapital float64) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"strategy_id":     strategyID,
		"parameters":      parameters,
		"historical_data": historicalData,
		"initial_capital": initialCapital,
	}

	resp, err := c.client.R().
		SetBody(payload).
		Post("/api/v1/finance/backtest")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data map[string]interface{} `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return apiResp.Data, nil
}

// QueryLLM queries a language model
func (c *Client) QueryLLM(prompt string, model *string, temperature float64, maxTokens int) (*LLMQuery, error) {
	payload := map[string]interface{}{
		"prompt":      prompt,
		"temperature": temperature,
		"max_tokens":  maxTokens,
	}
	if model != nil {
		payload["model"] = *model
	}

	resp, err := c.client.R().
		SetBody(payload).
		Post("/api/v1/llm/query")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data LLMQuery `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return &apiResp.Data, nil
}

// LLMConsensus queries multiple models with consensus
func (c *Client) LLMConsensus(prompt string, models []map[string]interface{}, strategy string, temperature float64, maxTokens int) (*LLMConsensus, error) {
	if strategy == "" {
		strategy = "majority_vote"
	}

	payload := map[string]interface{}{
		"prompt":      prompt,
		"models":      models,
		"strategy":    strategy,
		"temperature": temperature,
		"max_tokens":  maxTokens,
	}

	resp, err := c.client.R().
		SetBody(payload).
		Post("/api/v1/llm/consensus")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data LLMConsensus `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return &apiResp.Data, nil
}

// ListLLMModels lists available LLM models
func (c *Client) ListLLMModels() ([]map[string]interface{}, error) {
	resp, err := c.client.R().Get("/api/v1/llm/models")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data struct {
			Models []map[string]interface{} `json:"models"`
		} `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return apiResp.Data.Models, nil
}

// ForecastTimeSeries forecasts a time series
func (c *Client) ForecastTimeSeries(seriesID string, historicalData []float64, timestamps []int64, horizon int, method string) (*TimeSeriesForecast, error) {
	if method == "" {
		method = "arima"
	}

	payload := map[string]interface{}{
		"series_id":       seriesID,
		"historical_data": historicalData,
		"horizon":         horizon,
		"method": map[string]interface{}{
			method: map[string]interface{}{},
		},
	}
	if timestamps != nil {
		payload["timestamps"] = timestamps
	}

	resp, err := c.client.R().
		SetBody(payload).
		Post("/api/v1/timeseries/forecast")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data TimeSeriesForecast `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return &apiResp.Data, nil
}

// ProcessPixels processes pixel data
func (c *Client) ProcessPixels(frameID int, width int, height int, pixels []int, options map[string]interface{}) (*PixelProcessing, error) {
	payload := map[string]interface{}{
		"frame_id": frameID,
		"width":    width,
		"height":   height,
		"pixels":   pixels,
	}
	if options != nil {
		payload["processing_options"] = options
	}

	resp, err := c.client.R().
		SetBody(payload).
		Post("/api/v1/pixels/process")

	if err != nil {
		return nil, c.handleError(resp, err)
	}
	if resp.StatusCode() >= 400 {
		return nil, c.handleError(resp, nil)
	}

	var apiResp struct {
		Data PixelProcessing `json:"data"`
	}
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return nil, err
	}
	return &apiResp.Data, nil
}
