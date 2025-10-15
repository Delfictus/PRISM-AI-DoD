/**
 * Main PRISM-AI client implementation.
 */

const axios = require('axios');
const {
  PrismAPIError,
  AuthenticationError,
  AuthorizationError,
  NotFoundError,
  ValidationError,
  RateLimitError,
  ServerError,
  NetworkError,
  TimeoutError,
} = require('./exceptions');
const {
  ThreatDetection,
  PortfolioOptimization,
  LLMQuery,
  LLMConsensus,
  TimeSeriesForecast,
  PixelProcessing,
  SensorFusion,
  TrajectoryPrediction,
} = require('./models');

/**
 * PRISM-AI API Client.
 *
 * @example
 * const client = new PrismClient({ apiKey: 'your-key' });
 * const health = await client.health();
 * console.log(health.status);
 */
class PrismClient {
  /**
   * Create a new PRISM-AI client.
   *
   * @param {Object} options - Configuration options
   * @param {string} options.apiKey - API authentication key
   * @param {string} [options.baseUrl='http://localhost:8080'] - Base URL for the API
   * @param {number} [options.timeout=30000] - Request timeout in milliseconds
   * @param {boolean} [options.verifySsl=true] - Verify SSL certificates
   */
  constructor({ apiKey, baseUrl = 'http://localhost:8080', timeout = 30000, verifySsl = true }) {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeout = timeout;
    this.verifySsl = verifySsl;

    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: this.timeout,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'User-Agent': 'prism-javascript-client/0.1.0',
      },
      httpsAgent: verifySsl ? undefined : new (require('https').Agent)({
        rejectUnauthorized: false,
      }),
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => this._handleError(error)
    );
  }

  /**
   * Handle HTTP errors and convert to custom exceptions.
   * @private
   */
  _handleError(error) {
    if (error.code === 'ECONNABORTED') {
      throw new TimeoutError(`Request timed out after ${this.timeout}ms`);
    }

    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      throw new NetworkError(`Connection failed: ${error.message}`);
    }

    if (!error.response) {
      throw new PrismAPIError(`Request failed: ${error.message}`);
    }

    const { status, data } = error.response;
    const errorMessage = data?.error || data?.message || 'Unknown error';

    switch (status) {
      case 401:
        throw new AuthenticationError('Authentication failed. Check your API key.', status, error.response);
      case 403:
        throw new AuthorizationError('Permission denied.', status, error.response);
      case 404:
        throw new NotFoundError('Resource not found.', status, error.response);
      case 400:
        throw new ValidationError(errorMessage, status, error.response);
      case 429:
        const retryAfter = error.response.headers['retry-after'];
        throw new RateLimitError('Rate limit exceeded.', status, error.response, retryAfter ? parseInt(retryAfter) : null);
      default:
        if (status >= 500) {
          throw new ServerError(`Server error: ${status}`, status, error.response);
        }
        throw new PrismAPIError(errorMessage, status, error.response);
    }
  }

  // Health and Info

  /**
   * Check API health status.
   * @returns {Promise<Object>} Health status
   */
  async health() {
    const response = await this.client.get('/health');
    return response.data;
  }

  /**
   * Get API information.
   * @returns {Promise<Object>} API information
   */
  async info() {
    const response = await this.client.get('/');
    return response.data;
  }

  // PWSA Endpoints

  /**
   * Detect threat from IR sensor data.
   *
   * @param {number} svId - Space vehicle ID
   * @param {number} timestamp - Unix timestamp
   * @param {Object} irFrame - IR frame data with width, height, centroid, hotspots
   * @returns {Promise<ThreatDetection>} Threat detection result
   */
  async detectThreat(svId, timestamp, irFrame) {
    const response = await this.client.post('/api/v1/pwsa/detect', {
      sv_id: svId,
      timestamp,
      ir_frame: irFrame,
    });
    return ThreatDetection.fromJSON(response.data.data);
  }

  /**
   * Fuse multi-sensor data.
   *
   * @param {number} svId - Space vehicle ID
   * @param {number} timestamp - Unix timestamp
   * @param {Object} sensors - Sensor data (IR, radar, optical)
   * @returns {Promise<SensorFusion>} Sensor fusion result
   */
  async fuseSensors(svId, timestamp, sensors) {
    const response = await this.client.post('/api/v1/pwsa/fuse', {
      sv_id: svId,
      timestamp,
      sensors,
    });
    return SensorFusion.fromJSON(response.data.data);
  }

  /**
   * Predict threat trajectory.
   *
   * @param {string} trackId - Track identifier
   * @param {Array<Object>} history - Historical track data
   * @param {number} predictionHorizon - Prediction time (seconds)
   * @param {string} [model='kalman_filter'] - Prediction model
   * @returns {Promise<TrajectoryPrediction>} Trajectory prediction result
   */
  async predictTrajectory(trackId, history, predictionHorizon, model = 'kalman_filter') {
    const response = await this.client.post('/api/v1/pwsa/predict', {
      track_id: trackId,
      history,
      prediction_horizon: predictionHorizon,
      model,
    });
    return TrajectoryPrediction.fromJSON(response.data.data);
  }

  /**
   * Prioritize multiple threats.
   *
   * @param {Array<Object>} threats - List of threat data
   * @param {string} [strategy='time_weighted_risk'] - Prioritization strategy
   * @param {Object} [defensiveAssets=null] - Available defensive resources
   * @returns {Promise<Object>} Prioritized threat list
   */
  async prioritizeThreats(threats, strategy = 'time_weighted_risk', defensiveAssets = null) {
    const data = {
      threats,
      prioritization_strategy: strategy,
    };
    if (defensiveAssets) {
      data.defensive_assets = defensiveAssets;
    }
    const response = await this.client.post('/api/v1/pwsa/prioritize', data);
    return response.data.data;
  }

  // Finance Endpoints

  /**
   * Optimize portfolio allocation.
   *
   * @param {Array<Object>} assets - List of assets with expected returns and volatility
   * @param {Object} constraints - Portfolio constraints
   * @param {string} [objective='maximize_sharpe'] - Optimization objective
   * @returns {Promise<PortfolioOptimization>} Portfolio optimization result
   */
  async optimizePortfolio(assets, constraints, objective = 'maximize_sharpe') {
    const response = await this.client.post('/api/v1/finance/optimize', {
      assets,
      constraints,
      objective,
    });
    return PortfolioOptimization.fromJSON(response.data.data);
  }

  /**
   * Assess portfolio risk.
   *
   * @param {string} portfolioId - Portfolio identifier
   * @param {Array<Object>} positions - Current positions
   * @param {Array<string>} riskMetrics - Metrics to calculate (var, cvar, beta, etc.)
   * @returns {Promise<Object>} Risk assessment results
   */
  async assessRisk(portfolioId, positions, riskMetrics) {
    const response = await this.client.post('/api/v1/finance/risk', {
      portfolio_id: portfolioId,
      positions,
      risk_metrics: riskMetrics,
    });
    return response.data.data;
  }

  /**
   * Backtest trading strategy.
   *
   * @param {string} strategyId - Strategy identifier
   * @param {Object} parameters - Strategy parameters
   * @param {Object} historicalData - Historical market data
   * @param {number} initialCapital - Starting capital
   * @returns {Promise<Object>} Backtest results
   */
  async backtestStrategy(strategyId, parameters, historicalData, initialCapital) {
    const response = await this.client.post('/api/v1/finance/backtest', {
      strategy_id: strategyId,
      parameters,
      historical_data: historicalData,
      initial_capital: initialCapital,
    });
    return response.data.data;
  }

  // LLM Endpoints

  /**
   * Query a language model.
   *
   * @param {string} prompt - Input prompt
   * @param {string} [model=null] - Model name (optional, uses default)
   * @param {number} [temperature=0.7] - Sampling temperature (0-2)
   * @param {number} [maxTokens=500] - Maximum tokens to generate
   * @returns {Promise<LLMQuery>} LLM query result
   */
  async queryLLM(prompt, model = null, temperature = 0.7, maxTokens = 500) {
    const data = {
      prompt,
      temperature,
      max_tokens: maxTokens,
    };
    if (model) {
      data.model = model;
    }
    const response = await this.client.post('/api/v1/llm/query', data);
    return LLMQuery.fromJSON(response.data.data);
  }

  /**
   * Query multiple models with consensus.
   *
   * @param {string} prompt - Input prompt
   * @param {Array<Object>} models - List of models with weights
   * @param {string} [strategy='majority_vote'] - Consensus strategy
   * @param {number} [temperature=0.3] - Sampling temperature
   * @param {number} [maxTokens=500] - Maximum tokens
   * @returns {Promise<LLMConsensus>} LLM consensus result
   */
  async llmConsensus(prompt, models, strategy = 'majority_vote', temperature = 0.3, maxTokens = 500) {
    const response = await this.client.post('/api/v1/llm/consensus', {
      prompt,
      models,
      strategy,
      temperature,
      max_tokens: maxTokens,
    });
    return LLMConsensus.fromJSON(response.data.data);
  }

  /**
   * List available LLM models.
   * @returns {Promise<Array<Object>>} List of available models
   */
  async listLLMModels() {
    const response = await this.client.get('/api/v1/llm/models');
    return response.data.data.models;
  }

  // Time Series Endpoints

  /**
   * Forecast time series.
   *
   * @param {string} seriesId - Series identifier
   * @param {Array<number>} historicalData - Historical values
   * @param {Array<number>} [timestamps=null] - Optional timestamps
   * @param {number} [horizon=10] - Forecast horizon
   * @param {string} [method='arima'] - Forecasting method
   * @returns {Promise<TimeSeriesForecast>} Time series forecast result
   */
  async forecastTimeSeries(seriesId, historicalData, timestamps = null, horizon = 10, method = 'arima') {
    const data = {
      series_id: seriesId,
      historical_data: historicalData,
      horizon,
      method: { [method]: {} },
    };
    if (timestamps) {
      data.timestamps = timestamps;
    }
    const response = await this.client.post('/api/v1/timeseries/forecast', data);
    return TimeSeriesForecast.fromJSON(response.data.data);
  }

  // Pixel Processing Endpoints

  /**
   * Process pixel data.
   *
   * @param {number} frameId - Frame identifier
   * @param {number} width - Frame width
   * @param {number} height - Frame height
   * @param {Array<number>} pixels - Pixel data (flattened)
   * @param {Object} [options=null] - Processing options
   * @returns {Promise<PixelProcessing>} Pixel processing result
   */
  async processPixels(frameId, width, height, pixels, options = null) {
    const data = {
      frame_id: frameId,
      width,
      height,
      pixels,
    };
    if (options) {
      data.processing_options = options;
    }
    const response = await this.client.post('/api/v1/pixels/process', data);
    return PixelProcessing.fromJSON(response.data.data);
  }

  /**
   * Close the client and clean up resources.
   */
  close() {
    // Axios doesn't require explicit cleanup, but this method is provided for API consistency
  }
}

module.exports = PrismClient;
