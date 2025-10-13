/**
 * TypeScript definitions for PRISM-AI JavaScript Client Library
 */

// Exceptions

export class PrismAPIError extends Error {
  constructor(message: string, statusCode?: number | null, response?: any);
  message: string;
  statusCode: number | null;
  response: any;
  toString(): string;
}

export class AuthenticationError extends PrismAPIError {}
export class AuthorizationError extends PrismAPIError {}
export class NotFoundError extends PrismAPIError {}
export class ValidationError extends PrismAPIError {}

export class RateLimitError extends PrismAPIError {
  constructor(message?: string, statusCode?: number, response?: any, retryAfter?: number | null);
  retryAfter: number | null;
}

export class ServerError extends PrismAPIError {}
export class NetworkError extends PrismAPIError {}
export class TimeoutError extends PrismAPIError {}

// Models

export class ThreatDetection {
  constructor(data: any);
  static fromJSON(data: any): ThreatDetection;
  threatId: string;
  threatType: string;
  confidence: number;
  position: number[];
  velocity: number[] | null;
  timeToImpact: number | null;
  recommendedAction: string | null;
  timestamp: number | null;
}

export class PortfolioOptimization {
  constructor(data: any);
  static fromJSON(data: any): PortfolioOptimization;
  weights: Array<{ symbol: string; weight: number }>;
  expectedReturn: number;
  expectedRisk: number;
  sharpeRatio: number;
  optimizationTimeMs: number;
}

export class LLMQuery {
  constructor(data: any);
  static fromJSON(data: any): LLMQuery;
  text: string;
  modelUsed: string;
  tokensUsed: number;
  costUsd: number;
  latencyMs: number;
}

export class LLMConsensus {
  constructor(data: any);
  static fromJSON(data: any): LLMConsensus;
  consensusText: string;
  confidence: number;
  strategy: string;
  individualResponses: any[];
  totalCostUsd: number;
  totalTimeMs: number;
  agreementRate: number;
}

export class TimeSeriesForecast {
  constructor(data: any);
  static fromJSON(data: any): TimeSeriesForecast;
  seriesId: string;
  predictions: number[];
  timestamps: number[] | null;
  confidenceIntervals: number[][] | null;
  metrics: Record<string, number> | null;
  computationTimeMs: number | null;
}

export class PixelProcessing {
  constructor(data: any);
  static fromJSON(data: any): PixelProcessing;
  frameId: number;
  processedPixels: number[];
  hotspots: any[];
  entropy: number | null;
  tdaFeatures: any[] | null;
  processingTimeMs: number | null;
}

export class SensorFusion {
  constructor(data: any);
  static fromJSON(data: any): SensorFusion;
  numTracks: number;
  fusionQuality: number;
  tracks: any[];
  processingTimeMs: number;
}

export class TrajectoryPrediction {
  constructor(data: any);
  static fromJSON(data: any): TrajectoryPrediction;
  trackId: string;
  model: string;
  confidence: number;
  predictions: any[];
  timeToImpact: number | null;
  impactPoint: any | null;
  uncertainty: any[] | null;
}

// Client

export interface PrismClientOptions {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  verifySsl?: boolean;
}

export class PrismClient {
  constructor(options: PrismClientOptions);

  // Health and Info
  health(): Promise<any>;
  info(): Promise<any>;

  // PWSA Endpoints
  detectThreat(svId: number, timestamp: number, irFrame: any): Promise<ThreatDetection>;
  fuseSensors(svId: number, timestamp: number, sensors: any): Promise<SensorFusion>;
  predictTrajectory(
    trackId: string,
    history: any[],
    predictionHorizon: number,
    model?: string
  ): Promise<TrajectoryPrediction>;
  prioritizeThreats(
    threats: any[],
    strategy?: string,
    defensiveAssets?: any
  ): Promise<any>;

  // Finance Endpoints
  optimizePortfolio(
    assets: any[],
    constraints: any,
    objective?: string
  ): Promise<PortfolioOptimization>;
  assessRisk(
    portfolioId: string,
    positions: any[],
    riskMetrics: string[]
  ): Promise<any>;
  backtestStrategy(
    strategyId: string,
    parameters: any,
    historicalData: any,
    initialCapital: number
  ): Promise<any>;

  // LLM Endpoints
  queryLLM(
    prompt: string,
    model?: string | null,
    temperature?: number,
    maxTokens?: number
  ): Promise<LLMQuery>;
  llmConsensus(
    prompt: string,
    models: any[],
    strategy?: string,
    temperature?: number,
    maxTokens?: number
  ): Promise<LLMConsensus>;
  listLLMModels(): Promise<any[]>;

  // Time Series Endpoints
  forecastTimeSeries(
    seriesId: string,
    historicalData: number[],
    timestamps?: number[] | null,
    horizon?: number,
    method?: string
  ): Promise<TimeSeriesForecast>;

  // Pixel Processing Endpoints
  processPixels(
    frameId: number,
    width: number,
    height: number,
    pixels: number[],
    options?: any
  ): Promise<PixelProcessing>;

  close(): void;
}
