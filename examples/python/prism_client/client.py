"""
Main PRISM-AI client implementation.
"""

import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin

from .exceptions import (
    PrismAPIError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
    NetworkError,
    TimeoutError as PrismTimeoutError,
)
from .models import (
    ThreatDetection,
    PortfolioOptimization,
    LLMQuery,
    LLMConsensus,
    TimeSeriesForecast,
    PixelProcessing,
    SensorFusion,
    TrajectoryPrediction,
)


class PrismClient:
    """
    PRISM-AI API Client.

    Args:
        api_key: API authentication key
        base_url: Base URL for the API (default: http://localhost:8080)
        timeout: Request timeout in seconds (default: 30)
        verify_ssl: Verify SSL certificates (default: True)

    Example:
        >>> client = PrismClient(api_key="your-key")
        >>> health = client.health()
        >>> print(health["status"])
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8080",
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "prism-python-client/0.1.0",
            }
        )

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to API."""
        url = urljoin(self.base_url, endpoint)

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )

            # Handle error responses
            if response.status_code == 401:
                raise AuthenticationError(
                    "Authentication failed. Check your API key.",
                    status_code=401,
                    response=response,
                )
            elif response.status_code == 403:
                raise AuthorizationError(
                    "Permission denied.",
                    status_code=403,
                    response=response,
                )
            elif response.status_code == 404:
                raise NotFoundError(
                    "Resource not found.",
                    status_code=404,
                    response=response,
                )
            elif response.status_code == 400:
                error_msg = response.json().get("error", "Bad request")
                raise ValidationError(
                    error_msg,
                    status_code=400,
                    response=response,
                )
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    "Rate limit exceeded.",
                    status_code=429,
                    response=response,
                    retry_after=int(retry_after) if retry_after else None,
                )
            elif response.status_code >= 500:
                raise ServerError(
                    f"Server error: {response.status_code}",
                    status_code=response.status_code,
                    response=response,
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            raise PrismTimeoutError(f"Request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Connection failed: {str(e)}")
        except requests.exceptions.RequestException as e:
            if not isinstance(e, PrismAPIError):
                raise PrismAPIError(f"Request failed: {str(e)}")
            raise

    # Health and Info

    def health(self) -> Dict[str, Any]:
        """Check API health status."""
        return self._make_request("GET", "/health")

    def info(self) -> Dict[str, Any]:
        """Get API information."""
        return self._make_request("GET", "/")

    # PWSA Endpoints

    def detect_threat(
        self,
        sv_id: int,
        timestamp: int,
        ir_frame: Dict[str, Any],
    ) -> ThreatDetection:
        """
        Detect threat from IR sensor data.

        Args:
            sv_id: Space vehicle ID
            timestamp: Unix timestamp
            ir_frame: IR frame data with width, height, centroid, hotspots

        Returns:
            ThreatDetection object
        """
        data = {
            "sv_id": sv_id,
            "timestamp": timestamp,
            "ir_frame": ir_frame,
        }
        response = self._make_request("POST", "/api/v1/pwsa/detect", data=data)
        return ThreatDetection.from_dict(response["data"])

    def fuse_sensors(
        self,
        sv_id: int,
        timestamp: int,
        sensors: Dict[str, Any],
    ) -> SensorFusion:
        """
        Fuse multi-sensor data.

        Args:
            sv_id: Space vehicle ID
            timestamp: Unix timestamp
            sensors: Sensor data (IR, radar, optical)

        Returns:
            SensorFusion object
        """
        data = {
            "sv_id": sv_id,
            "timestamp": timestamp,
            "sensors": sensors,
        }
        response = self._make_request("POST", "/api/v1/pwsa/fuse", data=data)
        return SensorFusion.from_dict(response["data"])

    def predict_trajectory(
        self,
        track_id: str,
        history: List[Dict[str, Any]],
        prediction_horizon: int,
        model: str = "kalman_filter",
    ) -> TrajectoryPrediction:
        """
        Predict threat trajectory.

        Args:
            track_id: Track identifier
            history: Historical track data
            prediction_horizon: Prediction time (seconds)
            model: Prediction model (kalman_filter, neural_network)

        Returns:
            TrajectoryPrediction object
        """
        data = {
            "track_id": track_id,
            "history": history,
            "prediction_horizon": prediction_horizon,
            "model": model,
        }
        response = self._make_request("POST", "/api/v1/pwsa/predict", data=data)
        return TrajectoryPrediction.from_dict(response["data"])

    def prioritize_threats(
        self,
        threats: List[Dict[str, Any]],
        strategy: str = "time_weighted_risk",
        defensive_assets: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prioritize multiple threats.

        Args:
            threats: List of threat data
            strategy: Prioritization strategy
            defensive_assets: Available defensive resources

        Returns:
            Prioritized threat list
        """
        data = {
            "threats": threats,
            "prioritization_strategy": strategy,
        }
        if defensive_assets:
            data["defensive_assets"] = defensive_assets

        response = self._make_request("POST", "/api/v1/pwsa/prioritize", data=data)
        return response["data"]

    # Finance Endpoints

    def optimize_portfolio(
        self,
        assets: List[Dict[str, Any]],
        constraints: Dict[str, Any],
        objective: str = "maximize_sharpe",
    ) -> PortfolioOptimization:
        """
        Optimize portfolio allocation.

        Args:
            assets: List of assets with expected returns and volatility
            constraints: Portfolio constraints
            objective: Optimization objective

        Returns:
            PortfolioOptimization object
        """
        data = {
            "assets": assets,
            "constraints": constraints,
            "objective": objective,
        }
        response = self._make_request("POST", "/api/v1/finance/optimize", data=data)
        return PortfolioOptimization.from_dict(response["data"])

    def assess_risk(
        self,
        portfolio_id: str,
        positions: List[Dict[str, Any]],
        risk_metrics: List[str],
    ) -> Dict[str, Any]:
        """
        Assess portfolio risk.

        Args:
            portfolio_id: Portfolio identifier
            positions: Current positions
            risk_metrics: Metrics to calculate (var, cvar, beta, etc.)

        Returns:
            Risk assessment results
        """
        data = {
            "portfolio_id": portfolio_id,
            "positions": positions,
            "risk_metrics": risk_metrics,
        }
        response = self._make_request("POST", "/api/v1/finance/risk", data=data)
        return response["data"]

    def backtest_strategy(
        self,
        strategy_id: str,
        parameters: Dict[str, Any],
        historical_data: Dict[str, Any],
        initial_capital: float,
    ) -> Dict[str, Any]:
        """
        Backtest trading strategy.

        Args:
            strategy_id: Strategy identifier
            parameters: Strategy parameters
            historical_data: Historical market data
            initial_capital: Starting capital

        Returns:
            Backtest results
        """
        data = {
            "strategy_id": strategy_id,
            "parameters": parameters,
            "historical_data": historical_data,
            "initial_capital": initial_capital,
        }
        response = self._make_request("POST", "/api/v1/finance/backtest", data=data)
        return response["data"]

    # LLM Endpoints

    def query_llm(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMQuery:
        """
        Query a language model.

        Args:
            prompt: Input prompt
            model: Model name (optional, uses default)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate

        Returns:
            LLMQuery object
        """
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            data["model"] = model

        response = self._make_request("POST", "/api/v1/llm/query", data=data)
        return LLMQuery.from_dict(response["data"])

    def llm_consensus(
        self,
        prompt: str,
        models: List[Dict[str, Any]],
        strategy: str = "majority_vote",
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> LLMConsensus:
        """
        Query multiple models with consensus.

        Args:
            prompt: Input prompt
            models: List of models with weights
            strategy: Consensus strategy
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            LLMConsensus object
        """
        data = {
            "prompt": prompt,
            "models": models,
            "strategy": strategy,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._make_request("POST", "/api/v1/llm/consensus", data=data)
        return LLMConsensus.from_dict(response["data"])

    def list_llm_models(self) -> List[Dict[str, Any]]:
        """List available LLM models."""
        response = self._make_request("GET", "/api/v1/llm/models")
        return response["data"]["models"]

    # Time Series Endpoints

    def forecast_timeseries(
        self,
        series_id: str,
        historical_data: List[float],
        timestamps: Optional[List[int]] = None,
        horizon: int = 10,
        method: str = "arima",
    ) -> TimeSeriesForecast:
        """
        Forecast time series.

        Args:
            series_id: Series identifier
            historical_data: Historical values
            timestamps: Optional timestamps
            horizon: Forecast horizon
            method: Forecasting method (arima, lstm, etc.)

        Returns:
            TimeSeriesForecast object
        """
        data = {
            "series_id": series_id,
            "historical_data": historical_data,
            "horizon": horizon,
            "method": {method: {}},
        }
        if timestamps:
            data["timestamps"] = timestamps

        response = self._make_request("POST", "/api/v1/timeseries/forecast", data=data)
        return TimeSeriesForecast.from_dict(response["data"])

    # Pixel Processing Endpoints

    def process_pixels(
        self,
        frame_id: int,
        width: int,
        height: int,
        pixels: List[int],
        options: Optional[Dict[str, Any]] = None,
    ) -> PixelProcessing:
        """
        Process pixel data.

        Args:
            frame_id: Frame identifier
            width: Frame width
            height: Frame height
            pixels: Pixel data (flattened)
            options: Processing options

        Returns:
            PixelProcessing object
        """
        data = {
            "frame_id": frame_id,
            "width": width,
            "height": height,
            "pixels": pixels,
        }
        if options:
            data["processing_options"] = options

        response = self._make_request("POST", "/api/v1/pixels/process", data=data)
        return PixelProcessing.from_dict(response["data"])

    def close(self):
        """Close the client session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
