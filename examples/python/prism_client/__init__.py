"""
PRISM-AI Python Client Library

A comprehensive Python SDK for interacting with the PRISM-AI REST API.

Example usage:
    >>> from prism_client import PrismClient
    >>> client = PrismClient(api_key="your-key-here")
    >>> health = client.health()
    >>> print(health)

For more examples, see the examples/ directory.
"""

from .client import PrismClient
from .exceptions import (
    PrismAPIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ServerError,
)
from .models import (
    ThreatDetection,
    PortfolioOptimization,
    LLMQuery,
    TimeSeriesForecast,
)

__version__ = "0.1.0"
__author__ = "PRISM-AI Team"

__all__ = [
    "PrismClient",
    "PrismAPIError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "ServerError",
    "ThreatDetection",
    "PortfolioOptimization",
    "LLMQuery",
    "TimeSeriesForecast",
]
