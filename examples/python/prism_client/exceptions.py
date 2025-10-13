"""
Exception classes for PRISM-AI client library.
"""


class PrismAPIError(Exception):
    """Base exception for all PRISM API errors."""

    def __init__(self, message, status_code=None, response=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response

    def __str__(self):
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(PrismAPIError):
    """Raised when authentication fails (401)."""

    pass


class AuthorizationError(PrismAPIError):
    """Raised when user lacks permission (403)."""

    pass


class NotFoundError(PrismAPIError):
    """Raised when resource is not found (404)."""

    pass


class ValidationError(PrismAPIError):
    """Raised when request validation fails (400)."""

    pass


class RateLimitError(PrismAPIError):
    """Raised when rate limit is exceeded (429)."""

    def __init__(self, message, status_code=429, response=None, retry_after=None):
        super().__init__(message, status_code, response)
        self.retry_after = retry_after


class ServerError(PrismAPIError):
    """Raised when server encounters an error (5xx)."""

    pass


class NetworkError(PrismAPIError):
    """Raised when network connection fails."""

    pass


class TimeoutError(PrismAPIError):
    """Raised when request times out."""

    pass
