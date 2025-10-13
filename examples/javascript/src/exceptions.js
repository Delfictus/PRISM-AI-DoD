/**
 * Exception classes for PRISM-AI client library.
 */

/**
 * Base exception for all PRISM API errors.
 */
class PrismAPIError extends Error {
  constructor(message, statusCode = null, response = null) {
    super(message);
    this.name = this.constructor.name;
    this.message = message;
    this.statusCode = statusCode;
    this.response = response;
    Error.captureStackTrace(this, this.constructor);
  }

  toString() {
    if (this.statusCode) {
      return `[${this.statusCode}] ${this.message}`;
    }
    return this.message;
  }
}

/**
 * Raised when authentication fails (401).
 */
class AuthenticationError extends PrismAPIError {
  constructor(message = 'Authentication failed', statusCode = 401, response = null) {
    super(message, statusCode, response);
  }
}

/**
 * Raised when user lacks permission (403).
 */
class AuthorizationError extends PrismAPIError {
  constructor(message = 'Permission denied', statusCode = 403, response = null) {
    super(message, statusCode, response);
  }
}

/**
 * Raised when resource is not found (404).
 */
class NotFoundError extends PrismAPIError {
  constructor(message = 'Resource not found', statusCode = 404, response = null) {
    super(message, statusCode, response);
  }
}

/**
 * Raised when request validation fails (400).
 */
class ValidationError extends PrismAPIError {
  constructor(message = 'Validation failed', statusCode = 400, response = null) {
    super(message, statusCode, response);
  }
}

/**
 * Raised when rate limit is exceeded (429).
 */
class RateLimitError extends PrismAPIError {
  constructor(message = 'Rate limit exceeded', statusCode = 429, response = null, retryAfter = null) {
    super(message, statusCode, response);
    this.retryAfter = retryAfter;
  }
}

/**
 * Raised when server encounters an error (5xx).
 */
class ServerError extends PrismAPIError {
  constructor(message = 'Server error', statusCode = 500, response = null) {
    super(message, statusCode, response);
  }
}

/**
 * Raised when network connection fails.
 */
class NetworkError extends PrismAPIError {
  constructor(message = 'Network connection failed', statusCode = null, response = null) {
    super(message, statusCode, response);
  }
}

/**
 * Raised when request times out.
 */
class TimeoutError extends PrismAPIError {
  constructor(message = 'Request timed out', statusCode = null, response = null) {
    super(message, statusCode, response);
  }
}

module.exports = {
  PrismAPIError,
  AuthenticationError,
  AuthorizationError,
  NotFoundError,
  ValidationError,
  RateLimitError,
  ServerError,
  NetworkError,
  TimeoutError,
};
