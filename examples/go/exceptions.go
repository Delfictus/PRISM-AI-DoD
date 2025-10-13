package prismclient

import "fmt"

// PrismAPIError is the base error type for all PRISM API errors
type PrismAPIError struct {
	Message    string
	StatusCode int
	Response   interface{}
}

func (e *PrismAPIError) Error() string {
	if e.StatusCode > 0 {
		return fmt.Sprintf("[%d] %s", e.StatusCode, e.Message)
	}
	return e.Message
}

// NewPrismAPIError creates a new PrismAPIError
func NewPrismAPIError(message string, statusCode int, response interface{}) *PrismAPIError {
	return &PrismAPIError{
		Message:    message,
		StatusCode: statusCode,
		Response:   response,
	}
}

// AuthenticationError represents authentication failures (401)
type AuthenticationError struct {
	*PrismAPIError
}

// NewAuthenticationError creates a new AuthenticationError
func NewAuthenticationError(message string, statusCode int, response interface{}) *AuthenticationError {
	return &AuthenticationError{
		PrismAPIError: &PrismAPIError{
			Message:    message,
			StatusCode: statusCode,
			Response:   response,
		},
	}
}

// AuthorizationError represents authorization failures (403)
type AuthorizationError struct {
	*PrismAPIError
}

// NewAuthorizationError creates a new AuthorizationError
func NewAuthorizationError(message string, statusCode int, response interface{}) *AuthorizationError {
	return &AuthorizationError{
		PrismAPIError: &PrismAPIError{
			Message:    message,
			StatusCode: statusCode,
			Response:   response,
		},
	}
}

// NotFoundError represents resource not found errors (404)
type NotFoundError struct {
	*PrismAPIError
}

// NewNotFoundError creates a new NotFoundError
func NewNotFoundError(message string, statusCode int, response interface{}) *NotFoundError {
	return &NotFoundError{
		PrismAPIError: &PrismAPIError{
			Message:    message,
			StatusCode: statusCode,
			Response:   response,
		},
	}
}

// ValidationError represents validation failures (400)
type ValidationError struct {
	*PrismAPIError
}

// NewValidationError creates a new ValidationError
func NewValidationError(message string, statusCode int, response interface{}) *ValidationError {
	return &ValidationError{
		PrismAPIError: &PrismAPIError{
			Message:    message,
			StatusCode: statusCode,
			Response:   response,
		},
	}
}

// RateLimitError represents rate limiting errors (429)
type RateLimitError struct {
	*PrismAPIError
	RetryAfter int
}

// NewRateLimitError creates a new RateLimitError
func NewRateLimitError(message string, statusCode int, response interface{}, retryAfter int) *RateLimitError {
	return &RateLimitError{
		PrismAPIError: &PrismAPIError{
			Message:    message,
			StatusCode: statusCode,
			Response:   response,
		},
		RetryAfter: retryAfter,
	}
}

// ServerError represents server errors (5xx)
type ServerError struct {
	*PrismAPIError
}

// NewServerError creates a new ServerError
func NewServerError(message string, statusCode int, response interface{}) *ServerError {
	return &ServerError{
		PrismAPIError: &PrismAPIError{
			Message:    message,
			StatusCode: statusCode,
			Response:   response,
		},
	}
}

// NetworkError represents network connection failures
type NetworkError struct {
	*PrismAPIError
}

// NewNetworkError creates a new NetworkError
func NewNetworkError(message string) *NetworkError {
	return &NetworkError{
		PrismAPIError: &PrismAPIError{
			Message:    message,
			StatusCode: 0,
		},
	}
}

// TimeoutError represents request timeout errors
type TimeoutError struct {
	*PrismAPIError
}

// NewTimeoutError creates a new TimeoutError
func NewTimeoutError(message string) *TimeoutError {
	return &TimeoutError{
		PrismAPIError: &PrismAPIError{
			Message:    message,
			StatusCode: 0,
		},
	}
}
