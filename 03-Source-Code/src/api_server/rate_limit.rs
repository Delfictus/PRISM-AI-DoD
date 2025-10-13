//! Advanced rate limiting with hybrid token bucket + leaky bucket algorithms
//!
//! Provides sophisticated rate limiting with multiple strategies:
//! - Token bucket for burst handling
//! - Leaky bucket for sustained rate control
//! - Sliding window log for precision
//! - Exponential backoff for repeat violators

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};

/// Hybrid rate limiter combining multiple algorithms
#[derive(Clone)]
pub struct HybridRateLimiter {
    inner: Arc<Mutex<RateLimiterState>>,
}

struct RateLimiterState {
    /// Token bucket for burst handling
    token_buckets: HashMap<String, TokenBucket>,

    /// Leaky bucket for sustained rate
    leaky_buckets: HashMap<String, LeakyBucket>,

    /// Sliding window for precision
    windows: HashMap<String, SlidingWindow>,

    /// Backoff state for repeat violators
    backoff_states: HashMap<String, BackoffState>,

    /// Configuration
    config: RateLimitConfig,
}

#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum burst size (tokens)
    pub max_burst: u32,

    /// Sustained rate (requests per second)
    pub requests_per_second: u32,

    /// Sliding window duration (seconds)
    pub window_duration_secs: u64,

    /// Maximum requests in window
    pub max_requests_per_window: u32,

    /// Enable exponential backoff
    pub enable_backoff: bool,

    /// Backoff base duration (ms)
    pub backoff_base_ms: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_burst: 100,
            requests_per_second: 100,
            window_duration_secs: 60,
            max_requests_per_window: 1000,
            enable_backoff: true,
            backoff_base_ms: 1000,
        }
    }
}

impl HybridRateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RateLimiterState {
                token_buckets: HashMap::new(),
                leaky_buckets: HashMap::new(),
                windows: HashMap::new(),
                backoff_states: HashMap::new(),
                config,
            })),
        }
    }

    /// Check if request should be allowed
    pub fn check_rate_limit(&self, client_id: &str) -> RateLimitDecision {
        let mut state = self.inner.lock().unwrap();

        // 1. Check token bucket (fast burst handling)
        let token_bucket = state
            .token_buckets
            .entry(client_id.to_string())
            .or_insert_with(|| TokenBucket::new(state.config.max_burst, state.config.requests_per_second));

        if !token_bucket.try_consume() {
            return state.handle_rate_limit_exceeded(client_id, RateLimitReason::BurstExceeded);
        }

        // 2. Check leaky bucket (sustained rate)
        let leaky_bucket = state
            .leaky_buckets
            .entry(client_id.to_string())
            .or_insert_with(|| LeakyBucket::new(state.config.requests_per_second));

        if !leaky_bucket.try_add() {
            // Refund token bucket since we're rejecting
            token_bucket.refund();
            return state.handle_rate_limit_exceeded(client_id, RateLimitReason::SustainedRateExceeded);
        }

        // 3. Check sliding window (precision over longer period)
        let window = state
            .windows
            .entry(client_id.to_string())
            .or_insert_with(|| {
                SlidingWindow::new(
                    Duration::from_secs(state.config.window_duration_secs),
                    state.config.max_requests_per_window,
                )
            });

        if !window.try_add() {
            // Refund both buckets
            token_bucket.refund();
            leaky_bucket.leak_one();
            return state.handle_rate_limit_exceeded(client_id, RateLimitReason::WindowExceeded);
        }

        // 4. Check backoff state (if client was previously rate-limited)
        if state.config.enable_backoff {
            if let Some(backoff) = state.backoff_states.get(client_id) {
                if backoff.is_in_backoff() {
                    // Refund all
                    token_bucket.refund();
                    leaky_bucket.leak_one();
                    window.remove_last();
                    return RateLimitDecision::Denied {
                        reason: RateLimitReason::InBackoff,
                        retry_after_ms: backoff.retry_after_ms(),
                    };
                }
            }
        }

        // All checks passed - allow request
        // Clear backoff if it existed
        if state.config.enable_backoff {
            state.backoff_states.remove(client_id);
        }

        RateLimitDecision::Allowed {
            remaining: token_bucket.remaining(),
            reset_after_ms: token_bucket.reset_time_ms(),
        }
    }

    /// Get current rate limit status for client
    pub fn get_status(&self, client_id: &str) -> RateLimitStatus {
        let state = self.inner.lock().unwrap();

        let tokens_remaining = state
            .token_buckets
            .get(client_id)
            .map(|b| b.remaining())
            .unwrap_or(state.config.max_burst);

        let window_remaining = state
            .windows
            .get(client_id)
            .map(|w| w.remaining())
            .unwrap_or(state.config.max_requests_per_window);

        let in_backoff = state
            .backoff_states
            .get(client_id)
            .map(|b| b.is_in_backoff())
            .unwrap_or(false);

        RateLimitStatus {
            tokens_remaining,
            window_remaining,
            in_backoff,
        }
    }
}

impl RateLimiterState {
    fn handle_rate_limit_exceeded(&mut self, client_id: &str, reason: RateLimitReason) -> RateLimitDecision {
        if self.config.enable_backoff {
            let backoff = self
                .backoff_states
                .entry(client_id.to_string())
                .or_insert_with(|| BackoffState::new(self.config.backoff_base_ms));

            backoff.increment();

            RateLimitDecision::Denied {
                reason,
                retry_after_ms: backoff.retry_after_ms(),
            }
        } else {
            RateLimitDecision::Denied {
                reason,
                retry_after_ms: 1000, // Default 1 second
            }
        }
    }
}

/// Token bucket for burst handling
struct TokenBucket {
    tokens: f64,
    max_tokens: u32,
    refill_rate: u32,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(max_tokens: u32, refill_rate: u32) -> Self {
        Self {
            tokens: max_tokens as f64,
            max_tokens,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    fn try_consume(&mut self) -> bool {
        self.refill();

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn refund(&mut self) {
        self.tokens = (self.tokens + 1.0).min(self.max_tokens as f64);
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();

        let tokens_to_add = elapsed * self.refill_rate as f64;
        self.tokens = (self.tokens + tokens_to_add).min(self.max_tokens as f64);

        self.last_refill = now;
    }

    fn remaining(&self) -> u32 {
        self.tokens.floor() as u32
    }

    fn reset_time_ms(&self) -> u64 {
        if self.tokens >= self.max_tokens as f64 {
            0
        } else {
            let tokens_needed = self.max_tokens as f64 - self.tokens;
            ((tokens_needed / self.refill_rate as f64) * 1000.0) as u64
        }
    }
}

/// Leaky bucket for sustained rate control
struct LeakyBucket {
    queue: VecDeque<Instant>,
    leak_rate: u32,
    last_leak: Instant,
}

impl LeakyBucket {
    fn new(leak_rate: u32) -> Self {
        Self {
            queue: VecDeque::new(),
            leak_rate,
            last_leak: Instant::now(),
        }
    }

    fn try_add(&mut self) -> bool {
        self.leak();

        if self.queue.len() < self.leak_rate as usize {
            self.queue.push_back(Instant::now());
            true
        } else {
            false
        }
    }

    fn leak(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_leak).as_secs_f64();
        let items_to_leak = (elapsed * self.leak_rate as f64).floor() as usize;

        for _ in 0..items_to_leak.min(self.queue.len()) {
            self.queue.pop_front();
        }

        if items_to_leak > 0 {
            self.last_leak = now;
        }
    }

    fn leak_one(&mut self) {
        self.queue.pop_back();
    }
}

/// Sliding window log for precision
struct SlidingWindow {
    requests: VecDeque<Instant>,
    window_duration: Duration,
    max_requests: u32,
}

impl SlidingWindow {
    fn new(window_duration: Duration, max_requests: u32) -> Self {
        Self {
            requests: VecDeque::new(),
            window_duration,
            max_requests,
        }
    }

    fn try_add(&mut self) -> bool {
        self.cleanup();

        if self.requests.len() < self.max_requests as usize {
            self.requests.push_back(Instant::now());
            true
        } else {
            false
        }
    }

    fn cleanup(&mut self) {
        let now = Instant::now();
        let cutoff = now - self.window_duration;

        while let Some(&front) = self.requests.front() {
            if front < cutoff {
                self.requests.pop_front();
            } else {
                break;
            }
        }
    }

    fn remove_last(&mut self) {
        self.requests.pop_back();
    }

    fn remaining(&self) -> u32 {
        self.max_requests - self.requests.len() as u32
    }
}

/// Exponential backoff for repeat violators
struct BackoffState {
    violations: u32,
    last_violation: Instant,
    base_delay_ms: u64,
}

impl BackoffState {
    fn new(base_delay_ms: u64) -> Self {
        Self {
            violations: 0,
            last_violation: Instant::now(),
            base_delay_ms,
        }
    }

    fn increment(&mut self) {
        self.violations += 1;
        self.last_violation = Instant::now();
    }

    fn is_in_backoff(&self) -> bool {
        let elapsed = Instant::now().duration_since(self.last_violation);
        elapsed < self.backoff_duration()
    }

    fn retry_after_ms(&self) -> u64 {
        let elapsed = Instant::now().duration_since(self.last_violation);
        let backoff = self.backoff_duration();

        if elapsed < backoff {
            (backoff - elapsed).as_millis() as u64
        } else {
            0
        }
    }

    fn backoff_duration(&self) -> Duration {
        // Exponential backoff: base * 2^violations, capped at 5 minutes
        let exp = 2_u64.pow(self.violations.min(10));
        let delay_ms = (self.base_delay_ms * exp).min(300_000); // Max 5 minutes
        Duration::from_millis(delay_ms)
    }
}

/// Rate limit decision
#[derive(Debug, Clone)]
pub enum RateLimitDecision {
    Allowed {
        remaining: u32,
        reset_after_ms: u64,
    },
    Denied {
        reason: RateLimitReason,
        retry_after_ms: u64,
    },
}

/// Reason for rate limit
#[derive(Debug, Clone, Copy)]
pub enum RateLimitReason {
    BurstExceeded,
    SustainedRateExceeded,
    WindowExceeded,
    InBackoff,
}

/// Current rate limit status
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    pub tokens_remaining: u32,
    pub window_remaining: u32,
    pub in_backoff: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_token_bucket_burst() {
        let config = RateLimitConfig {
            max_burst: 10,
            requests_per_second: 10,
            ..Default::default()
        };

        let limiter = HybridRateLimiter::new(config);

        // Should allow 10 burst requests
        for _ in 0..10 {
            match limiter.check_rate_limit("client1") {
                RateLimitDecision::Allowed { .. } => {},
                RateLimitDecision::Denied { .. } => panic!("Should allow burst"),
            }
        }

        // 11th should be denied
        match limiter.check_rate_limit("client1") {
            RateLimitDecision::Denied { .. } => {},
            RateLimitDecision::Allowed { .. } => panic!("Should deny after burst"),
        }
    }

    #[test]
    fn test_token_refill() {
        let config = RateLimitConfig {
            max_burst: 5,
            requests_per_second: 10,
            ..Default::default()
        };

        let limiter = HybridRateLimiter::new(config);

        // Consume all tokens
        for _ in 0..5 {
            limiter.check_rate_limit("client1");
        }

        // Wait for refill (100ms should add ~1 token at 10/sec rate)
        thread::sleep(Duration::from_millis(150));

        // Should allow one more request
        match limiter.check_rate_limit("client1") {
            RateLimitDecision::Allowed { .. } => {},
            RateLimitDecision::Denied { .. } => panic!("Should allow after refill"),
        }
    }

    #[test]
    fn test_backoff_escalation() {
        let config = RateLimitConfig {
            max_burst: 1,
            requests_per_second: 1,
            enable_backoff: true,
            backoff_base_ms: 100,
            ..Default::default()
        };

        let limiter = HybridRateLimiter::new(config);

        // Consume token
        limiter.check_rate_limit("client1");

        // Violate rate limit multiple times
        let first_violation = limiter.check_rate_limit("client1");
        let second_violation = limiter.check_rate_limit("client1");

        // Backoff should increase
        match (first_violation, second_violation) {
            (
                RateLimitDecision::Denied { retry_after_ms: retry1, .. },
                RateLimitDecision::Denied { retry_after_ms: retry2, .. },
            ) => {
                assert!(retry2 >= retry1, "Backoff should escalate");
            }
            _ => panic!("Should deny both violations"),
        }
    }

    #[test]
    fn test_per_client_isolation() {
        let config = RateLimitConfig {
            max_burst: 2,
            ..Default::default()
        };

        let limiter = HybridRateLimiter::new(config);

        // Client 1 uses both tokens
        limiter.check_rate_limit("client1");
        limiter.check_rate_limit("client1");

        // Client 2 should still have tokens
        match limiter.check_rate_limit("client2") {
            RateLimitDecision::Allowed { .. } => {},
            RateLimitDecision::Denied { .. } => panic!("Client 2 should not be affected by client 1"),
        }
    }
}
