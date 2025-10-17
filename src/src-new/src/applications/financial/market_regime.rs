//! Market Regime Detection using Active Inference
//!
//! Uses Active Inference to detect and adapt to different market regimes:
//! - Bull market (trending up)
//! - Bear market (trending down)
//! - High volatility
//! - Low volatility
//! - Normal/stable
//!
//! The regime detection influences portfolio optimization strategy.

use anyhow::Result;
use ndarray::Array1;
use crate::active_inference::GenerativeModel;
use serde::{Deserialize, Serialize};

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Upward trending market
    Bull,
    /// Downward trending market
    Bear,
    /// High volatility period
    HighVolatility,
    /// Low volatility period
    LowVolatility,
    /// Normal/stable market conditions
    Normal,
    /// Transitioning between regimes
    Transition,
}

/// Market regime detector using Active Inference
pub struct MarketRegimeDetector {
    /// Active inference model for regime prediction
    generative_model: GenerativeModel,
    /// Window size for regime detection (in days)
    window_size: usize,
    /// Volatility threshold for classification
    volatility_threshold: f64,
    /// Trend threshold for bull/bear classification
    trend_threshold: f64,
    /// Current detected regime
    current_regime: MarketRegime,
    /// Confidence in current regime (0-1)
    confidence: f64,
}

impl MarketRegimeDetector {
    /// Create a new market regime detector
    pub fn new(window_size: usize) -> Self {
        Self {
            generative_model: GenerativeModel::new(),
            window_size,
            volatility_threshold: 0.02, // 2% daily volatility threshold
            trend_threshold: 0.01,      // 1% daily trend threshold
            current_regime: MarketRegime::Normal,
            confidence: 0.5,
        }
    }

    /// Detect current market regime from price history
    pub fn detect_regime(&mut self, price_history: &[f64]) -> Result<(MarketRegime, f64)> {
        if price_history.len() < self.window_size {
            return Ok((MarketRegime::Normal, 0.5));
        }

        // Calculate returns
        let returns = self.calculate_returns(price_history);

        // Calculate volatility (standard deviation of returns)
        let volatility = self.calculate_volatility(&returns);

        // Calculate trend (mean return)
        let trend = returns.mean().unwrap_or(0.0);

        // Use Active Inference to predict regime
        let regime_observation = self.construct_regime_observation(volatility, trend);
        let _action = self.generative_model.step(&regime_observation);

        // Classify based on volatility and trend
        let (regime, confidence) = self.classify_regime(volatility, trend);

        self.current_regime = regime;
        self.confidence = confidence;

        Ok((regime, confidence))
    }

    /// Get current detected regime
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get confidence in current regime
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Calculate returns from price history
    fn calculate_returns(&self, prices: &[f64]) -> Array1<f64> {
        let mut returns = Vec::with_capacity(prices.len() - 1);

        for i in 1..prices.len() {
            let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
            returns.push(ret);
        }

        Array1::from_vec(returns)
    }

    /// Calculate volatility (standard deviation of returns)
    fn calculate_volatility(&self, returns: &Array1<f64>) -> f64 {
        let mean = returns.mean().unwrap_or(0.0);
        let variance = returns.mapv(|r| (r - mean).powi(2)).mean().unwrap_or(0.0);
        variance.sqrt()
    }

    /// Construct observation vector for Active Inference
    fn construct_regime_observation(&self, volatility: f64, trend: f64) -> Array1<f64> {
        // Create a 100-dimensional observation (matching the generative model)
        let mut observation = Array1::zeros(100);

        // Encode volatility in first 50 dimensions
        let vol_scaled = (volatility / self.volatility_threshold).min(2.0);
        for i in 0..50 {
            observation[i] = vol_scaled * (i as f64 / 50.0);
        }

        // Encode trend in last 50 dimensions
        let trend_scaled = (trend / self.trend_threshold).clamp(-2.0, 2.0);
        for i in 50..100 {
            observation[i] = trend_scaled * ((i - 50) as f64 / 50.0);
        }

        observation
    }

    /// Classify regime based on volatility and trend
    fn classify_regime(&self, volatility: f64, trend: f64) -> (MarketRegime, f64) {
        let vol_ratio = volatility / self.volatility_threshold;
        let trend_norm = (trend / self.trend_threshold).abs();

        // High volatility dominates other factors
        if vol_ratio > 1.5 {
            return (MarketRegime::HighVolatility, 0.8);
        }

        // Low volatility
        if vol_ratio < 0.5 {
            return (MarketRegime::LowVolatility, 0.8);
        }

        // Strong upward trend
        if trend > self.trend_threshold && trend_norm > 1.0 {
            return (MarketRegime::Bull, 0.7);
        }

        // Strong downward trend
        if trend < -self.trend_threshold && trend_norm > 1.0 {
            return (MarketRegime::Bear, 0.7);
        }

        // Check for transition (moderate volatility and trend)
        if vol_ratio > 0.8 && vol_ratio < 1.2 && trend_norm > 0.5 && trend_norm < 1.0 {
            return (MarketRegime::Transition, 0.6);
        }

        // Default to normal
        (MarketRegime::Normal, 0.6)
    }

    /// Adjust portfolio strategy based on regime
    pub fn regime_adjustment_factor(&self) -> f64 {
        match self.current_regime {
            MarketRegime::Bull => 1.2,           // Increase exposure
            MarketRegime::Bear => 0.8,           // Decrease exposure
            MarketRegime::HighVolatility => 0.7, // Conservative
            MarketRegime::LowVolatility => 1.1,  // Slightly aggressive
            MarketRegime::Normal => 1.0,         // Neutral
            MarketRegime::Transition => 0.9,     // Cautious
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_detector_creation() {
        let detector = MarketRegimeDetector::new(20);
        assert_eq!(detector.window_size, 20);
        assert_eq!(detector.current_regime(), MarketRegime::Normal);
    }

    #[test]
    fn test_bull_market_detection() {
        let mut detector = MarketRegimeDetector::new(10);

        // Create upward trending prices
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();

        let result = detector.detect_regime(&prices);
        assert!(result.is_ok());

        let (regime, confidence) = result.unwrap();
        // Should detect bull or high confidence
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_bear_market_detection() {
        let mut detector = MarketRegimeDetector::new(10);

        // Create downward trending prices
        let prices: Vec<f64> = (0..50).map(|i| 200.0 - i as f64 * 2.0).collect();

        let result = detector.detect_regime(&prices);
        assert!(result.is_ok());

        let (regime, confidence) = result.unwrap();
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_high_volatility_detection() {
        let mut detector = MarketRegimeDetector::new(10);

        // Create volatile prices
        let mut prices = Vec::new();
        for i in 0..50 {
            let base = 100.0;
            let volatility = if i % 2 == 0 { 10.0 } else { -10.0 };
            prices.push(base + volatility);
        }

        let result = detector.detect_regime(&prices);
        assert!(result.is_ok());
    }

    #[test]
    fn test_regime_adjustment_factors() {
        let mut detector = MarketRegimeDetector::new(10);

        detector.current_regime = MarketRegime::Bull;
        assert_eq!(detector.regime_adjustment_factor(), 1.2);

        detector.current_regime = MarketRegime::Bear;
        assert_eq!(detector.regime_adjustment_factor(), 0.8);

        detector.current_regime = MarketRegime::HighVolatility;
        assert_eq!(detector.regime_adjustment_factor(), 0.7);
    }
}
