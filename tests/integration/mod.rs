// Integration test module
// Run these tests with: cargo test --test integration --features api_server

mod test_api_health;
mod test_authentication;
mod test_finance_endpoints;
mod test_llm_endpoints;
mod test_performance;
mod test_pwsa_endpoints;
mod test_websocket;
mod test_worker_integrations;

// Phase 2 Integration Tests (Worker 7 QA Lead)
mod phase2_integration;

// Test utilities and helpers
pub mod common;
