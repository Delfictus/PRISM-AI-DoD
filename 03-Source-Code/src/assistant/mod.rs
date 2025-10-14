//! PRISM Assistant Module
//!
//! Fully offline, autonomous AI assistant with:
//! - Local GPU LLM (zero cost, privacy-first)
//! - Code execution (Python, Rust, shell)
//! - PRISM tool calling (finance, drug discovery, robotics, etc.)
//! - Works completely offline - no internet required!

pub mod autonomous_agent;
pub mod prism_assistant;

pub use autonomous_agent::{AutonomousAgent, SafetyMode, ToolCall, ToolResult};
pub use prism_assistant::{PrismAssistant, AssistantMode, ChatResponse, ChatResponseWithTools, AssistantStatus};
