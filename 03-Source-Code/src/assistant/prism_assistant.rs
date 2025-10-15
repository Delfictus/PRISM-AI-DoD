//! PRISM Assistant - Unified AI Assistant with Local/Cloud Toggle
//!
//! Features:
//! - **Local-only mode**: GPU LLM (privacy-first, zero API cost, fully offline)
//! - **Cloud mode**: Mission Charlie (4 providers, thermodynamic consensus)
//! - **Hybrid mode**: Local for fast/simple, cloud for complex queries
//! - **Tool calling**: Full PRISM module access
//! - **Code execution**: Python, Rust, shell commands (sandboxed)
//!
//! WORKS COMPLETELY OFFLINE - No internet required!

use anyhow::Result;
use serde::{Serialize, Deserialize};

// Simplified - using basic types for now
use crate::assistant::autonomous_agent::{AutonomousAgent, SafetyMode, ToolCall, ToolResult};

// Local LLM integration
use crate::assistant::local_llm::GpuLocalLLMSystem;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssistantMode {
    /// Local GPU LLM only (privacy-first, zero cost, fully offline)
    LocalOnly,
    /// Cloud LLMs only (Mission Charlie) - requires internet
    CloudOnly,
    /// Hybrid: Auto-route based on query complexity
    Hybrid {
        complexity_threshold: f64
    },
}

pub struct PrismAssistant {
    mode: AssistantMode,
    tools_enabled: bool,
    agent: Option<AutonomousAgent>,
    model_path: Option<String>,
    llm_system: Option<GpuLocalLLMSystem>,
}

impl PrismAssistant {
    /// Create new PRISM Assistant
    pub async fn new(mode: AssistantMode, model_path: Option<String>) -> Result<Self> {
        let agent = Some(AutonomousAgent::new(SafetyMode::Balanced)?);

        // Load local LLM model if needed
        let llm_system = match &mode {
            AssistantMode::LocalOnly | AssistantMode::Hybrid { .. } => {
                let path = Self::find_model_path(model_path.as_deref())?;
                println!("🔄 Loading GPU LLM from: {}", path);

                let system = GpuLocalLLMSystem::from_gguf_file(&path)?;
                println!("✅ GPU LLM loaded successfully");

                Some(system)
            }
            AssistantMode::CloudOnly => None,
        };

        Ok(Self {
            mode,
            tools_enabled: true,
            agent,
            model_path,
            llm_system,
        })
    }

    /// Chat with PRISM Assistant (simple mode)
    pub async fn chat(&mut self, message: &str) -> Result<ChatResponse> {
        match &self.mode {
            AssistantMode::LocalOnly => self.chat_local(message).await,
            AssistantMode::CloudOnly => self.chat_cloud(message).await,
            AssistantMode::Hybrid { complexity_threshold } => {
                let complexity = self.estimate_complexity(message);
                if complexity < *complexity_threshold {
                    self.chat_local(message).await
                } else {
                    match self.chat_cloud(message).await {
                        Ok(response) => Ok(response),
                        Err(_) => {
                            println!("⚠️  Cloud unavailable, falling back to local");
                            self.chat_local(message).await
                        }
                    }
                }
            }
        }
    }

    /// Chat with tool execution enabled (RECOMMENDED)
    pub async fn chat_with_tools(&mut self, message: &str) -> Result<ChatResponseWithTools> {
        let llm_response = self.chat(message).await?;
        let mut tool_results = Vec::new();

        if self.tools_enabled {
            if let Some(agent) = &self.agent {
                let tool_calls = agent.extract_tool_calls(&llm_response.text);

                for call in tool_calls {
                    let result = match call {
                        ToolCall::Python { code } => {
                            println!("🐍 Executing Python code...");
                            agent.execute_python(&code)?
                        }
                        ToolCall::Rust { code } => {
                            println!("🦀 Compiling and executing Rust...");
                            agent.execute_rust(&code)?
                        }
                        ToolCall::Shell { command } => {
                            println!("💻 Executing shell: {}", command);
                            agent.execute_shell_command(&command)?
                        }
                        ToolCall::Finance { operation, params } => {
                            println!("💰 PRISM Finance: {}", operation);
                            agent.call_finance_tool(&operation, params)?
                        }
                        ToolCall::DrugDiscovery { operation, params } => {
                            println!("💊 PRISM Drug Discovery: {}", operation);
                            agent.call_drug_discovery_tool(&operation, params)?
                        }
                        ToolCall::Robotics { operation, params } => {
                            println!("🤖 PRISM Robotics: {}", operation);
                            agent.call_robotics_tool(&operation, params)?
                        }
                        ToolCall::TimeSeries { operation, params } => {
                            println!("📈 PRISM Time Series: {}", operation);
                            agent.call_time_series_tool(&operation, params)?
                        }
                    };

                    if result.success {
                        println!("✅ Tool executed in {}ms", result.execution_time_ms);
                    } else {
                        println!("❌ Tool failed: {:?}", result.error);
                    }

                    tool_results.push(result);
                }
            }
        }

        Ok(ChatResponseWithTools {
            text: llm_response.text,
            mode_used: llm_response.mode_used,
            latency_ms: llm_response.latency_ms,
            cost_usd: llm_response.cost_usd,
            model: llm_response.model,
            tools_called: llm_response.tools_called,
            tool_results,
        })
    }

    /// Chat using local GPU LLM (FULLY OFFLINE!)
    async fn chat_local(&mut self, message: &str) -> Result<ChatResponse> {
        let start = std::time::Instant::now();

        let llm = self.llm_system.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Local LLM not loaded"))?;

        // Format prompt for Llama 3.2 Instruct
        let prompt = format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
            You are PRISM Assistant, a helpful AI with access to code execution and PRISM platform tools.\
            You can write Python code in ```python blocks, Rust in ```rust blocks, and shell commands in ```bash blocks.\n\
            <|eot_id|><|start_header_id|>user<|end_header_id|>\n\
            {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
            message
        );

        // Generate response using GPU LLM
        let response_text = llm.generate_text(&prompt, 256)?;

        let latency = start.elapsed().as_millis() as u64;

        Ok(ChatResponse {
            text: response_text,
            mode_used: "local_gpu".to_string(),
            latency_ms: latency,
            cost_usd: 0.0,
            model: "Llama-3.2-3B-Instruct-Q4_K_M".to_string(),
            tools_called: vec![],
        })
    }

    /// Chat using cloud (requires internet)
    async fn chat_cloud(&mut self, _message: &str) -> Result<ChatResponse> {
        Err(anyhow::anyhow!("Cloud mode requires Mission Charlie integration"))
    }

    /// Estimate query complexity
    fn estimate_complexity(&self, message: &str) -> f64 {
        let word_count = message.split_whitespace().count();
        let has_code = message.contains("```");
        let mut complexity = (word_count as f64 / 100.0).min(0.5);
        if has_code { complexity += 0.2; }
        complexity.min(1.0)
    }

    /// Find and verify local LLM model exists
    fn find_model_path(model_path: Option<&str>) -> Result<String> {
        let search_paths = if let Some(path) = model_path {
            vec![path.to_string()]
        } else {
            vec![
                "/home/diddy/Desktop/PRISM-Worker-6/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf".into(),
                "/home/diddy/Desktop/prism-ai-v1.0.0/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf".into(),
                "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf".into(),
            ]
        };

        println!("🔍 Checking for local LLM model...");

        for path in &search_paths {
            if std::path::Path::new(path).exists() {
                println!("✅ Found model: {}", path);
                return Ok(path.clone());
            }
        }

        Err(anyhow::anyhow!("No local model found. Searched paths: {:?}", search_paths))
    }

    pub fn set_mode(&mut self, mode: AssistantMode) {
        self.mode = mode;
    }

    pub fn set_tools_enabled(&mut self, enabled: bool) {
        self.tools_enabled = enabled;
    }

    pub fn status(&self) -> AssistantStatus {
        AssistantStatus {
            mode: format!("{:?}", self.mode),
            local_available: self.model_path.is_some(),
            cloud_available: false,
            tools_enabled: self.tools_enabled,
            model_loaded: self.model_path.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub text: String,
    pub mode_used: String,
    pub latency_ms: u64,
    pub cost_usd: f64,
    pub model: String,
    pub tools_called: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponseWithTools {
    pub text: String,
    pub mode_used: String,
    pub latency_ms: u64,
    pub cost_usd: f64,
    pub model: String,
    pub tools_called: Vec<String>,
    pub tool_results: Vec<ToolResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantStatus {
    pub mode: String,
    pub local_available: bool,
    pub cloud_available: bool,
    pub tools_enabled: bool,
    pub model_loaded: Option<String>,
}
