/*!
 * PRISM-AI Command Line Interface
 *
 * A powerful CLI for interacting with the PRISM-AI REST API.
 */

use clap::{Parser, Subcommand};
use colored::Colorize;
use anyhow::{Context, Result};

mod client;
mod commands;
mod config;
mod output;

use client::PrismClient;
use config::Config;

#[derive(Parser)]
#[command(name = "prism")]
#[command(version, about = "PRISM-AI Command Line Interface", long_about = None)]
struct Cli {
    /// API base URL (overrides config file)
    #[arg(long, global = true, env = "PRISM_API_URL")]
    api_url: Option<String>,

    /// API key (overrides config file)
    #[arg(long, global = true, env = "PRISM_API_KEY")]
    api_key: Option<String>,

    /// Output format (json, table, yaml)
    #[arg(long, global = true, default_value = "table")]
    output: String,

    /// Disable colored output
    #[arg(long, global = true)]
    no_color: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigCommands,
    },

    /// Health check
    Health,

    /// PWSA (Passive Wide-area Surveillance & Assessment) commands
    #[command(subcommand)]
    Pwsa(PwsaCommands),

    /// Finance commands
    #[command(subcommand)]
    Finance(FinanceCommands),

    /// LLM commands
    #[command(subcommand)]
    Llm(LlmCommands),

    /// Time series commands
    #[command(subcommand)]
    Timeseries(TimeseriesCommands),

    /// Pixel processing commands
    #[command(subcommand)]
    Pixels(PixelsCommands),
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Initialize configuration file
    Init,
    /// Show current configuration
    Show,
    /// Set configuration value
    Set {
        /// Configuration key
        key: String,
        /// Configuration value
        value: String,
    },
}

#[derive(Subcommand)]
enum PwsaCommands {
    /// Detect threats from IR sensor data
    Detect {
        /// Space vehicle ID
        #[arg(long)]
        sv_id: i32,
        /// Unix timestamp
        #[arg(long)]
        timestamp: i64,
        /// IR frame JSON file path
        #[arg(long)]
        ir_frame: String,
    },

    /// Fuse multi-sensor data
    Fuse {
        /// Space vehicle ID
        #[arg(long)]
        sv_id: i32,
        /// Unix timestamp
        #[arg(long)]
        timestamp: i64,
        /// Sensor data JSON file path
        #[arg(long)]
        sensors: String,
    },

    /// Predict threat trajectory
    Predict {
        /// Track ID
        #[arg(long)]
        track_id: String,
        /// History data JSON file path
        #[arg(long)]
        history: String,
        /// Prediction horizon in seconds
        #[arg(long)]
        horizon: i32,
        /// Model type (kalman_filter, neural_network)
        #[arg(long, default_value = "kalman_filter")]
        model: String,
    },

    /// Prioritize multiple threats
    Prioritize {
        /// Threats JSON file path
        #[arg(long)]
        threats: String,
        /// Prioritization strategy
        #[arg(long, default_value = "time_weighted_risk")]
        strategy: String,
    },
}

#[derive(Subcommand)]
enum FinanceCommands {
    /// Optimize portfolio allocation
    Optimize {
        /// Assets JSON file path
        #[arg(long)]
        assets: String,
        /// Constraints JSON file path
        #[arg(long)]
        constraints: String,
        /// Optimization objective
        #[arg(long, default_value = "maximize_sharpe")]
        objective: String,
    },

    /// Assess portfolio risk
    Risk {
        /// Portfolio ID
        #[arg(long)]
        portfolio_id: String,
        /// Positions JSON file path
        #[arg(long)]
        positions: String,
        /// Risk metrics (comma-separated: var,cvar,beta)
        #[arg(long)]
        metrics: String,
    },

    /// Backtest trading strategy
    Backtest {
        /// Strategy ID
        #[arg(long)]
        strategy_id: String,
        /// Parameters JSON file path
        #[arg(long)]
        parameters: String,
        /// Historical data JSON file path
        #[arg(long)]
        historical_data: String,
        /// Initial capital
        #[arg(long)]
        initial_capital: f64,
    },
}

#[derive(Subcommand)]
enum LlmCommands {
    /// Query a language model
    Query {
        /// Prompt text
        prompt: String,
        /// Model name (optional)
        #[arg(long)]
        model: Option<String>,
        /// Temperature (0.0-2.0)
        #[arg(long, default_value = "0.7")]
        temperature: f64,
        /// Maximum tokens
        #[arg(long, default_value = "500")]
        max_tokens: i32,
    },

    /// Multi-model consensus
    Consensus {
        /// Prompt text
        prompt: String,
        /// Models JSON file path
        #[arg(long)]
        models: String,
        /// Consensus strategy
        #[arg(long, default_value = "majority_vote")]
        strategy: String,
        /// Temperature
        #[arg(long, default_value = "0.3")]
        temperature: f64,
        /// Maximum tokens
        #[arg(long, default_value = "500")]
        max_tokens: i32,
    },

    /// List available models
    Models,
}

#[derive(Subcommand)]
enum TimeseriesCommands {
    /// Forecast time series
    Forecast {
        /// Series ID
        #[arg(long)]
        series_id: String,
        /// Historical data JSON file path
        #[arg(long)]
        data: String,
        /// Forecast horizon
        #[arg(long, default_value = "10")]
        horizon: i32,
        /// Forecasting method
        #[arg(long, default_value = "arima")]
        method: String,
    },
}

#[derive(Subcommand)]
enum PixelsCommands {
    /// Process pixel data
    Process {
        /// Frame ID
        #[arg(long)]
        frame_id: i32,
        /// Frame width
        #[arg(long)]
        width: i32,
        /// Frame height
        #[arg(long)]
        height: i32,
        /// Pixels data JSON file path
        #[arg(long)]
        pixels: String,
        /// Enable hotspot detection
        #[arg(long)]
        detect_hotspots: bool,
        /// Compute entropy
        #[arg(long)]
        compute_entropy: bool,
        /// Apply TDA
        #[arg(long)]
        apply_tda: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.no_color {
        colored::control::set_override(false);
    }

    // Load configuration
    let mut config = Config::load().unwrap_or_default();

    // Override with CLI arguments if provided
    if let Some(api_url) = cli.api_url {
        config.api_url = api_url;
    }
    if let Some(api_key) = cli.api_key {
        config.api_key = api_key;
    }

    // Handle config commands separately (no API client needed)
    if let Commands::Config { action } = cli.command {
        return handle_config_command(action, &config);
    }

    // Validate configuration
    if config.api_key.is_empty() {
        eprintln!("{}", "Error: API key not set. Use 'prism config init' or set PRISM_API_KEY environment variable.".red());
        std::process::exit(1);
    }

    // Create API client
    let client = PrismClient::new(&config.api_url, &config.api_key);

    // Execute command
    match cli.command {
        Commands::Health => commands::health::execute(&client, &cli.output).await,
        Commands::Pwsa(cmd) => commands::pwsa::execute(&client, cmd, &cli.output).await,
        Commands::Finance(cmd) => commands::finance::execute(&client, cmd, &cli.output).await,
        Commands::Llm(cmd) => commands::llm::execute(&client, cmd, &cli.output).await,
        Commands::Timeseries(cmd) => commands::timeseries::execute(&client, cmd, &cli.output).await,
        Commands::Pixels(cmd) => commands::pixels::execute(&client, cmd, &cli.output).await,
        Commands::Config { .. } => unreachable!(),
    }
}

fn handle_config_command(action: ConfigCommands, config: &Config) -> Result<()> {
    match action {
        ConfigCommands::Init => {
            let config_path = Config::path()?;
            if config_path.exists() {
                println!("{}", "Configuration file already exists:".yellow());
            } else {
                Config::default().save()?;
                println!("{}", "Configuration file created:".green());
            }
            println!("  {}", config_path.display());
            println!("\nEdit the file to set your API key and URL.");
            Ok(())
        }
        ConfigCommands::Show => {
            println!("{}", "Current Configuration:".bold());
            println!("  API URL: {}", config.api_url.cyan());
            println!("  API Key: {}", if config.api_key.is_empty() {
                "not set".red()
            } else {
                "***set***".green()
            });
            Ok(())
        }
        ConfigCommands::Set { key, value } => {
            let mut config = Config::load().unwrap_or_default();
            match key.as_str() {
                "api_url" => config.api_url = value,
                "api_key" => config.api_key = value,
                _ => {
                    eprintln!("{}", format!("Unknown config key: {}", key).red());
                    std::process::exit(1);
                }
            }
            config.save()?;
            println!("{}", format!("Configuration updated: {} = {}", key, value).green());
            Ok(())
        }
    }
}
