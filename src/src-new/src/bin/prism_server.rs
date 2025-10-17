//! PRISM-AI Production REST API Server
//!
//! High-performance production server for PRISM-AI revolutionary computing systems.
//! Provides secure REST API endpoints for quantum, thermodynamic, and neuromorphic computing.

use actix_web::{
    middleware::{Logger, DefaultHeaders},
    web, App, HttpServer, HttpResponse, Result,
    http::header,
};
use actix_web_httpauth::{
    extractors::bearer::{BearerAuth, Config},
    middleware::HttpAuthentication,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use prism_ai::api::{PrismApi, PrismRequest, PrismResponse};
use env_logger::Env;
use clap::{Parser, Subcommand};
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey, Algorithm};
use chrono::{Utc, Duration};
use std::collections::HashMap;
use actix_governor::{Governor, GovernorConfigBuilder};

/// JWT claims structure
#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: usize,
    iat: usize,
    role: String,
}

/// Server configuration
#[derive(Clone)]
struct ServerConfig {
    jwt_secret: String,
    api_keys: HashMap<String, String>, // API key -> role mapping
    max_requests_per_minute: u32,
}

/// Application state
struct AppState {
    api: Arc<RwLock<PrismApi>>,
    config: ServerConfig,
}

/// CLI Arguments
#[derive(Parser)]
#[command(name = "prism_server")]
#[command(about = "PRISM-AI Production Server", long_about = None)]
struct Cli {
    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Host to bind to
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,

    /// JWT secret key
    #[arg(long, env = "PRISM_JWT_SECRET", default_value = "prism-secret-key-change-in-production")]
    jwt_secret: String,

    /// Max requests per minute per client
    #[arg(long, default_value = "60")]
    rate_limit: u32,

    /// Number of worker threads
    #[arg(short, long, default_value = "4")]
    workers: usize,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate an API key
    GenKey {
        /// Role for the API key (admin, user, readonly)
        #[arg(short, long, default_value = "user")]
        role: String,
    },
}

/// Authentication validator
async fn validator(
    req: actix_web::dev::ServiceRequest,
    credentials: BearerAuth,
) -> Result<actix_web::dev::ServiceRequest, actix_web::Error> {
    let config = req.app_data::<web::Data<AppState>>()
        .ok_or_else(|| actix_web::error::ErrorInternalServerError("Server configuration error"))?;

    // First check if it's an API key
    if let Some(role) = config.config.api_keys.get(credentials.token()) {
        // Valid API key - inject role into request extensions
        req.extensions_mut().insert(role.clone());
        return Ok(req);
    }

    // Otherwise try JWT validation
    let token_data = decode::<Claims>(
        credentials.token(),
        &DecodingKey::from_secret(config.config.jwt_secret.as_ref()),
        &Validation::new(Algorithm::HS256),
    ).map_err(|_| actix_web::error::ErrorUnauthorized("Invalid token"))?;

    // Check if token is expired (handled by jsonwebtoken automatically)
    req.extensions_mut().insert(token_data.claims.role);
    Ok(req)
}

/// Health check endpoint
async fn health() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "timestamp": Utc::now().to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION"),
    })))
}

/// System info endpoint
async fn info(data: web::Data<AppState>) -> Result<HttpResponse> {
    let api = data.api.read().await;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "name": "PRISM-AI Production Server",
        "version": env!("CARGO_PKG_VERSION"),
        "capabilities": [
            "quantum_computing",
            "thermodynamic_computing",
            "neuromorphic_quantum_hybrid",
            "adaptive_feature_fusion"
        ],
        "gpu_enabled": cfg!(feature = "cuda"),
        "timestamp": Utc::now().to_rfc3339(),
    })))
}

/// Main API handler
async fn api_handler(
    data: web::Data<AppState>,
    request: web::Json<PrismRequest>,
) -> Result<HttpResponse> {
    let mut api = data.api.write().await;

    match api.process_request(request.into_inner()).await {
        Ok(response) => Ok(HttpResponse::Ok().json(response)),
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Processing error: {}", e),
            "timestamp": Utc::now().to_rfc3339(),
        }))),
    }
}

/// Quantum supremacy benchmark endpoint
async fn quantum_supremacy(
    data: web::Data<AppState>,
    params: web::Query<HashMap<String, String>>,
) -> Result<HttpResponse> {
    let depth = params.get("depth")
        .and_then(|d| d.parse::<usize>().ok())
        .unwrap_or(100);

    let request = PrismRequest::QuantumSupremacy {
        n_qubits: 16,  // Default to 16 qubits
        circuit_depth: depth
    };
    let mut api = data.api.write().await;

    match api.process_request(request).await {
        Ok(response) => Ok(HttpResponse::Ok().json(response)),
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Quantum supremacy error: {}", e),
        }))),
    }
}

/// TSP solver endpoint
async fn solve_tsp(
    data: web::Data<AppState>,
    cities: web::Json<Vec<(f32, f32)>>,
) -> Result<HttpResponse> {
    let request = PrismRequest::SolveTSP {
        cities: cities.into_inner(),
        algorithm: "hybrid".to_string()
    };
    let mut api = data.api.write().await;

    match api.process_request(request).await {
        Ok(response) => Ok(HttpResponse::Ok().json(response)),
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("TSP solver error: {}", e),
        }))),
    }
}

/// Graph coloring endpoint
async fn graph_coloring(
    data: web::Data<AppState>,
    graph_data: web::Json<serde_json::Value>,
) -> Result<HttpResponse> {
    // Parse vertices and edges from JSON
    let vertices = graph_data["vertices"].as_u64().unwrap_or(10) as usize;
    let edges: Vec<(usize, usize)> = graph_data["edges"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|e| {
            let arr = e.as_array()?;
            if arr.len() == 2 {
                Some((
                    arr[0].as_u64()? as usize,
                    arr[1].as_u64()? as usize,
                ))
            } else {
                None
            }
        })
        .collect();

    // Convert to adjacency matrix
    let mut adjacency = vec![vec![false; vertices]; vertices];
    for (u, v) in edges {
        if u < vertices && v < vertices {
            adjacency[u][v] = true;
            adjacency[v][u] = true;
        }
    }

    let request = PrismRequest::GraphColoring {
        adjacency_matrix: adjacency,
        max_colors: 4  // Default to 4 colors
    };
    let mut api = data.api.write().await;

    match api.process_request(request).await {
        Ok(response) => Ok(HttpResponse::Ok().json(response)),
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Graph coloring error: {}", e),
        }))),
    }
}

/// Generate JWT token endpoint
async fn generate_token(
    data: web::Data<AppState>,
    credentials: web::Json<HashMap<String, String>>,
) -> Result<HttpResponse> {
    // In production, validate credentials against database
    let username = credentials.get("username")
        .ok_or_else(|| actix_web::error::ErrorBadRequest("Missing username"))?;
    let password = credentials.get("password")
        .ok_or_else(|| actix_web::error::ErrorBadRequest("Missing password"))?;

    // Simple validation for demo (replace with real auth in production)
    if password != "prism-demo-password" {
        return Ok(HttpResponse::Unauthorized().json(serde_json::json!({
            "error": "Invalid credentials"
        })));
    }

    let now = Utc::now();
    let claims = Claims {
        sub: username.clone(),
        exp: (now + Duration::hours(24)).timestamp() as usize,
        iat: now.timestamp() as usize,
        role: "user".to_string(),
    };

    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(data.config.jwt_secret.as_ref()),
    ).map_err(|_| actix_web::error::ErrorInternalServerError("Token generation failed"))?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "token": token,
        "expires_at": claims.exp,
    })))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    // Handle subcommands
    if let Some(command) = cli.command {
        match command {
            Commands::GenKey { role } => {
                let key = generate_api_key();
                println!("Generated API key for role '{}': {}", role, key);
                println!("Add this to your server configuration or environment");
                return Ok(());
            }
        }
    }

    // Initialize logging
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    // Create default API keys (replace with database in production)
    let mut api_keys = HashMap::new();
    api_keys.insert(
        "prism-demo-key-admin".to_string(),
        "admin".to_string()
    );
    api_keys.insert(
        "prism-demo-key-user".to_string(),
        "user".to_string()
    );

    // Initialize server config
    let config = ServerConfig {
        jwt_secret: cli.jwt_secret.clone(),
        api_keys,
        max_requests_per_minute: cli.rate_limit,
    };

    // Initialize PRISM API
    let api = Arc::new(RwLock::new(PrismApi::new()));

    // Create app state
    let app_state = web::Data::new(AppState {
        api: api.clone(),
        config: config.clone(),
    });

    // Configure rate limiting
    let governor_conf = GovernorConfigBuilder::default()
        .per_minute(cli.rate_limit as u64)
        .finish()
        .unwrap();

    println!("🚀 PRISM-AI Production Server");
    println!("   Version: {}", env!("CARGO_PKG_VERSION"));
    println!("   Listening on: http://{}:{}", cli.host, cli.port);
    println!("   Workers: {}", cli.workers);
    println!("   Rate limit: {} req/min", cli.rate_limit);
    println!("   GPU acceleration: {}", if cfg!(feature = "cuda") { "ENABLED" } else { "DISABLED" });
    println!();
    println!("📡 Available endpoints:");
    println!("   GET  /health                - Health check");
    println!("   GET  /info                  - System information");
    println!("   POST /auth/token            - Generate JWT token");
    println!("   POST /api/process           - Main API handler");
    println!("   GET  /api/quantum/supremacy - Quantum supremacy benchmark");
    println!("   POST /api/tsp               - TSP solver");
    println!("   POST /api/graph/coloring    - Graph coloring");
    println!();

    // Start HTTP server
    HttpServer::new(move || {
        // Configure authentication middleware
        let auth = HttpAuthentication::bearer(validator);

        App::new()
            .app_data(app_state.clone())
            .wrap(Logger::default())
            .wrap(
                DefaultHeaders::new()
                    .add((header::CONTENT_TYPE, "application/json"))
                    .add(("X-Server", "PRISM-AI"))
            )
            .wrap(Governor::new(&governor_conf))
            // Public endpoints (no auth required)
            .route("/health", web::get().to(health))
            .route("/info", web::get().to(info))
            .route("/auth/token", web::post().to(generate_token))
            // Protected endpoints
            .service(
                web::scope("/api")
                    .wrap(auth)
                    .route("/process", web::post().to(api_handler))
                    .route("/quantum/supremacy", web::get().to(quantum_supremacy))
                    .route("/tsp", web::post().to(solve_tsp))
                    .route("/graph/coloring", web::post().to(graph_coloring))
            )
    })
    .workers(cli.workers)
    .bind((cli.host.as_str(), cli.port))?
    .run()
    .await
}

/// Generate a secure random API key
fn generate_api_key() -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::thread_rng();

    (0..32)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}