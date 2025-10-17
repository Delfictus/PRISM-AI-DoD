//! WebSocket support for real-time updates

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::api_server::AppState;

/// WebSocket event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsEvent {
    /// PWSA threat detection event
    ThreatDetected {
        threat_id: String,
        confidence: f64,
        position: (f64, f64, f64),
        timestamp: i64,
    },
    /// Portfolio update event
    PortfolioUpdate {
        portfolio_id: String,
        value: f64,
        returns: f64,
        timestamp: i64,
    },
    /// Network congestion event
    NetworkCongestion {
        node_id: String,
        utilization: f64,
        packet_loss: f64,
        timestamp: i64,
    },
    /// LLM generation streaming
    LlmStream {
        request_id: String,
        token: String,
        is_final: bool,
    },
    /// Time series forecast update
    ForecastUpdate {
        series_id: String,
        predictions: Vec<f64>,
        confidence_intervals: Vec<(f64, f64)>,
        timestamp: i64,
    },
    /// System status
    SystemStatus {
        cpu_usage: f64,
        gpu_usage: f64,
        memory_usage: f64,
        active_requests: usize,
    },
    /// Ping/Pong for connection keepalive
    Ping,
    Pong,
}

/// WebSocket handler
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Handle individual WebSocket connection
async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    // Create broadcast channel for this connection
    let (tx, mut rx) = broadcast::channel::<WsEvent>(100);

    // Spawn task to send events to client
    let mut send_task = tokio::spawn(async move {
        while let Ok(event) = rx.recv().await {
            let json = serde_json::to_string(&event).unwrap();
            if sender.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages from client
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    log::debug!("Received WebSocket message: {}", text);
                    // Parse and handle client messages
                    if let Ok(event) = serde_json::from_str::<WsEvent>(&text) {
                        match event {
                            WsEvent::Ping => {
                                // Send pong back
                                let _ = tx.send(WsEvent::Pong);
                            }
                            _ => {
                                log::debug!("Received event: {:?}", event);
                            }
                        }
                    }
                }
                Message::Close(_) => {
                    log::info!("WebSocket connection closed");
                    break;
                }
                _ => {}
            }
        }
    });

    // Wait for either task to finish
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }

    log::info!("WebSocket connection ended");
}

/// Broadcast event to all connected WebSocket clients
pub async fn broadcast_event(event: WsEvent, tx: &broadcast::Sender<WsEvent>) {
    let _ = tx.send(event);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_event_serialization() {
        let event = WsEvent::ThreatDetected {
            threat_id: "threat-001".to_string(),
            confidence: 0.95,
            position: (1.0, 2.0, 3.0),
            timestamp: 1234567890,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("ThreatDetected"));
        assert!(json.contains("threat-001"));

        let parsed: WsEvent = serde_json::from_str(&json).unwrap();
        if let WsEvent::ThreatDetected { threat_id, .. } = parsed {
            assert_eq!(threat_id, "threat-001");
        } else {
            panic!("Wrong event type");
        }
    }
}
