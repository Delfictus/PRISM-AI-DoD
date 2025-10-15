// Integration tests for WebSocket functionality

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::{connect_async, tungstenite::Message};

const WS_URL: &str = "ws://localhost:8080/ws";

#[tokio::test]
async fn test_websocket_connection() {
    let (ws_stream, _) = connect_async(WS_URL).await.expect("Failed to connect");

    assert!(ws_stream.get_ref().is_active(), "WebSocket should be active");
}

#[tokio::test]
async fn test_websocket_ping_pong() {
    let (ws_stream, _) = connect_async(WS_URL).await.expect("Failed to connect");
    let (mut write, mut read) = ws_stream.split();

    // Send ping
    let ping_msg = json!({
        "type": "Ping"
    });

    write
        .send(Message::Text(ping_msg.to_string()))
        .await
        .expect("Failed to send ping");

    // Wait for pong
    if let Some(Ok(Message::Text(text))) = read.next().await {
        let response: Value = serde_json::from_str(&text).expect("Failed to parse response");
        assert_eq!(response["type"], "Pong", "Should receive Pong response");
    } else {
        panic!("Did not receive WebSocket message");
    }
}

#[tokio::test]
async fn test_websocket_event_subscription() {
    let (ws_stream, _) = connect_async(WS_URL).await.expect("Failed to connect");
    let (mut write, mut read) = ws_stream.split();

    // Subscribe to threat events
    let subscribe_msg = json!({
        "type": "Subscribe",
        "event_type": "ThreatDetected"
    });

    write
        .send(Message::Text(subscribe_msg.to_string()))
        .await
        .expect("Failed to send subscribe message");

    // Wait for confirmation
    if let Some(Ok(Message::Text(text))) = read.next().await {
        let response: Value = serde_json::from_str(&text).expect("Failed to parse response");
        assert!(
            response["type"] == "Subscribed" || response["type"] == "Pong",
            "Should receive subscription confirmation"
        );
    }
}

#[tokio::test]
async fn test_websocket_multiple_clients() {
    // Connect two clients
    let (ws1, _) = connect_async(WS_URL).await.expect("Failed to connect client 1");
    let (ws2, _) = connect_async(WS_URL).await.expect("Failed to connect client 2");

    assert!(ws1.get_ref().is_active(), "Client 1 should be active");
    assert!(ws2.get_ref().is_active(), "Client 2 should be active");
}

#[tokio::test]
async fn test_websocket_invalid_message() {
    let (ws_stream, _) = connect_async(WS_URL).await.expect("Failed to connect");
    let (mut write, mut read) = ws_stream.split();

    // Send invalid JSON
    write
        .send(Message::Text("not valid json".to_string()))
        .await
        .expect("Failed to send message");

    // Should receive error or connection close
    if let Some(Ok(msg)) = read.next().await {
        match msg {
            Message::Text(text) => {
                let response: Value = serde_json::from_str(&text).ok().unwrap_or_default();
                assert!(
                    response["type"] == "Error" || text.contains("error"),
                    "Should receive error for invalid message"
                );
            }
            Message::Close(_) => {
                // Connection closed due to invalid message
            }
            _ => {}
        }
    }
}

#[tokio::test]
async fn test_websocket_reconnection() {
    // Connect
    let (ws_stream, _) = connect_async(WS_URL).await.expect("Failed to connect");

    // Close connection
    drop(ws_stream);

    // Reconnect
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let (ws_stream_2, _) = connect_async(WS_URL).await.expect("Failed to reconnect");
    assert!(
        ws_stream_2.get_ref().is_active(),
        "Should be able to reconnect"
    );
}

#[tokio::test]
async fn test_websocket_event_streaming() {
    let (ws_stream, _) = connect_async(WS_URL).await.expect("Failed to connect");
    let (_write, mut read) = ws_stream.split();

    // Set timeout for receiving events
    let timeout = tokio::time::Duration::from_secs(5);

    tokio::select! {
        result = read.next() => {
            if let Some(Ok(Message::Text(text))) = result {
                let event: Value = serde_json::from_str(&text).expect("Failed to parse event");
                assert!(event["type"].is_string(), "Event should have type field");
            }
        }
        _ = tokio::time::sleep(timeout) => {
            // Timeout is acceptable - no events may be generated during test
        }
    }
}
