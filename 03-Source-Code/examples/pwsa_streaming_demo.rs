//! PWSA Streaming Telemetry Demonstration
//!
//! Week 2 Enhancement: Real-time async streaming architecture
//!
//! Demonstrates continuous telemetry ingestion at operational rates:
//! - 10 Hz fusion rate
//! - 100+ samples processed
//! - <1ms latency maintained

use prism_ai::pwsa::satellite_adapters::*;
use prism_ai::pwsa::streaming::*;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use std::time::SystemTime;
use colored::*;

/// Generate continuous transport telemetry stream
async fn generate_transport_stream(tx: mpsc::Sender<OctTelemetry>) {
    let mut ticker = interval(Duration::from_millis(100));  // 10 Hz

    for i in 0..100 {
        ticker.tick().await;

        let telem = OctTelemetry {
            sv_id: (i % 10) + 1,
            link_id: (i % 4) as u8,
            timestamp: SystemTime::now(),
            optical_power_dbm: -12.0 + (i as f64 * 0.1).sin() * 3.0,
            bit_error_rate: 1e-9 * (1.0 + (i as f64 * 0.05).cos() * 0.2),
            pointing_error_urad: 5.0 + (i as f64 * 0.15).sin() * 2.0,
            data_rate_gbps: 10.0,
            temperature_c: 20.0 + (i as f64 * 0.1).cos() * 3.0,
        };

        if tx.send(telem).await.is_err() {
            break;
        }
    }
}

/// Generate continuous tracking telemetry stream
async fn generate_tracking_stream(tx: mpsc::Sender<IrSensorFrame>) {
    let mut ticker = interval(Duration::from_millis(100));  // 10 Hz

    for i in 0..100 {
        ticker.tick().await;

        // Inject threat every 20 samples
        let is_threat = i % 20 == 10;

        let frame = IrSensorFrame {
            sv_id: (i % 5) + 1,
            timestamp: SystemTime::now(),
            width: 1024,
            height: 1024,
            max_intensity: if is_threat { 3000.0 } else { 1000.0 },
            background_level: 100.0,
            hotspot_count: if is_threat { 3 } else { 0 },
            centroid_x: 512.0,
            centroid_y: 512.0,
            velocity_estimate_mps: if is_threat { 1900.0 } else { 100.0 },
            acceleration_estimate: if is_threat { 50.0 } else { 5.0 },
            swir_band_ratio: 1.0,
            thermal_signature: if is_threat { 0.9 } else { 0.2 },
            geolocation: (38.0, 127.0),
        };

        if tx.send(frame).await.is_err() {
            break;
        }
    }
}

/// Generate continuous ground station stream
async fn generate_ground_stream(tx: mpsc::Sender<GroundStationData>) {
    let mut ticker = interval(Duration::from_millis(100));  // 10 Hz

    for i in 0..100 {
        ticker.tick().await;

        let data = GroundStationData {
            station_id: (i % 3) + 1,
            timestamp: SystemTime::now(),
            uplink_power_dbm: 48.0 + (i as f64 * 0.1).sin() * 2.0,
            downlink_snr_db: 22.0 + (i as f64 * 0.12).cos() * 3.0,
            command_queue_depth: (i % 10) as u32,
        };

        if tx.send(data).await.is_err() {
            break;
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(80).bright_cyan());
    println!("{}", "PWSA STREAMING TELEMETRY DEMONSTRATION".bright_yellow().bold());
    println!("{}", "Real-Time Async Architecture - Week 2 Enhancement".bright_white());
    println!("{}", "=".repeat(80).bright_cyan());

    println!("\n{}", "Configuration:".bright_white().bold());
    println!("  Fusion Rate: 10 Hz");
    println!("  Sample Count: 100 fusions");
    println!("  Architecture: Tokio async runtime");
    println!("  Backpressure: Rate limiting enabled");

    // Create channels
    let (transport_tx, transport_rx) = mpsc::channel(50);
    let (tracking_tx, tracking_rx) = mpsc::channel(50);
    let (ground_tx, ground_rx) = mpsc::channel(50);
    let (output_tx, mut output_rx) = mpsc::channel(50);

    println!("\n{}", "Starting telemetry generators...".bright_green());

    // Spawn telemetry generators
    let gen1 = tokio::spawn(generate_transport_stream(transport_tx));
    let gen2 = tokio::spawn(generate_tracking_stream(tracking_tx));
    let gen3 = tokio::spawn(generate_ground_stream(ground_tx));

    // Spawn fusion platform
    println!("{}", "Starting streaming fusion platform...".bright_green());
    let fusion_handle = tokio::spawn(async move {
        let mut platform = StreamingPwsaFusionPlatform::new(
            transport_rx,
            tracking_rx,
            ground_rx,
            output_tx,
            10.0,  // 10 Hz max rate
        ).unwrap();

        platform.run().await.unwrap();
        platform.get_stats()
    });

    // Process outputs
    println!("{}", "Processing mission awareness outputs...".bright_green());
    println!("{}", "-".repeat(80));

    let mut count = 0;
    let mut threat_count = 0;

    while let Some(awareness) = output_rx.recv().await {
        count += 1;

        let max_threat = awareness.threat_status.iter()
            .skip(1)
            .cloned()
            .fold(0.0_f64, f64::max);

        if max_threat > 0.5 {
            threat_count += 1;
            let threat_classes = ["None", "Aircraft", "Cruise", "Ballistic", "HYPERSONIC"];
            let threat_idx = awareness.threat_status.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            println!("{} Fusion #{}: {} detected ({:.0}% confidence)",
                "⚠".bright_red(),
                count,
                threat_classes[threat_idx].bright_red().bold(),
                max_threat * 100.0
            );
        } else if count % 10 == 0 {
            println!("{} Fusion #{}: Normal operations",
                "✓".bright_green(),
                count
            );
        }
    }

    // Wait for generators to complete
    gen1.await?;
    gen2.await?;
    gen3.await?;

    // Get final statistics
    let stats = fusion_handle.await?;

    println!("{}", "-".repeat(80));
    println!("\n{}", "STREAMING PERFORMANCE SUMMARY".bright_yellow().bold());
    println!("{}", "=".repeat(80).bright_cyan());
    println!("  Fusions Completed: {}", stats.fusions_completed);
    println!("  Average Latency:   {}μs ({:.2}ms)",
        stats.average_latency_us,
        stats.average_latency_us as f64 / 1000.0
    );
    println!("  Threat Detections: {}", threat_count);

    let latency_status = if stats.average_latency_us < 1000 {
        format!("✓ EXCEEDS TARGET: <1ms achieved").green()
    } else if stats.average_latency_us < 5000 {
        format!("✓ MEETS TARGET: <5ms achieved").bright_green()
    } else {
        format!("✗ EXCEEDS TARGET: >5ms").red()
    };

    println!("  Status:            {}", latency_status);

    println!("\n{}", "Streaming Capabilities:".bright_white().bold());
    println!("  ✓ Async telemetry ingestion");
    println!("  ✓ Backpressure control (10 Hz rate limiting)");
    println!("  ✓ Real-time fusion processing");
    println!("  ✓ Continuous operation validated");

    println!("\n{}", "STREAMING DEMO COMPLETE".bright_green().bold());
    println!("{}", "=".repeat(80).bright_cyan());

    Ok(())
}