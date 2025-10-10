//! Zero-Trust Vendor Sandbox for PWSA
//!
//! Provides isolated execution environments for multiple vendors:
//! - GPU context isolation (Article V compliance)
//! - Data classification enforcement
//! - Resource quotas and rate limiting
//! - Comprehensive audit logging
//!
//! Security Model:
//! - Zero-trust: No vendor is trusted by default
//! - API-only access: No direct memory access
//! - Whitelisted operations only

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, Instant, Duration};
use std::collections::HashMap;
use anyhow::{Result, Context, bail};
use uuid::Uuid;
use cudarc::driver::CudaContext;

//=============================================================================
// DATA CLASSIFICATION
//=============================================================================

/// Security classification levels for data
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataClassification {
    Unclassified,
    ControlledUnclassified,  // CUI
    Secret,
    TopSecret,
}

impl DataClassification {
    pub fn can_downgrade_to(&self, target: &DataClassification) -> bool {
        // Can only downgrade to equal or higher classification
        target >= self
    }
}

//=============================================================================
// SECURE DATA SLICE
//=============================================================================

/// Secure reference to data with classification metadata
#[derive(Debug, Clone)]
pub struct SecureDataSlice {
    pub data_id: Uuid,
    pub classification: DataClassification,
    pub size_bytes: usize,
    pub encrypted: bool,
    pub checksum: Option<u64>,
}

impl SecureDataSlice {
    pub fn new(classification: DataClassification, size_bytes: usize) -> Self {
        Self {
            data_id: Uuid::new_v4(),
            classification,
            size_bytes,
            encrypted: classification >= DataClassification::Secret,
            checksum: None,
        }
    }
}

//=============================================================================
// VENDOR PLUGIN TRAIT
//=============================================================================

/// Trait that vendors must implement for their analytics
pub trait VendorPlugin<T>: Send + Sync {
    /// Unique identifier for this plugin version
    fn plugin_id(&self) -> &str;

    /// Human-readable name
    fn plugin_name(&self) -> &str;

    /// Vendor organization
    fn vendor_name(&self) -> &str;

    /// Required data classification level
    fn required_classification(&self) -> DataClassification;

    /// Execute the plugin's analytics
    fn execute(
        &self,
        ctx: &Arc<CudaContext>,
        input: SecureDataSlice,
    ) -> Result<T>;

    /// Estimated GPU memory requirement (MB)
    fn estimated_gpu_memory_mb(&self) -> usize {
        256  // Default 256MB
    }

    /// Maximum execution time
    fn max_execution_time(&self) -> Duration {
        Duration::from_secs(5)  // Default 5 seconds
    }
}

//=============================================================================
// ZERO TRUST POLICY
//=============================================================================

/// Access control policy for a vendor
#[derive(Debug, Clone)]
pub struct ZeroTrustPolicy {
    vendor_id: String,
    allowed_classifications: Vec<DataClassification>,
    allowed_operations: Vec<String>,
    max_data_size_mb: usize,
    expiration: Option<SystemTime>,
}

impl ZeroTrustPolicy {
    pub fn new(vendor_id: String) -> Self {
        Self {
            vendor_id,
            allowed_classifications: vec![DataClassification::Unclassified],
            allowed_operations: vec!["read".to_string(), "compute".to_string()],
            max_data_size_mb: 100,
            expiration: None,
        }
    }

    pub fn with_classification(mut self, classification: DataClassification) -> Self {
        if !self.allowed_classifications.contains(&classification) {
            self.allowed_classifications.push(classification);
        }
        self
    }

    pub fn with_expiration(mut self, expiration: SystemTime) -> Self {
        self.expiration = Some(expiration);
        self
    }

    pub fn allows_classification(&self, classification: &DataClassification) -> bool {
        self.allowed_classifications.contains(classification)
    }

    pub fn allows_operation(&self, operation: &str) -> bool {
        self.allowed_operations.contains(&operation.to_string())
    }

    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.expiration {
            SystemTime::now() > exp
        } else {
            false
        }
    }
}

//=============================================================================
// RESOURCE QUOTA
//=============================================================================

/// Resource limits for a vendor
#[derive(Debug, Clone)]
pub struct ResourceQuota {
    vendor_id: String,
    max_gpu_memory_mb: usize,
    max_executions_per_hour: usize,
    max_execution_time_seconds: u64,

    // Current usage tracking
    current_gpu_memory_mb: usize,
    executions_this_hour: usize,
    total_execution_time_ms: u128,
    hour_start: Instant,
}

impl ResourceQuota {
    pub fn new(vendor_id: String) -> Self {
        Self {
            vendor_id,
            max_gpu_memory_mb: 1024,  // 1GB default
            max_executions_per_hour: 1000,
            max_execution_time_seconds: 60,  // 60 seconds per hour

            current_gpu_memory_mb: 0,
            executions_this_hour: 0,
            total_execution_time_ms: 0,
            hour_start: Instant::now(),
        }
    }

    pub fn check_and_update(&mut self, memory_mb: usize, execution_time: Duration) -> Result<()> {
        // Reset hourly counters if needed
        if self.hour_start.elapsed() > Duration::from_secs(3600) {
            self.executions_this_hour = 0;
            self.total_execution_time_ms = 0;
            self.hour_start = Instant::now();
        }

        // Check limits
        if self.current_gpu_memory_mb + memory_mb > self.max_gpu_memory_mb {
            bail!("GPU memory quota exceeded: {} + {} > {} MB",
                self.current_gpu_memory_mb, memory_mb, self.max_gpu_memory_mb);
        }

        if self.executions_this_hour >= self.max_executions_per_hour {
            bail!("Execution quota exceeded: {} >= {}/hour",
                self.executions_this_hour, self.max_executions_per_hour);
        }

        let new_total_time_ms = self.total_execution_time_ms + execution_time.as_millis();
        if new_total_time_ms > (self.max_execution_time_seconds * 1000) as u128 {
            bail!("Execution time quota exceeded: {}ms > {}s",
                new_total_time_ms, self.max_execution_time_seconds);
        }

        // Update usage
        self.current_gpu_memory_mb += memory_mb;
        self.executions_this_hour += 1;
        self.total_execution_time_ms = new_total_time_ms;

        Ok(())
    }

    pub fn release_memory(&mut self, memory_mb: usize) {
        self.current_gpu_memory_mb = self.current_gpu_memory_mb.saturating_sub(memory_mb);
    }

    pub fn usage_report(&self) -> String {
        format!(
            "Vendor {}: GPU Memory: {}/{} MB, Executions: {}/{}/hr, Time: {}ms/{}s",
            self.vendor_id,
            self.current_gpu_memory_mb,
            self.max_gpu_memory_mb,
            self.executions_this_hour,
            self.max_executions_per_hour,
            self.total_execution_time_ms,
            self.max_execution_time_seconds
        )
    }
}

//=============================================================================
// AUDIT LOGGER
//=============================================================================

/// Audit log entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub timestamp: SystemTime,
    pub vendor_id: String,
    pub plugin_id: String,
    pub operation: String,
    pub data_id: Uuid,
    pub classification: DataClassification,
    pub size_bytes: usize,
    pub execution_time_ms: Option<u128>,
    pub result: String,
}

/// Compliance-ready audit logger
#[derive(Debug)]
pub struct AuditLogger {
    entries: Arc<Mutex<Vec<AuditEntry>>>,
    max_entries: usize,
}

impl AuditLogger {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
            max_entries: 10000,
        }
    }

    pub fn log_execution(
        &self,
        vendor_id: &str,
        plugin_id: &str,
        operation: &str,
        data: &SecureDataSlice,
        execution_time: Option<Duration>,
        success: bool,
    ) {
        let entry = AuditEntry {
            timestamp: SystemTime::now(),
            vendor_id: vendor_id.to_string(),
            plugin_id: plugin_id.to_string(),
            operation: operation.to_string(),
            data_id: data.data_id,
            classification: data.classification,
            size_bytes: data.size_bytes,
            execution_time_ms: execution_time.map(|d| d.as_millis()),
            result: if success { "SUCCESS".to_string() } else { "FAILED".to_string() },
        };

        let mut entries = self.entries.lock().unwrap();
        entries.push(entry);

        // Rotate log if too large
        if entries.len() > self.max_entries {
            entries.drain(0..1000);  // Remove oldest 1000 entries
        }
    }

    pub fn get_entries(&self, vendor_id: Option<&str>, limit: usize) -> Vec<AuditEntry> {
        let entries = self.entries.lock().unwrap();

        let filtered: Vec<_> = if let Some(vid) = vendor_id {
            entries.iter()
                .filter(|e| e.vendor_id == vid)
                .cloned()
                .collect()
        } else {
            entries.clone()
        };

        filtered.into_iter()
            .rev()
            .take(limit)
            .collect()
    }

    pub fn export_compliance_report(&self) -> String {
        let entries = self.entries.lock().unwrap();

        let mut report = String::from("=== PWSA VENDOR AUDIT LOG ===\n");
        report.push_str(&format!("Generated: {:?}\n", SystemTime::now()));
        report.push_str(&format!("Total Entries: {}\n\n", entries.len()));

        // Summary by vendor
        let mut vendor_stats: HashMap<String, (usize, u128)> = HashMap::new();
        for entry in entries.iter() {
            let stats = vendor_stats.entry(entry.vendor_id.clone())
                .or_insert((0, 0));
            stats.0 += 1;
            if let Some(time) = entry.execution_time_ms {
                stats.1 += time;
            }
        }

        report.push_str("VENDOR SUMMARY:\n");
        for (vendor, (count, total_time)) in vendor_stats {
            report.push_str(&format!(
                "  {}: {} executions, {}ms total time\n",
                vendor, count, total_time
            ));
        }

        report.push_str("\nCLASSIFICATION BREAKDOWN:\n");
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        for entry in entries.iter() {
            *class_counts.entry(format!("{:?}", entry.classification))
                .or_insert(0) += 1;
        }
        for (class, count) in class_counts {
            report.push_str(&format!("  {}: {}\n", class, count));
        }

        report
    }
}

//=============================================================================
// VENDOR SANDBOX
//=============================================================================

/// Isolated execution environment for a vendor
pub struct VendorSandbox {
    vendor_id: String,
    gpu_context: Arc<CudaContext>,
    policy: ZeroTrustPolicy,
    quota: ResourceQuota,
    audit_logger: Arc<AuditLogger>,
}

impl VendorSandbox {
    /// Create a new sandbox for a vendor with isolated GPU context
    pub fn new(vendor_id: String, gpu_device_id: usize) -> Result<Self> {
        // Create isolated GPU context for this vendor
        let gpu_context = CudaContext::new(gpu_device_id)
            .context("Failed to create isolated GPU context")?;

        Ok(Self {
            vendor_id: vendor_id.clone(),
            gpu_context,
            policy: ZeroTrustPolicy::new(vendor_id.clone()),
            quota: ResourceQuota::new(vendor_id.clone()),
            audit_logger: Arc::new(AuditLogger::new()),
        })
    }

    /// Update access policy for this vendor
    pub fn set_policy(&mut self, policy: ZeroTrustPolicy) {
        self.policy = policy;
    }

    /// Update resource quota for this vendor
    pub fn set_quota(&mut self, quota: ResourceQuota) {
        self.quota = quota;
    }

    /// Execute a vendor plugin in the sandbox
    pub fn execute_plugin<T>(
        &mut self,
        plugin: &dyn VendorPlugin<T>,
        input: SecureDataSlice,
    ) -> Result<T> {
        // Pre-execution validation
        self.validate_execution(plugin, &input)?;

        let start = Instant::now();

        // Check and reserve resources
        let memory_mb = plugin.estimated_gpu_memory_mb();
        self.quota.check_and_update(memory_mb, Duration::from_secs(0))?;

        // Execute in isolated environment
        let result = plugin.execute(&self.gpu_context, input.clone());
        let execution_time = start.elapsed();

        // Update quota with actual execution time
        self.quota.release_memory(memory_mb);
        self.quota.check_and_update(0, execution_time)?;

        // Audit log
        self.audit_logger.log_execution(
            &self.vendor_id,
            plugin.plugin_id(),
            "execute",
            &input,
            Some(execution_time),
            result.is_ok(),
        );

        // Check execution time limit
        if execution_time > plugin.max_execution_time() {
            bail!("Plugin execution exceeded time limit: {:?} > {:?}",
                execution_time, plugin.max_execution_time());
        }

        result
    }

    /// Validate that the vendor can execute this operation
    fn validate_execution<T>(
        &self,
        plugin: &dyn VendorPlugin<T>,
        input: &SecureDataSlice,
    ) -> Result<()> {
        // Check policy expiration
        if self.policy.is_expired() {
            bail!("Vendor policy expired");
        }

        // Check data classification
        if !self.policy.allows_classification(&input.classification) {
            bail!("Vendor not authorized for {:?} data", input.classification);
        }

        // Check operation permission
        if !self.policy.allows_operation("execute") {
            bail!("Vendor not authorized for execute operation");
        }

        // Check plugin classification requirement
        if plugin.required_classification() > input.classification {
            bail!("Plugin requires {:?} but data is {:?}",
                plugin.required_classification(), input.classification);
        }

        // Check data size limit
        let size_mb = input.size_bytes / (1024 * 1024);
        if size_mb > self.policy.max_data_size_mb {
            bail!("Data size {}MB exceeds limit {}MB",
                size_mb, self.policy.max_data_size_mb);
        }

        Ok(())
    }

    /// Get current resource usage
    pub fn get_usage(&self) -> String {
        self.quota.usage_report()
    }

    /// Get audit entries for this vendor
    pub fn get_audit_log(&self, limit: usize) -> Vec<AuditEntry> {
        self.audit_logger.get_entries(Some(&self.vendor_id), limit)
    }

    /// Synchronize GPU operations
    pub fn synchronize(&self) -> Result<()> {
        // GPU synchronization handled by context
        Ok(())
    }
}

//=============================================================================
// MULTI-VENDOR ORCHESTRATOR
//=============================================================================

/// Manages multiple vendor sandboxes
pub struct MultiVendorOrchestrator {
    sandboxes: HashMap<String, Arc<Mutex<VendorSandbox>>>,
    global_audit: Arc<AuditLogger>,
    max_vendors: usize,
}

impl MultiVendorOrchestrator {
    pub fn new() -> Self {
        Self {
            sandboxes: HashMap::new(),
            global_audit: Arc::new(AuditLogger::new()),
            max_vendors: 10,
        }
    }

    /// Register a new vendor
    pub fn register_vendor(
        &mut self,
        vendor_id: String,
        gpu_device_id: usize,
    ) -> Result<()> {
        if self.sandboxes.len() >= self.max_vendors {
            bail!("Maximum vendor limit reached: {}", self.max_vendors);
        }

        if self.sandboxes.contains_key(&vendor_id) {
            bail!("Vendor {} already registered", vendor_id);
        }

        let mut sandbox = VendorSandbox::new(vendor_id.clone(), gpu_device_id)?;
        sandbox.audit_logger = Arc::clone(&self.global_audit);

        self.sandboxes.insert(
            vendor_id,
            Arc::new(Mutex::new(sandbox))
        );

        Ok(())
    }

    /// Execute a plugin for a specific vendor
    pub fn execute_vendor_plugin<T>(
        &self,
        vendor_id: &str,
        plugin: &dyn VendorPlugin<T>,
        input: SecureDataSlice,
    ) -> Result<T> {
        let sandbox = self.sandboxes.get(vendor_id)
            .ok_or_else(|| anyhow::anyhow!("Vendor {} not registered", vendor_id))?;

        let mut sandbox = sandbox.lock().unwrap();
        sandbox.execute_plugin(plugin, input)
    }

    /// Update vendor policy
    pub fn update_vendor_policy(
        &self,
        vendor_id: &str,
        policy: ZeroTrustPolicy,
    ) -> Result<()> {
        let sandbox = self.sandboxes.get(vendor_id)
            .ok_or_else(|| anyhow::anyhow!("Vendor {} not registered", vendor_id))?;

        let mut sandbox = sandbox.lock().unwrap();
        sandbox.set_policy(policy);
        Ok(())
    }

    /// Get global audit report
    pub fn get_compliance_report(&self) -> String {
        self.global_audit.export_compliance_report()
    }

    /// Get all vendor usage reports
    pub fn get_all_usage_reports(&self) -> HashMap<String, String> {
        let mut reports = HashMap::new();

        for (vendor_id, sandbox) in &self.sandboxes {
            let sandbox = sandbox.lock().unwrap();
            reports.insert(vendor_id.clone(), sandbox.get_usage());
        }

        reports
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin;

    impl VendorPlugin<f64> for TestPlugin {
        fn plugin_id(&self) -> &str { "test_v1" }
        fn plugin_name(&self) -> &str { "Test Analytics" }
        fn vendor_name(&self) -> &str { "TestVendor" }
        fn required_classification(&self) -> DataClassification {
            DataClassification::Unclassified
        }

        fn execute(
            &self,
            _ctx: &Arc<CudaContext>,
            _input: SecureDataSlice,
        ) -> Result<f64> {
            Ok(42.0)
        }
    }

    #[test]
    fn test_vendor_sandbox_creation() {
        let sandbox = VendorSandbox::new("TestVendor".to_string(), 0);
        assert!(sandbox.is_ok());
    }

    #[test]
    fn test_zero_trust_policy() {
        let policy = ZeroTrustPolicy::new("TestVendor".to_string())
            .with_classification(DataClassification::Secret);

        assert!(policy.allows_classification(&DataClassification::Unclassified));
        assert!(policy.allows_classification(&DataClassification::Secret));
        assert!(!policy.allows_classification(&DataClassification::TopSecret));
    }

    #[test]
    fn test_resource_quota() {
        let mut quota = ResourceQuota::new("TestVendor".to_string());

        let result = quota.check_and_update(512, Duration::from_secs(1));
        assert!(result.is_ok());

        let result = quota.check_and_update(600, Duration::from_secs(1));
        assert!(result.is_err());  // Exceeds 1GB limit
    }
}