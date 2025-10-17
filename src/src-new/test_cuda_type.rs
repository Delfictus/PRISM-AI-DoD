// Test file to check CudaContext type
#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use std::sync::Arc;

fn main() {
    #[cfg(feature = "cuda")]
    {
        // Test what CudaContext::new actually returns
        let result = CudaContext::new(0);
        match result {
            Ok(context) => {
                // Check if context needs Arc wrapping or not
                let _test: Arc<CudaContext> = Arc::new(context);
                println!("CudaContext::new returns CudaContext directly");
            }
            Err(e) => {
                println!("Failed to create context: {:?}", e);
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    println!("CUDA not enabled");
}