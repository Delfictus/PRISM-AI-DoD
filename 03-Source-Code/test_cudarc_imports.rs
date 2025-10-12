// Test program to figure out correct cudarc imports

// Try different import patterns
#[cfg(feature = "driver")]
mod test1 {
    use cudarc::driver::*;
    pub fn test() {
        println!("cudarc::driver::* works");
    }
}

#[cfg(feature = "driver")]
mod test2 {
    use cudarc::driver::CudaDevice;
    pub fn test() {
        println!("cudarc::driver::CudaDevice works");
    }
}

mod test3 {
    // Try without feature gate
    use cudarc::*;
    pub fn test() {
        println!("cudarc::* works");
    }
}

fn main() {
    println!("Testing cudarc imports...");
}