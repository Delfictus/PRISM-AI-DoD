#!/usr/bin/env python3
"""
GPU Usage Verification Script
Monitors actual GPU utilization to verify if PRISM-AI is using GPU
"""

import subprocess
import time
import sys
import threading
import signal

class GPUMonitor:
    def __init__(self):
        self.monitoring = True
        self.gpu_active = False
        self.max_util = 0
        self.max_memory = 0
        self.samples = []

    def monitor_gpu(self):
        """Monitor GPU usage continuously"""
        while self.monitoring:
            try:
                # Get GPU utilization and memory
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=True
                )

                # Parse output
                parts = result.stdout.strip().split(',')
                if len(parts) == 3:
                    util = int(parts[0].strip())
                    mem_used = int(parts[1].strip())
                    mem_total = int(parts[2].strip())

                    self.samples.append((util, mem_used))

                    if util > 0:
                        self.gpu_active = True
                        self.max_util = max(self.max_util, util)

                    if mem_used > self.max_memory:
                        self.max_memory = mem_used

                    # Print real-time status
                    sys.stdout.write(f"\rGPU: {util:3d}% | Mem: {mem_used:5d}/{mem_total:5d} MB | Max: {self.max_util:3d}% | Active: {'YES' if self.gpu_active else 'NO '}")
                    sys.stdout.flush()

            except Exception as e:
                print(f"\nError monitoring GPU: {e}")

            time.sleep(0.1)  # Sample every 100ms

    def stop(self):
        self.monitoring = False

    def report(self):
        """Generate final report"""
        print("\n\n" + "="*60)
        print("GPU USAGE VERIFICATION REPORT")
        print("="*60)

        if not self.samples:
            print("❌ No GPU samples collected")
            return False

        # Calculate average utilization
        avg_util = sum(s[0] for s in self.samples) / len(self.samples)
        avg_mem = sum(s[1] for s in self.samples) / len(self.samples)

        print(f"Samples collected: {len(self.samples)}")
        print(f"Max GPU Utilization: {self.max_util}%")
        print(f"Avg GPU Utilization: {avg_util:.1f}%")
        print(f"Max Memory Used: {self.max_memory} MB")
        print(f"Avg Memory Used: {avg_mem:.1f} MB")

        print("\n" + "="*60)

        if self.gpu_active and self.max_util > 5:
            print("✅ GPU IS BEING USED!")
            print(f"   Peak utilization of {self.max_util}% indicates GPU kernels are running")
            return True
        elif self.max_memory > 500:
            print("⚠️  GPU MEMORY IS ALLOCATED BUT LOW UTILIZATION")
            print("   GPU memory is being used but compute kernels may not be running")
            print("   This suggests data is on GPU but processing might be on CPU")
            return False
        else:
            print("❌ GPU IS NOT BEING USED!")
            print("   No significant GPU activity detected")
            print("   System is likely using CPU fallback paths")
            return False

def main():
    print("Starting GPU Usage Monitor...")
    print("Run PRISM-AI operations in another terminal")
    print("Press Ctrl+C to stop monitoring and see report\n")

    monitor = GPUMonitor()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        monitor.stop()
        time.sleep(0.5)
        monitor.report()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start monitoring in thread
    thread = threading.Thread(target=monitor.monitor_gpu)
    thread.start()

    # Wait for thread to complete
    thread.join()

if __name__ == "__main__":
    main()