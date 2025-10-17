#!/usr/bin/env python3
"""
AUTOMATED CPU FALLBACK PURGE
Removes ALL #[cfg(not(feature = "cuda"))] blocks from source files
GPU-ONLY enforcement
"""

import re
import sys
from pathlib import Path

def remove_cpu_fallback_blocks(content):
    """Remove all #[cfg(not(feature = "cuda"))] blocks"""

    # Pattern to match #[cfg(not(feature = "cuda"))] blocks
    # This handles both single-line and multi-line blocks
    pattern = r'#\[cfg\(not\(feature = "cuda"\)\)\]\s*\{[^}]*\}(?:\s*\n)?'

    # Remove simple blocks
    content = re.sub(pattern, '', content, flags=re.DOTALL)

    # More aggressive pattern for nested blocks
    # Remove entire #[cfg(not(feature = "cuda"))] sections
    lines = content.split('\n')
    result_lines = []
    skip_until_bracket = 0
    in_cpu_block = False

    for line in lines:
        if '#[cfg(not(feature = "cuda"))]' in line:
            in_cpu_block = True
            skip_until_bracket = 0
            continue

        if in_cpu_block:
            if '{' in line:
                skip_until_bracket += line.count('{')
                skip_until_bracket -= line.count('}')
                if skip_until_bracket <= 0:
                    in_cpu_block = False
                continue
            elif '}' in line:
                skip_until_bracket -= line.count('}')
                if skip_until_bracket <= 0:
                    in_cpu_block = False
                continue
            else:
                continue

        result_lines.append(line)

    return '\n'.join(result_lines)

def process_file(filepath):
    """Process a single file to remove CPU fallback"""
    print(f"Processing: {filepath}")

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        original_count = content.count('#[cfg(not(feature = "cuda"))]')

        if original_count == 0:
            print(f"  ✅ No CPU fallback found")
            return 0

        new_content = remove_cpu_fallback_blocks(content)
        new_count = new_content.count('#[cfg(not(feature = "cuda"))]')

        with open(filepath, 'w') as f:
            f.write(new_content)

        removed = original_count - new_count
        print(f"  ✅ Removed {removed} CPU fallback blocks")
        return removed

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0

def main():
    repo_root = Path("/home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code/src")

    files_to_fix = [
        "pwsa/active_inference_classifier.rs",
        "cma/transfer_entropy_gpu.rs",
        "cma/gpu_integration.rs",
        "cma/neural/neural_quantum.rs",
        "cma/quantum/pimc_gpu.rs",
        "information_theory/gpu_transfer_entropy.rs",
        "quantum_mlir/runtime.rs",
        "statistical_mechanics/gpu_bindings.rs",
        "statistical_mechanics/gpu_integration.rs",
        "active_inference/gpu_inference.rs",
        "active_inference/gpu_policy_eval.rs",
        "bin/prism.rs",
    ]

    print("="*50)
    print("  AUTOMATED CPU FALLBACK PURGE")
    print("  GPU-ONLY ENFORCEMENT")
    print("="*50)
    print()

    total_removed = 0

    for file_path in files_to_fix:
        full_path = repo_root / file_path
        if full_path.exists():
            removed = process_file(full_path)
            total_removed += removed
        else:
            print(f"Skipping {file_path} (not found)")

    print()
    print("="*50)
    print(f"✅ PURGE COMPLETE: Removed {total_removed} CPU fallback blocks")
    print("="*50)

if __name__ == "__main__":
    main()
