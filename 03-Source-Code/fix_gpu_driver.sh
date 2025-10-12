#!/bin/bash
# Fix GPU driver 580 after update
# Run with: sudo ./fix_gpu_driver.sh

echo "🔧 Attempting to fix GPU driver 580 without reboot..."
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run with sudo: sudo ./fix_gpu_driver.sh"
    exit 1
fi

echo "1. Loading NVIDIA kernel modules..."
modprobe nvidia && echo "   ✅ nvidia module loaded" || echo "   ❌ Failed to load nvidia"
modprobe nvidia_uvm && echo "   ✅ nvidia_uvm module loaded" || echo "   ❌ Failed to load nvidia_uvm"
modprobe nvidia_drm && echo "   ✅ nvidia_drm module loaded" || echo "   ❌ Failed to load nvidia_drm"
modprobe nvidia_modeset && echo "   ✅ nvidia_modeset module loaded" || echo "   ❌ Failed to load nvidia_modeset"

echo ""
echo "2. Creating device files..."
nvidia-modprobe && echo "   ✅ Device files created" || echo "   ❌ Failed to create device files"

echo ""
echo "3. Checking device files..."
ls -la /dev/nvidia* 2>/dev/null && echo "   ✅ Device files exist" || echo "   ❌ Device files missing"

echo ""
echo "4. Starting persistence daemon..."
systemctl restart nvidia-persistenced && echo "   ✅ Persistence daemon started" || echo "   ❌ Failed to start daemon"

echo ""
echo "5. Testing nvidia-smi..."
nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ nvidia-smi working!"
    nvidia-smi
else
    echo "   ❌ nvidia-smi still not working"
    echo "   ⚠️  A reboot is required to complete driver installation"
    echo ""
    echo "   Run: sudo reboot"
fi

echo ""
echo "📌 If this didn't work, please reboot your system"