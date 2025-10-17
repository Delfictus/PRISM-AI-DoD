# NVIDIA Driver 580 + Secure Boot Issue

## 🔒 Problem Identified
**Secure Boot** is blocking the unsigned NVIDIA 580 driver modules from loading.

Error: `modprobe: ERROR: could not insert 'nvidia': Key was rejected by service`

## Solutions (Choose One)

### Option 1: Disable Secure Boot (Easiest, Recommended)
1. **Reboot and enter BIOS/UEFI**
   ```bash
   sudo reboot
   # During boot, press F2/F12/Del/Esc (depends on motherboard)
   ```

2. **In BIOS/UEFI Settings**
   - Navigate to: Security → Secure Boot
   - Change: Secure Boot = **Disabled**
   - Save and Exit (F10)

3. **After reboot, verify GPU works**
   ```bash
   nvidia-smi
   ```

### Option 2: Sign NVIDIA Modules (Keep Secure Boot)
If you need Secure Boot enabled for security:

1. **Create signing keys**
   ```bash
   sudo openssl req -new -x509 -newkey rsa:2048 -keyout MOK.priv -outform DER -out MOK.der -nodes -days 36500 -subj "/CN=NVIDIA Driver Signing Key/"
   ```

2. **Sign the modules**
   ```bash
   sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 ./MOK.priv ./MOK.der $(modinfo -n nvidia)
   sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 ./MOK.priv ./MOK.der $(modinfo -n nvidia_uvm)
   sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 ./MOK.priv ./MOK.der $(modinfo -n nvidia_drm)
   sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 ./MOK.priv ./MOK.der $(modinfo -n nvidia_modeset)
   ```

3. **Enroll the key with MOK**
   ```bash
   sudo mokutil --import MOK.der
   # Set a password (remember it!)
   sudo reboot
   ```

4. **During reboot**
   - MOK Management will appear
   - Select "Enroll MOK"
   - Enter the password you set
   - Reboot

### Option 3: Use Ubuntu's Pre-Signed Drivers
Install NVIDIA drivers from Ubuntu's repository (pre-signed):
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

## Current Status
| Component | Status | Issue |
|-----------|--------|-------|
| Driver 580 | ✅ Installed | ❌ Can't load (unsigned) |
| CUDA 13 | ✅ Installed | ⏸️ Waiting for driver |
| RTX 5070 | ✅ Detected | ❌ Not accessible |
| Secure Boot | ✅ Enabled | 🔒 Blocking driver |

## Recommended Action
**Disable Secure Boot** (Option 1) - This is the fastest solution and won't impact your system's functionality unless you specifically need Secure Boot for enterprise compliance.

## After Fix
Once Secure Boot is disabled or modules are signed:
```bash
# GPU should work immediately
nvidia-smi

# Test CUDA 13
cd /home/<user>/PRISM-AI-DoD/src
./test_cuda13

# Test PRISM-AI
cargo test --features cuda
```

## Why This Happened
- Driver 580 is newer and may not be in Ubuntu's signed driver list yet
- Manual driver installations are typically unsigned
- Secure Boot prevents loading unsigned kernel modules (security feature)

---
*This is a common issue and easily fixed. Your GPU setup is otherwise perfect!*