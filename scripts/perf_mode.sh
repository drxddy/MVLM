#!/usr/bin/env bash
# Enable performance mode on Qualcomm Adreno A6x/A7x GPUs.
# Based on Qualcomm OpenCL Programming Guide (80-NB295-11 Rev C), Appendix A.3.
# Requires root access on the device.
set -euo pipefail

echo ">>> Setting performance mode (requires root)..."

# --- CPU: set all cores to performance governor ---
adb shell "su -c '\
for policy in /sys/devices/system/cpu/cpufreq/policy*; do
    echo performance > \$policy/scaling_governor 2>/dev/null || true
done
'"
echo "    CPU governors set to performance."

# --- GPU: set governor and lock to max frequency ---
# Adreno A6x/A7x sysfs paths
GPU_GOV="/sys/class/kgsl/kgsl-3d0/devfreq/governor"
GPU_MAX_FREQ="/sys/class/kgsl/kgsl-3d0/max_gpuclk"
GPU_MIN_FREQ="/sys/class/kgsl/kgsl-3d0/min_gpuclk" # used to lock freq
GPU_FREQ_TABLE="/sys/class/kgsl/kgsl-3d0/gpu_available_frequencies"

adb shell "su -c '\
echo performance > $GPU_GOV 2>/dev/null || true

MAX_FREQ=\$(cat $GPU_MAX_FREQ 2>/dev/null || echo \"\")
if [ -n \"\$MAX_FREQ\" ]; then
    echo \$MAX_FREQ > $GPU_MIN_FREQ 2>/dev/null || true
    echo \"    GPU locked to max frequency: \${MAX_FREQ} Hz\"
else
    echo \"    WARNING: Could not read max GPU frequency\"
fi

echo \"    GPU governor set to performance.\"
echo \"    Available GPU frequencies:\"
cat $GPU_FREQ_TABLE 2>/dev/null || echo \"    (could not read frequency table)\"
'"

# --- Disable thermal throttling (optional, use with caution) ---
# Uncomment the following to disable thermal mitigation:
# adb shell "su -c 'echo disabled > /sys/devices/virtual/thermal/thermal_zone0/mode 2>/dev/null || true'"
# echo "    WARNING: Thermal throttling disabled. Monitor temperatures!"

echo ">>> Performance mode enabled."
echo ">>> To revert, reboot the device."
