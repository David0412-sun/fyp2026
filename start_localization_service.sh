#!/usr/bin/env bash
set -euo pipefail

cd /home/david/fyp/raspberry_pi_localization

# 让 service 的环境尽量接近你手动开终端时的环境
export PATH="/home/david/fyp/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONUNBUFFERED=1
export XDG_RUNTIME_DIR="/run/user/1000"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus"
export PULSE_RUNTIME_PATH="/run/user/1000/pulse"
export PULSE_SERVER="unix:/run/user/1000/pulse/native"

# 给音频栈和 USB 设备一点时间
sleep 8

# 等待设备真正出现，最长等 30 秒
for i in $(seq 1 30); do
    if /home/david/fyp/.venv/bin/python /home/david/fyp/raspberry_pi_localization/app.py --list-devices 2>/dev/null | grep -q "Yundea 8MICA"; then
        break
    fi
    sleep 1
done

exec /home/david/fyp/.venv/bin/python /home/david/fyp/raspberry_pi_localization/app.py --host 0.0.0.0 --port 5000
