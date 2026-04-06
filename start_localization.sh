#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/david/fyp/raspberry_pi_localization"
VENV_PY="/home/david/fyp/.venv/bin/python"
LOG_FILE="/home/david/fyp/localization.log"
LOCK_FILE="/tmp/localization_app.lock"

# 等待系统、USB 和音频栈稳定
sleep 15

# 避免重复启动
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "$(date -Is) localization already running" >> "${LOG_FILE}"
  exit 0
fi

# 如果5000端口已经被占用，则不再启动第二个实例
if ss -ltn sport = :5000 | grep -q LISTEN; then
  echo "$(date -Is) port 5000 already in use, skip start" >> "${LOG_FILE}"
  exit 0
fi

cd "${APP_DIR}"

# 这里故意用 login shell，尽量贴近你手动打开终端时的环境
exec /usr/bin/bash -lc '
  export PATH="/home/david/fyp/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  export PYTHONUNBUFFERED=1
  cd /home/david/fyp/raspberry_pi_localization
  /home/david/fyp/.venv/bin/python app.py --host 0.0.0.0 --port 5000
' >> "${LOG_FILE}" 2>&1
