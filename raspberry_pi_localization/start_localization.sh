#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/david/fyp/raspberry_pi_localization"
VENV_PY="/home/david/fyp/.venv/bin/python"
LOG_FILE="/home/david/fyp/localization.log"
LOCK_FILE="/tmp/localization_app.lock"

export CLOUD_UPLOAD_ENABLED=1
export CLOUD_API_INGEST_URL="http://112.124.26.47:9000/api/ingest"
export CLOUD_API_INGEST_TOKEN="my-localization-test-token-123"
export CLOUD_SOURCE_NAME="raspberry-pi"

export PATH="/home/david/fyp/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONUNBUFFERED=1

# 等系统、USB 和音频栈稳定
sleep 15

# 防止重复启动
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "$(date -Is) localization already running" >> "${LOG_FILE}"
    exit 0
fi

# 如果 5000 已经被占用，就不再启动第二个实例
if ss -ltn sport = :5000 | grep -q LISTEN; then
    echo "$(date -Is) port 5000 already in use, skip start" >> "${LOG_FILE}"
    exit 0
fi

cd "${APP_DIR}"

echo "$(date -Is) starting localization app" >> "${LOG_FILE}"
echo "Using CLOUD_API_INGEST_TOKEN=$CLOUD_API_INGEST_TOKEN" >> $LOG_FILE
exec "${VENV_PY}" /home/david/fyp/raspberry_pi_localization/app.py --host 0.0.0.0 --port 5000 >> "${LOG_FILE}" 2>&1