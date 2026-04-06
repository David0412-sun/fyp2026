#!/usr/bin/env bash
set -euo pipefail

sudo systemctl disable --now localization.service 2>/dev/null || true
systemctl --user disable --now localization.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/localization.service
rm -f /home/david/.config/systemd/user/localization.service
sudo systemctl daemon-reload || true
systemctl --user daemon-reload || true
pkill -9 -f "/home/david/fyp/raspberry_pi_localization/app.py" 2>/dev/null || true
