#!/usr/bin/env bash
# Instala o timer semanal como serviço do utilizador (systemd --user).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
dest="$HOME/.config/systemd/user"
mkdir -p "$dest"
for f in pechincha-weekly.service pechincha-weekly.timer; do
  sed "s#%h/Documents/overpaid-or-bargain#$repo#g" "$repo/scripts/systemd/$f" >"$dest/$f"
done
systemctl --user daemon-reload
systemctl --user enable --now pechincha-weekly.timer
systemctl --user list-timers pechincha-weekly.timer
