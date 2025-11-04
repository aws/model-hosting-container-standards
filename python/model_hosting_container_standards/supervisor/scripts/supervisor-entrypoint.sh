#!/bin/bash
set -euo pipefail

CONFIG_PATH="${SUPERVISOR_CONFIG_PATH:-/tmp/supervisord.conf}"

log() {
    echo "[$(date '+%H:%M:%S')] $*" >&2
}

# Check requirements
if [[ -z "${LAUNCH_COMMAND:-}" ]]; then
    log "ERROR: LAUNCH_COMMAND must be set"
    exit 1
fi

if ! command -v supervisord >/dev/null 2>&1; then
    log "ERROR: supervisord not found. Install supervisor package."
    exit 1
fi

# Configuration validation
log "Configuration validation:"
log "  LAUNCH_COMMAND: ${LAUNCH_COMMAND}"
log "  ENGINE_AUTO_RECOVERY: ${ENGINE_AUTO_RECOVERY:-true}"
log "  ENGINE_MAX_START_RETRIES: ${ENGINE_MAX_START_RETRIES:-3}"

# Generate config
python_cmd="python3"
if ! command -v python3 >/dev/null 2>&1; then
    python_cmd="python"
fi

log "Generating supervisor config..."
if ! $python_cmd -m model_hosting_container_standards.supervisor.scripts.generate_supervisor_config -o "$CONFIG_PATH" -p "llm-engine" --log-level "ERROR"; then
    log "ERROR: Failed to generate config"
    exit 1
fi

log "Configuration generated successfully"

# Start supervisord with monitoring
log "Starting supervisord..."
trap 'log "Shutting down"; exit 0' TERM INT

supervisord -c "$CONFIG_PATH" &
supervisord_pid=$!

# LLM Service Monitoring Strategy:
# LLM services should run indefinitely - any exit is an error
# Monitor for FATAL state (indicates repeated failures)
while kill -0 $supervisord_pid 2>/dev/null; do
    status_output=$(supervisorctl -c "$CONFIG_PATH" status llm-engine 2>/dev/null || echo "")
    if echo "$status_output" | grep -q "FATAL"; then
        log "ERROR: LLM service failed repeatedly"
        supervisorctl -c "$CONFIG_PATH" shutdown 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

wait $supervisord_pid
