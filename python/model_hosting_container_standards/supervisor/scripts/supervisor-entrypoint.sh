#!/bin/bash

# Supervisor Process Management Entrypoint Script
set -euo pipefail

# Default values
DEFAULT_CONFIG_PATH="/tmp/supervisord.conf"

# Enhanced logging with timestamps
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" >&2
}

# Check basic requirements with comprehensive validation
check_requirements() {
    # Check for required environment variables
    if [[ -z "${LAUNCH_COMMAND:-}" ]]; then
        log_error "LAUNCH_COMMAND must be set"
        log_error "Set LAUNCH_COMMAND to your framework's start command, for example:"
        log_error "  export LAUNCH_COMMAND=\"python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8080\""
        log_error "  export LAUNCH_COMMAND=\"python -m tensorrt_llm.hlapi.llm_api --host 0.0.0.0 --port 8080\""
        return 1
    fi

    # Check for Python
    if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
        log_error "Python interpreter not found (python or python3)"
        return 1
    fi

    # Check for supervisord
    if ! command -v supervisord >/dev/null 2>&1; then
        log_error "supervisord command not found. Install supervisor package."
        return 1
    fi

    # Log configuration being used
    log_info "Configuration validation:"
    log_info "  LAUNCH_COMMAND: ${LAUNCH_COMMAND}"
    log_info "  ENGINE_AUTO_RECOVERY: ${ENGINE_AUTO_RECOVERY:-true}"
    log_info "  ENGINE_MAX_START_RETRIES: ${ENGINE_MAX_START_RETRIES:-3}"


    return 0
}

# Generate supervisord configuration with comprehensive error handling
generate_supervisor_config() {
    local config_path="${SUPERVISOR_CONFIG_PATH:-$DEFAULT_CONFIG_PATH}"
    local program_name="llm-engine"

    # Use Python module directly to generate configuration (works without package installation)
    local python_cmd="python3"
    if ! command -v python3 >/dev/null 2>&1; then
        python_cmd="python"
    fi

    if ! $python_cmd -m model_hosting_container_standards.supervisor.scripts.generate_supervisor_config -o "$config_path" -p "$program_name" --log-level "ERROR"; then
        log_error "Failed to generate supervisord configuration"
        return 1
    fi

    # Verify configuration file was created
    if [[ ! -f "$config_path" ]]; then
        log_error "Configuration file was not created: $config_path"
        return 1
    fi

    # Verify configuration file is not empty
    if [[ ! -s "$config_path" ]]; then
        log_error "Configuration file is empty: $config_path"
        return 1
    fi

    local file_size=$(stat -c%s "$config_path" 2>/dev/null || stat -f%z "$config_path" 2>/dev/null || echo "unknown")
    log_info "Configuration generated successfully: $config_path ($file_size bytes)"

    return 0
}

# Start supervisord with comprehensive error handling and process lifecycle logging
start_supervisord() {
    local config_path="${SUPERVISOR_CONFIG_PATH:-$DEFAULT_CONFIG_PATH}"

    # Final validation of supervisord command
    if ! command -v supervisord >/dev/null 2>&1; then
        log_error "supervisord command not found in PATH"
        log_error "Install supervisor package: pip install supervisor"
        return 1
    fi

    # Validate configuration file one more time
    if [[ ! -f "$config_path" ]]; then
        log_error "Configuration file not found: $config_path"
        return 1
    fi

    if [[ ! -r "$config_path" ]]; then
        log_error "Configuration file is not readable: $config_path"
        return 1
    fi

    log_info "Starting supervisord with configuration: $config_path"
    log_info "Process lifecycle logging will be handled by supervisord"

    # Set up signal handlers for graceful shutdown
    trap 'log_info "Received termination signal, shutting down supervisord"; exit 0' TERM INT

    # LLM Service Monitoring Strategy:
    # 1. LLM services should run indefinitely - any exit is an error
    # 2. supervisord will automatically restart failed processes up to max_recovery_attempts
    # 3. If restart limit is exceeded, program enters FATAL state
    # 4. We monitor for FATAL state and exit container with code 1 to signal failure
    # Start supervisord in background mode so we can monitor it
    log_info "Executing supervisord (PID: $$)"
    supervisord -c "$config_path" &
    local supervisord_pid=$!

    # Monitor supervisord and program status every 3 seconds
    # This loop continues until supervisord exits or we detect FATAL state
    local check_count=0
    local max_checks=60  # Maximum 3 minutes of monitoring (60 * 3 seconds)

    while kill -0 $supervisord_pid 2>/dev/null && [ $check_count -lt $max_checks ]; do
        # Check if our LLM program has entered FATAL state (too many restart failures)
        # FATAL state means supervisord gave up trying to restart the program
        local status_output=$(supervisorctl -c "$config_path" status llm-engine 2>/dev/null || echo "")

        if echo "$status_output" | grep -q "FATAL"; then
            log_error "Program llm-engine entered FATAL state after maximum retry attempts"
            log_error "This indicates the LLM service is failing to start or crashing repeatedly"
            log_error "Shutting down supervisord and exiting with code 1"
            supervisorctl -c "$config_path" shutdown 2>/dev/null || true
            wait $supervisord_pid 2>/dev/null || true
            exit 1
        fi

        check_count=$((check_count + 1))
        sleep 3
    done

    # If we exceeded max checks, something is wrong
    if [ $check_count -ge $max_checks ]; then
        log_error "Monitoring timeout exceeded - shutting down"
        supervisorctl -c "$config_path" shutdown 2>/dev/null || true
        wait $supervisord_pid 2>/dev/null || true
        exit 1
    fi

    # Wait for supervisord to finish and get its exit code
    wait $supervisord_pid
    local exit_code=$?
    log_info "Supervisord exited with code: $exit_code"
    exit $exit_code
}

# Main execution with comprehensive error handling and logging
main() {
    log_info "=== Starting Supervisor Process Management ==="
    log_info "Entrypoint script: $0"
    log_info "Process ID: $$"
    log_info "User: $(whoami 2>/dev/null || echo 'unknown')"
    log_info "Working directory: $(pwd)"

    # Execute each step with error handling
    log_info "Step 1: Checking requirements"
    if ! check_requirements; then
        log_error "Requirements check failed"
        exit 1
    fi

    log_info "Step 2: Generating supervisor configuration"
    if ! generate_supervisor_config; then
        log_error "Configuration generation failed"
        exit 1
    fi

    log_info "Step 3: Starting supervisord"
    if ! start_supervisord; then
        log_error "Supervisord startup failed"
        exit 1
    fi

    # This should never be reached due to exec in start_supervisord
    log_error "Unexpected return from supervisord"
    exit 1
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
