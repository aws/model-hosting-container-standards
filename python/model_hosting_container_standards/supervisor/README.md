# Supervisor Process Management

Provides supervisord-based process management for ML frameworks with automatic recovery and container-friendly logging.

## Overview

This module wraps your ML framework (vLLM, TensorRT-LLM, etc.) with supervisord to provide:

- **Automatic Process Monitoring**: Detects when your service crashes or exits unexpectedly
- **Auto-Recovery**: Automatically restarts failed processes with configurable retry limits
- **Container-Friendly**: Exits with code 1 after max retries so orchestrators (Docker, Kubernetes) can detect failures
- **Production Ready**: Structured logging, configurable behavior, and battle-tested supervisord underneath

**Use Case**: Deploy ML frameworks on SageMaker or any container platform with automatic crash recovery and proper failure signaling.

## Quick Setup (Simplified CLI Approach)

### 1. Install the Package
```bash
pip install model-hosting-container-standards
```

### 2. Use standard-supervisor with Your Command
Simply prepend `standard-supervisor` to your existing framework command:

```dockerfile
# Basic usage - just add standard-supervisor before your command
CMD ["standard-supervisor", "vllm", "serve", "model", "--host", "0.0.0.0", "--port", "8080"]
```

### 3. Alternative: Entrypoint Style
```dockerfile
# Use as entrypoint for more flexibility
ENTRYPOINT ["standard-supervisor"]
CMD ["vllm", "serve", "model", "--host", "0.0.0.0", "--port", "8080"]
```

That's it! No complex setup, no script extraction, no custom entrypoints needed.

## Configuration

Configure supervisor behavior using the unified `SUPERVISOR_*` environment variable pattern. These can be set in your Dockerfile with `ENV` or overridden at container runtime.

### Default Behavior
- **Config file**: `/tmp/supervisord.conf` (generated automatically)
- **Auto-recovery**: Enabled by default
- **Max retries**: 3 attempts
- **Log level**: info

### Configuration Options

#### Application-Level Configuration (Recommended)
Use these simple environment variables for common settings:

```bash
# Basic application behavior
export PROCESS_AUTO_RECOVERY=true                   # Auto-restart on failure (default: true)
export PROCESS_MAX_START_RETRIES=3                  # Max restart attempts (default: 3)
export LOG_LEVEL=info                               # Log level (default: info, options: debug, info, warn, error, critical)
```

#### Advanced SUPERVISOR_* Configuration
Use the pattern `SUPERVISOR_{SECTION}_{KEY}=VALUE` for advanced supervisord customization:

**Important**:
- The default program name is `app`
- To target specific programs, use double underscores `__` to represent colons in section names
- Program names in environment variables use the same format (e.g., `APP` for `app`)

```bash
# Program section overrides (for default program "app")
export SUPERVISOR_PROGRAM__APP_STARTSECS=10              # Seconds to wait before considering started (default: 1)
export SUPERVISOR_PROGRAM__APP_STOPWAITSECS=30           # Seconds to wait for graceful shutdown (default: 10)
export SUPERVISOR_PROGRAM__APP_AUTORESTART=unexpected    # Advanced restart control (true/false/unexpected)

# For program-specific overrides, use the program name (default: "app")
# Or use application-level variables like PROCESS_MAX_START_RETRIES for simpler configuration

# Supervisord daemon configuration
export SUPERVISOR_SUPERVISORD_LOGLEVEL=debug        # Daemon log level (can differ from application LOG_LEVEL)
export SUPERVISOR_SUPERVISORD_LOGFILE=/tmp/supervisord.log  # Log file location

# Unix HTTP server configuration
export SUPERVISOR_UNIX_HTTP_SERVER_FILE=/tmp/supervisor.sock  # Socket file location
```

### Common Configuration Examples

```bash
# High availability setup with more retries (recommended approach)
export PROCESS_MAX_START_RETRIES=10
export SUPERVISOR_PROGRAM__APP_STARTSECS=30
export SUPERVISOR_PROGRAM__APP_STARTRETRIES=10

# Debug mode with verbose logging
export LOG_LEVEL=debug
export SUPERVISOR_SUPERVISORD_LOGLEVEL=debug

# Quick restart for development
export SUPERVISOR_PROGRAM__APP_STARTSECS=1
export SUPERVISOR_PROGRAM__APP_STOPWAITSECS=5
export SUPERVISOR_PROGRAM__APP_STARTRETRIES=1

# Disable auto-recovery for debugging
export PROCESS_AUTO_RECOVERY=false
export PROCESS_MAX_START_RETRIES=1
```

### Runtime Override Examples

Environment variables set in the Dockerfile can be overridden when launching the container:

```bash
# Override max retries at runtime (recommended)
docker run -e PROCESS_MAX_START_RETRIES=5 my-image

# Disable auto-recovery at runtime (recommended)
docker run -e PROCESS_AUTO_RECOVERY=false my-image

# Change log level for debugging (recommended)
docker run -e LOG_LEVEL=debug my-image

# Override multiple settings (recommended approach)
docker run \
  -e PROCESS_MAX_START_RETRIES=10 \
  -e PROCESS_AUTO_RECOVERY=true \
  -e LOG_LEVEL=debug \
  my-image

# Advanced: Direct supervisord configuration override
docker run \
  -e SUPERVISOR_PROGRAM__APP_STARTSECS=30 \
  -e SUPERVISOR_PROGRAM__APP_STARTRETRIES=5 \
  -e SUPERVISOR_SUPERVISORD_LOGLEVEL=debug \
  my-image
```

## Complete Examples

### Basic vLLM Example
```dockerfile
FROM vllm/vllm-openai:latest

# Install model hosting container standards (includes supervisor)
RUN pip install model-hosting-container-standards

# Use standard-supervisor with your vLLM command
CMD ["standard-supervisor", "vllm", "serve", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "--host", "0.0.0.0", "--port", "8080", "--dtype", "auto"]
```

### TensorRT-LLM Example
```dockerfile
FROM nvcr.io/nvidia/tensorrt:23.08-py3

# Install dependencies and model hosting container standards
RUN pip install tensorrt-llm model-hosting-container-standards

# Use standard-supervisor with TensorRT-LLM
CMD ["standard-supervisor", "python", "-m", "tensorrt_llm.hlapi.llm_api", "--host", "0.0.0.0", "--port", "8080"]
```

### Advanced Configuration Example
```dockerfile
FROM vllm/vllm-openai:latest

# Install model hosting container standards
RUN pip install model-hosting-container-standards

# Configure supervisor behavior (recommended approach)
ENV PROCESS_MAX_START_RETRIES=5
ENV LOG_LEVEL=debug
ENV SUPERVISOR_PROGRAM__APP_STARTSECS=30
ENV SUPERVISOR_PROGRAM__APP_STARTRETRIES=5

# Use standard-supervisor with custom configuration
CMD ["standard-supervisor", "vllm", "serve", "model", "--host", "0.0.0.0", "--port", "8080"]
```

### SageMaker Integration with Custom Script
```dockerfile
FROM vllm/vllm-openai:latest

# Install model hosting container standards
RUN pip install model-hosting-container-standards

# Copy your custom startup script
COPY sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh

# Configure supervisor for production (recommended approach)
ENV PROCESS_MAX_START_RETRIES=3
ENV PROCESS_AUTO_RECOVERY=true

# Use standard-supervisor with your custom script
CMD ["standard-supervisor", "./sagemaker-entrypoint.sh"]
```

### Entrypoint Style for Flexibility
```dockerfile
FROM vllm/vllm-openai:latest

# Install model hosting container standards
RUN pip install model-hosting-container-standards

# Optional: Configure supervisor (recommended approach)
ENV PROCESS_MAX_START_RETRIES=5
ENV LOG_LEVEL=info

# Use as entrypoint for runtime flexibility
ENTRYPOINT ["standard-supervisor"]
CMD ["vllm", "serve", "model", "--host", "0.0.0.0", "--port", "8080"]
```

### Service Monitoring Behavior

**Expected Behavior**: LLM services should run indefinitely. Any exit is treated as an error.

**Restart Logic**:
1. If your service exits for any reason (crash, OOM, etc.), it will be automatically restarted
2. Maximum restart attempts: `PROCESS_MAX_START_RETRIES` (default: 3)
3. If restart limit is exceeded, the container exits with code 1
4. This signals to container orchestrators (Docker, Kubernetes) that the service failed

**Why This Matters**: Container orchestrators can detect the failure and take appropriate action (restart container, alert operators, etc.)


## Troubleshooting

### Common Errors

**"No command provided"**
```bash
# Fix: Provide a command after standard-supervisor
standard-supervisor vllm serve model --host 0.0.0.0 --port 8080
```

**"supervisord command not found"**
```bash
# Fix: Install supervisor (usually included with model-hosting-container-standards)
pip install supervisor
```

**Process keeps restarting**
```bash
# Fix: Disable auto-recovery to see the actual error (recommended)
export PROCESS_AUTO_RECOVERY=false
export PROCESS_MAX_START_RETRIES=1
```

**Configuration not taking effect**
```bash
# Fix: Use recommended application-level variables first
# Recommended: PROCESS_MAX_START_RETRIES=5
# Advanced (specific program): SUPERVISOR_PROGRAM__APP_STARTRETRIES=5
```

## Framework-Specific Examples

### vLLM Examples
```bash
# Basic vLLM server
standard-supervisor vllm serve model --host 0.0.0.0 --port 8080

# vLLM with specific model and parameters
standard-supervisor vllm serve microsoft/DialoGPT-medium --host 0.0.0.0 --port 8080 --dtype auto --max-model-len 2048

# vLLM with OpenAI-compatible API
standard-supervisor python -m vllm.entrypoints.openai.api_server --model model --host 0.0.0.0 --port 8080
```

### TensorRT-LLM Examples
```bash
# TensorRT-LLM API server
standard-supervisor python -m tensorrt_llm.hlapi.llm_api --host 0.0.0.0 --port 8080

# TensorRT-LLM with custom model path
standard-supervisor python -m tensorrt_llm.hlapi.llm_api --model-dir /opt/model --host 0.0.0.0 --port 8080
```

### Custom Python Scripts
```bash
# Your custom ML serving script
standard-supervisor python my_model_server.py --port 8080

# FastAPI application
standard-supervisor uvicorn app:app --host 0.0.0.0 --port 8080

# Any other command
standard-supervisor ./my-custom-entrypoint.sh
```



## Key Files

- `scripts/standard_supervisor.py` - Main CLI entry point (`standard-supervisor` command)
- `scripts/generate_supervisor_config.py` - Configuration generator (used internally)

That's all you need! The supervisor system handles the rest automatically.
