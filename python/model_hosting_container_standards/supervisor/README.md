# Supervisor Process Management

Provides supervisord-based process management for ML frameworks with automatic recovery and container-friendly logging.

## Overview

This module wraps your ML framework (vLLM, TensorRT-LLM, etc.) with supervisord to provide:

- **Automatic Process Monitoring**: Detects when your service crashes or exits unexpectedly
- **Auto-Recovery**: Automatically restarts failed processes with configurable retry limits
- **Container-Friendly**: Exits with code 1 after max retries so orchestrators (Docker, Kubernetes) can detect failures
- **Production Ready**: Structured logging, configurable behavior, and battle-tested supervisord underneath

**Use Case**: Deploy ML frameworks on SageMaker or any container platform with automatic crash recovery and proper failure signaling.

## Quick Setup

### 1. Install the Package
```bash
pip install model-hosting-container-standards
```

### 2. Extract the Entrypoint Script
Extract the entrypoint script from the installed package:
```bash
# In your Dockerfile (extracts to default: /opt/aws/supervisor-entrypoint.sh)
RUN extract-supervisor-entrypoint
```

Or specify a custom location:
```bash
# In your Dockerfile
RUN extract-supervisor-entrypoint -o /usr/local/bin/supervisor-entrypoint.sh
```

### 3. Configure Launch Command and Entrypoint
```dockerfile
# Set your framework's launch command
ENV LAUNCH_COMMAND="vllm serve model --host 0.0.0.0 --port 8080"

# Use supervisor entrypoint (using default path)
ENTRYPOINT ["/opt/aws/supervisor-entrypoint.sh"]
```

### Alternative: One-line Setup
```dockerfile
# Install and extract in one step (uses default path: /opt/aws/supervisor-entrypoint.sh)
RUN pip install model-hosting-container-standards && extract-supervisor-entrypoint

# Still need to configure your launch command and entrypoint
ENV LAUNCH_COMMAND="vllm serve model --host 0.0.0.0 --port 8080"
ENTRYPOINT ["/opt/aws/supervisor-entrypoint.sh"]
```

## Configuration

Configure your framework using environment variables. These can be set in your Dockerfile with `ENV` or overridden at container runtime.

### Default Paths
- **Entrypoint script**: `/opt/aws/supervisor-entrypoint.sh` (extracted by `extract-supervisor-entrypoint`)
- **Config file**: `/tmp/supervisord.conf` (generated automatically)

### Required: Launch Command
```bash
# Set your framework's start command
export LAUNCH_COMMAND="vllm serve model --host 0.0.0.0 --port 8080"
# or
export LAUNCH_COMMAND="python -m tensorrt_llm.hlapi.llm_api --host 0.0.0.0 --port 8080"
```

### Optional Settings
```bash
export ENGINE_AUTO_RECOVERY=true        # Auto-restart on failure (default: true)
export ENGINE_MAX_START_RETRIES=3       # Max restart attempts (default: 3, range: 0-100)
export SUPERVISOR_LOG_LEVEL=info        # Log level (default: info, options: debug, info, warn, error, critical)
export SUPERVISOR_CONFIG_PATH=/tmp/supervisord.conf  # Config file path (default: /tmp/supervisord.conf)
```

### Runtime Override Examples

Environment variables set in the Dockerfile can be overridden when launching the container:

```bash
# Override max retries at runtime
docker run -e ENGINE_MAX_START_RETRIES=5 my-image

# Disable auto-recovery at runtime
docker run -e ENGINE_AUTO_RECOVERY=false my-image

# Change log level for debugging
docker run -e SUPERVISOR_LOG_LEVEL=debug my-image

# Override multiple settings
docker run \
  -e ENGINE_MAX_START_RETRIES=10 \
  -e ENGINE_AUTO_RECOVERY=true \
  -e SUPERVISOR_LOG_LEVEL=debug \
  my-image
```

## Complete Example: vLLM + SageMaker Integration

### Dockerfile
```dockerfile
FROM vllm/vllm-openai:latest

# Install model hosting container standards and supervisor
RUN pip install supervisor model-hosting-container-standards

# Extract supervisor entrypoint (creates /opt/aws/supervisor-entrypoint.sh)
RUN extract-supervisor-entrypoint

# Copy your custom entrypoint script
COPY sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh

# Configure supervisor to launch your service
ENV LAUNCH_COMMAND="./sagemaker-entrypoint.sh"
ENV ENGINE_AUTO_RECOVERY=true
ENV ENGINE_MAX_START_RETRIES=3

# Use supervisor entrypoint for process management
ENTRYPOINT ["/opt/aws/supervisor-entrypoint.sh"]
```

### Custom Entrypoint Script (sagemaker-entrypoint.sh)
```bash
#!/bin/bash
# Your vLLM startup script with SageMaker integration

# Start vLLM with your model
exec vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype auto
```

### Service Monitoring Behavior

**Expected Behavior**: LLM services should run indefinitely. Any exit is treated as an error.

**Restart Logic**:
1. If your service exits for any reason (crash, OOM, etc.), it will be automatically restarted
2. Maximum restart attempts: `ENGINE_MAX_START_RETRIES` (default: 3)
3. If restart limit is exceeded, the container exits with code 1
4. This signals to container orchestrators (Docker, Kubernetes) that the service failed

**Why This Matters**: Container orchestrators can detect the failure and take appropriate action (restart container, alert operators, etc.)


## Troubleshooting

### Common Errors

**"No launch command available"**
```bash
# Fix: Set LAUNCH_COMMAND with your framework's start command
export LAUNCH_COMMAND="vllm serve model --host 0.0.0.0 --port 8080"
```

**"supervisord command not found"**
```bash
# Fix: Install supervisor
pip install supervisor
```

**Process keeps restarting**
```bash
# Fix: Disable auto-recovery to see the actual error
export ENGINE_AUTO_RECOVERY=false
export ENGINE_MAX_START_RETRIES=1
```

## Key Files

- `scripts/supervisor-entrypoint.sh` - Main entrypoint script for your container
- `scripts/extract_entrypoint.py` - CLI tool to extract the entrypoint script (`extract-supervisor-entrypoint`)
- `scripts/generate_supervisor_config.py` - Configuration generator (used internally)

That's all you need! The supervisor system handles the rest automatically.
