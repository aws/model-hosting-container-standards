# Supervisor Process Management

Provides supervisord-based process management for ML frameworks with automatic recovery and container-friendly logging.

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

### 3. Set as Container Entrypoint
```dockerfile
# In your Dockerfile (using default path)
ENTRYPOINT ["/opt/aws/supervisor-entrypoint.sh"]
```

### Alternative: One-line Setup
```dockerfile
# Install and extract in one step (uses default path: /opt/aws/supervisor-entrypoint.sh)
RUN pip install model-hosting-container-standards && extract-supervisor-entrypoint
```

## Configuration

Set environment variables to configure your framework:

### Default Paths
- **Entrypoint script**: `/opt/aws/supervisor-entrypoint.sh` (extracted by `extract-supervisor-entrypoint`)
- **Config file**: `/tmp/supervisord.conf` (generated automatically)

### Set Your Launch Command
```bash
export LAUNCH_COMMAND="python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8080"
# or
export LAUNCH_COMMAND="python -m tensorrt_llm.hlapi.llm_api --host 0.0.0.0 --port 8080"
# or any other framework start command
```

### Optional Settings
```bash
export ENGINE_AUTO_RECOVERY=true        # Auto-restart on failure (default: true)
export ENGINE_MAX_RECOVERY_ATTEMPTS=3   # Max restart attempts (default: 3)
export SUPERVISOR_LOG_LEVEL=info        # Log level (default: info)
export SUPERVISOR_CONFIG_PATH=/tmp/supervisord.conf  # Config file path (default: /tmp/supervisord.conf)
```

## What You Get

Your container will now:
- ✅ Automatically generate supervisor configuration
- ✅ Start your ML framework with process monitoring
- ✅ Auto-restart on failures
- ✅ Provide structured logging

## Example Dockerfile
```dockerfile
FROM python:3.10

# Install your ML framework and supervisor package
RUN pip install vllm model-hosting-container-standards

# Extract the entrypoint script from the package (default: /opt/aws/supervisor-entrypoint.sh)
RUN extract-supervisor-entrypoint

# Set environment
ENV LAUNCH_COMMAND="python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8080"

# Use supervisor entrypoint (default path)
ENTRYPOINT ["/opt/aws/supervisor-entrypoint.sh"]
```

## Usage Examples

### vLLM Example
```bash
export LAUNCH_COMMAND="python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8080"
export ENGINE_AUTO_RECOVERY=true
/opt/aws/supervisor-entrypoint.sh  # Using default path
```

### TensorRT-LLM Example
```bash
export LAUNCH_COMMAND="python -m tensorrt_llm.hlapi.llm_api --host 0.0.0.0 --port 8080"
export ENGINE_MAX_RECOVERY_ATTEMPTS=5
/opt/aws/supervisor-entrypoint.sh  # Using default path
```

### Minimal Recovery Mode
```bash
export LAUNCH_COMMAND="python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8080"
export ENGINE_AUTO_RECOVERY=false
export ENGINE_MAX_RECOVERY_ATTEMPTS=1
/opt/aws/supervisor-entrypoint.sh  # Using default path
```

## Troubleshooting

### Common Errors

**"No launch command available"**
```bash
# Fix: Set LAUNCH_COMMAND with your framework's start command
export LAUNCH_COMMAND="python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8080"
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
export ENGINE_MAX_RECOVERY_ATTEMPTS=1
```

## API Usage

```python
from model_hosting_container_standards.supervisor import (
    generate_supervisord_config,
    write_supervisord_config,
    SupervisorConfig
)

# Create configuration
config = SupervisorConfig(
    auto_recovery=True,
    max_recovery_attempts=5,
    launch_command="python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8080"
)

# Generate configuration content
config_content = generate_supervisord_config(config)

# Write configuration to file
write_supervisord_config("/tmp/supervisord.conf", config)
```

## Key Files

- `scripts/supervisor-entrypoint.sh` - Main entrypoint script for your container
- `scripts/extract_entrypoint.py` - CLI tool to extract the entrypoint script (`extract-supervisor-entrypoint`)
- `scripts/generate_supervisor_config.py` - Configuration generator (used internally)

That's all you need! The supervisor system handles the rest automatically.
