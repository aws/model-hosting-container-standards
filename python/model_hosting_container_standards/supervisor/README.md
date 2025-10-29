# Supervisor Process Management

Provides supervisord-based process management for ML frameworks with automatic recovery and container-friendly logging.

## Quick Setup

### 1. Install the Package
```bash
pip install model-hosting-container-standards
```

### 2. Copy the Entrypoint Script
Copy `supervisor-entrypoint.sh` to your container and make it executable:
```bash
# In your Dockerfile
COPY supervisor-entrypoint.sh /opt/aws/
RUN chmod +x /opt/aws/supervisor-entrypoint.sh
```

### 3. Set as Container Entrypoint
```dockerfile
# In your Dockerfile
ENTRYPOINT ["/opt/aws/supervisor-entrypoint.sh"]
```

## Configuration

Set environment variables to configure your framework:

### Option 1: Use Framework Name (Recommended)
```bash
export FRAMEWORK_NAME=vllm  # or tensorrt-llm
```

### Option 2: Use Custom Command
```bash
export FRAMEWORK_COMMAND="python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8080"
```

### Optional Settings
```bash
export ENGINE_AUTO_RECOVERY=true        # Auto-restart on failure (default: true)
export ENGINE_MAX_RECOVERY_ATTEMPTS=3   # Max restart attempts (default: 3)
export ENGINE_RECOVERY_BACKOFF_SECONDS=10  # Wait between restarts (default: 10)
export SUPERVISOR_LOG_LEVEL=info        # Log level (default: info)
export SUPERVISOR_CONFIG_PATH=/tmp/supervisord.conf  # Config file path
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

# Install your ML framework
RUN pip install vllm model-hosting-container-standards

# Copy the entrypoint script
COPY supervisor-entrypoint.sh /opt/aws/
RUN chmod +x /opt/aws/supervisor-entrypoint.sh

# Set environment
ENV FRAMEWORK_NAME=vllm

# Use supervisor entrypoint
ENTRYPOINT ["/opt/aws/supervisor-entrypoint.sh"]
```

## Usage Examples

### vLLM Example
```bash
export FRAMEWORK_NAME=vllm
export ENGINE_AUTO_RECOVERY=true
./supervisor-entrypoint.sh
```

### Custom Framework Example
```bash
export FRAMEWORK_COMMAND="python -m my_framework.server --port 8080"
export ENGINE_MAX_RECOVERY_ATTEMPTS=5
./supervisor-entrypoint.sh
```

### Debug Mode
```bash
export FRAMEWORK_NAME=vllm
export SUPERVISOR_DEBUG=true
export SUPERVISOR_LOG_LEVEL=debug
export ENGINE_MAX_RECOVERY_ATTEMPTS=1
./supervisor-entrypoint.sh
```

## Troubleshooting

### Common Errors

**"No framework command available"**
```bash
# Fix: Set either FRAMEWORK_NAME or FRAMEWORK_COMMAND
export FRAMEWORK_NAME=vllm
```

**"Invalid FRAMEWORK_NAME"**
```bash
# Fix: Use supported framework (vllm, tensorrt-llm) or custom command
export FRAMEWORK_NAME=vllm
# OR
export FRAMEWORK_COMMAND="python -m your_framework"
```

**"supervisord command not found"**
```bash
# Fix: Install supervisor
pip install supervisor
```

**Process keeps restarting**
```bash
# Fix: Enable debug mode and check logs
export SUPERVISOR_DEBUG=true
export ENGINE_MAX_RECOVERY_ATTEMPTS=1
```

## API Usage

```python
from model_hosting_container_standards.supervisor import (
    generate_supervisord_config,
    get_framework_command,
    SupervisorConfig
)

# Get framework command
command = get_framework_command()

# Generate configuration
config_content = generate_supervisord_config(command)

# Custom configuration
config = SupervisorConfig(
    auto_recovery=True,
    max_recovery_attempts=5,
    framework_command="python -m vllm.entrypoints.api_server"
)
```

## Key Files

- `scripts/supervisor-entrypoint.sh` - Main entrypoint script to copy to your container
- `scripts/generate_supervisor_config.py` - Configuration generator (used internally)

That's all you need! The supervisor system handles the rest automatically.
