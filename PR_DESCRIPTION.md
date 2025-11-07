# Add Supervisor Process Management Module

This introduces a **supervisor module** that wraps ML frameworks with supervisord for automatic crash recovery and robust process management. It can be integrated into any Dockerfile easily.

## Integration

Install and use with these commands:

```bash
pip install model-hosting-container-standards
standard-supervisor vllm serve model --host 0.0.0.0 --port 8080
```

Or in a Dockerfile:
```dockerfile
COPY model_hosting_container_standards-0.1.2-py3-none-any.whl /tmp/
RUN pip install supervisor
RUN pip install /tmp/model_hosting_container_standards-0.1.2-py3-none-any.whl

# Use supervisor entrypoint for SageMaker
ENV ENGINE_AUTO_RECOVERY=true
ENV ENGINE_MAX_RECOVERY_ATTEMPTS=3
ENTRYPOINT ["standard-supervisor", "./sagemaker-entrypoint.sh"]
```

## Workflow

1. **Parse command and environment** → Read ML framework command and supervisor configuration
2. **Generate supervisord config** → Create robust configuration with configparser
3. **Start supervisord** → Launch supervisor daemon with your framework as managed process
4. **Monitor and restart** → Supervisor detects crashes and restarts automatically with configurable limits
5. **Handle failures** → After max retries, container exits gracefully with proper error codes

### **Key Components**

**Core Modules:**
- `models.py` - Configuration data models with comprehensive validation and environment variable parsing
- `generator.py` - Robust supervisord configuration generation using configparser

**CLI Tools & Scripts:**
- `scripts/standard_supervisor.py` - Main CLI tool for running ML frameworks under supervisor (`standard-supervisor`)
- `scripts/generate_supervisor_config.py` - Standalone configuration generator CLI

**Documentation & Tests:**
- `README.md` - Comprehensive setup guide with examples
- `tests/integration/test_supervisor_cli_integration.py` - **Real behavior integration tests** that verify actual restart and retry behavior
- `tests/supervisor/` - Comprehensive unit tests for all components

## Usage Examples

### Simple CLI Usage
```bash
# Direct command execution with supervisor
standard-supervisor vllm serve model --host 0.0.0.0 --port 8080

# With custom configuration
PROCESS_MAX_START_RETRIES=5 SUPERVISOR_PROGRAM__LLM_ENGINE_STARTSECS=30 \
standard-supervisor python -m tensorrt_llm.hlapi.llm_api
```

### Dockerfile Integration
```dockerfile
FROM vllm/vllm-openai:latest

# Install with supervisor support
RUN pip install model-hosting-container-standards

# Configure your ML framework with supervisor settings
ENV PROCESS_MAX_START_RETRIES=3
ENV SUPERVISOR_PROGRAM__LLM_ENGINE_STARTSECS=30
ENV SUPERVISOR_PROGRAM__LLM_ENGINE_STOPWAITSECS=60
ENV LOG_LEVEL=info

# Use supervisor for process management
ENTRYPOINT ["python", "-m", "model_hosting_container_standards.supervisor.scripts.standard_supervisor"]
CMD ["vllm", "serve", "model", "--host", "0.0.0.0", "--port", "8080"]
```

## Configuration Options

**Basic Configuration:**
- Command line arguments become the supervised process command
- `PROCESS_MAX_START_RETRIES=3` - Maximum startup attempts before giving up (0-100)
- `LOG_LEVEL=info` - Logging level (debug, info, warn, error, critical)

**Advanced Supervisor Settings:**
- `SUPERVISOR_PROGRAM__LLM_ENGINE_STARTSECS=30` - Time process must run to be considered "started"
- `SUPERVISOR_PROGRAM__LLM_ENGINE_STOPWAITSECS=60` - Time to wait for graceful shutdown
- `SUPERVISOR_PROGRAM__LLM_ENGINE_AUTORESTART=true` - Enable automatic restart on failure
- `SUPERVISOR_PROGRAM__LLM_ENGINE_STARTRETRIES=3` - Startup retry attempts
- `SUPERVISOR_CONFIG_PATH=/tmp/supervisord.conf` - Custom config file location

**Custom Sections:**
- `SUPERVISOR_SUPERVISORD_LOGLEVEL=debug` - Supervisord daemon log level
- `SUPERVISOR_EVENTLISTENER__MEMMON_COMMAND=memmon -a 200MB` - Add custom event listeners

## Testing & Validation

**Comprehensive Test Suite:**
- **Integration Tests** - Actual supervisor processes that verify continuous restart and retry limit behavior
**Test Coverage:**
- **Continuous restart behavior** - Verifies supervisor actually restarts failed processes
- **Startup retry limits** - Confirms supervisor respects retry limits and gives up appropriately
- **Signal handling** - Tests graceful shutdown with SIGTERM
- **ML framework integration** - Tests with realistic ML framework startup patterns
- **Configuration generation** - Validates all supervisor configuration options
- **Error handling** - Tests invalid configurations and edge cases

**Manual Testing:**
- Tested with vLLM dockerfile build
- Verified with `docker exec` process killing to confirm restart behavior
- Validated in production-like container environments
