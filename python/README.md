# Model Hosting Container Standards - Python

A standardized Python framework for seamless integration between ML frameworks (TensorRT-LLM, vLLM) and Amazon SageMaker hosting.

## Overview

This package simplifies model deployment by providing:
- **Unified Handler System**: Consistent `/ping` and `/invocations` endpoints across frameworks
- **Flexible Configuration**: Environment variables, decorators, or custom scripts
- **Framework Agnostic**: Works with vLLM, TensorRT-LLM, and other ML frameworks
- **Production Ready**: Comprehensive logging, error handling, and debugging tools

## Quick Start

```bash
# Install
poetry install

# Basic usage - add to your model.py
import model_hosting_container_standards.sagemaker as sagemaker_standards

@sagemaker_standards.ping
async def health_check(request):
    return Response(status_code=200, content="OK")

@sagemaker_standards.invoke
async def process_request(request):
    body = await request.json()
    # Your model logic here
    return {"result": "processed"}
```


## Installation

```bash
# Install with Poetry (development)
poetry install

# Build wheel for distribution
poetry build
```

**Requirements:** Python >= 3.10, FastAPI >= 0.117.1

## Usage Patterns

### 1. Decorator-Based

The package provides decorators for easy SageMaker endpoint integration. Put this to your model artifact folder as model.py so you can customize ping and invoke:

```python
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request
from fastapi.responses import Response

# Override decorators - immediately register handlers
@sagemaker_standards.ping
async def custom_ping(request: Request) -> Response:
    """Custom ping handler."""
    return Response(status_code=200, content="OK")

@sagemaker_standards.invoke
async def custom_invoke(request: Request) -> dict:
    """Custom invocation handler."""
    body = await request.json()
    # Process your model inference here
    return {"result": "processed"}
```

### 2. Environment Variable Configuration

```bash
# Point to custom handlers in your code
export CUSTOM_FASTAPI_PING_HANDLER="model.py:my_ping_function"
export CUSTOM_FASTAPI_INVOCATION_HANDLER="model.py:my_invoke_function"

# Or use absolute paths
export CUSTOM_FASTAPI_PING_HANDLER="/opt/ml/model/handlers.py:ping"

# Or use module
export CUSTOM_FASTAPI_INVOCATION_HANDLER="model:my_invoke_function" #`model` is alias to $SAGEMAKER_MODEL_PATH/$CUSTOM_SCRIPT_FILENAME
CUSTOM_FASTAPI_PING_HANDLER="vllm.entrypoints.openai.api_server:health"
```

### 3. Handler Resolution Priority

The system resolves handlers in this order:
1. **Environment Variables** (highest priority)
2. **Decorator Registration** (`@ping`, `@invoke`)
3. **Function Discovery** (functions in custom script named `ping`, `invoke`)
4. **Default Handlers** (framework fallbacks)

## Framework Examples

### Integration with vLLM

To integrate with vLLM, you need to add the decorators to your vLLM server. Here's a complete working example `model.py`:

```python
import sys
import os
import model_hosting_container_standards.sagemaker as sagemaker_standards
from model_hosting_container_standards.logging_config import logger
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from vllm.lora.request import LoRARequest
import json
from fastapi import APIRouter, Request, HTTPException, Depends
from http import HTTPStatus
import json
import pydantic
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.protocol import CompletionRequest, ErrorResponse
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

# Customer override ping handler
@sagemaker_standards.ping
async def myping(raw_request: Request):
    logger.info("Custom ping handler called")
    return Response(status_code=201)

logger.info("Customer script loaded: Custom ping handler set")

def completion(raw_request: Request) -> OpenAIServingCompletion:
    """Get completion handler from request."""
    # This should return your completion handler instance
    # You'll need to adapt this based on your actual setup
    return raw_request.app.state.openai_serving_completion

async def create_completion(request: CompletionRequest, raw_request: Request):
    """Create completion response."""
    handler = completion(raw_request)
    return await handler.create_completion(request, raw_request)

@sagemaker_standards.invoke
async def invocations(raw_request: Request):
    """Simple completion-only handler for SageMaker."""
    logger.info("Custom invocation handler called")
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}"
        ) from e

    # Validate as CompletionRequest
    validator = pydantic.TypeAdapter(CompletionRequest)
    try:
        request = validator.validate_python(body)
    except pydantic.ValidationError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Invalid CompletionRequest: {e}"
        ) from e

    return await create_completion(request, raw_request)

logger.info("Customer script loaded - handlers registered")
```

#### Example Commands

```bash
# Enable debug logging
SAGEMAKER_CONTAINER_LOG_LEVEL=DEBUG vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto

# Custom ping handler from model.py
CUSTOM_FASTAPI_PING_HANDLER=model.py:myping vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto

# Custom ping handler with absolute path
CUSTOM_FASTAPI_PING_HANDLER=/opt/ml/model/model.py:myping vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto

# Use vLLM's built-in health endpoint as ping handler
CUSTOM_FASTAPI_PING_HANDLER=vllm.entrypoints.openai.api_server:health vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto
```

**Handler Path Formats:**
- `model.py:function_name` - Relative path
- `/opt/ml/model/handlers.py:ping` - Absolute path
- `vllm.entrypoints.openai.api_server:health` - Module path

## Configuration Reference



### Environment Variables

```python
from model_hosting_container_standards.common.fastapi import EnvVars, ENV_CONFIG
from model_hosting_container_standards.sagemaker import SageMakerEnvVars, SAGEMAKER_ENV_CONFIG

# FastAPI environment variables
EnvVars.CUSTOM_FASTAPI_PING_HANDLER
EnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER

# SageMaker environment variables
SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME
SageMakerEnvVars.SAGEMAKER_MODEL_PATH
```

### Debug Logging

```python
from model_hosting_container_standards.logging_config import enable_debug_logging

# Enable detailed handler resolution logging
enable_debug_logging()
```

```bash
# Or via environment variable
export SAGEMAKER_CONTAINER_LOG_LEVEL=DEBUG
```

## Testing

### Quick Endpoint Testing

```bash
# Start your service (example with vLLM)
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto

# Test ping
curl -i http://127.0.0.1:8000/ping

# Test invocation
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 50}'
```

## Development

### Quick Development Setup

```bash
# Install dependencies and dev tools
make install

# Install pre-commit hooks (recommended)
make pre-commit-install

# Run all checks
make all
```

### Development Commands

```bash
make install           # Install dependencies
make format            # Format code (black, isort)
make lint              # Run linters (flake8, mypy)
make test              # Run test suite
make all               # Format, lint, and test
make clean             # Clean build artifacts
```

### Code Quality Tools

- **Black** (88 char line length) + **isort** for formatting
- **flake8** + **mypy** for linting and type checking
- **pytest** for testing with coverage
- **pre-commit** hooks for automated checks

## Architecture

### Package Structure
```
model_hosting_container_standards/
├── common/             # Common utilities
│   ├── fastapi/        # FastAPI integration & env config
│   ├── custom_code_ref_resolver/  # Dynamic code loading
│   └── handler/        # Handler specifications & resolution
│       └── spec/       # Handler interface definitions
├── sagemaker/          # SageMaker decorators & handlers
│   └── lora/           # LoRA adapter support
│       ├── models/     # LoRA request/response models
│       └── transforms/ # API transformation logic
├── config.py           # Configuration management
├── utils.py            # Utility functions
└── logging_config.py   # Centralized logging
```

### Key Components

- **Handler Registry**: Central system for registering and resolving handlers
- **Code Resolver**: Dynamically loads handlers from customer code
- **Environment Config**: Manages configuration via environment variables
- **Logging System**: Comprehensive debug and operational logging

## Contributing

When contributing to this project:

1. Follow the established code quality standards
2. Include comprehensive tests for new functionality
3. Update documentation and type hints
4. Run the full test suite before submitting changes
5. Use the provided development tools and pre-commit hooks

## License

TBD
