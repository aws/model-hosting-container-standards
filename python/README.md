# Model Hosting Container Standards - Python

Python implementation of the Model Hosting Container Standards toolkit.

## Overview

This Python package provides a standardized extension that enables TensorRT-LLM and vLLM integration with Amazon SageMaker hosting platform for efficient model deployment and inference.

## Requirements

- Python >= 3.10
- FastAPI >= 0.117.1

## Installation

Install dependencies using Poetry:

```bash
poetry install
```

## Usage

```bash
# Activate virtual environment
poetry shell
```

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

## Debug Logging

To troubleshoot handler resolution or see detailed internal operations, enable debug logging:

### Environment Variable (Recommended)
```bash
# Package-specific
export SAGEMAKER_CONTAINER_LOG_LEVEL=DEBUG
python your_script.py

# Generic
export LOG_LEVEL=DEBUG
python your_script.py

# One-liner
SAGEMAKER_CONTAINER_LOG_LEVEL=DEBUG python examples/customer_script.py
```

### Programmatic Control
```python
from model_hosting_container_standards.logging_config import enable_debug_logging

# Enable debug logging before imports
enable_debug_logging()
import model_hosting_container_standards.sagemaker as sagemaker_standards
```

**Debug output shows:**
- Handler resolution steps
- Decorator calls and registrations
- Customer script loading process
- Internal flow and decision points

### Quick Reproduction Steps

To test the integration with vLLM:

1. **Add decorators to vLLM server**:
   Add `@sagemaker_standards.register_invocation_handler` and `@sagemaker_standards.register_ping_handler` to `vllm/entrypoints/openai/api_server.py`

2. **Start vLLM server**:
   ```bash
   vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto
   ```

3. **Test ping endpoint**:
   ```bash
   curl -i http://127.0.0.1:8000/ping
   ```

4. **Test invocation endpoint**:
   ```bash
   curl -X POST "http://localhost:8000/invocations" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
       "prompt": "Once upon a time",
       "max_tokens": 100,
       "temperature": 0.7,
       "top_p": 0.9,
       "n": 1,
       "stream": false,
       "stop": null
     }'
   ```

## Development

This project uses Poetry for dependency management and building, with integrated code quality tools.

### Quick Start

```bash
# Install dependencies (including dev tools)
make install

# Install pre-commit hooks (recommended)
make pre-commit-install

# Format, lint, and test
make all
```

### Development Tools

This project includes the following code quality tools:

- **Black**: Code formatter
- **isort**: Import sorter
- **flake8**: Linter for style and errors
- **mypy**: Static type checker
- **pytest**: Testing framework
- **pre-commit**: Git hooks for code quality

### Available Commands

```bash
make help              # Show all available commands
make install           # Install dependencies
make format            # Format code with black and isort
make lint              # Run all linters (flake8, mypy, black --check, isort --check)
make test              # Run tests
make clean             # Clean build artifacts
make pre-commit-install # Install pre-commit hooks
make pre-commit-run    # Run pre-commit on all files
make all               # Run format, lint, and test
make ci                # Run CI checks (lint and test)
```

### Development Workflow

#### 1. Development Setup
```bash
# Install dependencies including dev tools
make install

# Install pre-commit hooks (optional but recommended)
make pre-commit-install
```

#### 2. Code Development
```bash
# Format code automatically
make format

# Check code quality
make lint

# Run tests
make test

# Or run everything at once
make all
```

#### 3. Build and Test Workflow
```bash
# Clean and build
make clean
poetry build

# Install built wheel for testing
pip install dist/*.whl --force-reinstall

# Test import
python -c "from model_hosting_container_standards import ping; import logging; logging.basicConfig(); logging.getLogger().info('Import successful')"

# Run tests
make test
```

#### 4. Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before each commit:

```bash
# Install hooks (one-time setup)
make pre-commit-install

# Manually run hooks on all files
make pre-commit-run
```

### Code Quality Standards

- **Line length**: 88 characters (Black default)
- **Import sorting**: Automatic with isort
- **Type hints**: Required for all functions (enforced by mypy)
- **Testing**: All new code should include tests

## License

TBD
