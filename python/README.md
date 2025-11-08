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

# Framework integration (e.g., in vLLM server code)
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request, Response
import json

@sagemaker_standards.register_ping_handler
async def ping(raw_request: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    return Response(
        content='{"status": "healthy", "source": "vllm_default"}',
        media_type="application/json",
    )

@sagemaker_standards.register_invocation_handler
@sagemaker_standards.inject_adapter_id("model")
async def invocations(raw_request: Request) -> Response:
    """Model invocations endpoint with LoRA adapter injection"""
    body_bytes = await raw_request.body()
    body = json.loads(body_bytes.decode()) if body_bytes else {}

    # Adapter ID injected by decorator from SageMakerLoRAApiHeader
    adapter_id = body.get("model", "base-model")

    # Your model inference logic here
    response_data = {
        "predictions": ["Generated text response"],
        "adapter_id": adapter_id,
    }

    return Response(
        content=json.dumps(response_data),
        media_type="application/json",
    )

# Customer customization (in model.py)
@sagemaker_standards.custom_ping_handler
async def custom_ping(raw_request: Request):
    return Response(status_code=200, content="Custom OK")

# Or simple functions (automatically discovered)
async def ping():
    return {"status": "healthy"}
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

### 1. Framework Integration

For framework developers (e.g., vLLM, TensorRT-LLM), use register decorators to automatically set up routes:

```python
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request, Response

# Register decorators - automatically create /ping and /invocations routes
@sagemaker_standards.register_ping_handler
async def ping(request: Request) -> Response:
    """Framework ping handler with automatic routing."""
    return Response(status_code=200, content="OK")

@sagemaker_standards.register_invocation_handler
async def invocations(request: Request) -> dict:
    """Framework invocation handler with automatic routing."""
    body = await request.json()
    # Process your model inference here
    return {"result": "processed"}

# Optional: Add LoRA adapter support
@sagemaker_standards.register_invocation_handler
@sagemaker_standards.inject_adapter_id("model")
async def invocations_with_lora(request: Request) -> dict:
    """Invocation handler with LoRA adapter ID injection."""
    body = await request.json()
    adapter_id = body.get("model", "base-model")  # Injected from header
    # Use adapter_id for model inference
    return {"result": f"processed with {adapter_id}"}
```

### 2. Customer Script Customization

For customers customizing model behavior, put this in your model artifact folder as model.py:

```python
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request
from fastapi.responses import Response

# Override decorators - immediately register handlers
@sagemaker_standards.custom_ping_handler
async def custom_ping(request: Request) -> Response:
    """Custom ping handler."""
    return Response(status_code=200, content="OK")

@sagemaker_standards.custom_invocation_handler
async def custom_invoke(request: Request) -> dict:
    """Custom invocation handler."""
    body = await request.json()
    # Process your model inference here
    return {"result": "processed"}

# Or use simple functions (automatically discovered)
async def custom_sagemaker_ping_handler():
    """Simple ping function - automatically discovered."""
    return {"status": "healthy"}

async def custom_sagemaker_invocation_handler(request: Request):
    """Simple invoke function - automatically discovered."""
    body = await request.json()
    return {"result": "processed"}
```

### 3. Environment Variable Configuration

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

### 4. Handler Resolution Priority

The system automatically resolves handlers in this order:
1. **Environment Variables** (highest priority)
2. **Registry Decorators** (`@custom_ping_handler`, `@custom_invocation_handler` - customer overrides)
3. **Function Discovery** (functions in custom script named `custom_sagemaker_ping_handler`, `custom_sagemaker_invocation_handler`)
4. **Framework Register Decorators** (`@register_ping_handler`, `@register_invocation_handler`)

**Key Differences:**
- **`@register_ping_handler`**: Used by framework developers, automatically creates routes
- **`@custom_ping_handler`**: Used by customers to override framework behavior
- **Function discovery**: Simple functions automatically detected in customer scripts

**Note**: All handler detection and route setup happens automatically during bootstrap



## Decorator Reference

### Framework Decorators (for framework developers)

Framework decorators automatically create routes and register handlers as framework defaults:

```python
# Basic registration
@sagemaker_standards.register_ping_handler
@sagemaker_standards.register_invocation_handler

# LoRA adapter support
@sagemaker_standards.inject_adapter_id("model")
```

#### Advanced: FastAPI Route Configuration

All register decorators support FastAPI route parameters for request validation, response models, and OpenAPI documentation:

```python
from fastapi import Depends, HTTPException
from pydantic import BaseModel
import json

# Define response models
class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int

class SuccessResponse(BaseModel):
    predictions: list
    model_id: str

# Define validation dependencies
async def validate_json_request(request: Request):
    """Validate JSON format before handler execution."""
    try:
        body = await request.body()
        if body:
            json.loads(body.decode())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

# Register with full configuration
@sagemaker_standards.register_invocation_handler(
    dependencies=[Depends(validate_json_request)],  # Request validation
    responses={  # OpenAPI documentation
        200: {"model": SuccessResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Model Invocation Endpoint",
    description="Invoke the model with input data and get predictions",
    tags=["inference"]
)
@sagemaker_standards.inject_adapter_id("model")  # LoRA support
async def invocations(request: Request):
    """Handler with validation, documentation, and LoRA support."""
    body = await request.json()  # Already validated by dependency
    adapter_id = body.get("model", "base-model")
    # Process with model
    return SuccessResponse(
        predictions=["result1", "result2"],
        model_id=adapter_id
    )
```

**Supported Parameters:**
- `dependencies`: List of FastAPI dependencies for request validation
- `responses`: Dictionary of response models for OpenAPI documentation
- `response_model`: Pydantic model for response validation
- `tags`: List of tags for OpenAPI grouping
- `summary`, `description`: OpenAPI documentation strings
- `status_code`: Default HTTP status code
- All other FastAPI route parameters

**Benefits:**
- ✅ Request validation executes before handler, blocking invalid requests
- ✅ Response models automatically appear in OpenAPI schema
- ✅ Type safety with Pydantic runtime validation
- ✅ Works seamlessly with transform decorators like `@inject_adapter_id`

### Customer Decorators (for model customization)

```python
# Override framework defaults (higher priority)
@sagemaker_standards.custom_ping_handler
@sagemaker_standards.custom_invocation_handler

# LoRA transform decorators
@sagemaker_standards.register_load_adapter_handler(request_shape={...}, response_shape={...})
@sagemaker_standards.register_unload_adapter_handler(request_shape={...}, response_shape={...})
```

## Framework Examples

### vLLM Framework Integration

For vLLM framework developers, use register decorators to set up default handlers:

```python
# In vLLM server code (e.g., vllm/entrypoints/openai/api_server.py)
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import json

# Create router like real vLLM does
router = APIRouter()

@sagemaker_standards.register_ping_handler
async def ping(raw_request: Request) -> Response:
    """Default vLLM ping handler with automatic routing."""
    return Response(
        content='{"status": "healthy", "source": "vllm_default"}',
        media_type="application/json",
    )

@sagemaker_standards.register_invocation_handler
@sagemaker_standards.inject_adapter_id("model")
async def invocations(raw_request: Request) -> Response:
    """Default vLLM invocation handler with LoRA support."""
    body_bytes = await raw_request.body()
    body = json.loads(body_bytes.decode()) if body_bytes else {}

    # Adapter ID injected by decorator from SageMakerLoRAApiHeader
    adapter_id = body.get("model", "base-model")

    # Process with vLLM engine (your actual vLLM logic here)
    # result = await vllm_engine.generate(body["prompt"], adapter_id=adapter_id)

    response_data = {
        "predictions": ["Generated text from vLLM"],
        "adapter_id": adapter_id,
    }

    return Response(
        content=json.dumps(response_data),
        media_type="application/json",
    )

# Setup FastAPI app like real vLLM
app = FastAPI(title="vLLM Server", version="1.0.0")
app.include_router(router)

# Bootstrap SageMaker routes at the end (IMPORTANT!)
from model_hosting_container_standards.sagemaker.sagemaker_router import setup_ping_invoke_routes
setup_ping_invoke_routes(app)
```

### Customer vLLM Customization

Customers can override vLLM's default behavior using customer scripts (`model.py`):

```python
import model_hosting_container_standards.sagemaker as sagemaker_standards
from model_hosting_container_standards.logging_config import logger
from fastapi.responses import Response
from fastapi import Request, HTTPException
from http import HTTPStatus
import json
import pydantic
from vllm.entrypoints.openai.protocol import CompletionRequest
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

# Customer override decorators - higher priority than framework register decorators
@sagemaker_standards.custom_ping_handler
async def myping(raw_request: Request):
    logger.info("Customer ping handler called")
    return Response(status_code=200, content="Customer ping OK")

@sagemaker_standards.custom_invocation_handler
async def invocations(raw_request: Request):
    """Customer invocation handler for SageMaker."""
    logger.info("Customer invocation handler called")
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}"
        ) from e

    # Custom processing logic
    result = await custom_model_processing(body)
    return result

# Or use simple functions (automatically discovered)
async def custom_sagemaker_ping_handler():
    """Simple ping function - automatically discovered."""
    return {"status": "healthy", "custom": True}

async def custom_sagemaker_invocation_handler(request: Request):
    """Simple invoke function - automatically discovered."""
    body = await request.json()
    # Custom model logic
    return {"result": "custom processing"}

logger.info("Customer handlers loaded - will override framework defaults")
```

**Key Points:**
- ✅ **Framework Integration**: Use `@register_ping_handler` for framework defaults
- ✅ **Customer Overrides**: Use `@custom_ping_handler`/`@custom_invocation_handler` or simple functions to customize
- ✅ **Automatic Priority**: Customer handlers automatically override framework defaults
- ✅ **LoRA Support**: Use `@inject_adapter_id` for adapter ID injection from headers

### Adding Middleware to vLLM Integration

You can also add middleware to your vLLM integration:

```python
import model_hosting_container_standards.sagemaker as sagemaker_standards
from model_hosting_container_standards.common.fastapi.middleware import custom_middleware, input_formatter, output_formatter
from model_hosting_container_standards.logging_config import logger

# Add throttling middleware
@custom_middleware("throttle")
async def rate_limit_middleware(request, call_next):
    # Simple rate limiting example
    client_ip = request.client.host
    logger.info(f"Processing request from {client_ip}")

    response = await call_next(request)
    response.headers["X-Rate-Limited"] = "true"
    return response

# Add request preprocessing
@input_formatter
async def preprocess_request(request):
    # Log incoming requests
    logger.info(f"Preprocessing request: {request.method} {request.url}")
    return request

# Add response postprocessing
@output_formatter
async def postprocess_response(response):
    # Add custom headers
    response.headers["X-Processed-By"] = "model-hosting-standards"
    return response

# Your existing handlers
@sagemaker_standards.custom_ping_handler
async def myping(raw_request: Request):
    logger.info("Custom ping handler called")
    return Response(status_code=201)

@sagemaker_standards.custom_invocation_handler
async def invocations(raw_request: Request):
    # Your invocation logic here
    pass
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

# Add middleware via environment variables (file path)
CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE=middleware.py:throttle_func vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto

# Add middleware via module path
CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE=my_middleware:RateLimitClass vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto

# Combined middleware configuration
CUSTOM_FASTAPI_PING_HANDLER=model.py:myping \
CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE=middleware_module:RateLimiter \
CUSTOM_PRE_PROCESS=processors:log_requests \
CUSTOM_POST_PROCESS=processors:add_headers \
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto
```

**Handler Path Formats:**
- `model.py:function_name` - Relative path
- `/opt/ml/model/handlers.py:ping` - Absolute path
- `vllm.entrypoints.openai.api_server:health` - Module path

## Middleware Configuration

The package provides a flexible middleware system that supports both environment variable and decorator-based configuration.

### Middleware Environment Variables

```bash
# Throttling middleware
export CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE="throttle.py:rate_limit_middleware"

# Combined pre/post processing middleware
export CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS="processing.py:combined_middleware"

# Using module paths (no file extension)
export CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE="my_middleware_module:RateLimitMiddleware"
export CUSTOM_PRE_PROCESS="request_processors:log_and_validate"

# Separate pre/post processing (automatically combined)
export CUSTOM_PRE_PROCESS="preprocessing.py:pre_process_func"
export CUSTOM_POST_PROCESS="postprocessing.py:post_process_func"
```

### Middleware Decorators

```python
from model_hosting_container_standards.common.fastapi.middleware import (
    custom_middleware,
    input_formatter,
    output_formatter,
)

# Register throttle middleware
@custom_middleware("throttle")
async def my_throttle_middleware(request, call_next):
    # Rate limiting logic
    response = await call_next(request)
    return response

# Register combined pre/post middleware (function)
@custom_middleware("pre_post_process")
async def my_pre_post_middleware(request, call_next):
    # Pre-processing
    request = await pre_process(request)

    # Call next middleware/handler
    response = await call_next(request)

    # Post-processing
    response = await post_process(response)
    return response

# Register middleware class
@custom_middleware("throttle")
class ThrottleMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # ASGI middleware implementation
        # Rate limiting logic here
        await self.app(scope, receive, send)

# Register input formatter (pre-processing only)
@input_formatter
async def pre_process(request):
    # Modify request
    return request

# Register output formatter (post-processing only)
@output_formatter
async def post_process(response):
    # Modify response
    return response
```

### Middleware Priority

**Environment Variables > Decorators**

Environment variables always take priority over decorator-registered middleware:

```python
# This decorator will be ignored if CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE is set
@custom_middleware("throttle")
async def decorator_throttle(request, call_next):
    return await call_next(request)

# Environment variable takes priority (can use module or file path)
# CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE=throttle_module:ThrottleClass
# CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE=env_throttle.py:env_throttle_func
```



### Middleware Execution Order

```
Request → Throttle → Engine Middlewares → Pre/Post Process → Handler → Response
```

## Configuration Reference

### Environment Variables

```python
from model_hosting_container_standards.common.fastapi.config import FastAPIEnvVars, FASTAPI_ENV_CONFIG
from model_hosting_container_standards.sagemaker import SageMakerEnvVars, SAGEMAKER_ENV_CONFIG

# FastAPI handler environment variables
FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER
FastAPIEnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER

# FastAPI middleware environment variables
FastAPIEnvVars.CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE
FastAPIEnvVars.CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS
FastAPIEnvVars.CUSTOM_PRE_PROCESS
FastAPIEnvVars.CUSTOM_POST_PROCESS

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
│   ├── lora/           # LoRA adapter support
│   │   ├── models/     # LoRA request/response models
│   │   └── transforms/ # API transformation logic
│   └── sessions/       # Stateful session management
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
