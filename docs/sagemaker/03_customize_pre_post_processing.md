# Customize Pre/Post Processing

This guide explains how to add custom pre-processing and post-processing logic to your model endpoints using middleware.

> 📓 **Working Example**: See the [Pre/Post Processing Notebook](../../examples/vllm/notebooks/preprocessing_postprocessing_methods.ipynb) for a complete working example.

## Overview

Pre/post processing middleware allows you to transform requests and responses without modifying your core handler logic. Common use cases include:

- **Request preprocessing**: Input validation, format conversion, authentication, logging
- **Response postprocessing**: Output formatting, header injection, metrics collection, error handling
- **Combined processing**: Request/response transformation pipelines

**⚠️ Important: Middleware Runs on ALL Endpoints**

Pre/post processing middleware runs on **all endpoints** including `/ping`, `/invocations`, and any custom routes. To avoid errors, **always check the request path** before processing:

```python
# ✅ CORRECT: Check endpoint path first
if request.url.path == "/invocations":
    # Only process /invocations requests
    body = await request.json()
    # ... your processing logic

# ❌ WRONG: Processes all endpoints including /ping
body = await request.json()  # Will fail on /ping (no body)!
```

**Common patterns:**
- **Request preprocessing**: Check `request.url.path == "/invocations"` before reading body
- **Response postprocessing**: Check `hasattr(response, 'body')` for streaming responses
- **Skip processing**: Return early for endpoints you don't need to process

**Middleware Resolution Priority**:

Environment variables always take precedence over decorators:
1. **Environment Variables** (highest priority)
   - `CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS` - Combined pre/post middleware
   - `CUSTOM_PRE_PROCESS` + `CUSTOM_POST_PROCESS` - Separate pre/post (automatically combined if `CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS` is not set)
2. **Decorator Registration** (lower priority)
   - `@custom_middleware("pre_post_process")` - Combined middleware
   - `@input_formatter` + `@output_formatter` - Separate formatters (automatically combined)

**Note**: If `CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS` is set, it takes precedence and `CUSTOM_PRE_PROCESS`/`CUSTOM_POST_PROCESS` are ignored. If only `CUSTOM_PRE_PROCESS` and/or `CUSTOM_POST_PROCESS` are set, they are automatically combined into a single middleware.

## Quick Start

### Step 1: Create Middleware Script

Create a Python file (e.g., `middleware.py`) with your processing logic:

```python
# middleware.py
from model_hosting_container_standards.common.fastapi.middleware import (
    input_formatter,
    output_formatter,
    custom_middleware,
)
from model_hosting_container_standards.logging_config import logger
from fastapi import Request, Response
import json

@input_formatter
async def preprocess_request(request: Request) -> Request:
    """Transform incoming requests before they reach the handler."""
    logger.info(f"Preprocessing request: {request.method} {request.url.path}")

    # Only process /invocations endpoint
    if request.url.path == "/invocations":
        body = await request.json()

        # Validate required fields
        if "prompt" not in body:
            raise ValueError("Missing required field: prompt")

        # Transform input format
        if "max_tokens" not in body:
            body["max_tokens"] = 100  # Set default

        # Store modified body back to request
        request._body = json.dumps(body).encode()

    return request

@output_formatter
async def postprocess_response(response: Response) -> Response:
    """Transform responses before they're sent to the client."""
    logger.info(f"Postprocessing response: {response.status_code}")

    # Add custom headers (safe for all responses)
    response.headers["X-Processed-By"] = "custom-middleware"
    response.headers["X-Model-Version"] = "1.0.0"

    # ⚠️ IMPORTANT: Check if response has body (streaming responses don't)
    if not hasattr(response, 'body'):
        logger.debug("Streaming response, skipping body modification")
        return response

    # Transform response body if needed
    if response.status_code == 200:
        try:
            body = json.loads(response.body)
            # Add metadata
            body["metadata"] = {
                "processed": True,
                "version": "1.0.0"
            }
            response.body = json.dumps(body).encode()
        except json.JSONDecodeError:
            logger.debug("Non-JSON response, skipping body modification")

    return response
```

### Step 2: Upload to S3

```python
import boto3

s3_client = boto3.client('s3')
s3_client.upload_file('middleware.py', 'my-bucket', 'my-model/middleware.py')
```

### Step 3: Deploy to SageMaker

```python
sagemaker_client = boto3.client('sagemaker')

sagemaker_client.create_model(
    ModelName='my-vllm-model',
    ExecutionRoleArn='arn:aws:iam::123456789012:role/SageMakerExecutionRole',
    PrimaryContainer={
        'Image': f'{account_id}.dkr.ecr.{region}.amazonaws.com/vllm:latest',
        'ModelDataSource': {
            'S3DataSource': {
                'S3Uri': 's3://my-bucket/my-model/',
                'S3DataType': 'S3Prefix',
                'CompressionType': 'None',
            }
        },
        'Environment': {
            'SM_VLLM_MODEL': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'HUGGING_FACE_HUB_TOKEN': 'hf_your_token_here',
            'CUSTOM_PRE_PROCESS': 'middleware.py:preprocess_request',
            'CUSTOM_POST_PROCESS': 'middleware.py:postprocess_response',
        }
    }
)
```

## Middleware Methods

### Method 1: Separate Pre/Post Processing (Decorators)

Use separate decorators for preprocessing and postprocessing. The system automatically combines them into a single middleware.

**⚠️ Important:** When using decorators, they must be defined in the file specified by `CUSTOM_SCRIPT_FILENAME` (default: `model.py`). The system only loads and scans this file for decorated functions. If you want to use a different file, set `CUSTOM_SCRIPT_FILENAME` accordingly.

```python
# middleware.py
from model_hosting_container_standards.common.fastapi.middleware import (
    input_formatter,
    output_formatter,
)
from fastapi import Request, Response
import json

@input_formatter
async def validate_and_transform_input(request: Request) -> Request:
    """Preprocess incoming requests."""
    # Only process /invocations endpoint
    if request.url.path == "/invocations":
        body = await request.json()

        # Validation
        if not body.get("prompt"):
            raise ValueError("Prompt is required")

        # Transformation
        body["prompt"] = body["prompt"].strip()
        request._body = json.dumps(body).encode()

    return request

@output_formatter
async def add_metadata_to_response(response: Response) -> Response:
    """Postprocess outgoing responses."""
    response.headers["X-API-Version"] = "2.0"
    return response
```

**Environment Variable (Required for decorators to be loaded):**
```python
environment = {
    'CUSTOM_SCRIPT_FILENAME': 'middleware.py',  # File containing decorated functions
}
```

**Alternative: Use environment variables to point to functions directly (no decorator scanning needed):**
```python
environment = {
    'CUSTOM_PRE_PROCESS': 'middleware.py:validate_and_transform_input',
    'CUSTOM_POST_PROCESS': 'middleware.py:add_metadata_to_response',
}
```

### Method 2: Combined Middleware (Decorator)

Use a single middleware function that handles both pre and post processing.

**⚠️ Important:** When using decorators, they must be defined in the file specified by `CUSTOM_SCRIPT_FILENAME` (default: `model.py`). The system only loads and scans this file for decorated functions.

```python
# middleware.py
from model_hosting_container_standards.common.fastapi.middleware import custom_middleware
from fastapi import Request
import json

@custom_middleware("pre_post_process")
async def combined_processing(request: Request, call_next):
    """Combined pre/post processing middleware."""
    # Pre-processing - only for /invocations
    if request.url.path == "/invocations":
        body = await request.json()
        body["preprocessed"] = True
        request._body = json.dumps(body).encode()

    # Call the handler
    response = await call_next(request)

    # Post-processing - safe for all responses
    response.headers["X-Combined-Middleware"] = "true"

    return response
```

**Environment Variable (Required for decorator to be loaded):**
```python
environment = {
    'CUSTOM_SCRIPT_FILENAME': 'middleware.py',  # File containing decorated function
}
```

**Alternative: Use environment variable to point to function directly (no decorator scanning needed):**
```python
environment = {
    'CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS': 'middleware.py:combined_processing',
}
```

### Method 3: Environment Variables Only

Point directly to middleware functions without using decorators.

```python
# processors.py
from fastapi import Request, Response
import json

async def log_requests(request: Request) -> Request:
    """Simple request logger."""
    print(f"Request: {request.method} {request.url}")
    return request

async def add_headers(response: Response) -> Response:
    """Add custom headers."""
    response.headers["X-Custom"] = "value"
    return response
```

**Environment Variables:**
```bash
export CUSTOM_PRE_PROCESS="processors.py:log_requests"
export CUSTOM_POST_PROCESS="processors.py:add_headers"
```

## Common Use Cases

### Input Validation

```python
from model_hosting_container_standards.common.fastapi.middleware import input_formatter
from fastapi import Request, HTTPException
import json

@input_formatter
async def validate_input(request: Request) -> Request:
    """Validate request format and required fields."""
    # Only validate /invocations endpoint
    if request.url.path == "/invocations":
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        # Validate required fields
        if "prompt" not in body:
            raise HTTPException(status_code=400, detail="Missing 'prompt' field")

        # Validate ranges
        if "max_tokens" in body and body["max_tokens"] < 1:
            raise HTTPException(status_code=400, detail="max_tokens must be positive")

    return request
```

### Request Logging and Metrics

```python
from model_hosting_container_standards.common.fastapi.middleware import custom_middleware
from model_hosting_container_standards.logging_config import logger
from fastapi import Request
import time
import json

@custom_middleware("pre_post_process")
async def log_and_measure(request: Request, call_next):
    """Log requests and measure processing time."""
    start_time = time.time()
    request_id = request.headers.get("X-Request-Id", "unknown")

    logger.info(f"Request {request_id}: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Add metrics
    duration = time.time() - start_time
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Processing-Time"] = f"{duration:.3f}"

    logger.info(f"Request {request_id} completed in {duration:.3f}s")

    return response
```

## Middleware Execution Order

When multiple middleware types are configured, they execute in this order:

```
Request Flow:
  Client Request
    ↓
  Throttle Middleware (if configured)
    ↓
  Engine-specific Middleware (framework middleware)
    ↓
  Pre/Post Process Middleware
    ↓
  Handler (/ping or /invocations)
    ↓
  Pre/Post Process Middleware (post-processing)
    ↓
  Engine-specific Middleware (post-processing)
    ↓
  Throttle Middleware (post-processing)
    ↓
  Client Response
```

## Configuration Reference

### Environment Variables

```bash
# Separate pre/post processing (recommended)
export CUSTOM_PRE_PROCESS="middleware.py:preprocess_function"
export CUSTOM_POST_PROCESS="middleware.py:postprocess_function"

# Combined middleware
export CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS="middleware.py:combined_function"

# Using module paths (no .py extension)
export CUSTOM_PRE_PROCESS="my_middleware_module:preprocess"
export CUSTOM_POST_PROCESS="my_middleware_module:postprocess"

# Using absolute paths
export CUSTOM_PRE_PROCESS="/opt/ml/model/middleware.py:preprocess"
export CUSTOM_POST_PROCESS="/opt/ml/model/middleware.py:postprocess"
```

### Path Formats

- `middleware.py:function_name` - Relative to `/opt/ml/model`
- `/opt/ml/model/middleware.py:function` - Absolute path
- `middleware_module:function` - Python module path (no .py)
- `model:function` - Alias to `$SAGEMAKER_MODEL_PATH/$CUSTOM_SCRIPT_FILENAME`

## Troubleshooting

**Middleware not loading:**
- Verify file path is correct relative to `/opt/ml/model`
- Check CloudWatch logs for import errors
- Enable debug logging: `SAGEMAKER_CONTAINER_LOG_LEVEL=DEBUG`

**Request body not accessible:**
- Use `await request.json()` or `await request.body()` to read body
- Store modified body back: `request._body = json.dumps(body).encode()`
- Body can only be read once - store it if needed multiple times

**Response not modifying:**
- Ensure you're returning the modified response object
- For combined middleware, use `await call_next(request)` to get response
- Response body must be bytes: `response.body = json.dumps(data).encode()`

**Middleware execution order issues:**
- Check middleware priority (env vars > decorators)
- Use `SAGEMAKER_CONTAINER_LOG_LEVEL=DEBUG` to see execution order
- Remember: pre-processing runs top-down, post-processing runs bottom-up

## Best Practices

1. **Keep middleware focused**: Each middleware should have a single responsibility
2. **Handle errors gracefully**: Always catch and handle exceptions appropriately
3. **Log important events**: Use the centralized logger for debugging
4. **Validate early**: Validate inputs in preprocessing to fail fast
5. **Preserve request body**: Store modified body back to `request._body`
6. **Use type hints**: Add type hints for better code clarity
7. **Test thoroughly**: Test middleware with various input scenarios
8. **Monitor performance**: Log processing times to identify bottlenecks

## Additional Resources

- **[Customize Handlers](02_customize_handlers.md)** - Handler customization guide
- **[Python Package README](../../python/README.md)** - Detailed middleware documentation
- **[Quick Start Guide](01_quickstart.md)** - Basic deployment
- **[Pre/Post Processing Notebook](../../examples/vllm/notebooks/preprocessing_postprocessing_methods.ipynb)** - Complete working examples
- **[Handler Examples](../../examples/vllm/model_artifacts_examples/)** - Working code examples
