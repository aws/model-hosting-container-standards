# Customize Handlers

This guide explains how to customize the `/ping` and `/invocations` endpoints for your vLLM models on Amazon SageMaker.

## Overview

The vLLM container comes with default handlers for health checks and inference. You can override these defaults using custom Python code in your model artifacts.

**Handler Resolution Priority** (first match wins):
1. **Environment Variables** - `CUSTOM_FASTAPI_PING_HANDLER`, `CUSTOM_FASTAPI_INVOCATION_HANDLER`
2. **Decorator Registration** - `@custom_ping_handler`, `@custom_invocation_handler`
3. **Function Discovery** - Functions named `custom_sagemaker_ping_handler`, `custom_sagemaker_invocation_handler`
4. **Framework Defaults** - vLLM's built-in handlers

## Quick Start

### Step 1: Create Custom Handler Script

Create a Python file (e.g., `model.py`) with your custom handlers:

```python
# model.py
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request, Response
import json

@sagemaker_standards.custom_ping_handler
async def my_health_check(request: Request) -> Response:
    """Custom health check logic."""
    return Response(
        content=json.dumps({"status": "healthy", "custom": True}),
        media_type="application/json",
        status_code=200
    )

@sagemaker_standards.custom_invocation_handler
async def my_inference(request: Request) -> Response:
    """Custom inference logic."""
    body = await request.json()
    # Your custom logic here
    result = {"predictions": ["custom response"]}
    return Response(
        content=json.dumps(result),
        media_type="application/json"
    )
```

### Step 2: Upload to S3

```python
import boto3

s3_client = boto3.client('s3')
s3_client.upload_file('model.py', 'my-bucket', 'my-model/model.py')
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
            'CUSTOM_SCRIPT_FILENAME': 'model.py',  # Default: model.py
        }
    }
)
```

**Key Environment Variables:**
- `CUSTOM_SCRIPT_FILENAME`: Name of your custom script (default: `model.py`)
- `SAGEMAKER_MODEL_PATH`: Model directory (default: `/opt/ml/model`)
- `SAGEMAKER_CONTAINER_LOG_LEVEL`: Logging level (ERROR, INFO, DEBUG)

## Customization Methods

### Method 1: Environment Variables (Highest Priority)

Point directly to specific handler functions using environment variables. This overrides all other methods.

**⚠️ Important:** When using environment variables, the recommended approach is to use the `model:` module alias instead of file paths.

```python
# ✅ RECOMMENDED: Use module alias
environment = {
    'CUSTOM_SCRIPT_FILENAME': 'handlers_env_var.py',
    'CUSTOM_FASTAPI_PING_HANDLER': 'model:health_check',
    'CUSTOM_FASTAPI_INVOCATION_HANDLER': 'model:inference',
}

# ⚠️ ALTERNATIVE: Use absolute path
environment = {
    'CUSTOM_FASTAPI_PING_HANDLER': '/opt/ml/model/handlers_env_var.py:health_check',
    'CUSTOM_FASTAPI_INVOCATION_HANDLER': '/opt/ml/model/handlers_env_var.py:inference',
}
```

**Path Formats:**
- `model:function_name` - **Recommended** - Module alias (`model` = `$SAGEMAKER_MODEL_PATH/$CUSTOM_SCRIPT_FILENAME`)
- `/opt/ml/model/handlers.py:ping` - Absolute path
- `handlers.py:function_name` - Relative to `/opt/ml/model` (requires file to exist in that directory)
- `vllm.entrypoints.openai.api_server:health` - Python module path (for installed packages)

**Why use `model:` alias?**
- The `model` alias automatically resolves to your custom script file
- It's more portable and doesn't depend on absolute paths
- It works consistently across different deployment scenarios

### Method 2: Decorators

Use decorators to mark your custom handler functions. The system automatically discovers and registers them when your script loads.

**⚠️ Important:** Decorators must be defined in the file specified by `CUSTOM_SCRIPT_FILENAME` (default: `model.py`). The system only loads and scans this file for decorated functions.

```python
# model.py (or the file specified in CUSTOM_SCRIPT_FILENAME)
import model_hosting_container_standards.sagemaker as sagemaker_standards

@sagemaker_standards.custom_ping_handler
async def my_ping(request):
    return {"status": "ok"}

@sagemaker_standards.custom_invocation_handler
async def my_invoke(request):
    return {"result": "processed"}
```

**Environment Variable:**
```python
environment = {
    'CUSTOM_SCRIPT_FILENAME': 'model.py',  # File containing decorated functions
}
```

### Method 3: Function Discovery (Lowest Priority)

Name your functions with the expected pattern and they'll be automatically discovered - no decorator needed.

**⚠️ Important:** Functions must be defined in the file specified by `CUSTOM_SCRIPT_FILENAME` (default: `model.py`). The system only loads and scans this file for functions matching the expected names.

```python
# model.py (or the file specified in CUSTOM_SCRIPT_FILENAME)
async def custom_sagemaker_ping_handler(request):
    """Automatically discovered by name."""
    return {"status": "healthy"}

async def custom_sagemaker_invocation_handler(request):
    """Automatically discovered by name."""
    return {"result": "processed"}
```

**Environment Variable:**
```python
environment = {
    'CUSTOM_SCRIPT_FILENAME': 'model.py',  # File containing handler functions
}
```

## Complete Example

```python
# model.py
import model_hosting_container_standards.sagemaker as sagemaker_standards
from model_hosting_container_standards.logging_config import logger
from fastapi import Request, Response
import json

@sagemaker_standards.custom_ping_handler
async def health_check(request: Request) -> Response:
    """Custom health check."""
    logger.info("Custom health check called")
    return Response(
        content=json.dumps({"status": "healthy"}),
        media_type="application/json",
        status_code=200
    )

@sagemaker_standards.custom_invocation_handler
async def inference_with_rag(request: Request) -> Response:
    """Custom inference with optional RAG integration and vLLM engine."""
    body = await request.json()
    prompt = body["prompt"]
    max_tokens = body.get("max_tokens", 100)
    temperature = body.get("temperature", 0.7)

    # Optional RAG integration
    if body.get("use_rag", False):
        context = "Retrieved context..."
        prompt = f"Context: {context}\n\nQuestion: {prompt}"

    logger.info(f"Processing prompt: {prompt[:50]}...")

    # Call vLLM engine directly
    from vllm import SamplingParams
    import uuid

    engine = request.app.state.engine_client

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Generate response using vLLM engine
    request_id = str(uuid.uuid4())
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Collect final output
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    # Extract generated text
    if final_output and final_output.outputs:
        generated_text = final_output.outputs[0].text
        prompt_tokens = len(final_output.prompt_token_ids) if hasattr(final_output, "prompt_token_ids") else 0
        completion_tokens = len(final_output.outputs[0].token_ids)
    else:
        generated_text = ""
        prompt_tokens = 0
        completion_tokens = 0

    response_data = {
        "predictions": [generated_text],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "rag_enabled": body.get("use_rag", False)
    }

    return Response(
        content=json.dumps(response_data),
        media_type="application/json",
        headers={"X-Request-Id": request_id}
    )
```

## Troubleshooting

**Custom handlers not loading:**
- Verify `CUSTOM_SCRIPT_FILENAME` is set correctly (default: `model.py`)
- Check CloudWatch logs for import errors
- Enable debug logging: `SAGEMAKER_CONTAINER_LOG_LEVEL=DEBUG`

**Wrong handler being called:**
- Check handler resolution priority (env vars > decorators > function discovery > framework defaults)
- Use `SAGEMAKER_CONTAINER_LOG_LEVEL=DEBUG` to see which handler is selected

**Import errors:**
- Ensure all dependencies are installed in the container
- Verify the script path is correct

## Additional Resources

- **[Python Package README](../../python/README.md)** - Detailed decorator documentation and middleware options
- **[Handler Override Notebook](../../examples/vllm/notebooks/handler_customization_methods.ipynb)** - Complete working example
- **[Quick Start Guide](01_quickstart.md)** - Basic deployment
- **[Customize Pre/Post Processing](03_customize_pre_post_processing.md)** - Custom middleware for request/response transformation
