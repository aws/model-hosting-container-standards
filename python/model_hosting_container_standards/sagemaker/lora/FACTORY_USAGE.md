# LoRA Decorator Factory Usage Guide

This guide provides practical examples of using the LoRA decorator factory to implement LoRA adapter management handlers in your SageMaker model hosting container.

> **Note:** This guide focuses on LoRA-specific usage. For details on how the underlying transform decorator infrastructure works (the decorator factory pattern, request flow, JMESPath basics, etc.), see the [Transform Decorator Usage Guide](../../common/transforms/BASE_FACTORY_USAGE.md).

## Table of Contents

- [Quick Start](#quick-start)
- [Understanding the LoRA Factory](#understanding-the-lora-factory)
- [Using the Convenience Functions](#using-the-convenience-functions)
- [Basic Examples](#basic-examples)
- [Setting Up Your FastAPI Application](#setting-up-your-fastapi-application)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Quick Start

Here's the minimal setup to create a LoRA register handler:

```python
from fastapi import FastAPI, Request, Response
from types import SimpleNamespace
from model_hosting_container_standards.sagemaker import (
    register_load_adapter_handler,
    bootstrap
)

# Define your handler with transformations
@register_load_adapter_handler(
    request_shape={
        "adapter_id": "body.name",
        "adapter_source": "body.src"
    }
)
async def load_lora_adapter(data: SimpleNamespace, raw_request: Request):
    # Your backend-specific logic to load the adapter
    adapter_id = data.adapter_id
    adapter_source = data.adapter_source

    # Call your backend's load function
    await my_backend.load_adapter(adapter_id, adapter_source)

    return Response(status_code=200)

# Create FastAPI app and configure with SageMaker integrations
app = FastAPI()
bootstrap(app)

# Your app now automatically has: POST /adapters -> load_lora_adapter
```

## Understanding the LoRA Factory

### The `create_lora_transform_decorator` Function

The LoRA module provides `create_lora_transform_decorator(handler_type)` which is a specialized factory for creating LoRA handler decorators. It's built on top of the generic transform decorator infrastructure.

```python
from model_hosting_container_standards.sagemaker.lora import (
    create_lora_transform_decorator,
    LoRAHandlerType
)

# Creates a decorator factory for register operations
register_decorator = create_lora_transform_decorator(LoRAHandlerType.REGISTER_ADAPTER)

# Configure with your transformation shapes
@register_decorator(request_shape={...}, response_shape={...})
async def my_handler(data: SimpleNamespace, raw_request: Request):
    pass
```

**What it does:**
- Takes a `LoRAHandlerType` (REGISTER_ADAPTER, UNREGISTER_ADAPTER, or INJECT_ADAPTER_ID)
- Returns a decorator factory that knows which LoRA transformer class to use
- Handles request/response transformations specific to LoRA operations
- Registers your handler in the system

> **For details on how the decorator factory pattern works**, see [How the Decorator Factory Works](../../common/transforms/BASE_FACTORY_USAGE.md#how-the-decorator-factory-works) in the Transform Decorator Usage Guide.

### LoRA Handler Types

The LoRA module defines three handler types:

1. **`LoRAHandlerType.REGISTER_ADAPTER`** - For loading/registering LoRA adapters
2. **`LoRAHandlerType.UNREGISTER_ADAPTER`** - For unloading/unregistering LoRA adapters
3. **`LoRAHandlerType.INJECT_ADAPTER_ID`** - For injecting adapter IDs from headers into request body

Each handler type uses a different transformer class under the hood, but you typically don't need to worry about this when using the convenience functions.

## Using the Convenience Functions

The `sagemaker` module provides convenience functions that wrap `create_lora_transform_decorator` for easier use. These are the recommended way to create LoRA handlers in most cases.

```python
from model_hosting_container_standards.sagemaker import (
    register_load_adapter_handler,
    register_unload_adapter_handler,
    inject_adapter_id
)
```

### Available Convenience Functions

**1. `register_load_adapter_handler(request_shape, response_shape={})`**

Creates a decorator for registering/loading LoRA adapters:

```python
from model_hosting_container_standards.sagemaker import register_load_adapter_handler
from fastapi import Request, Response
from types import SimpleNamespace

@register_load_adapter_handler(
    request_shape={
        "adapter_id": "body.name",
        "adapter_source": "body.src"
    }
)
async def load_adapter(data: SimpleNamespace, raw_request: Request):
    # Your implementation
    return Response(status_code=200)
```

**2. `register_unload_adapter_handler(request_shape, response_shape={})`**

Creates a decorator for unregistering/unloading LoRA adapters:

```python
from model_hosting_container_standards.sagemaker import register_unload_adapter_handler

@register_unload_adapter_handler(
    request_shape={
        "adapter_id": "path_params.adapter_name"
    }
)
async def unload_adapter(data: SimpleNamespace, raw_request: Request):
    # Your implementation
    return Response(status_code=200)
```

**3. `inject_adapter_id(adapter_path, append=False, separator=None)`**

Creates a decorator for injecting adapter IDs from headers into the request body. Supports both replace and append modes:

```python
from model_hosting_container_standards.sagemaker import inject_adapter_id

# Replace mode (default)
@inject_adapter_id("lora_id")
async def inject_adapter_replace(raw_request: Request):
    # The request body now contains the adapter ID from the header
    return Response(status_code=200)

# Append mode
@inject_adapter_id("model", append=True, separator=":")
async def inject_adapter_append(raw_request: Request):
    # Appends adapter ID to existing model field
    return Response(status_code=200)
```

**How `inject_adapter_id` works:**
- Takes an `adapter_path` string parameter specifying where to inject the adapter ID in the request body
- Supports both simple keys (e.g., `"model"`) and nested paths using dot notation (e.g., `"body.model.lora_name"`)
- Automatically extracts the adapter ID from the SageMaker header `X-Amzn-SageMaker-Adapter-Identifier`
- **Replace mode (default)**: Replaces the existing value at the target path
- **Append mode**: Appends the adapter ID to existing value using a separator
- Raises `ValueError` if `adapter_path` is empty, not a string, or if `append=True` without `separator`

**Injection Modes:**

```python
# Replace mode (default)
@inject_adapter_id("model")

# Append mode with colon separator
@inject_adapter_id("model", append=True, separator=":")

# Custom separators
@inject_adapter_id("model", append=True, separator="-")  # Dash
@inject_adapter_id("model", append=True, separator="")   # Direct concatenation
```

### Benefits of Convenience Functions

1. **Shorter imports**: Import from `sagemaker` instead of `sagemaker.lora.factory`
2. **Clearer intent**: Function names explicitly state what they do
3. **Less boilerplate**: No need to import and reference `LoRAHandlerType`
4. **Built-in validation**: `inject_adapter_id` validates parameters and auto-fills the header mapping
5. **Future-proof**: If the implementation changes, your code doesn't need updates

### When to Use Direct Factory Access

You should only use `create_lora_transform_decorator` directly when:

1. **Creating custom handler types**: You've implemented a new transformer class and handler type
2. **Advanced use cases**: You need fine-grained control over the factory behavior
3. **Library development**: You're building on top of this framework

For normal application development, always prefer the convenience functions.

## Basic Examples

### Register Adapter Handler

This example shows how to implement a LoRA adapter registration handler that transforms SageMaker's request format to your backend's format.

```python
from fastapi import Request, Response, HTTPException
from http import HTTPStatus
from types import SimpleNamespace
from model_hosting_container_standards.sagemaker import register_load_adapter_handler

@register_load_adapter_handler(
    request_shape={
        "adapter_id": "body.name",        # SageMaker's "name" -> backend's "adapter_id"
        "adapter_source": "body.src",     # SageMaker's "src" -> backend's "adapter_source"
        "preload": "body.preload"         # Pass through preload setting
    }
)
async def load_lora_adapter(data: SimpleNamespace, raw_request: Request):
    """Load a LoRA adapter into the model.

    Receives:
        - data.adapter_id: Name of the adapter
        - data.adapter_source: S3 path or local path to adapter weights
        - data.preload: Whether to preload the adapter into memory
    """
    try:
        # Validate adapter source format
        if not data.adapter_source.startswith(('s3://', '/')):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Adapter source must be S3 path or local file path"
            )

        # Call your backend's adapter loading function
        # This is where you integrate with your specific inference engine
        result = await my_inference_engine.register_adapter(
            adapter_id=data.adapter_id,
            source=data.adapter_source,
            preload=data.preload
        )

        if result.success:
            return Response(
                status_code=HTTPStatus.OK,
                content=f"Adapter {data.adapter_id} loaded successfully"
            )
        else:
            return Response(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content=f"Failed to load adapter: {result.error}"
            )

    except Exception as e:
        return Response(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content=f"Error loading adapter: {str(e)}"
        )
```

**SageMaker Request:**
```json
POST /lora/register
{
  "name": "customer-support-adapter",
  "src": "s3://my-bucket/adapters/customer-support.safetensors",
  "preload": true
}
```

**Transformed Data Passed to Handler:**
```python
data.adapter_id = "customer-support-adapter"
data.adapter_source = "s3://my-bucket/adapters/customer-support.safetensors"
data.preload = True
```

### Unregister Adapter Handler

This example shows how to handle adapter unregistration, which typically extracts the adapter name from the URL path.

```python
from fastapi import Request, Response, HTTPException
from http import HTTPStatus
from types import SimpleNamespace
from model_hosting_container_standards.sagemaker import register_unload_adapter_handler

@register_unload_adapter_handler(request_shape={"lora_name":"path_params.adapter_name"})  # No transformations needed - uses default behavior
async def unload_lora_adapter(data: SimpleNamespace, raw_request: Request):
    """Unload a LoRA adapter from the model."""
    # Extract adapter name from path parameters
    adapter_name = raw_request.path_params.get("adapter_name")
    try:
        # Call your backend's adapter unloading function
        result = await my_inference_engine.unregister_adapter(data.lora_name)

        if result.success:
            return Response(
                status_code=HTTPStatus.OK,
                content=f"Adapter {adapter_name} unloaded successfully"
            )
        else:
            return Response(
                status_code=HTTPStatus.NOT_FOUND,
                content=f"Adapter {adapter_name} not found"
            )

    except Exception as e:
        return Response(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content=f"Error unloading adapter: {str(e)}"
        )
```

**SageMaker Request:**
```
DELETE /adapters/customer-support-adapter
```

**Handler Receives:**
```python
raw_request.path_params = {"adapter_name": "customer-support-adapter"}
```

### Adapter Header to Body Handler

This example shows how to extract adapter information from HTTP headers and inject it into the request body for inference requests.

```python
from fastapi import Request, Response
from model_hosting_container_standards.sagemaker import inject_adapter_id

# Replace mode example
@inject_adapter_id("lora_id")
async def inject_adapter_replace(raw_request: Request):
    """Inject adapter ID from header into request body (replace mode).

    This transformer modifies the request body in-place, replacing the lora_id
    field with the adapter ID from the X-Amzn-SageMaker-Adapter-Identifier header.
    """
    # The transformation has already modified raw_request._body
    # Just pass it through to the next handler
    return Response(status_code=200)

# Append mode example
@inject_adapter_id("model", append=True, separator=":")
async def inject_adapter_append(raw_request: Request):
    """Inject adapter ID using append mode."""
    return Response(status_code=200)
```

**SageMaker Request:**
```
POST /invocations
Headers:
  X-Amzn-SageMaker-Adapter-Identifier: customer-support-adapter
  Content-Type: application/json

Body:
{
  "inputs": "What is the return policy?",
  "parameters": {
    "max_new_tokens": 100
  }
}
```

**Transformed Request Body:**
```json
{
  "inputs": "What is the return policy?",
  "parameters": {
    "max_new_tokens": 100
  },
  "lora_id": "customer-support-adapter"
}
```

## Troubleshooting

### JMESPath Expression Issues

For general JMESPath troubleshooting (expressions not working, handling hyphens in headers, testing expressions), see the [Troubleshooting section](../../common/transforms/BASE_FACTORY_USAGE.md#troubleshooting) in the Transform Decorator Usage Guide.

### LoRA-Specific Issues

#### SageMaker Header Not Found

**Problem:** The `X-Amzn-SageMaker-Adapter-Identifier` header is not being extracted.

**Solutions:**
1. Verify the header is present in the request
2. Use `inject_adapter_id()` convenience function instead of manually specifying the header path
3. Check that the header name is correctly escaped if specifying manually:
   ```python
   "headers.\"X-Amzn-SageMaker-Adapter-Identifier\""
   ```

#### Adapter Source Validation

**Problem:** Invalid adapter source paths cause errors.

**Solution:** Add validation in your handler:

```python
@register_load_adapter_handler(request_shape={...})
async def load_adapter(data: SimpleNamespace, raw_request: Request):
    # Validate S3 paths
    if data.adapter_source.startswith('s3://'):
        if not is_valid_s3_path(data.adapter_source):
            raise HTTPException(400, "Invalid S3 path format")

    # Validate local paths
    elif data.adapter_source.startswith('/'):
        if not os.path.exists(data.adapter_source):
            raise HTTPException(400, "Adapter file not found")

    else:
        raise HTTPException(400, "Adapter source must be S3 or local path")
```

### Debug Tips

For general debugging tips (enabling logging, inspecting transformed data, testing transformations separately), see the [Debug Tips section](../../common/transforms/BASE_FACTORY_USAGE.md#debug-tips) in the Transform Decorator Usage Guide.

**LoRA-Specific Debugging:**

```python
@register_load_adapter_handler(request_shape={...})
async def load_adapter(data: SimpleNamespace, raw_request: Request):
    # Log the LoRA adapter details
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Loading LoRA adapter: {data.model} from {data.source}")

    # Inspect what was extracted
    print(f"Transformed data: {vars(data)}")

    # Your logic here
```

## Setting Up Your FastAPI Application

After defining your handlers, you need to configure your FastAPI application to use them. The SageMaker module provides a simple one-line setup function.

### Using `bootstrap()`

The `bootstrap()` function automatically configures your FastAPI application with all registered SageMaker handlers:

```python
from fastapi import FastAPI, Request, Response
from types import SimpleNamespace
from model_hosting_container_standards.sagemaker import (
    register_load_adapter_handler,
    register_unload_adapter_handler,
    bootstrap
)

# Step 1: Define your handlers
@register_load_adapter_handler(
    request_shape={
        "adapter_id": "body.name",
        "adapter_source": "body.src"
    }
)
async def load_adapter(data: SimpleNamespace, request: Request):
    await my_backend.load_adapter(data.adapter_id, data.adapter_source)
    return Response(status_code=200, content=f"Loaded {data.adapter_id}")

@register_unload_adapter_handler(
    request_shape={"adapter_id": "path_params.adapter_name"}
)
async def unload_adapter(data: SimpleNamespace, request: Request):
    await my_backend.unload_adapter(data.adapter_id)
    return Response(status_code=200, content=f"Unloaded {data.adapter_id}")

# Step 2: Create your FastAPI app
app = FastAPI()

# Step 3: Configure SageMaker integrations (must be called after handlers are registered)
bootstrap(app)

# Your app now automatically has these routes:
# POST /adapters -> load_adapter
# DELETE /adapters/{adapter_name} -> unload_adapter
```

### Important Notes

1. **Call `bootstrap()` after registering handlers**: The function mounts all handlers that are registered at the time it's called. Handlers registered after calling `bootstrap()` will not be automatically mounted.

2. **Default routes**: Handlers are mounted at standard SageMaker paths:
   - Register adapter: `POST /adapters`
   - Unregister adapter: `DELETE /adapters/{adapter_name}`

3. **One-time setup**: Call `bootstrap()` only once per application.

## Best Practices

### LoRA-Specific Best Practices

1. **Use the Convenience Functions:** Always use `register_load_adapter_handler`, `register_unload_adapter_handler`, and `inject_adapter_id` from the `sagemaker` module instead of directly using `create_lora_transform_decorator`. They provide better error messages, validation, and automatic header handling.

2. **Choose the Right Injection Mode:** Use `inject_adapter_id` replace mode (default) for most cases, but use append mode with appropriate separators for frameworks that expect concatenated model names.

3. **Validate Adapter Sources:** Always validate that adapter sources are accessible and in the correct format (S3 paths, local paths, etc.).

3. **Handle Adapter Loading Errors:** Wrap adapter loading in try-except blocks and return appropriate HTTP status codes:
   - 400 for invalid requests
   - 404 for adapter not found
   - 500 for backend errors

4. **Track Loaded Adapters:** Maintain a registry of loaded adapters for debugging and management:
   ```python
   loaded_adapters = {}

   @register_load_adapter_handler(request_shape={...})
   async def load_adapter(data: SimpleNamespace, request: Request):
       result = await my_backend.load_adapter(data.model, data.source)
       loaded_adapters[data.model] = {
           "source": data.source,
           "loaded_at": datetime.now()
       }
       return Response(status_code=200)
   ```

5. **Log Adapter Operations:** Log all adapter operations for troubleshooting:
   ```python
   import logging
   logger = logging.getLogger(__name__)

   @register_load_adapter_handler(request_shape={...})
   async def load_adapter(data: SimpleNamespace, request: Request):
       logger.info(f"Loading adapter {data.model} from {data.source}")
       # ... your logic
       logger.info(f"Successfully loaded adapter {data.model}")
   ```

### General Best Practices

For general transform decorator best practices (descriptive field names, documenting transformations, validation, handling missing fields, escaping special characters), see the [Best Practices section](../../common/transforms/BASE_FACTORY_USAGE.md#best-practices) in the Transform Decorator Usage Guide.

## See Also

- **Transform Decorator Usage Guide**: [../../common/transforms/BASE_FACTORY_USAGE.md](../../common/transforms/BASE_FACTORY_USAGE.md) - Understanding the underlying decorator infrastructure
- **LoRA README**: [README.md](./README.md) - Overview of the LoRA module
- **JMESPath Documentation**: [https://jmespath.org/](https://jmespath.org/)

For more information on the LoRA module, see the main [README.md](./README.md).
