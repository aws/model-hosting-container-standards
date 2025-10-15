# LoRA Decorator Factory Usage Guide

This guide provides practical examples of using the LoRA decorator factory to implement LoRA adapter management handlers in your SageMaker model hosting container.

## Table of Contents

- [Quick Start](#quick-start)
- [How the Factory Works](#how-the-factory-works)
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

## How the Factory Works

Understanding the factory mechanism helps you use it effectively and troubleshoot issues when they arise.

### The Decorator Factory Pattern

The LoRA module uses a **decorator factory pattern** to create handler decorators. Here's what happens step by step:

**1. Creating a Decorator (`create_transform_decorator`)**

When you call `create_transform_decorator(LoRAHandlerType.REGISTER_ADAPTER)`, you're creating a specialized decorator factory for that specific handler type (register, unregister, or adapter_id). This factory knows which transformer class to use based on the handler type you specify.

```python
# This creates a decorator factory for register operations
register_load = create_transform_decorator(LoRAHandlerType.REGISTER_ADAPTER)
# At this point, register_load is a function that can create decorators
```

**2. Configuring the Decorator (Calling the Factory)**

When you call the factory with `request_shape` and/or `response_shape`, you're configuring how data should be transformed:

```python
# This creates an actual decorator configured with your transformation rules
@register_load(request_shape={...}, response_shape={...})
```

At this point, the factory:
- Looks up the appropriate transformer class (e.g., `RegisterLoRAApiTransform`)
- Creates an instance of that transformer with your shapes
- Compiles your JMESPath expressions for efficient execution
- Returns a decorator that will wrap your handler function

**3. Wrapping Your Handler (Applying the Decorator)**

When the decorator is applied to your handler function, it creates a wrapper function that:
- Intercepts incoming requests before they reach your handler
- Applies request transformations using the compiled JMESPath expressions
- Calls your handler with the transformed data
- Applies response transformations to your handler's return value
- Registers the wrapped function in the handler registry

```python
@register_load(request_shape={...})
async def my_handler(data: SimpleNamespace, raw_request: Request):
    # Your code here
    pass

# The decorator has now wrapped my_handler with transformation logic
# and registered it in the system
```

### The Request Flow

When a request comes in, here's what happens:

1. **Request Arrives**: FastAPI receives the HTTP request with headers, body, path parameters, etc.

2. **Serialization**: The raw request is serialized into a dictionary structure:
   ```python
   {
       "body": {...},           # Request body as JSON
       "headers": {...},        # HTTP headers
       "path_params": {...},    # URL path parameters
       "query_params": {...}    # Query string parameters
   }
   ```

3. **JMESPath Transformation**: Each JMESPath expression in your `request_shape` is applied to extract data:
   ```python
   # Your request_shape
   {"adapter_id": "body.name", "source": "body.src"}

   # Becomes
   {"adapter_id": "my-adapter", "source": "s3://..."}
   ```

4. **SimpleNamespace Creation**: The transformed dictionary is converted to a `SimpleNamespace` object so you can access fields using dot notation (`data.adapter_id` instead of `data["adapter_id"]`).

5. **Handler Invocation**: Your handler is called with:
   - `data`: The transformed data as a SimpleNamespace
   - `raw_request`: The original FastAPI Request object (for accessing anything not in the transformation)

6. **Response Transformation**: If you provided a `response_shape`, your handler's response is transformed before being returned to the client.

### Passthrough vs Transform Mode

The decorator behaves differently based on whether you provide transformation shapes:

**Transform Mode** (shapes provided):
```python
@register_load(request_shape={"adapter_id": "body.name"})
async def handler(data: SimpleNamespace, raw_request: Request):
    # data.adapter_id is already extracted
    pass
```
- Handler receives transformed data as first argument
- Handler receives raw request as second argument
- Request and response transformations are applied

**Passthrough Mode** (no shapes):
```python
@register_load()
async def handler(raw_request: Request):
    # No transformations, parse request yourself
    body = await raw_request.json()
    pass
```
- Handler receives only the raw request
- No transformations are applied
- Handler is still registered in the system

## Using the Convenience Functions

The `sagemaker` module provides convenience functions that wrap `create_transform_decorator` for easier use. These are the recommended way to create LoRA handlers in most cases.

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

**3. `inject_adapter_id(request_shape, response_shape={})`**

Creates a decorator for injecting adapter IDs from headers into the request body. This function has **special behavior**:

```python
from model_hosting_container_standards.sagemaker import inject_adapter_id

@inject_adapter_id(
    request_shape={
        "lora_id": None  # Value is ignored - automatically filled with the SageMaker header
    }
)
async def inject_adapter_id(raw_request: Request):
    # The request body now contains the adapter ID from the header
    return Response(status_code=200)
```

**Special behavior of `inject_adapter_id`:**
- Only accepts a **single key** in `request_shape` (raises `ValueError` if more than one)
- **Ignores the value** you provide - it automatically replaces it with the correct JMESPath expression for the SageMaker adapter identifier header: `headers."X-Amzn-SageMaker-Adapter-Identifier"`
- Logs a warning if you provide a `response_shape` (since this handler type doesn't use response transformations)

This makes it foolproof - you don't need to remember the exact header name or how to escape the hyphens:

```python
# These are all equivalent and produce the same result:
@inject_adapter_id(request_shape={"lora_id": None})
@inject_adapter_id(request_shape={"lora_id": ""})
@inject_adapter_id(request_shape={"lora_id": "any_value"})

# The function automatically converts all of these to:
# request_shape={"lora_id": "headers.\"X-Amzn-SageMaker-Adapter-Identifier\""}
```

### Benefits of Convenience Functions

1. **Shorter imports**: Import from `sagemaker` instead of `sagemaker.lora.factory`
2. **Clearer intent**: Function names explicitly state what they do
3. **Less boilerplate**: No need to import and reference `LoRAHandlerType`
4. **Built-in validation**: `inject_adapter_id` validates and auto-fills the header mapping
5. **Future-proof**: If the implementation changes, your code doesn't need updates

### When to Use Direct Factory Access

You should only use `create_transform_decorator` directly when:

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

@inject_adapter_id(
    request_shape={
        "lora_id": None  # Value is automatically filled with the SageMaker header
    }
)
async def inject_adapter_to_body(raw_request: Request):
    """Inject adapter ID from header into request body for inference.

    This transformer modifies the request body in-place, adding the adapter ID
    extracted from the X-Amzn-SageMaker-Adapter-Identifier header.
    """
    # The transformation has already modified raw_request._body
    # Just pass it through to the next handler
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

### Common Issues

#### 1. JMESPath Expression Not Working

**Problem:** Your JMESPath expression doesn't extract the expected data.

**Cause:** Incorrect path or data structure mismatch.

**Solution:** Test your JMESPath expressions:

```python
import jmespath

# Test your expression
request_data = {
    "body": {"name": "test-adapter"},
    "headers": {"X-Custom-Header": "value"}
}

result = jmespath.search("body.name", request_data)
print(result)  # Should print: "test-adapter"
```

#### 2. JMESPath with Hyphens in Field Names

**Problem:** Your JMESPath expression fails when extracting fields with hyphens (e.g., HTTP headers like `X-Amzn-SageMaker-Adapter-Identifier`).

**Cause:** JMESPath interprets hyphens as subtraction operators, not as part of the field name.

**Solution:** Wrap field names containing hyphens in escaped double quotes:

```python
# Wrong - will fail
request_shape = {
    "adapter_id": "headers.X-Amzn-SageMaker-Adapter-Identifier"  # Syntax error!
}

# Correct - escape the hyphenated field name
request_shape = {
    "adapter_id": "headers.\"X-Amzn-SageMaker-Adapter-Identifier\""  # Works!
}
```

**Examples of correctly escaping hyphens:**

```python
# For headers with hyphens
request_shape = {
    "adapter_id": "headers.\"X-Amzn-SageMaker-Adapter-Identifier\"",
    "adapter_alias": "headers.\"X-Amzn-SageMaker-Adapter-Alias\"",
    "request_id": "headers.\"X-Request-Id\"",
    "content_type": "headers.\"Content-Type\""
}

# For body fields with hyphens (less common but possible)
request_shape = {
    "special_field": "body.\"my-special-field\""
}

# Testing with jmespath
import jmespath

test_data = {
    "headers": {
        "X-Amzn-SageMaker-Adapter-Identifier": "my-adapter"
    }
}

# This works
result = jmespath.search("headers.\"X-Amzn-SageMaker-Adapter-Identifier\"", test_data)
print(result)  # Prints: "my-adapter"
```

### Debug Tips

1. **Enable Logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Inspect Transformed Data:**
   ```python
   @register_load(request_shape={...})
   async def handler(data: SimpleNamespace, raw_request: Request):
       print(f"Transformed data: {vars(data)}")
       # Your logic
   ```

3. **Test Transformations Separately:**
   ```python
   from model_hosting_container_standards.sagemaker.lora.transforms import RegisterLoRAApiTransform

   transformer = RegisterLoRAApiTransform(
       request_shape={"adapter_id": "body.name"},
       response_shape={}
   )

   # Test with mock data
   result = transformer._transform(test_data, transformer._request_shape)
   print(result)
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

1. **Use the Convenience Functions:** Unless you are creating a new handler, always use `register_load_adapter_handler`, `register_unload_adapter_handler`, and `inject_adapter_id` from the `sagemaker` module instead of directly using `create_transform_decorator`. They provide better error messages, validation, and automatic header handling.

2. **Use Descriptive Field Names:** Choose clear names for your transformed fields that match your backend's API.

3. **Validate Early:** Add validation in your handler to catch issues early.

4. **Handle Errors Gracefully:** Always wrap backend calls in try-except blocks.

5. **Document Your Transformations:** Add comments explaining your transformation mappings, especially for complex JMESPath expressions.

6. **Remember to Escape Hyphens:** When extracting from headers with hyphens, wrap field names in escaped double quotes (e.g., `headers.\"X-Request-Id\"`).

For more information, see the main [README.md](./README.md).
