# Transform Decorator Usage Guide

This guide explains the generic transform decorator infrastructure that powers request/response transformations using JMESPath expressions.

## Table of Contents

- [Overview](#overview)
- [How the Decorator Factory Works](#how-the-decorator-factory-works)
- [The Request Flow](#the-request-flow)
- [Passthrough vs Transform Mode](#passthrough-vs-transform-mode)
- [JMESPath Expression Guide](#jmespath-expression-guide)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

The transform decorator infrastructure provides a reusable pattern for creating decorators that automatically transform HTTP requests and responses. This allows you to:

- Extract data from different parts of HTTP requests (body, headers, path parameters, query parameters)
- Transform data structures using JMESPath expressions
- Provide clean, strongly-typed interfaces to your handler functions
- Decouple your handler logic from the wire format

## How the Decorator Factory Works

Understanding the factory mechanism helps you use it effectively and troubleshoot issues when they arise.

### The Decorator Factory Pattern

The transform decorator uses a **decorator factory pattern** to create handler decorators. Here's what happens step by step:

**1. Creating a Decorator (`create_transform_decorator`)**

When you call `create_transform_decorator(handler_type, transform_resolver)`, you're creating a specialized decorator factory for that specific handler type. This factory knows which transformer class to use based on the `transform_resolver` function you provide.

```python
from model_hosting_container_standards.common.transforms import create_transform_decorator

# This creates a decorator factory for a specific operation
my_decorator_factory = create_transform_decorator(
    handler_type="my_operation",
    transform_resolver=my_transform_resolver_function
)
# At this point, my_decorator_factory is a function that can create decorators
```

**2. Configuring the Decorator (Calling the Factory)**

When you call the factory with `request_shape` and/or `response_shape`, you're configuring how data should be transformed:

```python
# This creates an actual decorator configured with your transformation rules
@my_decorator_factory(request_shape={...}, response_shape={...})
async def my_handler(data, raw_request):
    pass
```

At this point, the factory:
- Calls your `transform_resolver` function to get the appropriate transformer class
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
@my_decorator_factory(request_shape={...})
async def my_handler(data: SimpleNamespace, raw_request: Request):
    # Your code here
    pass

# The decorator has now wrapped my_handler with transformation logic
# and registered it in the system
```

## The Request Flow

When a request comes in, here's what happens:

**1. Request Arrives**

FastAPI receives the HTTP request with headers, body, path parameters, etc.

**2. Serialization**

The raw request is serialized into a dictionary structure:

```python
{
    "body": {...},           # Request body as JSON
    "headers": {...},        # HTTP headers
    "path_params": {...},    # URL path parameters
    "query_params": {...}    # Query string parameters
}
```

**3. JMESPath Transformation**

Each JMESPath expression in your `request_shape` is applied to extract data:

```python
# Your request_shape
{"user_id": "body.userId", "token": "headers.Authorization"}

# Serialized request
{
    "body": {"userId": "123"},
    "headers": {"Authorization": "Bearer abc123"}
}

# Becomes (after transformation)
{"user_id": "123", "token": "Bearer abc123"}
```

**4. SimpleNamespace Creation**

The transformed dictionary is converted to a `SimpleNamespace` object so you can access fields using dot notation (`data.user_id` instead of `data["user_id"]`).

**5. Handler Invocation**

Your handler is called with:
- `data`: The transformed data as a SimpleNamespace
- `raw_request`: The original FastAPI Request object (for accessing anything not in the transformation)

**6. Response Transformation**

If you provided a `response_shape`, your handler's response is transformed before being returned to the client.

## Passthrough vs Transform Mode

The decorator behaves differently based on whether you provide transformation shapes:

### Transform Mode (shapes provided)

```python
@my_decorator(request_shape={"user_id": "body.id"})
async def handler(data: SimpleNamespace, raw_request: Request):
    # data.user_id is already extracted
    print(data.user_id)
```

**Behavior:**
- Handler receives transformed data as first argument
- Handler receives raw request as second argument
- Request and response transformations are applied

### Passthrough Mode (no shapes)

```python
@my_decorator()
async def handler(raw_request: Request):
    # No transformations, parse request yourself
    body = await raw_request.json()
    user_id = body.get("id")
```

**Behavior:**
- Handler receives only the raw request
- No transformations are applied
- Handler is still registered in the system

## JMESPath Expression Guide

JMESPath is a query language for JSON that allows you to extract and transform data.
### Field Names with Special Characters

When field names contain hyphens, wrap them in escaped double quotes:

```python
# Headers with hyphens
{"request_id": "headers.\"X-Request-Id\""}
{"content_type": "headers.\"Content-Type\""}

# Body fields with special characters
{"special": "body.\"my-special-field\""}
```

## Troubleshooting

### Debug Tips

**1. Enable Logging:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

The transform decorator will log:
- When transformations are applied
- The handler type being processed
- Compilation of JMESPath expressions

**2. Inspect Transformed Data:**

```python
@my_decorator(request_shape={...})
async def handler(data: SimpleNamespace, raw_request: Request):
    # Print all transformed fields
    print(f"Transformed data: {vars(data)}")

    # Check individual fields
    print(f"user_id: {data.user_id if hasattr(data, 'user_id') else 'NOT PRESENT'}")
```

**3. Test Transformations Separately:**

Create a transformer instance and test it directly:

```python
from model_hosting_container_standards.common.transforms import BaseApiTransform

# Create test transformer
transformer = MyTransformerClass(
    request_shape={"user_id": "body.id"},
    response_shape={}
)

# Test with mock data
test_data = {
    "body": {"id": "123"},
    "headers": {}
}

result = transformer._transform(test_data, transformer._request_shape)
print(result)  # Should print: {"user_id": "123"}
```

**4. Verify Request Structure:**

Add a passthrough handler to see the raw request structure:

```python
@my_decorator()  # No shapes = passthrough mode
async def debug_handler(raw_request: Request):
    body = await raw_request.json()
    print(f"Body: {body}")
    print(f"Headers: {dict(raw_request.headers)}")
    print(f"Path params: {dict(raw_request.path_params)}")
    print(f"Query params: {dict(raw_request.query_params)}")
    return Response(status_code=200)
```

## Best Practices

### 1. Use Descriptive Field Names

Choose clear names for your transformed fields:

```python
# Good - descriptive
request_shape = {
    "user_id": "body.userId",
    "auth_token": "headers.Authorization"
}

# Bad - unclear
request_shape = {
    "id": "body.userId",
    "token": "headers.Authorization"
}
```

### 2. Document Your Transformations

Add comments explaining complex transformations:

```python
@my_decorator(
    request_shape={
        # Extract user ID from nested user object in request body
        "user_id": "body.user.id",

        # Extract correlation ID from custom header (note escaped quotes for hyphen)
        "correlation_id": "headers.\"X-Correlation-Id\"",

        # Extract optional pagination parameter with default handled in code
        "page": "query_params.page"
    }
)
async def my_handler(data, raw_request):
    pass
```

### 3. Validate Early

Add validation in your handler to catch issues early:

```python
@my_decorator(request_shape={...})
async def handler(data: SimpleNamespace, raw_request: Request):
    # Validate required fields exist
    if not hasattr(data, 'user_id') or not data.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    # Validate field formats
    if not data.email or '@' not in data.email:
        raise HTTPException(status_code=400, detail="Invalid email format")

    # Your logic here
```

### 4. Handle Missing Fields Gracefully

Not all fields may be present in every request:

```python
@my_decorator(request_shape={...})
async def handler(data: SimpleNamespace, raw_request: Request):
    # Use getattr with defaults for optional fields
    page = getattr(data, 'page', 1)
    per_page = getattr(data, 'per_page', 10)

    # Or check existence
    if hasattr(data, 'optional_field'):
        process_optional_field(data.optional_field)
```

### 5. Remember to Escape Special Characters

When field names contain hyphens, dots, or other special characters:

```python
# Always escape hyphens in headers
request_shape = {
    "request_id": "headers.\"X-Request-Id\"",
    "content_type": "headers.\"Content-Type\"",
    "custom_header": "headers.\"X-My-Custom-Header\""
}
```

## See Also

- **LoRA Transform Decorator Guide**: For SageMaker-specific LoRA adapter management, see [sagemaker/lora/FACTORY_USAGE.md](../../sagemaker/lora/FACTORY_USAGE.md)
- **Base Transform Class**: For information on creating custom transformer classes, see `base_api_transform.py`
- **JMESPath Documentation**: [https://jmespath.org/](https://jmespath.org/)
