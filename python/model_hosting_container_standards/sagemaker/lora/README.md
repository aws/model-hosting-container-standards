# LoRA Module

The LoRA (Low-Rank Adaptation) module provides a flexible framework for handling LoRA adapter operations in SageMaker model hosting containers. It enables automatic request/response transformations between SageMaker's API format and backend inference engine formats.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Request Response Transformations](#request-response-transformations)
- [LoRA Handler Decorator Factory](#lora-handler-decorator-factory)
- [Public API and Convenience Functions](#public-api-and-convenience-functions)
- [Setting Up Your FastAPI Application](#setting-up-your-fastapi-application)
- [Custom Transformer Implementation](#custom-transformer-implementation)
- [Utilities](#utilities)
- [Error Handling](#error-handling)
- [Constants and Headers](#constants-and-headers)
- [Examples](#examples)
- [Testing](#testing)
- [Dependencies](#dependencies)

## Overview

This module provides:

- **Automatic Request/Response Transformations**: JMESPath-based mapping between different API shapes
- **LoRA-Specific Decorators**: Easy integration of LoRA handlers with minimal boilerplate
- **Built on Generic Infrastructure**: Uses the reusable transform decorator framework from `common/transforms`
- **Extensible Architecture**: Base classes for implementing custom transformations
- **Type-Safe Request Validation**: Pydantic models for request validation

> **Note:** The LoRA module is built on top of the generic transform decorator infrastructure in `common/transforms`. For details on how the underlying decorator factory pattern works, see the [Transform Decorator Usage Guide](../../common/transforms/BASE_FACTORY_USAGE.md).

## Architecture

The module consists of several key components:

```
common/transforms/                # Generic transform infrastructure (shared)
├── decorator.py                  # Generic decorator factory (create_transform_decorator)
├── base_api_transform.py         # Abstract base class for all transformations
└── utils.py                      # JMESPath utilities

lora/                             # LoRA-specific implementation
├── factory.py                    # LoRA decorator factory (wraps generic decorator)
├── base_lora_api_transform.py   # LoRA-specific base transform class
├── models/                       # Pydantic models for requests and responses
│   ├── request.py
│   ├── response.py
│   └── transform.py
├── transforms/                   # Concrete LoRA transformer implementations
│   ├── __init__.py              # resolve_lora_transform function
│   ├── register.py              # Register adapter transformer
│   ├── unregister.py            # Unregister adapter transformer
│   └── inject_to_body.py        # Inject data to body transformer
├── constants.py                  # LoRA-specific constants and enums
└── utils.py                      # LoRA-specific utility functions
```

**Key architectural layers:**
- **Generic Infrastructure** (`common/transforms`): Reusable decorator factory and base classes
- **LoRA Implementation** (`lora`): LoRA-specific transformers and handlers
- **Public API** (`sagemaker/__init__.py`): Convenience functions for users

## Request Response Transformations

### How Transformations Work

The transformation system uses **JMESPath expressions** to map data between different API shapes. This allows you to:

1. Extract data from incoming SageMaker requests (headers, body, path parameters)
2. Transform it into the format expected by your backend inference engine
3. Transform engine responses back into SageMaker's expected format

### Request Shape (`request_shape`)

The `request_shape` dictionary defines how to extract data from the incoming SageMaker request and map it to your backend engine's expected format.

**Structure:**
- **Keys**: The target field names in your engine's format
- **Values**: JMESPath expressions that extract data from the SageMaker request

**Available JMESPath Sources:**
- `body.*` - Extract from request body (e.g., `body.name`, `body.src`)
- `headers.*` - Extract from HTTP headers (e.g., `headers.X-Amzn-SageMaker-Adapter-Identifier`)
- `path_params.*` - Extract from URL path parameters (e.g., `path_params.adapter_name`)
- `query_params.*` - Extract from query string parameters

**Example:**
```python
request_shape = {
    "model": "body.name",           # Maps body.name to engine's "model" field
    "source": "body.src",            # Maps body.src to engine's "source" field
    "adapter_id": "headers.X-Amzn-SageMaker-Adapter-Identifier"  # From header
}
```

For a SageMaker request like:
```json
{
  "name": "my-lora-adapter",
  "src": "s3://bucket/adapter",
  "preload": true
}
```

The transformation produces:
```python
{
  "model": "my-lora-adapter",
  "source": "s3://bucket/adapter"
}
```

### Response Shape (`response_shape`)

The `response_shape` dictionary defines how to transform your engine's response back into SageMaker's expected format.

**Structure:**
- **Keys**: The target field names in SageMaker's format
- **Values**: JMESPath expressions that extract data from the engine response

**Example:**
```python
response_shape = {
    "adapter_name": "id",          # Maps engine's "id" to SageMaker's "adapter_name"
    "status": "state",             # Maps engine's "state" to SageMaker's "status"
    "metadata": {                  # Nested transformations supported
        "rank": "lora_rank",
        "alpha": "lora_alpha"
    }
}
```

### Nested Transformations

Both `request_shape` and `response_shape` support nested dictionary structures for complex mappings:

```python
request_shape = {
    "adapter": {
        "name": "body.name",
        "path": "body.src"
    },
    "config": {
        "preload": "body.preload",
        "pin": "body.pin"
    }
}
```

## LoRA Handler Decorator Factory

The LoRA module provides `create_lora_transform_decorator`, which creates LoRA-specific handler decorators. This function is a specialized wrapper around the generic `create_transform_decorator` from `common/transforms`.

```python
from model_hosting_container_standards.sagemaker.lora.factory import create_lora_transform_decorator
from model_hosting_container_standards.sagemaker.lora.constants import LoRAHandlerType

# Create decorators for each handler type
register_load = create_lora_transform_decorator(LoRAHandlerType.REGISTER_ADAPTER)
register_unload = create_lora_transform_decorator(LoRAHandlerType.UNREGISTER_ADAPTER)
inject_adapter_id = create_lora_transform_decorator(LoRAHandlerType.INJECT_ADAPTER_ID)
```

**How it works:**
```python
# In lora/factory.py
def create_lora_transform_decorator(handler_type: str):
    return create_transform_decorator(handler_type, resolve_lora_transform)
```

The `resolve_lora_transform` function maps LoRA handler types to their specific transformer classes.

> **For details on the underlying decorator factory pattern**, see [How the Decorator Factory Works](../../common/transforms/BASE_FACTORY_USAGE.md#how-the-decorator-factory-works) in the Transform Decorator Usage Guide.

### Handler Types

#### 1. `REGISTER_ADAPTER` (register_adapter -> register_load_adapter_handler)
Used for registering/loading new LoRA adapters.

**Transformer**: `RegisterLoRAApiTransform`
- Validates required fields: `name` and `src`
- Generates success/error messages
- Maps adapter registration errors to appropriate HTTP status codes

#### 2. `UNREGISTER_ADAPTER` (unregister_adapter -> register_unload_adapter_handler)
Used for unregistering/unloading LoRA adapters.

**Transformer**: `UnregisterLoRAApiTransform`
- Extracts adapter name from path parameters
- No request body required
- Generates unregistration confirmation messages

#### 3. `INJECT_ADAPTER_ID` (inject_adapter_id -> inject_adapter_id)
Used for injecting adapter information into the request body from various sources (headers, query params, etc.).

**Transformer**: `InjectToBodyApiTransform`
- Extracts data from sources (headers, query params, etc.) using JMESPath expressions in request_shape
- Injects extracted data into request body at specified paths (supports nested paths using dot notation)
- Useful for inference requests with adapter identifiers in headers
- Only accepts flat request_shape mappings (target paths to source expressions as strings)
- Does not support response transformations (response_shape is ignored with a warning)

**Example Usage:**
```python
from model_hosting_container_standards.sagemaker import inject_adapter_id_handler

@inject_adapter_id_handler(
    request_shape={
        "adapter_id": 'headers."x-adapter-id"',           # Inject to top-level
        "config.adapter_name": 'headers."x-adapter-name"', # Inject to nested path
        "config.priority": 'headers."x-priority"'          # Inject to nested path
    }
)
async def inference_handler(raw_request: Request):
    # Request body is automatically modified before reaching this handler
    body = await raw_request.json()
    # body now contains:
    # {
    #   "adapter_id": "<value-from-header>",
    #   "config": {
    #     "adapter_name": "<value-from-header>",
    #     "priority": "<value-from-header>"
    #   }
    # }
    return Response(status_code=200)
```

**Important Notes:**
- The request_shape keys use dot notation for nested paths (e.g., `"config.adapter_name"`)
- Parent objects must exist in the request body before injecting nested values
- The transformer modifies the raw request body in place
- Non-string values in request_shape will raise a `ValueError`

### Decorator Behavior

When you apply a decorator to a handler function:

1. **With Transformations** (`request_shape` or `response_shape` provided):
   - The decorator wraps your handler with transformation logic
   - Request data is transformed before calling your handler
   - Your handler receives a `SimpleNamespace` object with transformed data and the raw request
   - Response is transformed before returning to the client

2. **Without Transformations** (no shapes provided):
   - The decorator registers your handler in passthrough mode
   - Your handler receives only the raw FastAPI `Request` object
   - Response is returned unchanged

### Handler Function Signatures

**With Transformations:**
```python
async def my_handler(transformed_data: SimpleNamespace, raw_request: Request):
    # Access transformed data as attributes
    adapter_name = transformed_data.model
    adapter_src = transformed_data.source

    # Still have access to raw request
    headers = raw_request.headers

    return Response(status_code=200, content="Success")
```

**Without Transformations (Passthrough):**
```python
async def my_handler(raw_request: Request):
    # Parse request manually
    body = await raw_request.json()

    return Response(status_code=200, content="Success")
```

**When `transformed_request` is None:**
```python
async def my_handler(raw_request: Request):
    # Only raw request is passed when transformation produces None
    # (e.g., for unregister which has no body)
    adapter_name = raw_request.path_params.get("adapter_name")

    return Response(status_code=200, content="Success")
```

## Public API and Convenience Functions

### Architecture

The `sagemaker` module (`sagemaker/__init__.py`) provides a public API layer that wraps the LoRA decorator factory. This layer serves as the recommended interface for users while keeping the underlying factory implementation flexible.

```
User Code
    ↓
sagemaker/__init__.py (Public API - Convenience Functions)
    ↓
lora/factory.py (Decorator Factory)
    ↓
lora/transforms/* (Transformer Implementations)
```

### Convenience Function Implementation

The convenience functions are thin wrappers around `create_lora_transform_decorator` that provide:

1. **Cleaner imports**: Users import from `sagemaker` instead of `sagemaker.lora.factory`
2. **Validation and preprocessing**: Input validation before passing to the factory
3. **Handler type abstraction**: Users don't need to know about `LoRAHandlerType` enum
4. **Special behavior**: Automatic configuration (e.g., header auto-fill for `inject_adapter_id`)

### Design Rationale

**Why a wrapper layer?**

1. **API Stability**: The factory implementation can change without breaking user code
2. **Input Validation**: Catch common mistakes early with clear error messages
3. **Special Cases**: Handle handler-specific logic (like `inject_adapter_id` auto-fill)
4. **Future Extensions**: Easy to add preprocessing, logging, or metrics without changing the factory
5. **User Experience**: Simpler, more intuitive API for common use cases

**Why not just export `create_lora_transform_decorator`?**

While `create_lora_transform_decorator` is available for advanced use cases, the convenience functions:
- Hide internal enums (`LoRAHandlerType`) from typical users
- Provide type-specific validation that isn't possible with a generic factory
- Allow handler-specific behavior without complicating the factory
- Create a clear separation between "public API" and "internal implementation"

### Adding New Convenience Functions

When adding support for a new LoRA operation:

**Step 1: Implement the transformer** (see Custom Transformer Implementation section)

**Step 2: Add the handler type to `constants.py`:**

```python
class LoRAHandlerType(str, Enum):
    REGISTER_ADAPTER = "register_adapter"
    UNREGISTER_ADAPTER = "unregister_adapter"
    INJECT_ADAPTER_ID = "inject_adapter_id"
    MY_NEW_OPERATION = "my_new_operation"  # New
```

**Step 3: Map it in `transforms/__init__.py`:**

```python
def resolve_lora_transform(handler_type: str) -> type:
    match handler_type:
        case LoRAHandlerType.MY_NEW_OPERATION:
            return MyNewOperationTransform
        # ... other cases
```

**Step 4: Add convenience function to `sagemaker/__init__.py`:**

```python
def register_my_new_operation_handler(request_shape: dict, response_shape: dict = {}):
    """Convenience function for my new operation.

    :param request_shape: JMESPath expressions for request transformation
    :param response_shape: JMESPath expressions for response transformation
    :return: Decorator for handler function
    """
    # Add any validation specific to this operation
    if some_validation_condition:
        raise ValueError("Validation error message")

    # Add any preprocessing specific to this operation
    processed_request_shape = preprocess_if_needed(request_shape)

    return create_lora_transform_decorator(LoRAHandlerType.MY_NEW_OPERATION)(
        processed_request_shape,
        response_shape
    )
```

**Step 5: Export in `__all__`:**

```python
__all__ = [
    # ... existing exports
    "register_my_new_operation_handler",
]
```

### Validation Guidelines

When adding validation to convenience functions:

1. **Validate early**: Fail fast with clear error messages
2. **Be specific**: Provide actionable feedback ("missing required field X" vs "invalid input")
3. **Log warnings**: Use logger.warning() for non-fatal issues
4. **Document behavior**: Explain validation rules in docstrings
5. **Test edge cases**: Write tests for all validation branches

Example validation patterns:

```python
# Check required fields
if "required_field" not in request_shape:
    raise ValueError("request_shape must contain 'required_field'")

# Validate structure
if not isinstance(request_shape.get("nested"), dict):
    raise ValueError("request_shape['nested'] must be a dictionary")

# Warn about unused parameters
if unused_param:
    logger.warning(f"Parameter {unused_param} is not used for this handler type")

# Validate JMESPath expressions (future enhancement)
for key, expression in request_shape.items():
    try:
        jmespath.compile(expression)
    except jmespath.exceptions.ParseError as e:
        raise ValueError(f"Invalid JMESPath in '{key}': {e}")
```

## Setting Up Your FastAPI Application

The `sagemaker` module provides utilities for automatically configuring your FastAPI application with registered handlers.

### Recommended Approach: Using `bootstrap()`

The simplest and recommended way to configure your FastAPI application is using `bootstrap()`:

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

# Step 3: Configure SageMaker integrations
bootstrap(app)

# Your app now has these routes automatically configured:
# POST /adapters -> load_adapter
# DELETE /adapters/{adapter_name} -> unload_adapter
```

**Important:** Call `bootstrap()` after registering all handlers. Handlers registered after this call will not be automatically mounted.

### Advanced: Router Mounting Internals

For developers who need more control over the routing infrastructure, the following functions are available:

**1. `create_sagemaker_router()`**

Creates a new APIRouter with all registered SageMaker handlers automatically mounted. This is used internally by `bootstrap()` but can be used directly if you need to manually control router mounting:

```python
from fastapi import FastAPI
from model_hosting_container_standards.sagemaker import create_sagemaker_router

# Register your handlers first
@register_load_adapter_handler(request_shape={"name": "body.name"})
async def load_adapter(data, request):
    return {"status": "loaded"}

# Manually create and mount the router
app = FastAPI()
sagemaker_router = create_sagemaker_router()
app.include_router(sagemaker_router)
```

This function internally uses the generic `create_router()` from `common.fastapi.routing` with the LoRA route resolver.

**2. `mount_handlers(router, handler_names=None, route_resolver=None)`**

For even more control, use the generic mounting function from `common.fastapi.routing` to mount handlers to an existing router:

```python
from fastapi import APIRouter
from model_hosting_container_standards.common.fastapi.routing import mount_handlers
from model_hosting_container_standards.sagemaker.lora.routes import get_lora_route_config

# Create your router
app_router = APIRouter()

# Register your handlers first
@register_load_adapter_handler(request_shape={"name": "body.name"})
async def load_adapter(data, request):
    return {"status": "loaded"}

@register_unload_adapter_handler(request_shape={"name": "path_params.adapter_name"})
async def unload_adapter(data, request):
    return {"status": "unloaded"}

# Mount all registered handlers using the LoRA route resolver
mount_handlers(app_router, route_resolver=get_lora_route_config)

# Or mount only specific handler types
mount_handlers(
    app_router,
    handler_names=["register_adapter", "unregister_adapter"],
    route_resolver=get_lora_route_config
)
```

**3. `get_lora_route_config(handler_type)`**

Maps LoRA handler types to their default API route configurations, returning a `RouteConfig` object:

```python
from model_hosting_container_standards.sagemaker.lora.routes import get_lora_route_config
from model_hosting_container_standards.common.fastapi.routing import RouteConfig

# Get route configuration
route_config = get_lora_route_config("register_adapter")
if route_config:
    print(f"{route_config.method} {route_config.path}")  # Prints: POST /adapters
    print(f"Tags: {route_config.tags}")  # Prints: Tags: ['adapters', 'lora']
    print(f"Summary: {route_config.summary}")  # Prints: Summary: Register a new LoRA adapter

# Returns None for handlers without routes
route_config = get_lora_route_config("inject_adapter_id")
print(route_config)  # Prints: None (no default route - this is a transform only)
```

The `RouteConfig` dataclass (from `common.fastapi.routing`) provides:
- `path`: The URL path for the route (e.g., "/adapters")
- `method`: The HTTP method (e.g., "POST", "DELETE")
- `tags`: Optional list of tags for API documentation
- `summary`: Optional short summary for API documentation

### Default Route Mappings

The following default routes are automatically configured:

| Handler Type | HTTP Method | Route Path | Description |
|-------------|-------------|------------|-------------|
| `register_adapter` | POST | `/adapters` | Register/load a LoRA adapter |
| `unregister_adapter` | DELETE | `/adapters/{adapter_name}` | Unregister/unload a LoRA adapter |
| `inject_adapter_id` | N/A | N/A | request transform only, no direct route |

## Custom Transformer Implementation

To implement a custom transformer for specialized LoRA operations:

### Step 1: Create Transformer Class

Inherit from `BaseLoRAApiTransform` and implement required methods:

```python
from model_hosting_container_standards.sagemaker.lora.base_lora_api_transform import BaseLoRAApiTransform
from model_hosting_container_standards.sagemaker.lora.models import BaseLoRATransformRequestOutput
from fastapi import Request, Response

class MyCustomTransform(BaseLoRAApiTransform):
    async def transform_request(self, raw_request: Request) -> BaseLoRATransformRequestOutput:
        # Parse and validate request
        request_data = await raw_request.json()

        # Apply transformations using _transform_request helper
        transformed = self._transform_request(request_data, raw_request)

        # Return transformation output
        return BaseLoRATransformRequestOutput(
            request=transformed,
            raw_request=raw_request,
            adapter_name=request_data.get("name")
        )

    def _transform_ok_response(self, response: Response, **kwargs):
        # Transform successful responses
        adapter_name = kwargs.get("adapter_name")
        adapter_alias = kwargs.get("adapter_alias")
        return Response(
            status_code=200,
            content=f"Operation succeeded for {adapter_alias or adapter_name}"
        )

    def _transform_error_response(self, response: Response, **kwargs):
        # Transform error responses
        return response  # Or customize error handling
```

### Step 2: Register Handler Type

Add your handler type to `constants.py`:

```python
class LoRAHandlerType(str, Enum):
    REGISTER_ADAPTER = "register_adapter"
    UNREGISTER_ADAPTER = "unregister_adapter"
    INJECT_ADAPTER_ID = "inject_adapter_id"
    MY_CUSTOM_HANDLER = "my_custom_handler"  # New handler type
```

### Step 3: Register Transformer in Resolver

Update the `resolve_lora_transform` function in `transforms/__init__.py` to map your handler type to the transformer:

```python
def resolve_lora_transform(handler_type: str) -> type:
    """Resolve LoRA handler type to transformer class.

    This function is passed to create_transform_decorator to map
    handler types to their corresponding transformer implementations.
    """
    match handler_type:
        case LoRAHandlerType.REGISTER_ADAPTER:
            return RegisterLoRAApiTransform
        case LoRAHandlerType.UNREGISTER_ADAPTER:
            return UnregisterLoRAApiTransform
        case LoRAHandlerType.INJECT_ADAPTER_ID:
            return InjectToBodyApiTransform
        case LoRAHandlerType.MY_CUSTOM_HANDLER:
            return MyCustomTransform  # New mapping
        case _:
            raise ValueError(f"Unsupported LoRAHandlerType: {handler_type}")
```

### Step 4: Create and Use Decorator

```python
from model_hosting_container_standards.sagemaker.lora.factory import create_lora_transform_decorator

register_my_handler = create_lora_transform_decorator(LoRAHandlerType.MY_CUSTOM_HANDLER)

@register_my_handler(
    request_shape={"model": "body.name"},
    response_shape={"status": "state"}
)
async def my_custom_handler(data: SimpleNamespace, raw_request: Request):
    # Your handler logic
    return Response(status_code=200)
```

### Step 5: Add Convenience Function (Optional)

For better user experience, add a convenience function in `sagemaker/__init__.py`:

```python
def register_my_custom_handler(request_shape: dict, response_shape: dict = {}):
    """Convenience function for my custom operation."""
    # Add validation as needed
    return create_lora_transform_decorator(LoRAHandlerType.MY_CUSTOM_HANDLER)(
        request_shape,
        response_shape
    )
```

## Utilities

### Adapter Name Extraction

The `utils.py` module provides helper functions for extracting adapter information:

```python
from model_hosting_container_standards.sagemaker.lora.utils import (
    get_adapter_name_from_request,
    get_adapter_alias_from_request_header,
    get_adapter_name_from_request_path
)

# Extract adapter name with priority: path_params > transform_output > header
adapter_name = get_adapter_name_from_request(transform_request_output)

# Extract adapter alias from X-Amzn-SageMaker-Adapter-Alias header
adapter_alias = get_adapter_alias_from_request_header(raw_request)

# Extract adapter name from URL path parameters
adapter_name = get_adapter_name_from_request_path(raw_request)
```

## Error Handling

The module provides standard error handling through the transformer classes:

### Status Codes

- `200 OK` - Successful operation
- `400 BAD_REQUEST` - Request validation failed (missing required fields, invalid format)
- `424 FAILED_DEPENDENCY` - Adapter-related errors (invalid weights, rank issues)
- `500 INTERNAL_SERVER_ERROR` - Unexpected server errors

### Error Messages

Standard error messages are defined in `constants.py`:

```python
class ResponseMessage(str, Enum):
    ADAPTER_REGISTERED = "Adapter {alias} registered"
    ADAPTER_UNREGISTERED = "Adapter {alias} unregistered"
    ADAPTER_NOT_FOUND = "The adapter {alias} was not found"
    ADAPTER_INVALID_WEIGHTS = "doesn't contain tensors"
    ADAPTER_MAX_LORA_RANK = "greater than max_lora_rank"
```

### Custom Error Handling

Implement `_transform_error_response` in your transformer to customize error handling:

```python
def _transform_error_response(self, response: Response, **kwargs):
    response_body = response.body.decode()

    if "specific_error_pattern" in response_body:
        return Response(
            status_code=HTTPStatus.CUSTOM_CODE,
            content="Custom error message"
        )

    return response  # Default error handling
```

## Constants and Headers

### SageMaker LoRA API Headers

```python
class SageMakerLoRAApiHeader:
    ADAPTER_IDENTIFIER = "X-Amzn-SageMaker-Adapter-Identifier"
    ADAPTER_ALIAS = "X-Amzn-SageMaker-Adapter-Alias"
```

These headers are used to pass adapter information in inference requests:
- `X-Amzn-SageMaker-Adapter-Identifier`: The name/ID of the adapter to use
- `X-Amzn-SageMaker-Adapter-Alias`: A human-readable alias for the adapter

## Examples and Documentation

### LoRA-Specific Documentation

- **[FACTORY_USAGE.md](./FACTORY_USAGE.md)** - Detailed examples of using the LoRA decorators in real-world scenarios, including:
  - Quick start guide
  - Using convenience functions (`register_load_adapter_handler`, etc.)
  - Complete handler examples
  - LoRA-specific troubleshooting
  - Best practices for adapter management

### Generic Transform Decorator Documentation

- **[Transform Decorator Usage Guide](../../common/transforms/BASE_FACTORY_USAGE.md)** - Understanding the underlying infrastructure:
  - How the decorator factory pattern works
  - Request transformation flow
  - JMESPath expression guide
  - General troubleshooting and debugging
  - Best practices for transformations

## Testing

The module includes comprehensive unit tests:

- `tests/sagemaker/lora/test_factory.py` - Factory and decorator tests
- `tests/sagemaker/lora/test_api_transform.py` - Transformation logic tests
- `tests/sagemaker/lora/transforms/` - Individual transformer tests

Run tests with:
```bash
pytest python/tests/sagemaker/lora/
```

## Dependencies

- **fastapi**: Web framework for request/response handling
- **pydantic**: Request validation and data models
- **jmespath**: JSON query language for data extraction
