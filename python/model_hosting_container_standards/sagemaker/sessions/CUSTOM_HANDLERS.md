# Custom Session Handlers

This guide explains how to implement custom create and close session handlers when your inference engine has its own session management API.

## Overview

By default, SageMaker's session management uses the built-in `SessionManager` to handle session lifecycle. However, if your inference engine provides its own session API, you can register custom handlers to delegate session operations to the engine.

### When to Use Custom Handlers

Use custom handlers when:
- Your engine has native session management capabilities
- You want to leverage engine-specific session features
- Session state needs to be managed within the engine's memory space
- You need custom session initialization or cleanup logic

### Architecture

```
Client Request (NEW_SESSION or CLOSE)
    ↓
SessionApiTransform (detects session request)
    ↓
Handler Registry Check
    ↓
    ├─→ Custom Handler (if registered)
    │   └─→ Your Engine's Session API
    │
    └─→ Default Handler (if not registered)
        └─→ SageMaker SessionManager
```

## Registration API

Use the `@register_create_session_handler` and `@register_close_session_handler` decorators to register custom handlers:

```python
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from model_hosting_container_standards.sagemaker import (
    register_create_session_handler,
    register_close_session_handler,
    stateful_session_manager,
    bootstrap
)

app = FastAPI()

# Define your engine's request models using Pydantic
class CreateSessionRequest(BaseModel):
    capacity: int
    session_id: Optional[str] = None

class CloseSessionRequest(BaseModel):
    session_id: str

# Register custom create session handler
@register_create_session_handler(
    engine_response_session_id_path="body"  # Extract session ID from response body
)
@app.post("/engine/create_session")
async def create_session(obj: CreateSessionRequest, request: Request):
    # Call your engine's session creation API
    session_id = await my_engine.create_session(capacity=obj.capacity)
    return session_id  # Return session ID directly

# Register custom close session handler
@register_close_session_handler(
    engine_request_session_id_path="body.session_id"  # Inject session ID into request body
)
@app.post("/engine/close_session")
async def close_session(obj: CloseSessionRequest, request: Request):
    # Call your engine's session closure API
    await my_engine.close_session(obj.session_id)
    return {"status": "closed"}

# Your main invocations endpoint
@app.post("/invocations")
@stateful_session_manager()
async def invocations(request: Request):
    # Handle regular inference requests
    pass

bootstrap(app)
```

## Decorator Parameters

### `@register_create_session_handler`

```python
@register_create_session_handler(
    engine_response_session_id_path: str,
    engine_request_model_cls: Optional[BaseModel] = None
)
```

**Parameters:**

- **`engine_response_session_id_path`** (required): JMESPath expression to extract the session ID from your engine's response. This is **required** because the framework needs to return the session ID to the client.
  - Common values:
    - `"body"` - if your handler returns just the session ID string
    - `"body.session_id"` - if your handler returns `{"session_id": "..."}`
    - `"body.data.id"` - for nested response structures
    - `"headers.X-Session-Id"` - to extract from response headers

- **`engine_request_model_cls`** (optional): A Pydantic BaseModel class defining the expected request schema for your engine endpoint. If provided, FastAPI will validate incoming requests against this model and provide the validated object to your handler.

### `@register_close_session_handler`

```python
@register_close_session_handler(
    engine_request_session_id_path: str,
    engine_request_model_cls: Optional[BaseModel] = None
)
```

**Parameters:**

- **`engine_request_session_id_path`** (required): Target path in the engine request body where the session ID will be injected. The session ID is extracted from the SageMaker session header (`X-Amzn-SageMaker-Session-Id`) and placed at this path. This is **required** because the engine needs to know which session to close.
  - Common values:
    - `"body.session_id"` - injects at root level: `{"session_id": "..."}`
    - `"body.metadata.session_id"` - for nested paths
    - `"body.id"` - if your engine expects `{"id": "..."}`

- **`engine_request_model_cls`** (optional): A Pydantic BaseModel class defining the expected request schema for your engine endpoint.

## How It Works

When you register custom handlers:

1. **Client sends session request** to `/invocations` with `{"requestType": "NEW_SESSION"}` or `{"requestType": "CLOSE"}`
2. **SessionApiTransform intercepts** the request and checks the handler registry
3. **If custom handler registered**: Request is routed to your custom endpoint (e.g., `/engine/create_session`)
4. **Session ID handling**:
   - For **create**: Session ID is extracted from your handler's response using `engine_response_session_id_path`
   - For **close**: Session ID is injected into your handler's request at `engine_request_session_id_path`
5. **Response returned**: With appropriate SageMaker session headers (`X-Amzn-SageMaker-New-Session-Id` or `X-Amzn-SageMaker-Closed-Session-Id`)

The key benefit: Your `/invocations` endpoint stays clean, and session management is handled transparently.

## Default Values Configuration

You can configure default values for your custom session handlers using environment variables. This is useful for providing default capacity, timeouts, or other parameters without hardcoding them in your handler.

### Environment Variables

Set defaults using JSON-formatted environment variables:

```bash
# Default values for create_session requests
export SAGEMAKER_TRANSFORMS_CREATE_SESSION_DEFAULTS='{"body.capacity": 1024, "body.timeout": 30}'

# Default values for close_session requests
export SAGEMAKER_TRANSFORMS_CLOSE_SESSION_DEFAULTS='{"body.force": false}'
```

**Format**: The keys use JMESPath notation to specify where in the request body the default values should be injected:
- `"body.capacity"` → `{"capacity": 1024}`
- `"body.metadata.size"` → `{"metadata": {"size": 100}}`

These defaults are merged with the actual request data. Values explicitly provided in the request take precedence over defaults.

### Using Defaults in Handlers

The framework automatically merges default values before calling your handler:

```python
class CreateSessionRequest(BaseModel):
    capacity: int = 1024  # Pydantic default as fallback
    timeout: int = 30

@register_create_session_handler(
    engine_response_session_id_path="body"
)
@app.post("/engine/create_session")
async def create_session(obj: CreateSessionRequest, request: Request):
    # obj.capacity will be:
    # 1. Value from request body if provided
    # 2. Value from SAGEMAKER_TRANSFORMS_CREATE_SESSION_DEFAULTS if set
    # 3. Pydantic default (1024) if neither is provided
    session_id = await my_engine.create_session(capacity=obj.capacity)
    return session_id
```

## Response Formats

Your custom handlers can return different response formats:

### String Response (Recommended for Create Session)
```python
async def create_session(obj: CreateSessionRequest, request: Request):
    session_id = str(uuid.uuid4())
    return session_id  # Return session ID directly

# Configuration: engine_response_session_id_path="body"
```

### Dictionary Response
```python
async def create_session(obj: CreateSessionRequest, request: Request):
    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "metadata": {"engine": "custom", "version": "1.0"}
    }

# Configuration: engine_response_session_id_path="body.session_id"
```

### FastAPI Response Object
```python
from fastapi import Response
import json

async def create_session(obj: CreateSessionRequest, request: Request):
    session_id = str(uuid.uuid4())
    return Response(
        status_code=201,
        content=json.dumps({"session_id": session_id}),
        media_type="application/json",
        headers={"X-Custom-Header": "value"}
    )

# Configuration: engine_response_session_id_path="body.session_id"
```

### Header-Based Response
```python
from fastapi import Response

async def create_session(obj: CreateSessionRequest, request: Request):
    session_id = str(uuid.uuid4())
    return Response(
        status_code=200,
        headers={"X-Session-Id": session_id}
    )

# Configuration: engine_response_session_id_path="headers.X-Session-Id"
```

## Error Handling

Raise `HTTPException` to return errors to the client:

```python
from fastapi.exceptions import HTTPException

@register_create_session_handler(
    engine_response_session_id_path="body"
)
async def create_session(obj: CreateSessionRequest, request: Request):
    try:
        session_id = await my_engine.create_session(capacity=obj.capacity)
        return session_id
    except EngineError as e:
        raise HTTPException(status_code=500, detail=f"Engine error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Unexpected error")
```

## Complete Example

Here's a complete example with error handling, validation, and session tracking:

```python
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import json

from model_hosting_container_standards.sagemaker import (
    register_create_session_handler,
    register_close_session_handler,
    stateful_session_manager,
    bootstrap
)

app = FastAPI()

# Track sessions in memory (for demo purposes)
active_sessions = {}

# Define request models with validation
class CreateSessionRequest(BaseModel):
    capacity: int = Field(default=1024, ge=1, le=10000)
    session_id: Optional[str] = None

class CloseSessionRequest(BaseModel):
    session_id: str

@register_create_session_handler(
    engine_response_session_id_path="body.session_id",
    engine_request_model_cls=CreateSessionRequest
)
@app.post("/engine/create_session")
async def create_session(obj: CreateSessionRequest, request: Request):
    # Generate or use provided session ID
    session_id = obj.session_id or str(uuid.uuid4())

    # Check if session already exists
    if session_id in active_sessions:
        raise HTTPException(status_code=400, detail="Session already exists")

    # Validate capacity
    if obj.capacity < 1 or obj.capacity > 10000:
        raise HTTPException(status_code=400, detail="Capacity must be between 1 and 10000")

    # Create session in your engine
    try:
        active_sessions[session_id] = {
            "capacity": obj.capacity,
            "created_at": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")

    return {
        "session_id": session_id,
        "message": f"Session created with capacity {obj.capacity}"
    }

@register_close_session_handler(
    engine_request_session_id_path="body.session_id",
    engine_request_model_cls=CloseSessionRequest
)
@app.post("/engine/close_session")
async def close_session(obj: CloseSessionRequest, request: Request):
    if obj.session_id not in active_sessions:
        # Idempotent: succeed even if session doesn't exist
        return Response(status_code=200, content="Session already closed or does not exist")

    # Close session in your engine
    try:
        del active_sessions[obj.session_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to close session: {e}")

    return Response(status_code=200, content="Session closed successfully")

@app.post("/invocations")
@stateful_session_manager(engine_request_session_id_path="session_id")
async def invocations(request: Request):
    body_bytes = await request.body()
    body = json.loads(body_bytes.decode())
    session_id = body.get("session_id")

    if session_id and session_id not in active_sessions:
        raise HTTPException(status_code=400, detail="Invalid session")

    # Process inference request with session context
    return JSONResponse({
        "result": "success",
        "session_id": session_id or "no-session",
        "session_data": active_sessions.get(session_id, {}),
        "echo": body
    })

bootstrap(app)
```

## Session Validation Behavior

When custom handlers are registered, the framework **does not** validate session IDs against the default `SessionManager`. This means:

- **With custom handlers**: Session validation is your responsibility. The framework only routes requests to your handlers.
- **Without custom handlers** (default mode): The framework validates session IDs against the `SessionManager` automatically.

This design allows your engine to manage sessions independently without interference from the default session manager.

## Using Pydantic Models for Validation

Using Pydantic models with `engine_request_model_cls` provides automatic request validation:

```python
from pydantic import BaseModel, Field, validator

class CreateSessionRequest(BaseModel):
    capacity: int = Field(ge=1, le=10000, description="Session capacity")
    timeout: int = Field(default=30, ge=1, le=3600)
    metadata: Optional[dict] = None

    @validator('capacity')
    def validate_capacity(cls, v):
        if v % 2 != 0:
            raise ValueError('Capacity must be even')
        return v

@register_create_session_handler(
    engine_response_session_id_path="body",
    engine_request_model_cls=CreateSessionRequest
)
@app.post("/engine/create_session")
async def create_session(obj: CreateSessionRequest, request: Request):
    # obj is fully validated - capacity is even, in range, etc.
    session_id = str(uuid.uuid4())
    return session_id
```

If validation fails, FastAPI automatically returns a 422 Unprocessable Entity response with detailed error information.

## Best Practices

1. **Use Pydantic models**: Define `engine_request_model_cls` for automatic validation and documentation
2. **Validate session IDs**: Check that the engine returns valid session IDs in create handlers
3. **Handle errors gracefully**: Use HTTPException for clear error messages
4. **Log operations**: Log session creation/closure for debugging
5. **Test thoroughly**: Test both success and failure scenarios
6. **Idempotency**: Handle duplicate close requests gracefully (return success if session already closed)
7. **Session isolation**: Ensure different sessions maintain independent state
8. **Thread safety**: If your engine stores session state, ensure thread-safe access for concurrent requests
9. **Use environment variables for defaults**: Configure default values via `SAGEMAKER_TRANSFORMS_*` env vars
10. **Return simple responses**: For create handlers, returning just the session ID string is often simplest

## Troubleshooting

### Session ID not extracted from response

**Problem**: Getting "Session ID not found in response" error.

**Solution**: Check that your `engine_response_session_id_path` matches your response structure:
```python
# If your handler returns: "abc123" (string)
engine_response_session_id_path="body"

# If your handler returns: {"session_id": "abc123"}
engine_response_session_id_path="body.session_id"

# If your handler returns: {"data": {"id": "abc123"}}
engine_response_session_id_path="body.data.id"

# If session ID is in response header
engine_response_session_id_path="headers.X-Session-Id"
```

### Request not reaching custom handler

**Problem**: Custom handler not being called.

**Solution**: Ensure you call `bootstrap(app)` **after** registering your handlers:
```python
# Register handlers first
@register_create_session_handler(...)
async def create_session(...):
    pass

@register_close_session_handler(...)
async def close_session(...):
    pass

# Bootstrap after all registrations
bootstrap(app)
```

### Session ID not injected into engine request

**Problem**: Close session handler receives request without session ID.

**Solution**:
1. Ensure your Pydantic model has a `session_id` field
2. Ensure `engine_request_session_id_path` points to the correct location
3. Ensure the client sends the session ID in the `X-Amzn-SageMaker-Session-Id` header

```python
class CloseSessionRequest(BaseModel):
    session_id: str  # This field must exist

@register_close_session_handler(
    engine_request_session_id_path="body.session_id",  # Must match field path
    engine_request_model_cls=CloseSessionRequest
)
```

### Default values not applied

**Problem**: Default values from environment variables not being used.

**Solution**:
1. Ensure environment variable is set before application starts
2. Check JSON format is valid
3. Verify the path notation matches your model structure

```bash
# Correct format
export SAGEMAKER_TRANSFORMS_CREATE_SESSION_DEFAULTS='{"body.capacity": 1024}'

# Incorrect format (will fail)
export SAGEMAKER_TRANSFORMS_CREATE_SESSION_DEFAULTS='capacity: 1024'
```

### Validation errors with Pydantic models

**Problem**: Getting 422 validation errors unexpectedly.

**Solution**: Check that default values and environment variable defaults match your Pydantic model:

```python
class CreateSessionRequest(BaseModel):
    capacity: int = Field(ge=1)  # Must be >= 1

# Environment variable default must also satisfy constraints
# Good: export SAGEMAKER_TRANSFORMS_CREATE_SESSION_DEFAULTS='{"body.capacity": 1024}'
# Bad:  export SAGEMAKER_TRANSFORMS_CREATE_SESSION_DEFAULTS='{"body.capacity": 0}'
```

## See Also

- [README.md](./README.md) - Main sessions documentation
- [Integration tests](../../../tests/integration/test_custom_session_handlers_integration.py) - Complete working examples
- [BaseApiTransform2](../../common/transforms/base_api_transform2.py) - Transform base class
- [Defaults Configuration](../../common/transforms/defaults_config.py) - Default values system
