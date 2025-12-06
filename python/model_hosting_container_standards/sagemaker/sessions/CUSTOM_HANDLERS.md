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
from model_hosting_container_standards.sagemaker import (
    register_create_session_handler,
    register_close_session_handler,
    stateful_session_manager,
    bootstrap
)

app = FastAPI()

# Define your engine's request/response models
class CreateSessionRequest(BaseModel):
    capacity: int

class CreateSessionResponse(BaseModel):
    session_id: str
    message: str

# Register custom create session handler
@register_create_session_handler(
    request_shape={
        "capacity": "`1024`"  # JMESPath literal value
    },
    response_session_id_path="body.session_id",  # Extract session ID from response
    content_path="body.message"  # Extract content for logging
)
@app.post("/engine/create_session")
async def create_session(obj: CreateSessionRequest, request: Request):
    # Call your engine's session creation API
    session_id = await my_engine.create_session(capacity=obj.capacity)
    return CreateSessionResponse(session_id=session_id, message="Session created")

# Register custom close session handler
@register_close_session_handler(
    request_shape={
        "session_id": 'headers."X-Amzn-SageMaker-Session-Id"'  # Extract from header
    },
    content_path="`Session closed successfully`"  # Static message
)
@app.post("/engine/close_session")
async def close_session(session_id: str, request: Request):
    # Call your engine's session closure API
    await my_engine.close_session(session_id)
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
    request_shape: dict,              # Required: JMESPath mappings for request transformation
    response_session_id_path: str,    # Required: JMESPath to extract session ID from response
    content_path: str = None          # Optional: JMESPath to extract content for logging
)
```

- **`request_shape`**: Maps target keys to source JMESPath expressions. Transforms the incoming SageMaker request into your engine's expected format.
- **`response_session_id_path`**: JMESPath expression to extract the session ID from your engine's response. This is **required** because the framework needs to return the session ID in the response header.
- **`content_path`**: Optional JMESPath expression to extract a message for logging. Defaults to a generic success message.

### `@register_close_session_handler`

```python
@register_close_session_handler(
    request_shape: dict,              # Required: JMESPath mappings for request transformation
    content_path: str = None          # Optional: JMESPath to extract content for logging
)
```

- **`request_shape`**: Maps target keys to source JMESPath expressions. Typically extracts the session ID from the request header.
- **`content_path`**: Optional JMESPath expression to extract a message for logging. Defaults to a generic success message.

**Note**: `response_session_id_path` is not needed for close handlers because the session ID comes from the request header, not the response.

## How It Works

When you register custom handlers:

1. **Client sends session request** to `/invocations` with `{"requestType": "NEW_SESSION"}` or `{"requestType": "CLOSE"}`
2. **SessionApiTransform intercepts** the request and checks the handler registry
3. **If custom handler registered**: Request is routed to your custom endpoint (e.g., `/engine/create_session`)
4. **Transform applies**: Request/response shapes are transformed using JMESPath
5. **Response returned**: With appropriate SageMaker session headers (`X-Amzn-SageMaker-New-Session-Id` or `X-Amzn-SageMaker-Closed-Session-Id`)

The key benefit: Your `/invocations` endpoint stays clean, and session management is handled transparently.

## JMESPath Expressions

The `request_shape` and `response_shape` parameters use JMESPath expressions to transform data:

### Request Shape

Maps target keys to source expressions:

```python
request_shape={
    "capacity": "`1024`",  # Literal value
    "session_id": 'headers."X-Amzn-SageMaker-Session-Id"',  # From header
    "user_id": "body.metadata.user"  # From request body
}
```

### Response Shape

For **create session**, you must specify:
- `response_session_id_path`: Where to extract the session ID from the engine's response
- `content_path`: Where to extract content for logging (optional)

```python
response_session_id_path="body.session_id"  # Extract from {"session_id": "..."}
response_session_id_path="body"  # If response is just the session ID string
content_path="body.message"  # Extract message from response
content_path="`Session created`"  # Use literal string
```

For **close session**, you only need:
- `content_path`: Where to extract content for logging (optional)

## Response Formats

Your custom handlers can return different response formats:

### Dictionary Response
```python
async def create_session(obj: CreateSessionRequest, request: Request):
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "metadata": {"engine": "custom"}}
```

### String Response
```python
async def create_session(obj: CreateSessionRequest, request: Request):
    session_id = str(uuid.uuid4())
    return session_id  # Just return the session ID
```

### FastAPI Response Object
```python
from fastapi import Response

async def create_session(obj: CreateSessionRequest, request: Request):
    session_id = str(uuid.uuid4())
    return Response(
        status_code=201,
        content=json.dumps({"session_id": session_id}),
        media_type="application/json"
    )
```

## Error Handling

Raise `HTTPException` to return errors to the client:

```python
from fastapi.exceptions import HTTPException

@register_create_session_handler(...)
async def create_session(obj: CreateSessionRequest, request: Request):
    try:
        session_id = await my_engine.create_session()
        return {"session_id": session_id}
    except EngineError as e:
        raise HTTPException(status_code=500, detail=f"Engine error: {e}")
```

## Complete Example

Here's a complete example with error handling and session tracking:

```python
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import json

from model_hosting_container_standards.sagemaker import (
    register_create_session_handler,
    register_close_session_handler,
    stateful_session_manager,
    bootstrap
)
from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader
)

app = FastAPI()

# Track sessions in memory (for demo purposes)
active_sessions = {}

class CreateSessionRequest(BaseModel):
    capacity: int
    session_id: Optional[str] = None

@register_create_session_handler(
    request_shape={
        "capacity": "`1024`",
        "session_id": f'headers."{SageMakerSessionHeader.SESSION_ID}"'
    },
    response_session_id_path="body.session_id",
    content_path="body.message"
)
@app.post("/engine/create_session")
async def create_session(obj: CreateSessionRequest, request: Request):
    # Generate or use provided session ID
    session_id = obj.session_id or str(uuid.uuid4())
    
    # Check if session already exists
    if session_id in active_sessions:
        raise HTTPException(status_code=400, detail="Session already exists")
    
    # Create session in your engine
    active_sessions[session_id] = {"capacity": obj.capacity}
    
    return {
        "session_id": session_id,
        "message": f"Session created with capacity {obj.capacity}"
    }

@register_close_session_handler(
    request_shape={"session_id": f'headers."{SageMakerSessionHeader.SESSION_ID}"'},
    content_path="`Session closed successfully`"
)
@app.post("/engine/close_session")
async def close_session(session_id: str, request: Request):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Close session in your engine
    del active_sessions[session_id]
    
    return Response(status_code=200, content="Session closed")

@app.post("/invocations")
@stateful_session_manager(request_session_id_path="session_id")
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
        "echo": body
    })

bootstrap(app)
```

## Session Validation Behavior

When custom handlers are registered, the framework **does not** validate session IDs against the default `SessionManager`. This means:

- **With custom handlers**: Session validation is your responsibility. The framework only routes requests to your handlers.
- **Without custom handlers** (default mode): The framework validates session IDs against the `SessionManager` automatically.

This design allows your engine to manage sessions independently without interference from the default session manager.

## Best Practices

1. **Validate session IDs**: Check that the engine returns valid session IDs in create handlers
2. **Handle errors gracefully**: Use HTTPException for clear error messages
3. **Log operations**: Log session creation/closure for debugging
4. **Test thoroughly**: Test both success and failure scenarios
5. **Idempotency**: Handle duplicate close requests gracefully (return 404 or succeed silently)
6. **Session isolation**: Ensure different sessions maintain independent state
7. **Thread safety**: If your engine stores session state, ensure thread-safe access for concurrent requests

## Troubleshooting

### Session ID not extracted from response

**Problem**: Getting "Engine failed to return a valid session ID" error.

**Solution**: Check that your `response_session_id_path` matches your response structure:
```python
# If your handler returns: {"session_id": "abc123"}
response_session_id_path="body.session_id"

# If your handler returns: "abc123"
response_session_id_path="body"
```

### Request not reaching custom handler

**Problem**: Custom handler not being called.

**Solution**: Ensure you call `bootstrap(app)` **after** registering your handlers:
```python
@register_create_session_handler(...)
async def create_session(...):
    pass

bootstrap(app)  # Must be after handler registration
```

### Session header not found in close handler

**Problem**: Getting "Session ID is required in request headers" error.

**Solution**: Ensure your `request_shape` extracts the session ID from the header:
```python
request_shape={"session_id": 'headers."X-Amzn-SageMaker-Session-Id"'}
```

## See Also

- [README.md](./README.md) - Main sessions documentation
- [Integration tests](../../../tests/integration/test_custom_session_handlers_integration.py) - Complete working examples
