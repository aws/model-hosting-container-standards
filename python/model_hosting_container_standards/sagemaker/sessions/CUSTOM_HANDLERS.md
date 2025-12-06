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
Client Request
    ↓
SessionApiTransform (detects session request)
    ↓
get_handler_for_request_type()
    ↓
    ├─→ Custom Handler (if registered)
    │   └─→ Engine's Session API
    │
    └─→ Default Handler (if not registered)
        └─→ SageMaker SessionManager
```

## Handler Signatures

Both handlers must be async functions that accept a FastAPI `Request` object:

```python
from fastapi import Request, Response

async def my_create_session_handler(raw_request: Request) -> Response:
    """Create a new session via the engine's API."""
    pass

async def my_close_session_handler(raw_request: Request) -> Response:
    """Close an existing session via the engine's API."""
    pass
```

## Using Transform Classes


```python
from model_hosting_container_standards.sagemaker.sessions.transforms import (
    CreateSessionApiTransform,
    CloseSessionApiTransform
)

# Define request/response shapes using JMESPath
create_transform = CreateSessionApiTransform(
    request_shape={},  # Transform incoming request
    response_shape={
        "X-Amzn-SageMaker-New-Session-Id": "session_id",
        "content": "message"
    }
)

close_transform = CloseSessionApiTransform(
    request_shape={
        "session_id": "headers.'X-Amzn-SageMaker-Session-Id'"
    },
    response_shape={
        "content": "message"
    }
)
```

## Best Practices

1. **Validate session IDs**: Always validate that the engine returns valid session IDs
2. **Handle timeouts**: Set appropriate timeouts when calling engine APIs
3. **Log operations**: Log session creation/closure for debugging
4. **Error propagation**: Provide clear error messages when engine operations fail
5. **Cleanup**: Ensure sessions are properly cleaned up even on errors
6. **Testing**: Test both success and failure scenarios
7. **Idempotency**: Handle duplicate close requests gracefully

## Utilities

The framework provides utility functions for working with sessions:

```python
from model_hosting_container_standards.sagemaker.sessions.utils import (
    get_session_id_from_request,  # Extract session ID from headers
    get_session,                   # Get session from manager
)
from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader,        # Header name constants
    SessionRequestType,            # Request type enum
)
```

## See Also

- [README.md](./README.md) - Main sessions documentation
- [handlers.py](./handlers.py) - Default handler implementations
- [transforms/](./transforms/) - Transform classes for engine integration
