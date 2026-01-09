# SageMaker Stateful Sessions

This module provides stateful session management for SageMaker model hosting containers, enabling multi-turn conversations and persistent state across requests.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Session Storage](#session-storage)
- [Expiration and Cleanup](#expiration-and-cleanup)
- [Advanced Usage](#advanced-usage)
  - [Custom Session Handlers](./CUSTOM_HANDLERS.md)

## Overview

Stateful sessions allow clients to maintain context across multiple inference requests without passing all state in every request. Each session has:
- **Unique ID**: UUID-based identifier
- **File-based storage**: Key-value data stored in-memory (not persistent across restarts)
- **Automatic expiration**: Configurable TTL (default: 20 minutes)
- **Thread-safe access**: Concurrent request handling

### Session Management Modes

The framework supports two modes of session management:

1. **SageMaker-Managed Sessions** (Default)
   - Sessions managed by the built-in `SessionManager`
   - File-based key-value storage in `/dev/shm`
   - Automatic expiration and cleanup
   - Best for general-purpose session state

2. **Engine-Managed Sessions** (Custom Handlers)
   - Sessions delegated to your inference engine's native API
   - Leverages engine-specific session features
   - Requires custom handler registration
   - Best when engine has built-in session support
   - See [CUSTOM_HANDLERS.md](./CUSTOM_HANDLERS.md) for details

## Architecture

```
Client Request to /invocations
    ↓
SessionApiTransform (intercepts and inspects)
    ↓
    ├─→ Session Management Request (NEW_SESSION or CLOSE)
    │   ├─→ Check Handler Registry
    │   │   ├─→ Custom Handler (if registered)
    │   │   │   └─→ Your engine's session API
    │   │   └─→ Default Handler (if not registered)
    │   │       └─→ SageMaker SessionManager
    │   └─→ Return with session headers
    │
    └─→ Regular Inference Request
        ├─→ Validate session ID (if present)
        ├─→ Inject session ID into body (if configured)
        └─→ Pass to your handler
```

### Key Components

- **`SessionManager`** (`manager.py`): Manages session lifecycle, expiration, and cleanup (default mode)
- **`Session`** (`manager.py`): Individual session with file-based key-value storage
- **`SessionApiTransform`** (`transform.py`): API transform that intercepts and routes session requests
- **Handler Registry**: Routes session requests to custom or default handlers
- **Session Handlers** (`handlers.py`): Default functions to create and close sessions
- **Engine Session Transforms** (`transforms/`): Transform classes for custom engine integration
- **Utilities** (`utils.py`): Helper functions for session ID extraction and retrieval

## Quick Start

### Enabling Sessions in Your Handler

Use the `stateful_session_manager()` decorator on your `/invocations` endpoint:

```python
from fastapi import FastAPI, Request
from model_hosting_container_standards.sagemaker import stateful_session_manager, bootstrap

app = FastAPI()

@app.post("/invocations")
@stateful_session_manager()
async def invocations(request: Request):
    # Handler logic with session support
    pass

bootstrap(app)
```

### Creating a Session

**Request:**
```json
{
  "requestType": "NEW_SESSION"
}
```

**Response Headers:**
```
X-Amzn-SageMaker-New-Session-Id: <uuid>; Expires=2025-10-22T12:34:56Z
```

### Using a Session

Include the session ID in subsequent requests:

**Request Headers:**
```
X-Amzn-SageMaker-Session-Id: <uuid>
```

### Closing a Session

**Request:**
```json
{
  "requestType": "CLOSE"
}
```

**Request Headers:**
```
X-Amzn-SageMaker-Session-Id: <uuid>
```

**Response Headers:**
```
X-Amzn-SageMaker-Closed-Session-Id: <uuid>
```

## Configuration

Configure via environment variables:

```bash
export SAGEMAKER_ENABLE_STATEFUL_SESSIONS=true
export SAGEMAKER_SESSIONS_EXPIRATION=1200  # TTL in seconds (default: 1200)
export SAGEMAKER_SESSIONS_PATH=/dev/shm/sagemaker_sessions  # Storage path (optional)
```

The session manager is automatically initialized from these environment variables when you call `bootstrap(app)`.

**Important**: If `SAGEMAKER_ENABLE_STATEFUL_SESSIONS` is not set to `true`, session management requests will fail with a 400 error. Regular inference requests without session headers will continue to work normally.

### Storage Location

Sessions are stored in memory-backed filesystem when available:
- **Preferred**: `/dev/shm/sagemaker_sessions` (tmpfs - fast, in-memory)
- **Fallback**: `{tempdir}/sagemaker_sessions` (disk-backed)

**Note**: Session data is not persistent across container restarts.

## Session Storage

Each session maintains its own directory with JSON files for key-value pairs:

```
/dev/shm/sagemaker_sessions/
├── <session-id-1>/
│   ├── key1.json
│   └── key2.json
└── <session-id-2>/
    └── key1.json
```

## Expiration and Cleanup

- Sessions automatically expire after configured TTL
- Expired sessions are cleaned up during:
  - New session creation
  - Session retrieval (lazy cleanup)
- Session data is deleted from disk on expiration/closure

## Advanced Usage

### Injecting Session ID into Request Body

If your handler needs the session ID in the request body (not just headers), use the `request_session_id_path` parameter:

```python
@app.post("/invocations")
@stateful_session_manager(request_session_id_path="session_id")
async def invocations(request: Request):
    body = await request.json()
    session_id = body.get("session_id")  # Automatically injected from header
    # Handler logic
```

For nested paths, use dot notation:

```python
@stateful_session_manager(request_session_id_path="metadata.session_id")
async def invocations(request: Request):
    body = await request.json()
    session_id = body["metadata"]["session_id"]  # Injected at nested path
```

**Note**: The session ID is only injected when the `X-Amzn-SageMaker-Session-Id` header is present in the request.

### Custom Session Handlers

If your inference engine has its own session management API, you can register custom handlers to delegate session creation and closure to the engine instead of using SageMaker's built-in session management.

See [CUSTOM_HANDLERS.md](./CUSTOM_HANDLERS.md) for detailed documentation on implementing custom create/close session handlers.
