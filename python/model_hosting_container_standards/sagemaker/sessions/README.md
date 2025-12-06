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
SessionApiTransform  (transform.py)
    ↓
    ├─→ Session Management Request
    │   ├─→ create_session (handlers.py)
    │   └─→ close_session (handlers.py)
    │
    └─→ Regular Inference Request
        └─→ Pass through with session context
```

### Key Components

- **`SessionManager`** (`manager.py`): Manages session lifecycle, expiration, and cleanup
- **`Session`** (`manager.py`): Individual session with file-based key-value storage
- **`SessionApiTransform`** (`transform.py`): API transform that intercepts session requests
- **Session Handlers** (`handlers.py`): Functions to create and close sessions
- **Utilities** (`utils.py`): Helper functions for session ID extraction and retrieval

## Quick Start

### Enabling Sessions in Your Handler

Use the `stateful_session_manager()` convenience decorator:

```python
from model_hosting_container_standards.sagemaker import stateful_session_manager

@stateful_session_manager()
def my_handler(request):
    # Handler logic with session support
    pass
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

Configure via `SessionManager` properties:

```python
session_manager = SessionManager({
    "sessions_expiration": "1200",  # TTL in seconds (default: 1200)
    "sessions_path": "/dev/shm/sagemaker_sessions"  # Storage path
})
```

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

### Custom Session Handlers

If your inference engine has its own session management API, you can register custom handlers to delegate session creation and closure to the engine instead of using SageMaker's built-in session management.

See [CUSTOM_HANDLERS.md](./CUSTOM_HANDLERS.md) for detailed documentation on implementing custom create/close session handlers.
