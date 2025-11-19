# MHCS Integration Runbook

**Version**: 1.0  
**Last Updated**: November 16, 2025  
**Target Audience**: ML framework developers integrating with Amazon SageMaker

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 [What is MHCS?](#11-what-is-mhcs)
   - 1.2 [Prerequisites](#12-prerequisites)
   - 1.3 [How to Use This Guide](#13-how-to-use-this-guide)

2. [Quick Start](#2-quick-start)
   - 2.1 [Use Case 1: Basic SageMaker Compatibility](#21-use-case-1-basic-sagemaker-compatibility)
   - 2.2 [Use Case 2: Multi-LoRA Support](#22-use-case-2-multi-lora-support)
   - 2.3 [Use Case 3: Sticky Session Support](#23-use-case-3-sticky-session-support)
   - 2.4 [Use Case 4: Runtime Custom Code Injection](#24-use-case-4-runtime-custom-code-injection)

3. [Core Integration](#3-core-integration)
   - 3.1 [Understanding the Bootstrap Function](#31-understanding-the-bootstrap-function)
   - 3.2 [Handler Registration Pattern](#32-handler-registration-pattern)
   - 3.3 [Handler Priority System](#33-handler-priority-system)
   - 3.4 [Integration Checklist](#34-integration-checklist)
   - 3.5 [Complete Integration Example](#35-complete-integration-example)

4. [Transform Decorators](#4-transform-decorators)
   - 4.1 [Transform System Overview](#41-transform-system-overview)
   - 4.2 [Transform System Components](#42-transform-system-components)
   - 4.3 [JMESPath Basics](#43-jmespath-basics)
   - 4.4 [Deconstructing @inject_adapter_id](#44-deconstructing-inject_adapter_id)
   - 4.5 [Creating Custom Transforms](#45-creating-custom-transforms)

5. [LoRA Adapter Support](#5-lora-adapter-support)
   - 5.1 [LoRA Overview](#51-lora-overview)
   - 5.2 [Adapter ID Injection](#52-adapter-id-injection)
   - 5.3 [Load/Unload Adapter Handlers](#53-loadunload-adapter-handlers)
   - 5.4 [Complete LoRA Example](#54-complete-lora-example)

6. [Session Management](#6-session-management)
   - 6.1 [Session Overview](#61-session-overview)
   - 6.2 [Session Environment Variables](#62-session-environment-variables)
   - 6.3 [Session Validation Behavior](#63-session-validation-behavior)
   - 6.4 [Using @stateful_session_manager](#64-using-stateful_session_manager)
   - 6.5 [Session Request Types and Headers](#65-session-request-types-and-headers)
   - 6.6 [Complete Session Example](#66-complete-session-example)

7. [Supervisor Process Management](#7-supervisor-process-management)
   - 7.1 [Supervisor Overview](#71-supervisor-overview)
   - 7.2 [Using standard-supervisor](#72-using-standard-supervisor)

8. [Customer Customization Patterns](#8-customer-customization-patterns)
   - 8.1 [Customer Override Methods](#81-customer-override-methods)
   - 8.2 [Priority Resolution Examples](#82-priority-resolution-examples)
   - 8.3 [Framework Developer Guidance](#83-framework-developer-guidance)

9. [SGLang Integration Example](#9-sglang-integration-example)
   - 9.1 [SGLang Architecture](#91-sglang-architecture)
   - 9.2 [SGLang Integration Example](#92-sglang-integration-example)
   - 9.3 [SGLang-Specific Considerations](#93-sglang-specific-considerations)

10. [Configuration Reference](#10-configuration-reference)
    - 10.1 [Environment Variables Tables](#101-environment-variables-tables)
    - 10.2 [Handler Priority Resolution](#102-handler-priority-resolution)

11. [Testing & Validation](#11-testing--validation)
    - 11.1 [Local Testing Guide](#111-local-testing-guide)
    - 11.2 [LoRA Adapter Testing](#112-lora-adapter-testing)
    - 11.3 [Customer Override Testing](#113-customer-override-testing)
    - 11.4 [Integration Test Examples](#114-integration-test-examples)

12. [Troubleshooting](#12-troubleshooting)
    - 12.1 [Common Issues](#121-common-issues)
    - 12.2 [Debug Logging](#122-debug-logging)
    - 12.3 [Patterns and Anti-Patterns](#123-patterns-and-anti-patterns)

13. [API Reference](#13-api-reference)
    - 13.1 [Core Decorators](#131-core-decorators)
    - 13.2 [LoRA Decorators](#132-lora-decorators)
    - 13.3 [Session Decorator and Bootstrap Function](#133-session-decorator-and-bootstrap-function)
    - 13.4 [Parameter Reference Tables](#134-parameter-reference-tables)

14. [Additional Resources](#14-additional-resources)
    - 14.1 [Documentation Links](#141-documentation-links)
    - 14.2 [Example Code](#142-example-code)
    - 14.3 [Getting Help](#143-getting-help)

15. [Appendix: Complete Example Templates](#15-appendix-complete-example-templates)
    - 15.1 [Minimal Integration Template](#151-minimal-integration-template)
    - 15.2 [Full-Featured Integration Template](#152-full-featured-integration-template)
    - 15.3 [SGLang Integration Template](#153-sglang-integration-template)

---

## 1. Introduction

### 1.1 What is MHCS?

Model Hosting Container Standards (MHCS) is a Python library that standardizes how ML frameworks integrate with Amazon SageMaker. It provides a unified approach to implementing the required SageMaker endpoints (`/ping` and `/invocations`) while adding powerful features like LoRA adapter management, stateful sessions, and customer customization.

**Key Benefits:**

- **Unified Handler System**: Consistent endpoint implementation across ML frameworks (vLLM, SGLang, TensorRT-LLM)
- **LoRA Adapter Support**: Built-in decorators for dynamic adapter loading, unloading, and request-level injection
- **Stateful Sessions**: File-based session management with automatic expiration
- **Customer Customization**: Multi-level override system allowing end-users to customize framework behavior without code changes
- **Production Ready**: Comprehensive logging, error handling, and process supervision
- **Framework Agnostic**: Works with any FastAPI-based ML serving framework

### 1.2 Prerequisites

**Required:**
- Python >= 3.10
- FastAPI >= 0.117.1
- An existing FastAPI application (your ML framework's serving layer)

**Installation:**

```bash
# Install with Poetry (recommended for development)
cd python
poetry install

# Or install from wheel (for production)
pip install model_hosting_container_standards-*.whl
```

**FastAPI Fundamentals:**

Understanding these FastAPI concepts will help with MHCS integration:

- **Handlers vs Routes**: A handler is a Python function that processes requests. A route maps a URL path and HTTP method to a handler (e.g., `GET /ping` → `ping_handler`).

- **Routers**: FastAPI routers (`APIRouter`) organize related endpoints into modules. Routers are included in the main app using `app.include_router(router)`. MHCS uses routers internally to organize SageMaker endpoints.

- **Request/Response Objects**: `Request` contains headers, body, and query params. `Response` allows custom status codes and headers. Handlers can return `Response` objects, dictionaries (converted to JSON), or Pydantic models.

### 1.3 MHCS Integration Pattern

MHCS follows a decorator-based integration pattern that works with existing FastAPI applications:

**Step 1: Decorate Your Handlers**

Apply MHCS decorators to register your framework handlers:

```python
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request, Response

@sagemaker_standards.register_ping_handler
async def ping(request: Request) -> Response:
    return Response(status_code=200, content="OK")

@sagemaker_standards.register_invocation_handler
async def invocations(request: Request) -> dict:
    body = await request.json()
    # Your framework's inference logic here
    return {"predictions": ["result"]}
```

**Step 2: Call bootstrap()**

After defining all routes and handlers, call `bootstrap(app)` to connect everything:

```python
from fastapi import FastAPI
import model_hosting_container_standards.sagemaker as sagemaker_standards

app = FastAPI(title="My ML Framework")

# ... define all your routes and handlers ...

# Bootstrap MHCS - must be called after route definitions
sagemaker_standards.bootstrap(app)
```

**What bootstrap() Does:**

The `bootstrap(app)` function performs critical setup:

1. **Registers Customer Overrides**: Scans for customer-provided handlers (via environment variables, custom scripts, or decorators) with higher priority than framework defaults
2. **Creates SageMaker Routes**: Automatically creates required endpoints:
   - `GET /ping` - Health check
   - `POST /invocations` - Model inference
   - `POST /adapters` - LoRA adapter registration (if LoRA handlers defined)
   - `DELETE /adapters/{adapter_name}` - LoRA adapter unregistration (if LoRA handlers defined)
3. **Mounts the Router**: Includes the SageMaker router in your FastAPI app
4. **Loads Middlewares**: Configures custom middlewares from environment variables or decorators

**Step 3: Start Your Server**

Your framework's existing server startup remains unchanged:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Request Flow After Integration:**

```
Client Request
    ↓
FastAPI App
    ↓
MHCS Router
    ↓
Your Framework Handler
    ↓
Response
```

### 1.4 How to Use This Guide

This guide is structured for progressive learning:

**For Quick Validation (5-10 minutes):**
- Start with [Section 2: Quick Start](#2-quick-start)
- Choose the use case matching your needs
- Copy the example code and test immediately

**For Complete Integration:**
1. Read [Section 3: Core Integration](#3-core-integration) for fundamentals
2. Follow the integration checklist step-by-step
3. Add features as needed:
   - [Section 5: LoRA Adapter Support](#5-lora-adapter-support) for multi-adapter scenarios
   - [Section 6: Session Management](#6-session-management) for conversational AI
   - [Section 8: Customer Customization Patterns](#8-customer-customization-patterns) for extensibility

**For Advanced Features:**
- [Section 4: Transform Decorators](#4-transform-decorators) - Transform system deep dive
- [Section 7: Supervisor Process Management](#7-supervisor-process-management) - Production reliability
- [Section 9: SGLang Integration Example](#9-sglang-integration-example) - Real framework example

**For Reference:**
- [Section 10: Configuration Reference](#10-configuration-reference) - Environment variables
- [Section 13: API Reference](#13-api-reference) - Complete decorator documentation
- [Section 12: Troubleshooting](#12-troubleshooting) - Common issues and patterns
- [Section 15: Appendix](#15-appendix-complete-example-templates) - Copy-paste templates

## 2. Quick Start

This section provides four self-contained examples that demonstrate common MHCS integration patterns. Each example can be copied and tested immediately to validate MHCS compatibility before diving into comprehensive integration.

### 2.1 Use Case 1: Basic SageMaker Compatibility

**What you'll build**: A minimal FastAPI application with SageMaker-compatible `/ping` and `/invocations` endpoints. This is the foundation for any MHCS integration.

**Why it matters**: SageMaker requires these two endpoints for health checks and model inference. This example gets you to a working integration in under 5 minutes.

**Code** (`basic_server.py`):

```python
from fastapi import FastAPI, Request, Response
import model_hosting_container_standards.sagemaker as sagemaker_standards

app = FastAPI(title="Basic ML Framework")

@sagemaker_standards.register_ping_handler
async def ping(request: Request) -> Response:
    """Health check endpoint for SageMaker."""
    return Response(status_code=200, content="Healthy")

@sagemaker_standards.register_invocation_handler
async def invocations(request: Request) -> dict:
    """Model inference endpoint for SageMaker."""
    body = await request.json()
    prompt = body.get("prompt", "")
    
    # Your framework's inference logic here
    result = f"Processed: {prompt}"
    
    return {"predictions": [result]}

# Bootstrap MHCS - must be called after handler definitions
sagemaker_standards.bootstrap(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**How it works**:
- `@register_ping_handler` - Registers your ping handler as the framework default
- `@register_invocation_handler` - Registers your invocation handler as the framework default
- `bootstrap(app)` - Creates SageMaker routes (`GET /ping`, `POST /invocations`) and connects them to your handlers

**Test it**:

```bash
# Start the server
python basic_server.py

# In another terminal, test the ping endpoint
curl http://localhost:8000/ping
```

**Expected Output:**
```
Healthy
```

```bash
# Test the invocations endpoint
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world"}'
```

**Expected Output:**
```json
{"predictions": ["Processed: Hello world"]}
```

**Next steps**: See [Section 3: Core Integration](#3-core-integration) for detailed explanation of handlers and bootstrap.

---

### 2.2 Use Case 2: Multi-LoRA Support

**What you'll build**: A FastAPI application that supports dynamic LoRA adapter loading and automatic adapter ID injection from SageMaker headers.

**Why it matters**: MHCS provides decorators that: (1) automatically inject adapter IDs from SageMaker headers into your inference requests, and (2) transform request/response shapes when dynamically loading and unloading adapters, adapting between SageMaker's API format and your framework's specific structure.

**Code** (`lora_server.py`):

```python
from fastapi import FastAPI, Request, Response
import model_hosting_container_standards.sagemaker as sagemaker_standards

app = FastAPI(title="LoRA-Enabled ML Framework")

# Simulated adapter storage
loaded_adapters = {}

@sagemaker_standards.register_ping_handler
async def ping(request: Request) -> Response:
    return Response(status_code=200, content="Healthy")

@sagemaker_standards.register_invocation_handler
@sagemaker_standards.inject_adapter_id("model")  # Injects adapter ID into "model" field
async def invocations(request: Request) -> dict:
    """Inference with automatic adapter ID injection."""
    body = await request.json()
    prompt = body.get("prompt", "")
    adapter_id = body.get("model", "base-model")  # Injected by decorator
    
    # Your framework's inference logic with adapter
    result = f"[{adapter_id}] Processed: {prompt}"
    
    return {"predictions": [result], "adapter_used": adapter_id}

@sagemaker_standards.register_load_adapter_handler(
    request_shape={"adapter_name": "body.name", "adapter_path": "body.src"},
    response_shape={}
)
async def load_adapter(request: Request) -> dict:
    """Load a LoRA adapter."""
    body = await request.json()
    adapter_name = body["adapter_name"]
    adapter_path = body.get("adapter_path", "")
    
    # Your framework's adapter loading logic
    loaded_adapters[adapter_name] = {"path": adapter_path, "loaded": True}
    
    return {"status": "success", "adapter_name": adapter_name}

@sagemaker_standards.register_unload_adapter_handler(
    request_shape={"adapter_name": "path_params.adapter_name"},
    response_shape={}
)
async def unload_adapter(request: Request) -> dict:
    """Unload a LoRA adapter."""
    adapter_name = request.path_params.get("adapter_name")
    
    # Your framework's adapter unloading logic
    if adapter_name in loaded_adapters:
        del loaded_adapters[adapter_name]
        return {"status": "success", "adapter_name": adapter_name}
    
    return {"status": "not_found", "adapter_name": adapter_name}

sagemaker_standards.bootstrap(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**How it works**:
- The LoRA decorators use the transform decorator system under the hood (see [Section 4: Transform Decorators](#4-transform-decorators) for details).
- `@inject_adapter_id("model")` - Extracts adapter ID from `X-Amzn-SageMaker-Adapter-Identifier` header and injects it into the `model` field of the request body. 
- `@register_load_adapter_handler` - Creates `POST /adapters` endpoint for loading adapters
- `@register_unload_adapter_handler` - Creates `DELETE /adapters/{adapter_name}` endpoint for unloading adapters

**Test it**:

```bash
# Start the server
python lora_server.py

# Load an adapter
curl -X POST http://localhost:8000/adapters \
  -H "Content-Type: application/json" \
  -d '{"adapter_name": "my-adapter", "adapter_path": "/tmp/adapter"}'
```

**Expected Output:**
```json
{"status": "success", "adapter_name": "my-adapter"}
```

```bash
# Send inference request with adapter header
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -H "X-Amzn-SageMaker-Adapter-Identifier: my-adapter" \
  -d '{"prompt": "Hello with adapter"}'
```

**Expected Output:**
```json
{"predictions": ["[my-adapter] Processed: Hello with adapter"], "adapter_used": "my-adapter"}
```

```bash
# Unload the adapter
curl -X DELETE http://localhost:8000/adapters/my-adapter
```

**Expected Output:**
```json
{"status": "success", "adapter_name": "my-adapter"}
```

**Next steps**: See [Section 5: LoRA Adapter Support](#5-lora-adapter-support) for complete LoRA integration details and [Section 4: Transform Decorators](#4-transform-decorators) to understand how these decorators work under the hood.

---

### 2.3 Use Case 3: Sticky Session Support

**What you'll build**: A FastAPI application with stateful session management for conversational AI, where each session maintains context across multiple requests.

**Why it matters**: MHCS provides file-based session storage with automatic expiration.

**Code** (`session_server.py`):

```python
from fastapi import FastAPI, Request, Response
import model_hosting_container_standards.sagemaker as sagemaker_standards

app = FastAPI(title="Session-Enabled ML Framework")

@sagemaker_standards.register_ping_handler
async def ping(request: Request) -> Response:
    return Response(status_code=200, content="Healthy")

@sagemaker_standards.register_invocation_handler
@sagemaker_standards.stateful_session_manager()
async def invocations(request: Request) -> dict:
    """Inference with session management."""
    body = await request.json()
    prompt = body.get("prompt", "")
    
    # Access session data if available
    session_id = request.headers.get("X-Amzn-SageMaker-Session-Id")
    
    # Your framework's inference logic with session context
    if session_id:
        result = f"[Session {session_id}] Processed: {prompt}"
    else:
        result = f"Processed: {prompt}"
    
    return {"predictions": [result]}

sagemaker_standards.bootstrap(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Environment Configuration**:

Sessions are disabled by default. Enable them with environment variables:

```bash
export SAGEMAKER_ENABLE_STATEFUL_SESSIONS=true
export SAGEMAKER_SESSIONS_EXPIRATION=1800  # 30 minutes (default: 1200)
export SAGEMAKER_SESSIONS_PATH=/tmp/sessions  # Optional custom path
```

**How it works**:
- `@stateful_session_manager()` - Enables session management for the handler
- Sessions are created by sending `{"requestType": "NEW_SESSION"}` in the request body
- Session IDs are passed via `X-Amzn-SageMaker-Session-Id` header
- Sessions are closed by sending `{"requestType": "CLOSE"}` in the request body

**Test it**:

```bash
# Start the server with sessions enabled
export SAGEMAKER_ENABLE_STATEFUL_SESSIONS=true
python session_server.py

# Create a new session
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"requestType": "NEW_SESSION"}'
```

**Expected Output:**
```json
{}
```

**Response Headers:**
```
X-Amzn-SageMaker-New-Session-Id: <session-id>
```

```bash
# Use the session (replace <session-id> with actual ID from previous response)
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -H "X-Amzn-SageMaker-Session-Id: <session-id>" \
  -d '{"prompt": "Hello in session"}'
```

**Expected Output:**
```json
{"predictions": ["[Session <session-id>] Processed: Hello in session"]}
```

```bash
# Close the session
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -H "X-Amzn-SageMaker-Session-Id: <session-id>" \
  -d '{"requestType": "CLOSE"}'
```

**Expected Output:**
```json
{}
```

**Response Headers:**
```
X-Amzn-SageMaker-Closed-Session-Id: <session-id>
```

**Session Request Types and Headers**:

| Request Type | Description | Request Body | Response Header |
|-------------|-------------|--------------|-----------------|
| `NEW_SESSION` | Create new session | `{"requestType": "NEW_SESSION"}` | `X-Amzn-SageMaker-New-Session-Id` |
| `CLOSE` | Close existing session | `{"requestType": "CLOSE"}` | `X-Amzn-SageMaker-Closed-Session-Id` |
| Regular request | Use existing session | `{"prompt": "..."}` | None |

**Session Headers**:

| Header | Direction | Description |
|--------|-----------|-------------|
| `X-Amzn-SageMaker-Session-Id` | Client → Server | Pass existing session ID to server |
| `X-Amzn-SageMaker-New-Session-Id` | Server → Client | Returned when creating new session (includes expiration) |
| `X-Amzn-SageMaker-Closed-Session-Id` | Server → Client | Returned when closing session |

**Next steps**: See [Section 6: Session Management](#6-session-management) for complete session configuration and advanced usage.

---

### 2.4 Use Case 4: Runtime Custom Code Injection

**What you'll build**: A framework integration that allows end-users to override your default handlers without modifying framework code.

**Why it matters**: Framework developers provide defaults, but customers need flexibility to customize behavior. MHCS provides a priority system where customer overrides take precedence over framework defaults.

**Code** (`extensible_server.py`):

```python
from fastapi import FastAPI, Request, Response
import model_hosting_container_standards.sagemaker as sagemaker_standards

app = FastAPI(title="Extensible ML Framework")

# Framework provides defaults using @register_* decorators
@sagemaker_standards.register_ping_handler
async def framework_ping(request: Request) -> Response:
    """Framework's default ping handler."""
    return Response(
        status_code=200,
        content='{"status": "healthy", "source": "framework_default"}'
    )

@sagemaker_standards.register_invocation_handler
async def framework_invocations(request: Request) -> dict:
    """Framework's default invocation handler."""
    body = await request.json()
    return {
        "predictions": ["Framework default response"],
        "source": "framework_default"
    }

sagemaker_standards.bootstrap(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Customer Override Example 1: Environment Variable**

Create a custom script (`/opt/ml/model/model.py`):

```python
from fastapi import Request

async def custom_ping_handler(request: Request):
    """Customer's custom ping handler."""
    return {"status": "healthy", "source": "customer_env_var"}

async def custom_invocation_handler(request: Request):
    """Customer's custom invocation handler."""
    body = await request.json()
    return {
        "predictions": ["Customer override response"],
        "source": "customer_env_var"
    }
```

Set environment variables to point to custom handlers:

```bash
export SAGEMAKER_MODEL_PATH=/opt/ml/model/
export CUSTOM_SCRIPT_FILENAME=model.py
export CUSTOM_FASTAPI_PING_HANDLER=model.py:custom_ping_handler
export CUSTOM_FASTAPI_INVOCATION_HANDLER=model.py:custom_invocation_handler
```

**Customer Override Example 2: Custom Decorator**

Customers can use `@custom_*` decorators in their own code:

```python
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request

@sagemaker_standards.custom_ping_handler
async def my_custom_ping(request: Request):
    """Customer's custom ping using decorator."""
    return {"status": "healthy", "source": "customer_decorator"}

@sagemaker_standards.custom_invocation_handler
async def my_custom_invocations(request: Request):
    """Customer's custom invocation using decorator."""
    body = await request.json()
    return {
        "predictions": ["Customer decorator response"],
        "source": "customer_decorator"
    }
```

**Handler Priority Resolution**:

MHCS resolves handlers in this priority order (highest to lowest):

1. **Environment variable** (`CUSTOM_FASTAPI_PING_HANDLER`) - Highest priority
2. **Custom decorator** (`@custom_ping_handler`) - Customer override
3. **Script function** (function in `model.py` with specific name)
4. **Register decorator** (`@register_ping_handler`) - Framework default (lowest priority)

**How it works**:
- Framework developers use `@register_*` decorators for defaults
- Customers can override using environment variables or `@custom_*` decorators
- `bootstrap(app)` resolves the priority and uses the highest-priority handler
- Framework handlers are never "final" - customers can always override

**Test it**:

```bash
# Test with framework defaults (no overrides)
python extensible_server.py

curl http://localhost:8000/ping
# Output: {"status": "healthy", "source": "framework_default"}

# Test with environment variable override
export CUSTOM_FASTAPI_PING_HANDLER=model.py:custom_ping_handler
python extensible_server.py

curl http://localhost:8000/ping
# Output: {"status": "healthy", "source": "customer_env_var"}
```

**Next steps**: See [Section 8: Customer Customization Patterns](#8-customer-customization-patterns) for complete customization guidance and [Section 3.3: Handler Priority System](#33-handler-priority-system) for detailed priority resolution.

---
**What's Next?**

- **For complete integration**: Continue to [Section 3: Core Integration](#3-core-integration)
- **For LoRA deep dive**: Jump to [Section 5: LoRA Adapter Support](#5-lora-adapter-support)
- **For session details**: Jump to [Section 6: Session Management](#6-session-management)
- **For production deployment**: See [Section 7: Supervisor Process Management](#7-supervisor-process-management)

## 3. Core Integration

[Content to be added in subsequent tasks]

## 4. Transform Decorators

[Content to be added in subsequent tasks]

## 5. LoRA Adapter Support

[Content to be added in subsequent tasks]

## 6. Session Management

[Content to be added in subsequent tasks]

## 7. Supervisor Process Management

[Content to be added in subsequent tasks]

## 8. Customer Customization Patterns

[Content to be added in subsequent tasks]

## 9. SGLang Integration Example

[Content to be added in subsequent tasks]

## 10. Configuration Reference

[Content to be added in subsequent tasks]

## 11. Testing & Validation

[Content to be added in subsequent tasks]

## 12. Troubleshooting

[Content to be added in subsequent tasks]

## 13. API Reference

[Content to be added in subsequent tasks]

## 14. Additional Resources

[Content to be added in subsequent tasks]

## 15. Appendix: Complete Example Templates

[Content to be added in subsequent tasks]

---

**Document Status**: Initial structure created. Content will be added incrementally following the implementation plan.
