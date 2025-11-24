# MHCS Client Library Integration Runbook - OUTLINE

**Purpose**: Guide ML framework developers (like Asimov team for SGLang) on integrating their frameworks with Model Hosting Container Standards (MHCS) for SageMaker deployment.

**Target Audience**: Framework developers integrating vLLM, SGLang, TensorRT-LLM, or similar frameworks

---

## 1. Introduction

### 1.1 What is MHCS?
- Purpose: Standardize model hosting container implementations for SageMaker
- Benefits: Unified handlers, LoRA support, session management, customer customization
- Architecture overview (FastAPI-based)
  - FastAPI fundamentals: handlers vs routes, app lifetimes, routers
  - MHCS integration pattern with existing FastAPI apps
  - How bootstrap() connects framework handlers to SageMaker routes

### 1.2 Prerequisites
- Python >= 3.10
- FastAPI-based ML framework
- Installation: `pip install model-hosting-container-standards`

### 1.3 How to Use This Guide
- Quick Start (5 min) vs Comprehensive sections
- Code examples are framework-agnostic
- SGLang-specific section included

---

## 2. Quick Start - 4 Integration Use Cases

### 2.1 Use Case 1: Basic SageMaker Compatibility
#### 2.1.A Ping and Invocation Endpoints
- Import MHCS and add `@register_ping_handler`, `@register_invocation_handler`
- Call `bootstrap(app)` after route definitions
- Test with `/ping` and `/invocations` endpoints
- 5-minute minimal integration example

### 2.2 Use Case 2: SageMaker Multi-LoRA Support
#### 2.2.A Making Existing LoRA Handlers SageMaker API Compatible
- Add `@inject_adapter_id` decorator to invocation handlers
- Implement `@register_load_adapter_handler` and `@register_unload_adapter_handler`
- Configure adapter ID injection from SageMaker headers
- Test adapter loading/unloading via SageMaker API

### 2.3 Use Case 3: SageMaker Sticky Session Support
- Enable sessions via `SAGEMAKER_ENABLE_STATEFUL_SESSIONS=true`
- Add `@stateful_session_manager` decorator to handlers
- Configure session storage and expiration
- Test session creation, retrieval, and cleanup

### 2.4 Use Case 4: Runtime Custom Code Injection
- Configure `SAGEMAKER_MODEL_PATH` for custom scripts
- Implement customer override patterns via environment variables
- Set up handler priority resolution (env vars > custom decorators > register decorators)
- Test customer customization paths

---

## 3. Core Integration (Step-by-Step)

### 3.1 Understanding the Bootstrap Function
- What `bootstrap(app)` does
- Why it must be called after route definition
- SageMaker routes it sets up

### 3.2 Handler Registration Pattern
- `@register_ping_handler` - Framework default for health checks
- `@register_invocation_handler` - Framework default for inference
- Request/Response flow through handlers

### 3.3 Handler Priority System
- Priority resolution order (env vars > custom decorators > script functions > register decorators)
- Example: How customers can override your defaults
- Why this matters for framework developers

### 3.4 Integration Checklist
- [ ] Import MHCS
- [ ] Decorate ping handler
- [ ] Decorate invocation handler
- [ ] Call bootstrap(app)
- [ ] Test with curl/httpx

### 3.5 Complete Integration Example
```python
# Full example showing FastAPI app structure with MHCS
# Including router setup, handler definitions, bootstrap call
```

---

## 4. Transform Decorators & Transform Classes

### 4.1 Overview
- What are transforms? Request/response transformation system
- How transforms enable features like LoRA injection
- Architecture: Base transform classes and decorator factories

### 4.2 Transform System Components
- `BaseApiTransform` - Abstract base class for request/response transformations
- `request_shape` - Dictionary defining JMESPath expressions for request transformation
- `response_shape` - Dictionary defining JMESPath expressions for response transformation
- JMESPath expressions for data extraction and manipulation
- `create_transform_decorator` - Decorator factory for creating transform decorators

### 4.3 How Transform Decorators Work
- Decorator factory pattern
- Wrapping handlers with transformation logic
- Request flow: Raw request → Transform → Handler → Transform → Response

### 4.4 JMESPath Basics for Transforms
- Selecting nested fields (e.g., `"model"`, `"request.adapter_id"`)
- Array operations
- Common patterns used in MHCS

### 4.5 Example: Understanding @inject_adapter_id
```python
# How @inject_adapter_id uses transforms under the hood
# Input transform: Extract from header, inject into body
# JMESPath expression examples
```

### 4.6 Creating Custom Transforms (Advanced)
- Extending `BaseApiTransform` abstract class
- Implementing `transform_request()` and `transform_response()` methods
- Using `create_transform_decorator()` to create custom decorators
- Use cases for custom transforms

### 4.7 Transform Reference
- Built-in transform decorators
- Transform configuration options
- Reference to `BASE_FACTORY_USAGE.md`

---

## 5. LoRA Adapter Support

### 5.1 Overview
- What LoRA adapters are
- Why SageMaker needs adapter management
- MHCS's LoRA capabilities (built on transform system)

### 5.2 Adapter ID Injection
- `@inject_adapter_id` decorator
- Replace mode vs Append mode
- Header: `X-Amzn-SageMaker-Adapter-Identifier`
- JMESPath expressions for request transformation

### 5.3 Example: Adding LoRA to Invocation Handler
```python
# Before: Basic invocation handler
# After: With @inject_adapter_id("model")
```

### 5.4 Load/Unload Adapter Handlers
- `@register_load_adapter_handler` for `POST /adapters`
- `@register_unload_adapter_handler` for `DELETE /adapters/{adapter_name}`
- Request/Response schemas
- Integration with framework's adapter system

### 5.5 Complete LoRA Example
```python
# Full example showing:
# - inject_adapter_id on invocations
# - load_adapter_handler implementation
# - unload_adapter_handler implementation
```

### 5.6 Testing LoRA Integration
```bash
# curl commands for loading/unloading adapters
# Testing adapter injection
```

---

## 6. Session Management

### 6.1 Overview
- Stateful sessions for conversational AI
- File-based key-value storage
- Automatic expiration and cleanup
- **NEW**: Toggle sessions on/off via environment variables

### 6.2 Enabling Sessions via Environment Variables
- `SAGEMAKER_ENABLE_STATEFUL_SESSIONS` - Enable/disable sessions (default: `false`)
- `SAGEMAKER_SESSIONS_EXPIRATION` - Session lifetime in seconds (default: `1200` / 20 minutes)
- `SAGEMAKER_SESSIONS_PATH` - Custom storage path (default: `/dev/shm/sagemaker_sessions` or temp directory)
- How the global session manager is initialized from config

### 6.3 SageMakerConfig Model
- Pydantic configuration model that loads from environment variables
- Automatic SAGEMAKER_* prefix detection
- Type validation and conversion (bool, int)
- Using `SageMakerConfig.from_env()` in your code

### 6.4 Session Validation Behavior
- What happens when sessions are disabled but session headers are present
- 400 BAD_REQUEST error with clear message
- Preventing invalid session requests

### 6.5 Using @stateful_session_manager Decorator
- How to apply to handlers
- Session creation, retrieval, closing
- Session storage location
- Decorator behavior when sessions are disabled

### 6.6 Session API Routes
- `POST /sessions` - Create session
- `GET /sessions/{session_id}` - Retrieve session
- `DELETE /sessions/{session_id}` - Close session
- Headers: `X-Amzn-SageMaker-Session-Id`, `X-Amzn-SageMaker-New-Session-Id`, `X-Amzn-SageMaker-Closed-Session-Id`

### 6.7 Example: Enabling Sessions
```bash
# Environment variable configuration
export SAGEMAKER_ENABLE_STATEFUL_SESSIONS=true
export SAGEMAKER_SESSIONS_EXPIRATION=1800  # 30 minutes
export SAGEMAKER_SESSIONS_PATH=/custom/path
```

### 6.8 Example: Session-Enabled Handler
```python
# Handler with @stateful_session_manager decorator
# Accessing session data in requests
# Handling session_id from headers
```

### 6.9 **[TODO] Engine-Specific Session Integration**
- Placeholder for upcoming feature
- Hook into framework's native create/close session APIs
- Integration pattern (TBD)

---

## 7. Supervisor Process Management

### 7.1 Overview
- `standard-supervisor` CLI for production reliability
- Automatic crash recovery
- Configurable retry limits

### 7.2 Using standard-supervisor
```bash
# Command to start your server with supervisor
# Configuration options
```

### 7.3 Configuration
- Retry limits
- Logging
- Error handling

### 7.4 Integration with Your Framework
- Where to add supervisor in startup scripts
- Production deployment considerations

---

## 8. Customer Customization Patterns

### 8.1 How Customers Override Handlers
- Environment variables
- `@custom_ping_handler` / `@custom_invocation_handler`
- Custom script functions (model.py)

### 8.2 Priority Resolution Examples
```python
# Example 1: Customer overrides via env var
# Example 2: Customer overrides via custom decorator
# Example 3: Custom script function
```

### 8.3 Why This Matters for Framework Developers
- Your handlers are defaults, not final
- Design for extensibility
- Testing customer overrides

---

## 9. SGLang Integration Example

### 9.1 SGLang Server Structure
- Overview of SGLang's FastAPI structure (if documented)
- Where handlers are defined
- Routing patterns

### 9.2 Applying MHCS to SGLang
```python
# Concrete SGLang example showing:
# - Importing MHCS
# - Decorating SGLang's ping/inference handlers
# - LoRA support for SGLang
# - Bootstrap call
```

### 9.3 SGLang-Specific Considerations
- Any SGLang-specific patterns
- Testing with SGLang models

---

## 10. Configuration Reference

### 10.1 Environment Variables

**Handler Override Configuration:**
| Variable | Description | Default |
|----------|-------------|---------|
| `CUSTOM_FASTAPI_PING_HANDLER` | Override ping handler | None |
| `CUSTOM_FASTAPI_INVOCATION_HANDLER` | Override invocation handler | None |
| `SAGEMAKER_MODEL_PATH` | Path to custom scripts | `/opt/ml/model/` |
| `CUSTOM_SCRIPT_FILENAME` | Custom script filename | `model.py` |
| `SAGEMAKER_CONTAINER_LOG_LEVEL` | Logging level | `INFO` |

**Session Management Configuration:**
| Variable | Description | Default |
|----------|-------------|---------|
| `SAGEMAKER_ENABLE_STATEFUL_SESSIONS` | Enable stateful sessions | `false` |
| `SAGEMAKER_SESSIONS_EXPIRATION` | Session lifetime in seconds | `1200` (20 min) |
| `SAGEMAKER_SESSIONS_PATH` | Custom session storage path | `/dev/shm/sagemaker_sessions` or temp |

### 10.2 Handler Priority Resolution
1. Environment variables (highest priority)
2. Custom decorators
3. Script functions
4. Register decorators (framework defaults)

---

## 11. Testing & Validation

### 11.1 Local Testing
```bash
# Starting your server locally
# Basic health check
# Inference request
```

### 11.2 Testing LoRA Adapters
```bash
# Load adapter
# Send request with adapter header
# Unload adapter
```

### 11.3 Testing Customer Overrides
```python
# Creating a test model.py script
# Testing priority resolution
```

### 11.4 Integration Test Examples
- Reference to `tests/resources/mock_vllm_server.py`
- How to structure tests for your framework

---

## 12. Troubleshooting

### 12.1 Common Issues

**Handler not being called**
- Check bootstrap() is called after routes
- Verify decorator placement
- Check handler priority resolution

**LoRA adapter not injected**
- Verify `@inject_adapter_id` decorator
- Check header name matches
- Validate JMESPath expression

**Import errors**
- Verify installation
- Check Python version compatibility

**Customer overrides not working**
- Check environment variable names
- Verify SAGEMAKER_MODEL_PATH
- Check script function naming

### 12.2 Debug Logging
```python
# Enabling DEBUG log level
# What to look for in logs
```

### 12.3 Common Patterns and Anti-Patterns
- ✅ DO: Call bootstrap() last
- ❌ DON'T: Hardcode customer logic in framework defaults
- ✅ DO: Use @register_* decorators for framework defaults
- ❌ DON'T: Use @custom_* decorators in framework code

---

## 13. API Reference

### 13.1 Core Decorators
- `@register_ping_handler`
- `@register_invocation_handler`
- `@custom_ping_handler`
- `@custom_invocation_handler`

### 13.2 LoRA Decorators
- `@inject_adapter_id(field_path, mode="replace")`
- `@register_load_adapter_handler`
- `@register_unload_adapter_handler`

### 13.3 Session Decorator
- `@stateful_session_manager()`

### 13.4 Bootstrap Function
```python
def bootstrap(app: FastAPI, **kwargs) -> None:
    """
    Sets up SageMaker routes and middleware
    Must be called after all routes are defined
    """
```

### 13.5 Parameter Reference
- Decorator parameters
- Return types
- Request/Response schemas

---

## 14. Best Practices

### 14.1 Error Handling
- Proper exception handling in handlers
- Returning appropriate HTTP status codes
- Logging errors

### 14.2 Performance Considerations
- Handler performance implications
- Async/await patterns
- Resource management

### 14.3 Production Deployment
- Using supervisor for reliability
- Logging configuration
- Monitoring and health checks

### 14.4 Testing Your Integration
- Unit tests for handlers
- Integration tests with MHCS
- Testing customer customization paths

---

## 15. Additional Resources

### 15.1 Documentation Links
- Python README: `python/README.md`
- LoRA documentation: `python/model_hosting_container_standards/sagemaker/lora/`
- Session documentation: `python/model_hosting_container_standards/sagemaker/sessions/`
- Supervisor documentation: `python/model_hosting_container_standards/supervisor/`
- Transform documentation: `python/model_hosting_container_standards/common/transforms/BASE_FACTORY_USAGE.md`

### 15.2 Example Code
- Mock vLLM server: `tests/resources/mock_vllm_server.py`
- Integration tests: `tests/integration/`

### 15.3 Getting Help
- GitHub issues
- Contributing guidelines

---

## Appendix A: Complete Example Templates

### A.1 Minimal Integration Template
```python
# Complete minimal example
```

### A.2 Full-Featured Integration Template
```python
# Complete example with LoRA, sessions, error handling
```

### A.3 SGLang Integration Template
```python
# Complete SGLang-specific example
```

---

## Document Metadata

**Status**: OUTLINE - Pending Review
**Next Steps**:
1. Review outline with team
2. Get feedback on structure and content coverage
3. Write full runbook based on approved outline

**Questions for Review**:
- Is the structure clear and logical?
- Are there any missing sections?
- Is the SGLang coverage adequate?
- Should any sections be expanded or condensed?
