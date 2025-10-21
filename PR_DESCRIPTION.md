# Add @register_ping_handler and @register_invocation_handler Auto-Routing Support

## Summary

This PR implements automatic route registration for `@register_ping_handler` and `@register_invocation_handler` decorators, simplifying the framework integration for server code like vLLM.

## Key Changes

### 1. Enhanced Register Decorators
- **Before**: `@register_ping_handler` only registered handlers in the registry
- **After**: `@register_ping_handler` automatically registers handlers AND sets up `/ping` routes
- **Before**: Manual route setup required: `@router.get("/ping")` + `@router.post("/ping")`
- **After**: Single decorator: `@register_ping_handler`

### 2. Updated Mock vLLM Server
- Migrated from manual route decorators to `@register_ping_handler` and `@register_invocation_handler`
- Added `@inject_adapter_id` decorator for LoRA adapter support
- Now accurately simulates real vLLM server behavior

### 3. Comprehensive Test Coverage
- **Removed duplicate tests**: Reduced from 13 to 9 tests by eliminating redundant test cases
- **Added LoRA integration test**: Validates `@inject_adapter_id` decorator functionality
- **Maintained full coverage**: All existing functionality remains tested

## Usage Examples

### Server Code (e.g., vLLM)
```python
# Before
@router.get("/ping", response_class=Response)
@router.post("/ping", response_class=Response)
async def ping(raw_request: Request) -> Response:
    return await health(raw_request)

# After
@register_ping_handler
async def ping(raw_request: Request) -> Response:
    return await health(raw_request)
```

### Customer Scripts (unchanged)
```python
# Customers continue using regular functions or @ping/@invoke decorators
async def ping():
    return {"status": "healthy"}

# OR
@ping
async def custom_ping():
    return {"status": "healthy"}
```

## Priority Order (unchanged)
1. **Environment variables** (highest priority)
2. **Registry decorators** (`@ping`, `@invoke`)
3. **Customer script functions**
4. **Framework register decorators** (`@register_ping_handler`) (lowest priority)

## Files Modified

### Core Implementation
- `python/model_hosting_container_standards/common/handler/decorators.py`
  - Enhanced `create_register_decorator` to register handlers directly to registry

### Test Infrastructure
- `python/tests/resources/mock_vllm_server.py`
  - Migrated to use `@register_ping_handler` and `@register_invocation_handler`
  - Added `@inject_adapter_id` for LoRA adapter support

### Test Suite
- `python/tests/integration/test_handler_override_integration.py`
  - Removed 4 duplicate test methods
  - Added `test_framework_inject_adapter_id_decorator` for LoRA testing
  - Streamlined from 13 to 9 focused tests

## Benefits

1. **Simplified API**: Single decorator replaces multiple route decorators
2. **Better Testing**: Mock server now accurately represents real usage
3. **LoRA Support**: Added adapter ID injection functionality
4. **Cleaner Tests**: Removed redundant test cases while maintaining coverage
5. **Backward Compatible**: All existing functionality preserved

## Testing

All tests pass:
```bash
python -m pytest tests/integration/test_handler_override_integration.py -v
# 9 passed in 0.39s
```

The implementation maintains full backward compatibility while providing a more streamlined developer experience for framework integrations.