"""Unit tests for utils module decorator functions."""

import inspect

from model_hosting_container_standards.common.handler.decorators import (
    create_override_decorator,
)
from model_hosting_container_standards.common.handler.registry import HandlerInfo


class MockHandlerRegistry:
    """Mock handler registry that mimics real registry behavior."""

    def __init__(self):
        self.handlers = {}
        self.decorator_handlers = {}

    def set_handler(self, handler_type: str, handler_func):
        """Set a handler in the registry."""
        self.handlers[handler_type] = handler_func

    def get_handler(self, handler_type: str):
        """Get a handler from the registry."""
        return self.handlers.get(handler_type)

    def set_decorator_handler(self, handler_type: str, handler_func, route_kwargs=None):
        """Set a decorator handler in the registry."""
        handler_info = HandlerInfo(func=handler_func, route_kwargs=route_kwargs or {})
        self.decorator_handlers[handler_type] = handler_info

    def get_decorator_handler(self, handler_type: str):
        """Get a decorator handler from the registry."""
        return self.decorator_handlers.get(handler_type)


class TestCreateOverrideDecorator:
    """Test create_override_decorator function."""

    def test_override_decorator_with_async_function(self):
        """Test override decorator works with async functions."""
        registry = MockHandlerRegistry()
        ping_override = create_override_decorator("ping", registry)

        @ping_override
        async def async_ping_handler():
            return {"status": "healthy", "async": True}

        # Verify registration and async nature preservation
        handler_info = registry.get_decorator_handler("ping")
        assert handler_info is not None
        assert handler_info.func is async_ping_handler
        assert inspect.iscoroutinefunction(handler_info.func)
        assert handler_info.route_kwargs == {}

    def test_override_decorator_with_parameters(self):
        """Test override decorator works with async functions that have parameters."""
        registry = MockHandlerRegistry()
        invoke_override = create_override_decorator("invocation", registry)

        @invoke_override
        async def invoke_handler(data, context=None):
            return {"result": data * 2, "context": context}

        # Verify registration and function signature preservation
        handler_info = registry.get_decorator_handler("invocation")
        assert handler_info is not None
        assert handler_info.func is invoke_handler
        assert handler_info.func.__name__ == "invoke_handler"
        assert inspect.iscoroutinefunction(handler_info.func)
        assert handler_info.route_kwargs == {}

    def test_override_decorator_multiple_handlers(self):
        """Test override decorator works with multiple async handler types."""
        registry = MockHandlerRegistry()
        ping_override = create_override_decorator("ping", registry)
        invoke_override = create_override_decorator("invocation", registry)

        @ping_override
        async def ping_handler():
            return {"status": "ok"}

        @invoke_override
        async def invoke_handler(data):
            return {"prediction": data}

        # Verify both handlers are registered correctly
        ping_info = registry.get_decorator_handler("ping")
        invoke_info = registry.get_decorator_handler("invocation")

        assert ping_info is not None
        assert invoke_info is not None
        assert ping_info.func is ping_handler
        assert invoke_info.func is invoke_handler
        assert inspect.iscoroutinefunction(ping_info.func)
        assert inspect.iscoroutinefunction(invoke_info.func)
        assert ping_info.route_kwargs == {}
        assert invoke_info.route_kwargs == {}


class TestOverrideDecoratorWithRouteKwargs:
    """Test override_handler decorator with route kwargs."""

    def test_override_decorator_without_parameters(self):
        """Test @override_handler without parameters."""
        registry = MockHandlerRegistry()
        ping_override = create_override_decorator("ping", registry)

        @ping_override
        async def custom_ping():
            return {"status": "healthy"}

        # Verify HandlerInfo is created with empty route_kwargs
        handler_info = registry.get_decorator_handler("ping")
        assert handler_info is not None
        assert handler_info.func is custom_ping
        assert handler_info.route_kwargs == {}
        assert inspect.iscoroutinefunction(handler_info.func)

    def test_override_decorator_with_dependencies(self):
        """Test @override_handler(dependencies=[...]) with parameters."""
        from fastapi import Depends

        def validate_request():
            """Mock dependency function."""
            pass

        registry = MockHandlerRegistry()
        invoke_override = create_override_decorator("invocation", registry)

        @invoke_override(dependencies=[Depends(validate_request)])
        async def custom_invoke(data):
            return {"result": data}

        # Verify HandlerInfo is created with correct route_kwargs
        handler_info = registry.get_decorator_handler("invocation")
        assert handler_info is not None
        assert handler_info.func is custom_invoke
        assert "dependencies" in handler_info.route_kwargs
        assert len(handler_info.route_kwargs["dependencies"]) == 1
        assert inspect.iscoroutinefunction(handler_info.func)

    def test_override_decorator_with_multiple_route_kwargs(self):
        """Test @override_handler with multiple route configuration parameters."""
        from fastapi import Depends

        def validate_json():
            """Mock dependency."""
            pass

        registry = MockHandlerRegistry()
        invoke_override = create_override_decorator("invocation", registry)

        @invoke_override(
            dependencies=[Depends(validate_json)],
            responses={
                400: {"description": "Bad Request"},
                500: {"description": "Internal Server Error"},
            },
            summary="Custom invocation handler",
            tags=["custom"],
        )
        async def custom_invoke(data):
            return {"result": data}

        # Verify HandlerInfo is created with all route_kwargs
        handler_info = registry.get_decorator_handler("invocation")
        assert handler_info is not None
        assert handler_info.func is custom_invoke
        assert "dependencies" in handler_info.route_kwargs
        assert "responses" in handler_info.route_kwargs
        assert "summary" in handler_info.route_kwargs
        assert "tags" in handler_info.route_kwargs
        assert handler_info.route_kwargs["summary"] == "Custom invocation handler"
        assert handler_info.route_kwargs["tags"] == ["custom"]
        assert 400 in handler_info.route_kwargs["responses"]
        assert 500 in handler_info.route_kwargs["responses"]
        assert inspect.iscoroutinefunction(handler_info.func)


class TestDecoratorIntegration:
    """Test override decorator functionality."""

    def test_multiple_override_decorators(self):
        """Test that multiple override decorators can work in the same registry."""
        registry = MockHandlerRegistry()

        # Create override decorators for different handler types
        ping_override = create_override_decorator("ping", registry)
        invoke_override = create_override_decorator("invocation", registry)

        # Use override decorators
        @ping_override
        async def custom_ping():
            return {"status": "custom"}

        @invoke_override
        async def custom_invoke(data):
            return {"result": data}

        # Verify both handlers are registered correctly
        ping_info = registry.get_decorator_handler("ping")
        invoke_info = registry.get_decorator_handler("invocation")

        assert ping_info is not None
        assert invoke_info is not None
        assert ping_info.func is custom_ping
        assert invoke_info.func is custom_invoke
        assert inspect.iscoroutinefunction(ping_info.func)
        assert inspect.iscoroutinefunction(invoke_info.func)
