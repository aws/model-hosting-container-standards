"""Unit tests for utils module decorator functions."""

import inspect

from model_hosting_container_standards.common.handler.decorators import (
    create_override_decorator,
)


class MockHandlerRegistry:
    """Mock handler registry that mimics real registry behavior."""

    def __init__(self):
        self.handlers = {}

    def set_handler(self, handler_type: str, handler_func):
        """Set a handler in the registry."""
        self.handlers[handler_type] = handler_func

    def get_handler(self, handler_type: str):
        """Get a handler from the registry."""
        return self.handlers.get(handler_type)


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
        registered_handler = registry.get_handler("ping")
        assert registered_handler is async_ping_handler
        assert inspect.iscoroutinefunction(registered_handler)

    def test_override_decorator_with_parameters(self):
        """Test override decorator works with async functions that have parameters."""
        registry = MockHandlerRegistry()
        invoke_override = create_override_decorator("invocation", registry)

        @invoke_override
        async def invoke_handler(data, context=None):
            return {"result": data * 2, "context": context}

        # Verify registration and function signature preservation
        registered_handler = registry.get_handler("invocation")
        assert registered_handler is invoke_handler
        assert registered_handler.__name__ == "invoke_handler"
        assert inspect.iscoroutinefunction(registered_handler)

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
        ping_registered = registry.get_handler("ping")
        invoke_registered = registry.get_handler("invocation")

        assert ping_registered is ping_handler
        assert invoke_registered is invoke_handler
        assert inspect.iscoroutinefunction(ping_registered)
        assert inspect.iscoroutinefunction(invoke_registered)


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
        ping_handler = registry.get_handler("ping")
        invoke_handler = registry.get_handler("invocation")

        assert ping_handler is custom_ping
        assert invoke_handler is custom_invoke
        assert inspect.iscoroutinefunction(ping_handler)
        assert inspect.iscoroutinefunction(invoke_handler)
