"""Unit tests for utils module decorator functions."""

import inspect

from model_hosting_container_standards.utils import (
    create_override_decorator,
    create_register_decorator,
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


class TestCreateRegisterDecorator:
    """Test create_register_decorator function."""

    def test_register_decorator_no_existing_handler(self):
        """Test register decorator when no existing handler is found."""
        registry = MockHandlerRegistry()

        def resolver_no_handler():
            return None  # No existing handler

        register_ping = create_register_decorator("ping", resolver_no_handler, registry)

        @register_ping
        async def default_ping_handler():
            return {"status": "default"}

        # Verify default handler was registered
        registered_handler = registry.get_handler("ping")
        assert registered_handler is default_ping_handler
        assert inspect.iscoroutinefunction(registered_handler)

    def test_register_decorator_existing_handler_takes_precedence(self):
        """Test register decorator when existing async handler takes precedence."""
        registry = MockHandlerRegistry()

        async def existing_handler():
            return {"status": "existing"}

        def resolver_with_handler():
            return existing_handler

        register_ping = create_register_decorator(
            "ping", resolver_with_handler, registry
        )

        @register_ping
        async def default_ping_handler():
            return {"status": "default"}  # Should NOT be used

        # The decorator should return the existing handler, not register the default
        returned_handler = register_ping(default_ping_handler)
        assert returned_handler is existing_handler
        assert inspect.iscoroutinefunction(returned_handler)

        # Registry should NOT have the default handler
        registered_handler = registry.get_handler("ping")
        assert registered_handler is None  # No handler registered

    def test_register_decorator_with_async_default(self):
        """Test register decorator with async default handler."""
        registry = MockHandlerRegistry()

        def resolver_no_handler():
            return None

        register_ping = create_register_decorator("ping", resolver_no_handler, registry)

        @register_ping
        async def async_default_handler():
            return {"status": "async_default"}

        # Verify async default handler was registered and preserved
        registered_handler = registry.get_handler("ping")
        assert registered_handler is async_default_handler
        assert inspect.iscoroutinefunction(registered_handler)

    def test_register_decorator_with_async_existing_handler(self):
        """Test register decorator with async existing handler taking precedence."""
        registry = MockHandlerRegistry()

        async def existing_async_handler():
            return {"status": "existing_async"}

        def resolver_with_async_handler():
            return existing_async_handler

        register_ping = create_register_decorator(
            "ping", resolver_with_async_handler, registry
        )

        @register_ping
        async def default_async_handler():
            return {"status": "default_async"}

        # The existing async handler should take precedence
        returned_handler = register_ping(default_async_handler)
        assert returned_handler is existing_async_handler
        assert inspect.iscoroutinefunction(returned_handler)

    def test_register_decorator_called_without_function(self):
        """Test register decorator when called without function parameter."""
        registry = MockHandlerRegistry()

        def resolver_no_handler():
            return None

        register_ping = create_register_decorator("ping", resolver_no_handler, registry)

        # Call decorator without function (returns inner decorator)
        inner_decorator = register_ping()
        assert callable(inner_decorator)

        # Now use the inner decorator
        @inner_decorator
        async def test_handler():
            return {"status": "test"}

        # Verify handler was registered
        registered_handler = registry.get_handler("ping")
        assert registered_handler is test_handler
        assert inspect.iscoroutinefunction(registered_handler)

    def test_register_decorator_called_with_function_parameter(self):
        """Test register decorator when called with function parameter."""
        registry = MockHandlerRegistry()

        def resolver_no_handler():
            return None

        register_ping = create_register_decorator("ping", resolver_no_handler, registry)

        async def test_handler():
            return {"status": "test"}

        # Call decorator with function parameter directly
        result = register_ping(test_handler)

        # Should register and return the function
        assert result is test_handler
        registered_handler = registry.get_handler("ping")
        assert registered_handler is test_handler
        assert inspect.iscoroutinefunction(registered_handler)


class TestDecoratorIntegration:
    """Test both decorators working together."""

    def test_override_and_register_decorators_together(self):
        """Test that override and register decorators can work in the same registry."""
        registry = MockHandlerRegistry()

        # Create both types of decorators
        ping_override = create_override_decorator("ping", registry)

        def resolver_no_invoke():
            return None

        register_invoke = create_register_decorator(
            "invocation", resolver_no_invoke, registry
        )

        # Use override decorator for ping
        @ping_override
        async def custom_ping():
            return {"status": "custom"}

        # Use register decorator for invocation
        @register_invoke
        async def default_invoke(data):
            return {"result": data}

        # Verify both handlers are registered correctly
        ping_handler = registry.get_handler("ping")
        invoke_handler = registry.get_handler("invocation")

        assert ping_handler is custom_ping
        assert invoke_handler is default_invoke
        assert inspect.iscoroutinefunction(ping_handler)
        assert inspect.iscoroutinefunction(invoke_handler)
