"""Tests for middleware exceptions."""

import pytest

from model_hosting_container_standards.common.fastapi.middleware import (
    custom_middleware,
    input_formatter,
)
from model_hosting_container_standards.common.fastapi.middleware.registry import (
    MiddlewareRegistry,
)
from model_hosting_container_standards.exceptions import (
    FormatterRegistrationError,
    MiddlewareRegistrationError,
)


class TestMiddlewareExceptions:
    """Test middleware-specific exceptions."""

    def setup_method(self):
        """Clear global state before each test."""
        from model_hosting_container_standards.common.fastapi.middleware.registry import (
            middleware_registry,
        )
        from model_hosting_container_standards.common.fastapi.middleware.source.decorator_loader import (
            decorator_loader,
        )

        # Clear decorator loader state
        decorator_loader.clear()
        # Clear middleware registry state
        middleware_registry.clear_middlewares()

    def teardown_method(self):
        """Clear global state after each test."""
        from model_hosting_container_standards.common.fastapi.middleware.registry import (
            middleware_registry,
        )
        from model_hosting_container_standards.common.fastapi.middleware.source.decorator_loader import (
            decorator_loader,
        )

        # Clear decorator loader state
        decorator_loader.clear()
        # Clear middleware registry state
        middleware_registry.clear_middlewares()

    def test_middleware_registration_error_invalid_name(self):
        """Test MiddlewareRegistrationError for invalid middleware name."""
        with pytest.raises(MiddlewareRegistrationError) as exc_info:

            @custom_middleware("invalid_name")
            def test_middleware():
                pass

        assert "invalid_name" in str(exc_info.value)
        assert "not allowed" in str(exc_info.value)

    def test_middleware_registration_error_duplicate_registration(self):
        """Test ValueError for duplicate registration at registry level."""
        registry = MiddlewareRegistry()

        def test_middleware():
            pass

        # First registration should succeed
        registry.register_middleware("pre_post_process", test_middleware)

        # Second registration should raise ValueError (registry level)
        with pytest.raises(ValueError) as exc_info:
            registry.register_middleware("pre_post_process", test_middleware)

        assert "already registered" in str(exc_info.value)

    def test_formatter_registration_error_duplicate_input_formatter(self):
        """Test ValueError for duplicate input formatter at decorator loader level."""
        from model_hosting_container_standards.common.fastapi.middleware.source.decorator_loader import (
            decorator_loader,
        )

        def formatter1():
            pass

        def formatter2():
            pass

        # First registration should succeed
        decorator_loader.set_input_formatter(formatter1)

        # Second registration should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            decorator_loader.set_input_formatter(formatter2)

        assert "Input formatter is already registered" in str(exc_info.value)

    def test_formatter_registration_error_duplicate_output_formatter(self):
        """Test ValueError for duplicate output formatter at decorator loader level."""
        from model_hosting_container_standards.common.fastapi.middleware.source.decorator_loader import (
            decorator_loader,
        )

        def formatter1():
            pass

        def formatter2():
            pass

        # First registration should succeed
        decorator_loader.set_output_formatter(formatter1)

        # Second registration should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            decorator_loader.set_output_formatter(formatter2)

        assert "Output formatter is already registered" in str(exc_info.value)

    def test_decorator_middleware_registration_error(self):
        """Test that decorator raises MiddlewareRegistrationError."""
        with pytest.raises(MiddlewareRegistrationError):

            @custom_middleware("invalid")
            def bad_middleware():
                pass

    def test_decorator_formatter_registration_error(self):
        """Test that formatter decorators raise FormatterRegistrationError."""

        # Register first formatter
        @input_formatter
        def first_input_formatter():
            pass

        # Try to register second formatter - should raise FormatterRegistrationError
        with pytest.raises(FormatterRegistrationError):

            @input_formatter
            def second_input_formatter():
                pass

    def test_exception_chaining(self):
        """Test that exceptions are properly chained with 'from' clause."""

        # First register a middleware using decorator
        @custom_middleware("pre_post_process")
        def first_middleware():
            pass

        # Now try to register the same middleware again - this should chain the exception
        with pytest.raises(MiddlewareRegistrationError) as exc_info:

            @custom_middleware("pre_post_process")
            def duplicate_middleware():
                pass

        # Check that the original ValueError is chained
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
