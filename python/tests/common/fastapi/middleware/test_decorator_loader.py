"""Tests for MiddlewareDecoratorLoader."""

import pytest

from model_hosting_container_standards.common.fastapi.middleware.source.decorator_loader import (
    MiddlewareDecoratorLoader,
)


class TestMiddlewareDecoratorLoader:
    """Test MiddlewareDecoratorLoader functionality."""

    def setup_method(self):
        """Clear loader state before each test."""
        # Create fresh loader for each test
        self.loader = MiddlewareDecoratorLoader()

    def test_init(self):
        """Test MiddlewareDecoratorLoader initialization."""
        loader = MiddlewareDecoratorLoader()

        assert loader.pre_fn is None
        assert loader.post_fn is None
        assert loader.pre_post_middleware is None
        assert loader.throttle_middleware is None

    def test_set_middleware_throttle(self):
        """Test setting throttle middleware."""

        def throttle_func():
            pass

        self.loader.set_middleware("throttle", throttle_func)

        assert self.loader.throttle_middleware is throttle_func

    def test_set_middleware_pre_post_process(self):
        """Test setting pre_post_process middleware."""

        def pre_post_func():
            pass

        self.loader.set_middleware("pre_post_process", pre_post_func)

        assert self.loader.pre_post_middleware is pre_post_func

    def test_set_middleware_invalid_name(self):
        """Test setting middleware with invalid name."""

        def invalid_func():
            pass

        with pytest.raises(ValueError) as exc_info:
            self.loader.set_middleware("invalid_name", invalid_func)

        assert "not allowed" in str(exc_info.value)
        assert "pre_post_process, throttle" in str(exc_info.value)

    def test_set_middleware_duplicate_throttle(self):
        """Test setting duplicate throttle middleware."""

        def throttle_func1():
            pass

        def throttle_func2():
            pass

        # First registration should succeed
        self.loader.set_middleware("throttle", throttle_func1)

        # Second registration should fail
        with pytest.raises(ValueError) as exc_info:
            self.loader.set_middleware("throttle", throttle_func2)

        assert "already registered" in str(exc_info.value)

    def test_set_middleware_duplicate_pre_post_process(self):
        """Test setting duplicate pre_post_process middleware."""

        def pre_post_func1():
            pass

        def pre_post_func2():
            pass

        # First registration should succeed
        self.loader.set_middleware("pre_post_process", pre_post_func1)

        # Second registration should fail
        with pytest.raises(ValueError) as exc_info:
            self.loader.set_middleware("pre_post_process", pre_post_func2)

        assert "already registered" in str(exc_info.value)

    def test_set_input_formatter(self):
        """Test setting input formatter."""

        def input_func():
            pass

        self.loader.set_input_formatter(input_func)

        assert self.loader.pre_fn is input_func

    def test_set_input_formatter_duplicate(self):
        """Test setting duplicate input formatter."""

        def input_func1():
            pass

        def input_func2():
            pass

        # First registration should succeed
        self.loader.set_input_formatter(input_func1)

        # Second registration should fail
        with pytest.raises(ValueError) as exc_info:
            self.loader.set_input_formatter(input_func2)

        assert "Input formatter is already registered" in str(exc_info.value)

    def test_set_output_formatter(self):
        """Test setting output formatter."""

        def output_func():
            pass

        self.loader.set_output_formatter(output_func)

        assert self.loader.post_fn is output_func

    def test_set_output_formatter_duplicate(self):
        """Test setting duplicate output formatter."""

        def output_func1():
            pass

        def output_func2():
            pass

        # First registration should succeed
        self.loader.set_output_formatter(output_func1)

        # Second registration should fail
        with pytest.raises(ValueError) as exc_info:
            self.loader.set_output_formatter(output_func2)

        assert "Output formatter is already registered" in str(exc_info.value)

    def test_load_no_formatters(self):
        """Test load when no formatters are set."""
        self.loader.load()

        # Should not create pre_post_middleware
        assert self.loader.pre_post_middleware is None

    def test_load_with_formatters(self):
        """Test load when formatters are set."""

        def input_func():
            pass

        def output_func():
            pass

        self.loader.set_input_formatter(input_func)
        self.loader.set_output_formatter(output_func)

        self.loader.load()

        # Should create combined pre_post_middleware
        assert self.loader.pre_post_middleware is not None
        assert callable(self.loader.pre_post_middleware)

    def test_load_with_existing_pre_post_middleware(self):
        """Test load when pre_post_middleware already exists."""

        def pre_post_func():
            pass

        def input_func():
            pass

        # Set both pre_post_middleware and formatter
        self.loader.set_middleware("pre_post_process", pre_post_func)
        self.loader.set_input_formatter(input_func)

        self.loader.load()

        # Should keep existing pre_post_middleware, not create new one
        assert self.loader.pre_post_middleware is pre_post_func

    def test_clear(self):
        """Test clearing all middlewares and formatters."""

        def throttle_func():
            pass

        def input_func():
            pass

        def output_func():
            pass

        # Set various middlewares
        self.loader.set_middleware("throttle", throttle_func)
        self.loader.set_input_formatter(input_func)
        self.loader.set_output_formatter(output_func)

        # Clear all
        self.loader.clear()

        # All should be None
        assert self.loader.throttle_middleware is None
        assert self.loader.pre_post_middleware is None
        assert self.loader.pre_fn is None
        assert self.loader.post_fn is None

    def test_clear_allows_reregistration(self):
        """Test that clear allows re-registration of middlewares."""

        def throttle_func1():
            pass

        def throttle_func2():
            pass

        # Register middleware
        self.loader.set_middleware("throttle", throttle_func1)

        # Clear
        self.loader.clear()

        # Should be able to register again
        self.loader.set_middleware("throttle", throttle_func2)
        assert self.loader.throttle_middleware is throttle_func2
