"""Tests for BaseMiddlewareLoader."""

import pytest

from model_hosting_container_standards.common.fastapi.middleware.source.base import (
    BaseMiddlewareLoader,
)


class TestBaseMiddlewareLoader:
    """Test BaseMiddlewareLoader functionality."""

    def test_init(self):
        """Test BaseMiddlewareLoader initialization."""
        loader = BaseMiddlewareLoader()

        assert loader.pre_fn is None
        assert loader.post_fn is None
        assert loader.pre_post_middleware is None
        assert loader.throttle_middleware is None

    def test_get_middleware_throttle(self):
        """Test get_middleware for throttle middleware."""
        loader = BaseMiddlewareLoader()

        # Initially None
        assert loader.get_middleware("throttle") is None

        # Set throttle middleware
        def throttle_func():
            pass

        loader.throttle_middleware = throttle_func

        assert loader.get_middleware("throttle") is throttle_func

    def test_get_middleware_pre_post_process(self):
        """Test get_middleware for pre_post_process middleware."""
        loader = BaseMiddlewareLoader()

        # Initially None
        assert loader.get_middleware("pre_post_process") is None

        # Set pre_post_process middleware
        def pre_post_func():
            pass

        loader.pre_post_middleware = pre_post_func

        assert loader.get_middleware("pre_post_process") is pre_post_func

    def test_get_middleware_unknown(self):
        """Test get_middleware for unknown middleware name."""
        loader = BaseMiddlewareLoader()

        assert loader.get_middleware("unknown") is None

    def test_has_middlewares_empty(self):
        """Test has_middlewares when no middlewares are set."""
        loader = BaseMiddlewareLoader()

        assert not loader.has_middlewares()

    def test_has_middlewares_with_throttle(self):
        """Test has_middlewares when throttle middleware is set."""
        loader = BaseMiddlewareLoader()
        loader.throttle_middleware = lambda: None

        assert loader.has_middlewares()

    def test_has_middlewares_with_pre_fn(self):
        """Test has_middlewares when pre_fn is set."""
        loader = BaseMiddlewareLoader()
        loader.pre_fn = lambda: None

        assert loader.has_middlewares()

    def test_has_middlewares_with_post_fn(self):
        """Test has_middlewares when post_fn is set."""
        loader = BaseMiddlewareLoader()
        loader.post_fn = lambda: None

        assert loader.has_middlewares()

    def test_has_middlewares_with_pre_post_middleware(self):
        """Test has_middlewares when pre_post_middleware is set."""
        loader = BaseMiddlewareLoader()
        loader.pre_post_middleware = lambda: None

        assert loader.has_middlewares()

    def test_combine_pre_post_middleware(self):
        """Test _combine_pre_post_middleware functionality."""
        loader = BaseMiddlewareLoader()

        # Set pre and post functions
        async def pre_func(request):
            return request

        async def post_func(response):
            return response

        loader.pre_fn = pre_func
        loader.post_fn = post_func

        # Combine them
        combined = loader._combine_pre_post_middleware("TEST")

        assert combined is not None
        assert callable(combined)
        assert combined.__name__ == "combined_pre_post_process"

    def test_combine_pre_post_middleware_none(self):
        """Test _combine_pre_post_middleware when no functions are set."""
        loader = BaseMiddlewareLoader()

        combined = loader._combine_pre_post_middleware("TEST")

        assert combined is None

    @pytest.mark.asyncio
    async def test_create_pre_post_middleware_execution(self):
        """Test that created middleware executes correctly."""
        loader = BaseMiddlewareLoader()

        # Mock functions
        pre_called = False
        post_called = False

        async def pre_func(request):
            nonlocal pre_called
            pre_called = True
            return request

        async def post_func(response):
            nonlocal post_called
            post_called = True
            return response

        # Create middleware
        middleware = loader._create_pre_post_middleware(
            pre_func, post_func, "test_middleware", "TEST"
        )

        # Mock call_next
        async def call_next(request):
            return "response"

        # Execute middleware
        result = await middleware("request", call_next)

        assert pre_called
        assert post_called
        assert result == "response"

    @pytest.mark.asyncio
    async def test_create_pre_post_middleware_exception_handling(self):
        """Test middleware exception handling."""
        loader = BaseMiddlewareLoader()

        # Function that raises exception
        async def failing_pre_func(request):
            raise ValueError("Test error")

        # Create middleware
        middleware = loader._create_pre_post_middleware(
            failing_pre_func, None, "test_middleware", "TEST"
        )

        # Mock call_next
        async def call_next(request):
            return "response"

        # Execute middleware - should return error response
        result = await middleware("request", call_next)

        # Should return JSONResponse with error
        assert hasattr(result, "status_code")
        assert result.status_code == 500
