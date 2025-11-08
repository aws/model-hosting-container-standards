"""Unit tests for HandlerRegistry and HandlerInfo."""

from model_hosting_container_standards.common.handler.registry import (
    HandlerInfo,
    HandlerRegistry,
)


class TestHandlerInfo:
    """Test HandlerInfo dataclass."""

    def test_handler_info_without_route_kwargs(self):
        """Test HandlerInfo without route_kwargs."""

        def handler():
            return {"status": "ok"}

        info = HandlerInfo(func=handler)

        assert info.func == handler
        assert info.route_kwargs == {}

    def test_handler_info_with_route_kwargs(self):
        """Test HandlerInfo with route_kwargs."""

        def handler():
            return {"status": "ok"}

        route_kwargs = {"tags": ["api"], "summary": "Test"}

        info = HandlerInfo(func=handler, route_kwargs=route_kwargs)

        assert info.func == handler
        assert info.route_kwargs == route_kwargs


class TestHandlerRegistry:
    """Test HandlerRegistry set and get methods."""

    def test_set_get_handler_without_route_kwargs(self):
        """Test set/get handler without route_kwargs."""
        registry = HandlerRegistry()

        def handler():
            return {"status": "ok"}

        registry.set_handler("test", handler)
        info = registry.get_handler("test")

        assert info.func == handler
        assert info.route_kwargs == {}

    def test_set_get_handler_with_route_kwargs(self):
        """Test set/get handler with route_kwargs."""
        registry = HandlerRegistry()

        def handler():
            return {"status": "ok"}

        route_kwargs = {"tags": ["api"], "summary": "Test"}
        handler_info = HandlerInfo(func=handler, route_kwargs=route_kwargs)
        registry.set_handler("test", handler_info)

        info = registry.get_handler("test")

        assert info.func == handler
        assert info.route_kwargs == route_kwargs

    def test_set_get_framework_default_without_route_kwargs(self):
        """Test set/get framework default without route_kwargs."""
        registry = HandlerRegistry()

        def handler():
            return {"status": "ok"}

        registry.set_framework_default("ping", handler)
        info = registry.get_framework_default("ping")

        assert info.func == handler
        assert info.route_kwargs == {}

    def test_set_get_framework_default_with_route_kwargs(self):
        """Test set/get framework default with route_kwargs."""
        registry = HandlerRegistry()

        def handler():
            return {"status": "ok"}

        route_kwargs = {"tags": ["health"]}
        registry.set_framework_default("ping", handler, route_kwargs=route_kwargs)
        info = registry.get_framework_default("ping")

        assert info.func == handler
        assert info.route_kwargs == route_kwargs

    def test_set_get_decorator_handler_without_route_kwargs(self):
        """Test set/get decorator handler without route_kwargs."""
        registry = HandlerRegistry()

        def handler():
            return {"status": "ok"}

        registry.set_decorator_handler("invoke", handler)
        info = registry.get_decorator_handler("invoke")

        assert info.func == handler
        assert info.route_kwargs == {}

    def test_set_get_decorator_handler_with_route_kwargs(self):
        """Test set/get decorator handler with route_kwargs."""
        registry = HandlerRegistry()

        def handler():
            return {"status": "ok"}

        route_kwargs = {"tags": ["custom"], "summary": "Custom"}
        registry.set_decorator_handler("invoke", handler, route_kwargs=route_kwargs)
        info = registry.get_decorator_handler("invoke")

        assert info.func == handler
        assert info.route_kwargs == route_kwargs
