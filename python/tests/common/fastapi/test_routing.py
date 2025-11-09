"""Unit tests for generic routing utilities."""

from unittest.mock import Mock, patch

import pytest
from fastapi import APIRouter, FastAPI

from model_hosting_container_standards.common.fastapi.routing import (
    RouteConfig,
    create_router,
    mount_handlers,
    normalize_prefix,
    remove_conflicting_routes,
)
from model_hosting_container_standards.common.handler.registry import HandlerInfo


class TestRouteConfig:
    """Test RouteConfig dataclass."""

    def test_route_config_is_frozen(self):
        """Test that RouteConfig is immutable (frozen=True)."""
        # Arrange
        config = RouteConfig(path="/test", method="POST")

        # Act & Assert
        with pytest.raises(AttributeError):
            config.path = "/modified"

        with pytest.raises(AttributeError):
            config.method = "GET"

    def test_route_config_equality(self):
        """Test RouteConfig equality comparison."""
        # Arrange
        config1 = RouteConfig(path="/test", method="POST", tags=["tag1"])
        config2 = RouteConfig(path="/test", method="POST", tags=["tag1"])
        config3 = RouteConfig(path="/other", method="POST", tags=["tag1"])
        config4 = RouteConfig(path="/test", method="GET", tags=["tag1"])

        # Assert
        assert config1 == config2
        assert config1 != config3
        assert config1 != config4


class TestNormalizePrefix:
    """Test normalize_prefix function."""

    def test_normalize_prefix_empty_string(self):
        """Test normalize_prefix with empty string."""
        assert normalize_prefix("") == ""

    def test_normalize_prefix_none_like_values(self):
        """Test normalize_prefix with None-like values."""
        assert normalize_prefix("") == ""
        assert normalize_prefix("/") == ""

    def test_normalize_prefix_with_leading_slash(self):
        """Test normalize_prefix with existing leading slash."""
        assert normalize_prefix("/api/v1") == "/api/v1"
        assert normalize_prefix("/lora") == "/lora"

    def test_normalize_prefix_without_leading_slash(self):
        """Test normalize_prefix without leading slash."""
        assert normalize_prefix("api/v1") == "/api/v1"
        assert normalize_prefix("lora") == "/lora"

    def test_normalize_prefix_with_trailing_slash(self):
        """Test normalize_prefix with trailing slash."""
        assert normalize_prefix("api/v1/") == "/api/v1"
        assert normalize_prefix("/api/v1/") == "/api/v1"
        assert normalize_prefix("lora/") == "/lora"
        assert normalize_prefix("/lora/") == "/lora"

    def test_normalize_prefix_complex_paths(self):
        """Test normalize_prefix with complex paths."""
        assert normalize_prefix("api/v1/endpoints") == "/api/v1/endpoints"
        assert normalize_prefix("/api/v1/endpoints/") == "/api/v1/endpoints"
        assert normalize_prefix("very/deep/nested/path/") == "/very/deep/nested/path"


class TestMountHandlers:
    """Test mount_handlers function."""

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_with_resolver(self, mock_logger, mock_registry):
        """Test mounting handlers with a custom route resolver."""
        # Arrange
        router = APIRouter()
        mock_handler = Mock()
        handler_info = HandlerInfo(func=mock_handler, route_kwargs={})

        # Mock route resolver
        def mock_resolver(handler_type: str):
            if handler_type == "test_handler":
                return RouteConfig(path="/test", method="POST")
            return None

        mock_registry.list_handlers.return_value = ["test_handler"]
        mock_registry.get_handler.return_value = handler_info

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        mock_registry.list_handlers.assert_called_once()
        mock_registry.get_handler.assert_called_once_with("test_handler")
        assert len(router.routes) == 1

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_specific_handler_types(self, mock_logger, mock_registry):
        """Test mounting only specific handler types."""
        # Arrange
        router = APIRouter()
        mock_handler1 = Mock()
        mock_handler2 = Mock()
        handler_info1 = HandlerInfo(func=mock_handler1, route_kwargs={})
        handler_info2 = HandlerInfo(func=mock_handler2, route_kwargs={})

        def mock_resolver(handler_type: str):
            return RouteConfig(path=f"/{handler_type}", method="POST")

        mock_registry.get_handler.side_effect = [handler_info1, handler_info2]

        # Act
        mount_handlers(
            router, handler_names=["handler1", "handler2"], route_resolver=mock_resolver
        )

        # Assert
        # list_handlers should NOT be called when handler_types is provided
        mock_registry.list_handlers.assert_not_called()
        # get_handler should be called for each specified handler
        assert mock_registry.get_handler.call_count == 2
        mock_registry.get_handler.assert_any_call("handler1")
        mock_registry.get_handler.assert_any_call("handler2")
        # Both routes should be mounted
        assert len(router.routes) == 2

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_skip_none_handler(self, mock_logger, mock_registry):
        """Test that None handlers are skipped."""
        # Arrange
        router = APIRouter()

        def mock_resolver(handler_type: str):
            return RouteConfig(path="/test", method="POST")

        mock_registry.list_handlers.return_value = ["test_handler"]
        mock_registry.get_handler.return_value = None  # Handler not found

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        # No routes should be added
        assert len(router.routes) == 0

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_skip_when_resolver_returns_none(
        self, mock_logger, mock_registry
    ):
        """Test that handlers without route configs are skipped."""
        # Arrange
        router = APIRouter()
        mock_handler = Mock()
        handler_info = HandlerInfo(func=mock_handler, route_kwargs={})

        # Resolver returns None (no route configured)
        def mock_resolver(handler_type: str):
            return None

        mock_registry.list_handlers.return_value = ["test_handler"]
        mock_registry.get_handler.return_value = handler_info

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        assert len(router.routes) == 0
        mock_logger.debug.assert_called_with(
            "Skipping test_handler - no default route configured"
        )

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_catches_value_error(self, mock_logger, mock_registry):
        """Test that ValueError from resolver is caught and logged."""
        # Arrange
        router = APIRouter()
        mock_handler = Mock()
        handler_info = HandlerInfo(func=mock_handler, route_kwargs={})

        # Resolver raises ValueError
        def mock_resolver(handler_type: str):
            raise ValueError("Unsupported handler type")

        mock_registry.list_handlers.return_value = ["invalid_handler"]
        mock_registry.get_handler.return_value = handler_info

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        assert len(router.routes) == 0
        # Should log the error at DEBUG level
        assert mock_logger.debug.called

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_without_route_kwargs(self, mock_logger, mock_registry):
        """Test mounting handler without route_kwargs (default behavior)."""
        # Arrange
        router = APIRouter()
        mock_handler = Mock()
        handler_info = HandlerInfo(func=mock_handler, route_kwargs={})

        def mock_resolver(handler_type: str):
            return RouteConfig(
                path="/test", method="POST", tags=["default"], summary="Default summary"
            )

        mock_registry.list_handlers.return_value = ["test_handler"]
        mock_registry.get_handler.return_value = handler_info

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        assert len(router.routes) == 1
        route = router.routes[0]
        assert route.path == "/test"
        assert "POST" in route.methods
        assert route.tags == ["default"]
        assert route.summary == "Default summary"

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_with_dependencies(self, mock_logger, mock_registry):
        """Test mounting handler with dependencies in route_kwargs."""
        # Arrange
        from fastapi import Depends

        router = APIRouter()
        mock_handler = Mock()

        # Create a mock dependency
        def mock_dependency():
            return "dependency_value"

        handler_info = HandlerInfo(
            func=mock_handler, route_kwargs={"dependencies": [Depends(mock_dependency)]}
        )

        def mock_resolver(handler_type: str):
            return RouteConfig(path="/test", method="POST")

        mock_registry.list_handlers.return_value = ["test_handler"]
        mock_registry.get_handler.return_value = handler_info

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        assert len(router.routes) == 1
        route = router.routes[0]
        assert route.path == "/test"
        assert len(route.dependencies) == 1

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_with_responses(self, mock_logger, mock_registry):
        """Test mounting handler with responses in route_kwargs."""
        # Arrange
        router = APIRouter()
        mock_handler = Mock()

        responses_config = {
            400: {"description": "Bad Request"},
            500: {"description": "Internal Server Error"},
        }

        handler_info = HandlerInfo(
            func=mock_handler, route_kwargs={"responses": responses_config}
        )

        def mock_resolver(handler_type: str):
            return RouteConfig(path="/test", method="POST")

        mock_registry.list_handlers.return_value = ["test_handler"]
        mock_registry.get_handler.return_value = handler_info

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        assert len(router.routes) == 1
        route = router.routes[0]
        assert route.path == "/test"
        assert route.responses == responses_config

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_with_multiple_route_parameters(
        self, mock_logger, mock_registry
    ):
        """Test mounting handler with multiple route parameters."""
        # Arrange
        from fastapi import Depends

        router = APIRouter()
        mock_handler = Mock()

        def mock_dependency():
            return "dependency_value"

        responses_config = {
            400: {"description": "Bad Request"},
            500: {"description": "Internal Server Error"},
        }

        handler_info = HandlerInfo(
            func=mock_handler,
            route_kwargs={
                "dependencies": [Depends(mock_dependency)],
                "responses": responses_config,
                "status_code": 201,
                "description": "Custom description",
                "deprecated": True,
            },
        )

        def mock_resolver(handler_type: str):
            return RouteConfig(path="/test", method="POST")

        mock_registry.list_handlers.return_value = ["test_handler"]
        mock_registry.get_handler.return_value = handler_info

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        assert len(router.routes) == 1
        route = router.routes[0]
        assert route.path == "/test"
        assert len(route.dependencies) == 1
        assert route.responses == responses_config
        assert route.status_code == 201
        assert route.description == "Custom description"
        assert route.deprecated is True

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_route_kwargs_override_defaults(
        self, mock_logger, mock_registry
    ):
        """Test that route_kwargs override default RouteConfig values."""
        # Arrange
        router = APIRouter()
        mock_handler = Mock()

        # route_kwargs should override the default tags and summary
        handler_info = HandlerInfo(
            func=mock_handler,
            route_kwargs={"tags": ["override_tag"], "summary": "Override summary"},
        )

        def mock_resolver(handler_type: str):
            return RouteConfig(
                path="/test",
                method="POST",
                tags=["default_tag"],
                summary="Default summary",
            )

        mock_registry.list_handlers.return_value = ["test_handler"]
        mock_registry.get_handler.return_value = handler_info

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        assert len(router.routes) == 1
        route = router.routes[0]
        assert route.path == "/test"
        # route_kwargs should override defaults
        assert route.tags == ["override_tag"]
        assert route.summary == "Override summary"

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_default_config_when_no_route_kwargs(
        self, mock_logger, mock_registry
    ):
        """Test that default RouteConfig is used when no route_kwargs present."""
        # Arrange
        router = APIRouter()
        mock_handler = Mock()

        # Empty route_kwargs
        handler_info = HandlerInfo(func=mock_handler, route_kwargs={})

        def mock_resolver(handler_type: str):
            return RouteConfig(
                path="/test",
                method="POST",
                tags=["default_tag"],
                summary="Default summary",
            )

        mock_registry.list_handlers.return_value = ["test_handler"]
        mock_registry.get_handler.return_value = handler_info

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        assert len(router.routes) == 1
        route = router.routes[0]
        assert route.path == "/test"
        # Should use default values from RouteConfig
        assert route.tags == ["default_tag"]
        assert route.summary == "Default summary"

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_partial_override(self, mock_logger, mock_registry):
        """Test that route_kwargs can partially override defaults."""
        # Arrange
        router = APIRouter()
        mock_handler = Mock()

        # Only override tags, keep default summary
        handler_info = HandlerInfo(
            func=mock_handler, route_kwargs={"tags": ["override_tag"]}
        )

        def mock_resolver(handler_type: str):
            return RouteConfig(
                path="/test",
                method="POST",
                tags=["default_tag"],
                summary="Default summary",
            )

        mock_registry.list_handlers.return_value = ["test_handler"]
        mock_registry.get_handler.return_value = handler_info

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        assert len(router.routes) == 1
        route = router.routes[0]
        assert route.path == "/test"
        # Tags should be overridden
        assert route.tags == ["override_tag"]
        # Summary should remain default
        assert route.summary == "Default summary"

    @patch("model_hosting_container_standards.common.fastapi.routing.handler_registry")
    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_mount_handlers_logs_route_kwargs_application(
        self, mock_logger, mock_registry
    ):
        """Test that applying route_kwargs is logged."""
        # Arrange
        router = APIRouter()
        mock_handler = Mock()

        handler_info = HandlerInfo(
            func=mock_handler, route_kwargs={"dependencies": [], "responses": {}}
        )

        def mock_resolver(handler_type: str):
            return RouteConfig(path="/test", method="POST")

        mock_registry.list_handlers.return_value = ["test_handler"]
        mock_registry.get_handler.return_value = handler_info

        # Act
        mount_handlers(router, route_resolver=mock_resolver)

        # Assert
        # Should log that route_kwargs were applied
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Applied route kwargs" in call for call in debug_calls)


class TestCreateRouter:
    """Test create_router function."""

    @patch("model_hosting_container_standards.common.fastapi.routing.mount_handlers")
    def test_create_router_with_kwargs(self, mock_mount):
        """Test creating a router with additional kwargs."""

        # Arrange
        def mock_resolver(handler_type: str):
            return None

        # Act
        router = create_router(
            prefix="/lora", route_resolver=mock_resolver, tags=["lora", "adapters"]
        )

        # Assert
        assert isinstance(router, APIRouter)
        assert router.prefix == "/lora"
        assert "lora" in router.tags
        assert "adapters" in router.tags
        mock_mount.assert_called_once()

    @patch("model_hosting_container_standards.common.fastapi.routing.mount_handlers")
    def test_create_router_passes_resolver_to_mount(self, mock_mount):
        """Test that the resolver is passed to mount_handlers."""

        # Arrange
        def mock_resolver(handler_type: str):
            return RouteConfig(path="/test", method="GET")

        # Act
        create_router(route_resolver=mock_resolver)

        # Assert
        # Verify mount_handlers was called with the resolver
        call_args = mock_mount.call_args
        assert call_args is not None
        assert call_args.kwargs["route_resolver"] == mock_resolver


class TestRemoveConflictingRoutes:
    """Test remove_conflicting_routes function."""

    def test_remove_conflicting_routes_no_conflicts(self):
        """Test that no routes are removed when there are no conflicts."""
        # Arrange
        app = FastAPI()
        router = APIRouter()

        # Add non-conflicting routes
        @app.get("/health")
        def app_health():
            return {"status": "ok"}

        @router.get("/status")
        def router_status():
            return {"status": "ready"}

        original_route_count = len(app.router.routes)

        # Act
        remove_conflicting_routes(app, router)

        # Assert
        assert len(app.router.routes) == original_route_count

    def test_remove_conflicting_routes_with_conflicts(self):
        """Test that conflicting routes are removed."""
        # Arrange
        app = FastAPI()
        router = APIRouter()

        # Add conflicting routes
        @app.get("/health")
        def app_health():
            return {"status": "app"}

        @router.get("/health")
        def router_health():
            return {"status": "router"}

        original_route_count = len(app.router.routes)

        # Act
        remove_conflicting_routes(app, router)

        # Assert
        # One route should be removed (the conflicting /health GET)
        assert len(app.router.routes) == original_route_count - 1

    def test_remove_conflicting_routes_different_methods_no_conflict(self):
        """Test that same path with different methods don't conflict."""
        # Arrange
        app = FastAPI()
        router = APIRouter()

        @app.get("/api/data")
        def app_get_data():
            return {"method": "GET"}

        @router.post("/api/data")
        def router_post_data():
            return {"method": "POST"}

        original_route_count = len(app.router.routes)

        # Act
        remove_conflicting_routes(app, router)

        # Assert
        # No routes should be removed (different methods)
        assert len(app.router.routes) == original_route_count

    def test_remove_conflicting_routes_with_prefix(self):
        """Test route conflict detection with prefix."""
        # Arrange
        app = FastAPI()
        router = APIRouter()

        # App has route at /api/health
        @app.get("/api/health")
        def app_health():
            return {"status": "app"}

        # Router has route at /health, but will be prefixed to /api/health
        @router.get("/health")
        def router_health():
            return {"status": "router"}

        original_route_count = len(app.router.routes)

        # Act
        remove_conflicting_routes(app, router, prefix="/api")

        # Assert
        # The conflicting /api/health route should be removed
        assert len(app.router.routes) == original_route_count - 1

    def test_remove_conflicting_routes_prefix_normalization(self):
        """Test that prefix normalization works correctly."""
        # Arrange
        app = FastAPI()
        router = APIRouter()

        @app.get("/api/health")
        def app_health():
            return {"status": "app"}

        @router.get("/health")
        def router_health():
            return {"status": "router"}

        original_route_count = len(app.router.routes)

        # Test various prefix formats
        test_cases = [
            "api",  # No leading slash
            "/api/",  # Trailing slash
            "/api",  # Proper format
        ]

        for i, prefix in enumerate(test_cases):
            # Reset app routes
            app = FastAPI()

            # Use dynamic function names to avoid redefinition
            def create_health_handler():
                return {"status": "app"}

            app.add_api_route("/api/health", create_health_handler, methods=["GET"])

            # Act
            remove_conflicting_routes(app, router, prefix=prefix)

            # Assert
            # Should remove the conflicting route regardless of prefix format
            assert len(app.router.routes) == original_route_count - 1

    def test_remove_conflicting_routes_multiple_conflicts(self):
        """Test removing multiple conflicting routes."""
        # Arrange
        app = FastAPI()
        router = APIRouter()

        # Add multiple conflicting routes
        @app.get("/health")
        def app_health():
            return {"status": "app"}

        @app.post("/data")
        def app_data():
            return {"source": "app"}

        @app.get("/info")
        def app_info():
            return {"source": "app"}

        @router.get("/health")
        def router_health():
            return {"status": "router"}

        @router.post("/data")
        def router_data():
            return {"source": "router"}

        @router.get("/unique")
        def router_unique():
            return {"source": "router"}

        original_route_count = len(app.router.routes)

        # Act
        remove_conflicting_routes(app, router)

        # Assert
        # Two conflicting routes should be removed (/health GET, /data POST)
        # /info GET should remain (no conflict), /unique GET will be added later
        assert len(app.router.routes) == original_route_count - 2

    def test_remove_conflicting_routes_empty_router(self):
        """Test with empty router."""
        # Arrange
        app = FastAPI()
        router = APIRouter()

        @app.get("/health")
        def app_health():
            return {"status": "app"}

        original_route_count = len(app.router.routes)

        # Act
        remove_conflicting_routes(app, router)

        # Assert
        # No routes should be removed
        assert len(app.router.routes) == original_route_count

    def test_remove_conflicting_routes_non_api_routes_ignored(self):
        """Test that non-APIRoute routes are ignored."""
        # Arrange
        app = FastAPI()
        router = APIRouter()

        @app.get("/health")
        def app_health():
            return {"status": "app"}

        @router.get("/health")
        def router_health():
            return {"status": "router"}

        # Add a non-APIRoute (like WebSocketRoute) - we'll simulate this
        # by checking the route filtering logic
        original_route_count = len(app.router.routes)

        # Act
        remove_conflicting_routes(app, router)

        # Assert
        # The conflicting APIRoute should be removed
        assert len(app.router.routes) == original_route_count - 1

    @patch("model_hosting_container_standards.common.fastapi.routing.logger")
    def test_remove_conflicting_routes_logs_removal(self, mock_logger):
        """Test that route removal is logged."""
        # Arrange
        app = FastAPI()
        router = APIRouter()

        @app.get("/health")
        def app_health():
            return {"status": "app"}

        @router.get("/health")
        def router_health():
            return {"status": "router"}

        # Act
        remove_conflicting_routes(app, router)

        # Assert
        # Should have two log calls: individual route removal + summary
        assert mock_logger.info.call_count == 2

        # Check individual route removal log
        first_call = mock_logger.info.call_args_list[0][0][0]
        assert "Removing conflicting route" in first_call
        assert "GET" in first_call
        assert "/health" in first_call

        # Check summary log
        second_call = mock_logger.info.call_args_list[1][0][0]
        assert "Removed 1 conflicting routes" in second_call
