"""Unit tests for sagemaker_router module.

Tests the core routing functionality including:
- _replace_route function for route replacement
- setup_ping_invoke_routes function for SageMaker endpoint setup
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute


class TestReplaceRoute:
    """Test _replace_route function."""

    def setup_method(self):
        """Setup for each test."""
        self.app = FastAPI()

    async def dummy_handler(self):
        """Dummy handler for testing."""
        return {"message": "test"}

    async def new_handler(self):
        """New handler for replacement testing."""
        return {"message": "new"}

    def test_adds_new_route_when_none_exists(self):
        """Test that _replace_route adds a new route when no existing route is found."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            _replace_route,
        )

        # FastAPI automatically adds default routes (openapi, docs, etc.)
        # Count only APIRoute instances for our test
        initial_api_routes = [
            r for r in self.app.router.routes if isinstance(r, APIRoute)
        ]
        assert len(initial_api_routes) == 0

        # Add new route
        _replace_route(
            app=self.app,
            path="/test",
            handler=self.dummy_handler,
            methods=["GET"],
            tags=["test"],
            summary="Test endpoint",
        )

        # Verify route was added
        api_routes = [r for r in self.app.router.routes if isinstance(r, APIRoute)]
        assert len(api_routes) == 1

        route = api_routes[0]
        assert route.path == "/test"
        assert "GET" in route.methods
        assert route.tags == ["test"]
        assert route.summary == "Test endpoint"
        assert route.endpoint == self.dummy_handler

    def test_replaces_existing_route_completely(self):
        """Test that _replace_route completely replaces existing route with new configuration."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            _replace_route,
        )

        # Add initial routes
        self.app.get("/before")(self.dummy_handler)
        self.app.get("/test", tags=["old"], summary="Old summary")(self.dummy_handler)
        self.app.get("/after")(self.dummy_handler)

        initial_api_routes = [
            r for r in self.app.router.routes if isinstance(r, APIRoute)
        ]
        assert len(initial_api_routes) == 3

        # Replace the test route completely
        _replace_route(
            app=self.app,
            path="/test",
            handler=self.new_handler,
            methods=["POST"],  # Different method
            tags=["new"],
            summary="New test endpoint",
        )

        # Verify route was replaced completely
        updated_api_routes = [
            r for r in self.app.router.routes if isinstance(r, APIRoute)
        ]
        assert len(updated_api_routes) == 3

        # Find the new test route
        test_routes = [r for r in updated_api_routes if r.path == "/test"]
        assert len(test_routes) == 1

        test_route = test_routes[0]
        assert "POST" in test_route.methods
        assert "GET" not in test_route.methods  # Old method completely replaced
        assert test_route.tags == ["new"]  # New tags used
        assert test_route.summary == "New test endpoint"  # New summary used
        assert test_route.endpoint == self.new_handler

        # Verify other routes still exist
        before_routes = [r for r in updated_api_routes if r.path == "/before"]
        after_routes = [r for r in updated_api_routes if r.path == "/after"]
        assert len(before_routes) == 1
        assert len(after_routes) == 1

    def test_uses_default_methods_when_not_specified(self):
        """Test that _replace_route uses default GET method when methods not specified."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            _replace_route,
        )

        # Add route with multiple methods
        @self.app.api_route("/test", methods=["GET", "POST", "PUT"])
        async def original_handler():
            return {"original": True}

        # Replace without specifying methods (should default to GET)
        _replace_route(
            app=self.app,
            path="/test",
            handler=self.new_handler,
        )

        # Verify default GET method is used (old methods not preserved)
        api_routes = [r for r in self.app.router.routes if isinstance(r, APIRoute)]
        test_route = next(r for r in api_routes if r.path == "/test")
        assert test_route.methods == {"GET"}  # Only GET method (default)
        assert test_route.endpoint == self.new_handler

    def test_uses_empty_tags_when_not_specified(self):
        """Test that _replace_route uses empty tags when tags not specified."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            _replace_route,
        )

        # Add route with tags
        self.app.get("/test", tags=["original", "test"])(self.dummy_handler)

        # Replace without specifying tags (should use empty list)
        _replace_route(
            app=self.app,
            path="/test",
            handler=self.new_handler,
        )

        # Verify empty tags are used (old tags not preserved)
        api_routes = [r for r in self.app.router.routes if isinstance(r, APIRoute)]
        test_route = next(r for r in api_routes if r.path == "/test")
        assert test_route.tags == []  # Empty tags (default)
        assert test_route.endpoint == self.new_handler

    def test_uses_none_summary_when_not_specified(self):
        """Test that _replace_route uses None summary when summary not specified."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            _replace_route,
        )

        # Add route with summary
        self.app.get("/test", summary="Original summary")(self.dummy_handler)

        # Replace without specifying summary (should use None)
        _replace_route(
            app=self.app,
            path="/test",
            handler=self.new_handler,
        )

        # Verify None summary is used (old summary not preserved)
        api_routes = [r for r in self.app.router.routes if isinstance(r, APIRoute)]
        test_route = next(r for r in api_routes if r.path == "/test")
        assert test_route.summary is None  # None summary (default)
        assert test_route.endpoint == self.new_handler

    def test_overrides_route_properties_when_specified(self):
        """Test that _replace_route properly overrides all properties when specified."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            _replace_route,
        )

        # Add route with original properties
        self.app.get("/test", tags=["original"], summary="Original")(self.dummy_handler)

        # Replace with new properties
        _replace_route(
            app=self.app,
            path="/test",
            handler=self.new_handler,
            methods=["POST"],  # Different method
            tags=["new"],
            summary="New summary",
        )

        # Verify all properties are properly overridden
        api_routes = [r for r in self.app.router.routes if isinstance(r, APIRoute)]
        test_routes = [r for r in api_routes if r.path == "/test"]
        assert len(test_routes) == 1  # Should have replaced, not added

        test_route = test_routes[0]
        assert "POST" in test_route.methods  # New method used
        assert "GET" not in test_route.methods  # Old method removed
        assert test_route.tags == ["new"]  # New tags used
        assert test_route.summary == "New summary"  # New summary used
        assert test_route.endpoint == self.new_handler  # New handler used

    def test_replaces_route_regardless_of_methods(self):
        """Test that _replace_route replaces route regardless of HTTP methods."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            _replace_route,
        )

        # Add route with GET method
        self.app.get("/test", tags=["original"])(self.dummy_handler)

        # Replace with POST method (should completely replace the GET route)
        _replace_route(
            app=self.app,
            path="/test",
            handler=self.new_handler,
            methods=["POST"],
            tags=["new"],
            summary="New summary",
        )

        # Verify old route was completely replaced
        api_routes = [r for r in self.app.router.routes if isinstance(r, APIRoute)]
        test_routes = [r for r in api_routes if r.path == "/test"]
        assert len(test_routes) == 1  # Only one route should exist

        test_route = test_routes[0]
        assert "POST" in test_route.methods
        assert "GET" not in test_route.methods  # Old GET method removed
        assert test_route.endpoint == self.new_handler
        assert test_route.tags == ["new"]
        assert test_route.summary == "New summary"

    def test_clears_openapi_schema(self):
        """Test that _replace_route clears the OpenAPI schema for regeneration."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            _replace_route,
        )

        # Set a mock OpenAPI schema
        self.app.openapi_schema = {"mock": "schema"}

        # Add and replace route
        _replace_route(
            app=self.app,
            path="/test",
            handler=self.dummy_handler,
        )

        # Verify schema was cleared
        assert self.app.openapi_schema is None

    def test_replaces_route_with_exact_methods_specified(self):
        """Test that _replace_route uses exactly the methods specified."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            _replace_route,
        )

        # Add route with multiple methods including HEAD and OPTIONS (auto-added by FastAPI)
        @self.app.api_route("/test", methods=["GET", "POST"])
        async def original_handler():
            return {"original": True}

        # Replace specifying only GET (should use exactly GET, not preserve old methods)
        _replace_route(
            app=self.app,
            path="/test",
            handler=self.new_handler,
            methods=["GET"],
        )

        # Verify route was replaced with exact methods specified
        updated_api_routes = [
            r for r in self.app.router.routes if isinstance(r, APIRoute)
        ]
        updated_test_route = next(r for r in updated_api_routes if r.path == "/test")
        assert updated_test_route.endpoint == self.new_handler
        # Should only have GET method (and any auto-added by FastAPI like HEAD/OPTIONS)
        assert "GET" in updated_test_route.methods
        assert "POST" not in updated_test_route.methods  # Old POST method removed

    @patch("model_hosting_container_standards.sagemaker.sagemaker_router.logger")
    def test_logs_route_operations(self, mock_logger):
        """Test that _replace_route logs appropriate messages."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            _replace_route,
        )

        # Test adding new route
        _replace_route(
            app=self.app,
            path="/new",
            handler=self.dummy_handler,
        )

        # Verify log for new route
        mock_logger.info.assert_called_with("Added new /new route: dummy_handler")

        # Reset mock
        mock_logger.reset_mock()

        # Add existing route and replace it
        self.app.get("/existing")(self.dummy_handler)
        _replace_route(
            app=self.app,
            path="/existing",
            handler=self.new_handler,
        )

        # Verify logs for replacement
        mock_logger.info.assert_any_call(
            "Removing existing /existing route: dummy_handler"
        )
        mock_logger.info.assert_any_call("Replaced /existing route: new_handler")


class TestSetupPingInvokeRoutes:
    """Test setup_ping_invoke_routes function."""

    def setup_method(self):
        """Setup for each test."""
        self.app = FastAPI()

    async def mock_ping_handler(self):
        """Mock ping handler."""
        return {"status": "healthy", "source": "mock"}

    async def mock_invoke_handler(self, request: Request):
        """Mock invoke handler."""
        return {"predictions": ["mock response"]}

    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_ping_handler"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_invoke_handler"
    )
    def test_returns_unchanged_app_when_no_handlers(
        self, mock_get_invoke, mock_get_ping
    ):
        """Test that setup_ping_invoke_routes returns unchanged app when no handlers found."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            setup_ping_invoke_routes,
        )

        # Mock no handlers found
        mock_get_ping.return_value = None
        mock_get_invoke.return_value = None

        # Setup routes
        result = setup_ping_invoke_routes(self.app)

        # Verify app is returned unchanged
        assert result is self.app
        # FastAPI adds default routes, check only APIRoute instances
        api_routes = [r for r in self.app.router.routes if isinstance(r, APIRoute)]
        assert len(api_routes) == 0

        # Verify resolvers were called
        mock_get_ping.assert_called_once()
        mock_get_invoke.assert_called_once()

    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_ping_handler"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_invoke_handler"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router._replace_route"
    )
    def test_sets_up_ping_route_when_handler_exists(
        self, mock_replace_route, mock_get_invoke, mock_get_ping
    ):
        """Test that setup_ping_invoke_routes sets up /ping route when ping handler exists."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            setup_ping_invoke_routes,
        )

        # Mock ping handler exists, invoke handler doesn't
        mock_get_ping.return_value = self.mock_ping_handler
        mock_get_invoke.return_value = None

        # Setup routes
        result = setup_ping_invoke_routes(self.app)

        # Verify app is returned
        assert result is self.app

        # Verify _replace_route was called for ping only
        mock_replace_route.assert_called_once_with(
            app=self.app,
            path="/ping",
            handler=self.mock_ping_handler,
            methods=["GET"],
            tags=["health"],
            summary="Health check endpoint",
        )

    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_ping_handler"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_invoke_handler"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router._replace_route"
    )
    def test_sets_up_invoke_route_when_handler_exists(
        self, mock_replace_route, mock_get_invoke, mock_get_ping
    ):
        """Test that setup_ping_invoke_routes sets up /invocations route when invoke handler exists."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            setup_ping_invoke_routes,
        )

        # Mock invoke handler exists, ping handler doesn't
        mock_get_ping.return_value = None
        mock_get_invoke.return_value = self.mock_invoke_handler

        # Setup routes
        result = setup_ping_invoke_routes(self.app)

        # Verify app is returned
        assert result is self.app

        # Verify _replace_route was called for invoke only
        mock_replace_route.assert_called_once_with(
            app=self.app,
            path="/invocations",
            handler=self.mock_invoke_handler,
            methods=["POST"],
            tags=["inference"],
            summary="Model inference endpoint",
        )

    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_ping_handler"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_invoke_handler"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router._replace_route"
    )
    def test_sets_up_both_routes_when_both_handlers_exist(
        self, mock_replace_route, mock_get_invoke, mock_get_ping
    ):
        """Test that setup_ping_invoke_routes sets up both routes when both handlers exist."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            setup_ping_invoke_routes,
        )

        # Mock both handlers exist
        mock_get_ping.return_value = self.mock_ping_handler
        mock_get_invoke.return_value = self.mock_invoke_handler

        # Setup routes
        result = setup_ping_invoke_routes(self.app)

        # Verify app is returned
        assert result is self.app

        # Verify _replace_route was called for both routes
        assert mock_replace_route.call_count == 2

        # Check ping route call
        ping_call = mock_replace_route.call_args_list[0]
        assert ping_call[1]["app"] is self.app
        assert ping_call[1]["path"] == "/ping"
        assert ping_call[1]["handler"] == self.mock_ping_handler
        assert ping_call[1]["methods"] == ["GET"]
        assert ping_call[1]["tags"] == ["health"]
        assert ping_call[1]["summary"] == "Health check endpoint"

        # Check invoke route call
        invoke_call = mock_replace_route.call_args_list[1]
        assert invoke_call[1]["app"] is self.app
        assert invoke_call[1]["path"] == "/invocations"
        assert invoke_call[1]["handler"] == self.mock_invoke_handler
        assert invoke_call[1]["methods"] == ["POST"]
        assert invoke_call[1]["tags"] == ["inference"]
        assert invoke_call[1]["summary"] == "Model inference endpoint"

    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_ping_handler"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_invoke_handler"
    )
    @patch("model_hosting_container_standards.sagemaker.sagemaker_router.logger")
    def test_logs_setup_operations(self, mock_logger, mock_get_invoke, mock_get_ping):
        """Test that setup_ping_invoke_routes logs appropriate messages."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            setup_ping_invoke_routes,
        )

        # Test with no handlers
        mock_get_ping.return_value = None
        mock_get_invoke.return_value = None

        setup_ping_invoke_routes(self.app)

        # Verify logging
        mock_logger.info.assert_any_call(
            "Setting up /ping and /invocations routes in FastAPI app"
        )
        mock_logger.info.assert_any_call(
            "No custom handlers found, returning app unchanged"
        )

        # Reset mock
        mock_logger.reset_mock()

        # Test with handlers
        mock_get_ping.return_value = self.mock_ping_handler
        mock_get_invoke.return_value = self.mock_invoke_handler

        setup_ping_invoke_routes(self.app)

        # Verify logging
        mock_logger.info.assert_any_call(
            "Setting up /ping and /invocations routes in FastAPI app"
        )
        mock_logger.info.assert_any_call(
            "/ping and /invocations routes setup completed"
        )

    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_ping_handler"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sagemaker_router.get_invoke_handler"
    )
    def test_integration_with_real_replace_route(self, mock_get_invoke, mock_get_ping):
        """Test integration between setup_ping_invoke_routes and _replace_route without mocking _replace_route."""
        from model_hosting_container_standards.sagemaker.sagemaker_router import (
            setup_ping_invoke_routes,
        )

        # Mock handlers exist
        mock_get_ping.return_value = self.mock_ping_handler
        mock_get_invoke.return_value = self.mock_invoke_handler

        # Add some existing routes to test route replacement
        @self.app.get("/ping", tags=["old"])
        async def old_ping():
            return {"old": True}

        @self.app.post("/invocations", tags=["old"])
        async def old_invoke():
            return {"old": True}

        initial_route_count = len(self.app.router.routes)

        # Setup routes
        result = setup_ping_invoke_routes(self.app)

        # Verify app is returned
        assert result is self.app

        # Verify route count is unchanged (routes replaced, not added)
        assert len(self.app.router.routes) == initial_route_count

        # Verify routes were replaced with new handlers
        api_routes = [r for r in self.app.router.routes if isinstance(r, APIRoute)]

        ping_route = next((r for r in api_routes if r.path == "/ping"), None)
        assert ping_route is not None
        assert ping_route.endpoint == self.mock_ping_handler
        assert "GET" in ping_route.methods
        # New implementation properly uses new tags
        assert ping_route.tags == ["health"]  # New tags are used
        assert ping_route.summary == "Health check endpoint"

        invoke_route = next((r for r in api_routes if r.path == "/invocations"), None)
        assert invoke_route is not None
        assert invoke_route.endpoint == self.mock_invoke_handler
        assert "POST" in invoke_route.methods
        # New implementation properly uses new tags
        assert invoke_route.tags == ["inference"]  # New tags are used
        assert invoke_route.summary == "Model inference endpoint"


if __name__ == "__main__":
    pytest.main([__file__])
