"""Tests for handler resolver functionality."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from model_hosting_container_standards.common.fastapi.config import FastAPIEnvVars
from model_hosting_container_standards.common.handler import handler_registry
from model_hosting_container_standards.sagemaker.config import SageMakerEnvVars
from model_hosting_container_standards.sagemaker.handler_resolver import (
    SageMakerHandlerResolver,
)
from model_hosting_container_standards.sagemaker.sagemaker_loader import (
    SageMakerFunctionLoader,
)


class TestHandlerResolver:
    """Test handler resolution logic."""

    def setup_method(self):
        """Setup for each test."""
        self.resolver = SageMakerHandlerResolver()
        # Clear registry before each test
        handler_registry.clear()
        # Clear cached loader
        SageMakerFunctionLoader._default_function_loader = None

    def test_resolve_ping_from_env_var(self):
        """Test resolving ping handler from environment variable."""
        # Mock environment variable
        with patch.dict(
            os.environ, {FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER: "os.path:exists"}
        ):
            handler_info = self.resolver.resolve_handler("ping")
            assert handler_info is not None
            assert callable(handler_info.func)
            assert handler_info.route_kwargs == {}

    def test_resolve_ping_from_registry(self):
        """Test resolving ping handler from registry (decorator)."""
        # Mock a decorated handler
        mock_handler = MagicMock()
        handler_registry.set_decorator_handler("ping", mock_handler)

        handler_info = self.resolver.resolve_handler("ping")
        assert handler_info is not None
        assert handler_info.func == mock_handler

    def test_resolve_ping_from_customer_script(self):
        """Test resolving ping handler from customer script."""
        # Create a temporary customer script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
def custom_sagemaker_ping_handler():
    return "customer ping"
"""
            )
            script_path = f.name

        try:
            # Mock environment to point to our temp script
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            with patch.dict(
                os.environ,
                {
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache to pick up new environment
                SageMakerFunctionLoader._default_function_loader = None

                handler_info = self.resolver.resolve_handler("ping")
                assert handler_info is not None
                assert callable(handler_info.func)
                assert handler_info.func() == "customer ping"
                assert handler_info.route_kwargs == {}
        finally:
            os.unlink(script_path)

    def test_priority_order(self):
        """Test that handlers are resolved in correct priority order."""
        # Setup registry handler
        registry_handler = MagicMock(return_value="registry")
        handler_registry.set_decorator_handler("ping", registry_handler)

        # Create customer script with ping function
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
def custom_sagemaker_ping_handler():
    return "customer"
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Test: Registry handler should take precedence over customer script
            with patch.dict(
                os.environ,
                {
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache to pick up new environment
                SageMakerFunctionLoader._default_function_loader = None

                handler_info = self.resolver.resolve_handler("ping")
                assert handler_info is not None
                assert handler_info.func == registry_handler

            # Test: Env var should take precedence over registry
            with patch.dict(
                os.environ,
                {
                    FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER: "os.path:exists",
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache to pick up new environment
                SageMakerFunctionLoader._default_function_loader = None

                handler_info = self.resolver.resolve_handler("ping")
                assert handler_info is not None
                assert handler_info.func != registry_handler
                assert callable(handler_info.func)
        finally:
            os.unlink(script_path)

    def test_no_handler_found(self):
        """Test when no handler is found."""
        handler = self.resolver.resolve_handler("ping")
        assert handler is None

    def test_resolver_methods(self):
        """Test resolver methods for both ping and invoke handlers."""
        mock_handler = MagicMock()
        handler_registry.set_decorator_handler("ping", mock_handler)

        # Test resolve_handler for ping
        handler_info = self.resolver.resolve_handler("ping")
        assert handler_info is not None
        assert handler_info.func == mock_handler

        # Test resolve_handler for invoke
        handler_registry.set_decorator_handler("invoke", mock_handler)
        handler_info = self.resolver.resolve_handler("invoke")
        assert handler_info is not None
        assert handler_info.func == mock_handler

    def test_router_path_in_env_var(self):
        """Test that router paths in environment variables are handled correctly."""
        # Test with router path - should return None for callable handlers
        with patch.dict(
            os.environ, {FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER: "/health"}
        ):
            handler = self.resolver.resolve_handler("ping")
            # Should fall back to next priority (registry/customer script)
            assert handler is None

        # Test with another router path format
        with patch.dict(
            os.environ, {FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER: "/api/v1/status"}
        ):
            handler = self.resolver.resolve_handler("ping")
            assert handler is None

    def test_resolve_invoke_handler(self):
        """Test resolving invoke handler."""
        # Test with registry handler
        mock_handler = MagicMock()
        handler_registry.set_decorator_handler("invoke", mock_handler)

        handler_info = self.resolver.resolve_handler("invoke")
        assert handler_info is not None
        assert handler_info.func == mock_handler

    def test_resolve_invoke_from_customer_script(self):
        """Test resolving invoke handler from customer script."""
        # Create a temporary customer script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
def custom_sagemaker_invocation_handler(data):
    return f"customer invoke: {data}"
"""
            )
            script_path = f.name

        try:
            # Mock environment to point to our temp script
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            with patch.dict(
                os.environ,
                {
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache to pick up new environment
                SageMakerFunctionLoader._default_function_loader = None

                handler_info = self.resolver.resolve_handler("invoke")
                assert handler_info is not None
                assert callable(handler_info.func)
                assert handler_info.func("test") == "customer invoke: test"
                assert handler_info.route_kwargs == {}
        finally:
            os.unlink(script_path)


if __name__ == "__main__":
    pytest.main([__file__])
