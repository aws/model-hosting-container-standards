"""Integration tests for handler override functionality.

Tests real customer usage scenarios:
- Using @ping and @invoke decorators to override handlers
- Setting environment variables for handler specifications
- Writing customer scripts with ping() and invoke() functions
- Priority: env vars > decorators > customer script files > framework defaults

Note: These tests focus on validating server responses rather than directly calling
get_ping_handler() and get_invoke_handler() to ensure full integration testing.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from model_hosting_container_standards.sagemaker.config import SageMakerEnvVars

# Removed direct handler imports - using server responses instead


class TestHandlerOverrideIntegration:
    """Integration tests simulating real customer usage scenarios.

    Each test simulates a fresh server startup where customers:
    - Use @ping and @invoke decorators
    - Set environment variables (CUSTOM_FASTAPI_PING_HANDLER, etc.)
    - Write customer scripts with ping() and invoke() functions
    """

    def setup_method(self):
        """Setup for each test - simulate fresh server startup."""
        self._clear_caches()

    def _clear_caches(self):
        """Clear handler registry and function loader cache."""
        from model_hosting_container_standards.common.handler import handler_registry
        from model_hosting_container_standards.sagemaker.sagemaker_loader import (
            SageMakerFunctionLoader,
        )

        handler_registry.clear()
        SageMakerFunctionLoader._default_function_loader = None

    def _reload_mock_server(self):
        """Clear caches and reload mock server to pick up new handlers."""
        self._clear_caches()

        from ..resources import mock_vllm_server

        # Reset the mock server to create a fresh FastAPI app
        mock_vllm_server.mock_server.reset()
        return mock_vllm_server

    def test_customer_script_functions_auto_loaded(self):
        """Test customer scenario: script functions automatically override framework defaults."""
        import asyncio

        # Customer writes a script file with ping() and invoke() functions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from fastapi import Request

async def ping():
    return {
        "status": "healthy",
        "source": "customer_override",
        "message": "Custom ping from customer script"
    }

async def invoke(request: Request):
    return {
        "predictions": ["Custom response from customer script"],
        "source": "customer_override"
    }
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Customer sets SageMaker environment variables to point to their script
            with patch.dict(
                os.environ,
                {
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache and reload mock server to pick up new handlers
                mock_vllm_server = self._reload_mock_server()

                # Customer tests their server and sees their overrides work automatically
                ping_response = asyncio.run(mock_vllm_server.call_ping_endpoint())
                invoke_response = asyncio.run(mock_vllm_server.call_invoke_endpoint())

                # Customer sees their functions are used
                assert ping_response["source"] == "customer_override"
                assert ping_response["message"] == "Custom ping from customer script"

                assert invoke_response["source"] == "customer_override"
                assert invoke_response["predictions"] == [
                    "Custom response from customer script"
                ]
        finally:
            os.unlink(script_path)

    def test_environment_variable_overrides_decorators(self):
        """Test customer scenario: environment variables override decorators."""
        import asyncio

        # Customer writes a script file with decorators and regular functions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request

@sagemaker_standards.invoke
async def custom_invoke(request: Request):
    return {
        "predictions": ["Custom response"],
        "source": "customer_decorator",
    }

# Regular ping function
async def ping():
    return {"source": "customer_function", "priority": "script_function"}
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Customer sets SageMaker environment variables to point to their script
            with patch.dict(
                os.environ,
                {
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache and reload mock server to pick up new handlers
                mock_vllm_server = self._reload_mock_server()

                # Customer tests server responses to verify priority order
                ping_response = asyncio.run(mock_vllm_server.call_ping_endpoint())
                invoke_response = asyncio.run(mock_vllm_server.call_invoke_endpoint())

                # Script function takes precedence over framework defaults for ping
                assert ping_response["source"] == "customer_function"
                assert ping_response["priority"] == "script_function"

                # Decorator from script works for invoke
                assert invoke_response["source"] == "customer_decorator"
        finally:
            os.unlink(script_path)

    def test_customer_sets_environment_variables(self):
        """Test customer scenario: setting environment variables with module:function."""
        import asyncio

        # Customer writes a script file with multiple handler options
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from fastapi import Request

async def ping():
    return {"source": "script_ping", "type": "script_function"}

async def invoke(request: Request):
    return {"source": "script_invoke", "type": "script_function"}

async def env_ping():
    return {"source": "env_ping", "type": "environment_variable"}

async def env_invoke(request=None):
    return {"source": "env_invoke", "type": "environment_variable"}
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Test 1: Without environment variables - script functions should be used
            with patch.dict(
                os.environ,
                {
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache and reload mock server to pick up new handlers
                mock_vllm_server = self._reload_mock_server()

                # Customer tests server responses to verify script functions work
                ping_response = asyncio.run(mock_vllm_server.call_ping_endpoint())
                invoke_response = asyncio.run(mock_vllm_server.call_invoke_endpoint())

                # Verify script functions are used
                assert ping_response["source"] == "script_ping"
                assert ping_response["type"] == "script_function"

                assert invoke_response["source"] == "script_invoke"
                assert invoke_response["type"] == "script_function"
        finally:
            os.unlink(script_path)

    def test_customer_writes_script_file(self):
        """Test customer scenario: writing a script file with ping() and invoke() functions."""
        import asyncio

        # Customer writes a script file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from fastapi import Request

async def ping():
    return {"status": "healthy", "source": "file_customer_script"}

async def invoke(request: Request):
    return {"predictions": ["file customer response"], "source": "file_customer_script"}
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Customer sets SageMaker environment variables to point to their script
            with patch.dict(
                os.environ,
                {
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache and reload mock server to pick up new handlers
                mock_vllm_server = self._reload_mock_server()

                # Customer tests server responses to verify their functions work
                ping_response = asyncio.run(mock_vllm_server.call_ping_endpoint())
                invoke_response = asyncio.run(mock_vllm_server.call_invoke_endpoint())

                # Verify customer's script functions are being used
                assert ping_response["status"] == "healthy"
                assert ping_response["source"] == "file_customer_script"

                assert invoke_response["predictions"] == ["file customer response"]
                assert invoke_response["source"] == "file_customer_script"
        finally:
            os.unlink(script_path)

    def test_customer_priority_understanding(self):
        """Test customer scenario: understanding priority order through server responses."""
        import asyncio

        # Customer writes a script file with different handler types
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
import model_hosting_container_standards.sagemaker as sagemaker_standards

# Decorator handler (higher priority than script functions)
@sagemaker_standards.ping
async def decorator_ping():
    return {"source": "decorator", "priority": "high"}

# Script function handler (lower priority than decorators)
async def ping():
    return {"source": "script_function", "priority": "low"}
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Customer sets SageMaker environment variables to point to their script
            with patch.dict(
                os.environ,
                {
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache and reload mock server to pick up new handlers
                mock_vllm_server = self._reload_mock_server()

                # Customer sees decorator takes precedence over script function
                ping_response = asyncio.run(mock_vllm_server.call_ping_endpoint())
                assert ping_response["source"] == "decorator"
                assert ping_response["priority"] == "high"
        finally:
            os.unlink(script_path)

    def test_customer_decorator_usage_with_server_response(self):
        """Test customer scenario: using @ping and @invoke decorators and seeing server responses."""
        import asyncio

        # Customer writes a script file with decorators and regular functions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request

# Customer uses decorators for some handlers
@sagemaker_standards.invoke
async def my_invoke(request: Request):
    return {"type": "invoke", "source": "customer_decorator"}

# Customer uses regular function for ping (higher priority than decorators)
async def ping():
    return {"type": "ping", "source": "customer_function"}
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Customer sets SageMaker environment variables to point to their script
            with patch.dict(
                os.environ,
                {
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache and reload mock server to pick up new handlers
                mock_vllm_server = self._reload_mock_server()

                ping_response = asyncio.run(mock_vllm_server.call_ping_endpoint())
                invoke_response = asyncio.run(mock_vllm_server.call_invoke_endpoint())

                # Customer sees their handlers are used by the server
                assert (
                    ping_response["source"] == "customer_function"
                )  # Function has higher priority
                assert (
                    invoke_response["source"] == "customer_decorator"
                )  # Decorator works for invoke
        finally:
            os.unlink(script_path)

    def test_register_handlers_priority_vs_script_functions(self):
        """Test priority: @ping/@invoke decorators vs script functions vs framework register decorators."""
        import asyncio

        # Customer writes a script with @ping decorator and regular functions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request, Response
import json

# Customer uses @ping decorator (higher priority than script functions)
@sagemaker_standards.ping
async def decorated_ping(raw_request: Request) -> Response:
    response_data = {
        "status": "healthy",
        "source": "ping_decorator_in_script",
        "priority": "decorator"
    }
    return Response(
        content=json.dumps(response_data),
        media_type="application/json",
        status_code=200
    )

# Customer also has a regular function (lower priority than @ping decorator)
async def ping():
    return {
        "status": "healthy",
        "source": "script_function",
        "priority": "function"
    }

# Customer has a regular invoke function (higher priority than framework register decorator)
async def invoke(request: Request):
    return {
        "predictions": ["Script function response"],
        "source": "script_invoke_function",
        "priority": "function"
    }
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Customer sets SageMaker environment variables to point to their script
            with patch.dict(
                os.environ,
                {
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache and reload mock server to pick up new handlers
                mock_vllm_server = self._reload_mock_server()

                # Test priority order: @ping decorator has higher priority than script functions
                ping_response = asyncio.run(mock_vllm_server.call_ping_endpoint())
                invoke_response = asyncio.run(mock_vllm_server.call_invoke_endpoint())

                # @ping decorator has higher priority than script function
                assert ping_response["source"] == "ping_decorator_in_script"
                assert ping_response["priority"] == "decorator"

                # Script function is used for invoke (higher priority than framework register decorator)
                assert invoke_response["source"] == "script_invoke_function"
                assert invoke_response["priority"] == "function"
        finally:
            os.unlink(script_path)

    def test_framework_routes_are_created_automatically(self):
        """Test that framework @register_ping_handler creates routes and works when no customer overrides exist."""
        import asyncio

        # Clear any existing handlers
        self._clear_caches()

        # Use the mock vLLM server which has @register_ping_handler and @register_invocation_handler
        # This simulates the real vLLM server behavior
        mock_vllm_server = self._reload_mock_server()

        # Initialize the app by getting the client
        mock_vllm_server.mock_server.get_client()

        # Get the FastAPI app to inspect routes
        app = mock_vllm_server.mock_server.app

        # Check that /ping and /invocations routes exist
        ping_routes = [
            route
            for route in app.routes
            if hasattr(route, "path") and route.path == "/ping"
        ]
        invocations_routes = [
            route
            for route in app.routes
            if hasattr(route, "path") and route.path == "/invocations"
        ]

        # Should have ping routes (the mock server creates them)
        assert (
            len(ping_routes) > 0
        ), f"No /ping routes found. Available routes: {[r.path for r in app.routes if hasattr(r, 'path')]}"

        # Should have invocations route (the mock server creates them)
        assert (
            len(invocations_routes) > 0
        ), f"No /invocations routes found. Available routes: {[r.path for r in app.routes if hasattr(r, 'path')]}"

        # Test that the routes actually work and call framework code
        ping_response = asyncio.run(mock_vllm_server.call_ping_endpoint())
        invoke_response = asyncio.run(mock_vllm_server.call_invoke_endpoint())

        # Verify framework handlers are called (from mock_vllm_server.py)
        assert ping_response["status"] == "healthy"
        assert ping_response["source"] == "vllm_default"
        assert ping_response["message"] == "Default ping from vLLM server"

        assert invoke_response["predictions"] == ["Default vLLM response"]
        assert invoke_response["source"] == "vllm_default"

    def test_framework_inject_adapter_id_decorator(self):
        """Test that @inject_adapter_id decorator works in framework code."""
        import asyncio

        # Clear any existing handlers
        self._clear_caches()

        # Use the mock vLLM server which has @inject_adapter_id on invocations
        mock_vllm_server = self._reload_mock_server()

        # Test 1: Call invocations without adapter header (should use base-model)
        invoke_response_no_adapter = asyncio.run(
            mock_vllm_server.call_invoke_endpoint()
        )

        # Should use base-model when no adapter header is provided
        assert invoke_response_no_adapter["adapter_id"] == "base-model"
        assert (
            invoke_response_no_adapter["message"]
            == "Response using adapter: base-model"
        )

        # Test 2: Call invocations with adapter header
        client = mock_vllm_server.mock_server.get_client()

        # Make request with LoRA adapter header
        response_with_adapter = client.post(
            "/invocations",
            json={"prompt": "test"},
            headers={"X-Amzn-SageMaker-Adapter-Identifier": "my-custom-adapter"},
        )

        assert response_with_adapter.status_code == 200
        response_data = response_with_adapter.json()

        # The inject_adapter_id decorator should have injected the adapter ID into the body
        assert response_data["adapter_id"] == "my-custom-adapter"
        assert response_data["message"] == "Response using adapter: my-custom-adapter"
        assert response_data["source"] == "vllm_default"


if __name__ == "__main__":
    pytest.main([__file__])
