#!/usr/bin/env python3
"""
Integration test for middleware loader functionality.
Tests that customer middlewares get called correctly with a mock vLLM server.
"""

import importlib
import os
import tempfile
from unittest.mock import patch

from model_hosting_container_standards.sagemaker.config import SageMakerEnvVars


class TestMiddlewareIntegration:
    """Integration test for middleware with mock vLLM server."""

    def setup_method(self):
        """Setup for each test - simulate fresh server startup."""
        self._clear_caches()

    def _clear_caches(self):
        """Clear middleware registry and function loader cache."""
        from model_hosting_container_standards.common.fastapi.middleware import (
            middleware_registry,
        )
        from model_hosting_container_standards.common.fastapi.middleware.source.decorator_loader import (
            decorator_loader,
        )
        from model_hosting_container_standards.sagemaker.sagemaker_loader import (
            SageMakerFunctionLoader,
        )

        middleware_registry.clear_middlewares()
        # Clear decorator loader state
        decorator_loader.clear()
        SageMakerFunctionLoader._default_function_loader = None

    def _reload_mock_vllm_server(self):
        """Clear caches and reload mock vLLM server to pick up new middlewares."""
        self._clear_caches()

        # Trigger loading of customer scripts
        from model_hosting_container_standards.sagemaker.sagemaker_loader import (
            SageMakerFunctionLoader,
        )

        SageMakerFunctionLoader.get_function_loader()

        from ..resources import mock_vllm_server

        importlib.reload(mock_vllm_server)
        return mock_vllm_server

    def test_customer_middleware_auto_loaded(self):
        """Test that customer middlewares are automatically loaded by plugin."""
        # Customer writes a middleware script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from model_hosting_container_standards.common.fastapi.middleware import register_middleware, output_formatter

@register_middleware("throttle")
async def customer_throttle_middleware(request, call_next):
    response = await call_next(request)
    response.headers["X-Customer-Throttle"] = "applied"
    order = response.headers.get("X-Middleware-Order", "")
    response.headers["X-Middleware-Order"] = order + "throttle,"
    return response

@output_formatter
async def customer_output_formatter(response):
    response.headers["X-Customer-Processed"] = "true"
    order = response.headers.get("X-Middleware-Order", "")
    response.headers["X-Middleware-Order"] = order + "pre_post_process,"
    return response
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Set environment variables to point to customer script
            with patch.dict(
                os.environ,
                {
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                },
            ):
                # Clear cache and reload mock server to pick up new middlewares
                self._reload_mock_vllm_server()

                # Test that middlewares are registered
                # Trigger middleware loading and registration
                from model_hosting_container_standards.common.custom_code_ref_resolver.function_loader import (
                    FunctionLoader,
                )
                from model_hosting_container_standards.common.fastapi.middleware import (
                    middleware_registry,
                )
                from model_hosting_container_standards.common.fastapi.middleware.source.decorator_loader import (
                    decorator_loader,
                )

                function_loader = FunctionLoader()
                middleware_registry.load_middlewares(function_loader)

                assert middleware_registry.has_middleware("throttle")
                assert (
                    decorator_loader.post_fn is not None
                )  # output formatter was registered

                # Create a real FastAPI app to test middleware execution
                from fastapi import FastAPI, Request
                from fastapi.testclient import TestClient
                from starlette.middleware.base import BaseHTTPMiddleware

                app = FastAPI(title="Test vLLM Server")

                # Add some mock vLLM middlewares to simulate real server
                class MockVLLMMiddleware(BaseHTTPMiddleware):
                    async def dispatch(self, request: Request, call_next):
                        response = await call_next(request)
                        response.headers["X-vLLM-Server"] = "mock"
                        order = response.headers.get("X-Middleware-Order", "")
                        response.headers["X-Middleware-Order"] = order + "vllm,"
                        return response

                app.add_middleware(MockVLLMMiddleware)

                # Add test endpoint
                @app.post("/generate")
                async def generate():
                    return {"text": "Generated response", "model": "mock-vllm"}

                # Load middlewares into the app
                from model_hosting_container_standards.common.custom_code_ref_resolver.function_loader import (
                    FunctionLoader,
                )
                from model_hosting_container_standards.common.fastapi.middleware.core import (
                    load_middlewares,
                )

                function_loader = FunctionLoader()
                load_middlewares(app, function_loader)

                # Test HTTP request to verify middleware execution
                client = TestClient(app)
                response = client.post("/generate", json={"prompt": "test"})

                # Verify response
                assert response.status_code == 200
                assert response.json()["text"] == "Generated response"

                # Verify customer middlewares were executed
                assert "X-Customer-Throttle" in response.headers
                assert "X-Customer-Processed" in response.headers

                # Verify vLLM middleware still works
                assert "X-vLLM-Server" in response.headers

                # Verify middleware execution order
                execution_order = response.headers.get("X-Middleware-Order", "").rstrip(
                    ","
                )
                order_parts = execution_order.split(",") if execution_order else []

                print(f"Execution order: {order_parts}")  # Debug output

                # Headers show response processing order (reverse of request order)
                # Request order should be: throttle -> vllm -> pre_post_process
                # Response order (what we see in headers): pre_post_process -> vllm -> throttle
                expected_response_order = ["pre_post_process", "vllm", "throttle"]
                actual_middlewares = [
                    mw for mw in expected_response_order if mw in order_parts
                ]

                # Check that middlewares execute in the expected order
                for i in range(len(actual_middlewares) - 1):
                    current_mw = actual_middlewares[i]
                    next_mw = actual_middlewares[i + 1]
                    current_index = order_parts.index(current_mw)
                    next_index = order_parts.index(next_mw)
                    assert current_index < next_index, (
                        f"Middleware order violation: {current_mw} should execute before {next_mw}. "
                        f"Actual order: {execution_order}"
                    )

                # Verify the response processing order matches expected
                # This confirms the request processing order is: throttle -> vllm -> pre_post_process
                if "pre_post_process" in order_parts and "vllm" in order_parts:
                    process_index = order_parts.index("pre_post_process")
                    vllm_index = order_parts.index("vllm")
                    assert (
                        process_index < vllm_index
                    ), f"Pre/Post process should complete response processing before vLLM: {execution_order}"

                if "vllm" in order_parts and "throttle" in order_parts:
                    vllm_index = order_parts.index("vllm")
                    throttle_index = order_parts.index("throttle")
                    assert (
                        vllm_index < throttle_index
                    ), f"vLLM should complete response processing before throttle: {execution_order}"

        finally:
            os.unlink(script_path)
