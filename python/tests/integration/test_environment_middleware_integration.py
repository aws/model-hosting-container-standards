"""Integration tests for environment variable middleware loading."""

import os
import tempfile
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from model_hosting_container_standards.common.fastapi.middleware import (
    middleware_registry,
)
from model_hosting_container_standards.common.fastapi.middleware.source.decorator_loader import (
    decorator_loader,
)
from model_hosting_container_standards.sagemaker import bootstrap
from model_hosting_container_standards.sagemaker.config import SageMakerEnvVars


class TestEnvironmentMiddlewareIntegration:
    """Integration tests for environment variable middleware loading."""

    def setup_method(self):
        """Clear state before each test."""
        # Clear middleware registry
        middleware_registry.clear_middlewares()

        # Clear decorator loader
        decorator_loader.clear()

        # Clear SageMaker function loader cache
        from model_hosting_container_standards.sagemaker.sagemaker_loader import (
            SageMakerFunctionLoader,
        )

        SageMakerFunctionLoader._default_function_loader = None

        # Clear relevant environment variables
        env_vars = [
            "CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE",
            "CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS",
            "CUSTOM_PRE_PROCESS",
            "CUSTOM_POST_PROCESS",
            SageMakerEnvVars.SAGEMAKER_MODEL_PATH,
            SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME,
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_throttle_middleware_from_env_var(self):
        """Test loading throttle middleware from environment variable."""
        # Create a test middleware script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
async def throttle_middleware(request, call_next):
    # Add custom header to identify this middleware ran
    response = await call_next(request)
    response.headers["X-Throttle-Applied"] = "true"
    return response
"""
            )
            script_path = f.name

        try:
            # Set environment variable to point to the middleware
            script_name = os.path.basename(script_path)
            script_dir = os.path.dirname(script_path)

            with patch.dict(
                os.environ,
                {
                    "CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE": f"{script_name}:throttle_middleware",
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                },
            ):
                # Test with FastAPI app
                app = FastAPI()

                @app.get("/test")
                def test_endpoint():
                    return {"message": "test"}

                # Bootstrap SageMaker with middleware loading
                bootstrap(app)

                # Verify middleware was registered
                assert middleware_registry.has_middleware("throttle")

                # Test the middleware works
                client = TestClient(app)
                response = client.get("/test")

                assert response.status_code == 200
                assert response.headers.get("X-Throttle-Applied") == "true"
                assert response.json() == {"message": "test"}

        finally:
            # Clean up
            os.unlink(script_path)

    def test_pre_post_process_middleware_from_env_var(self):
        """Test loading pre/post process middleware from environment variable."""
        # Create a test middleware script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
async def pre_post_middleware(request, call_next):
    # Add request header
    request.headers.__dict__.setdefault("_list", []).append(("X-Pre-Process", "true"))

    response = await call_next(request)

    # Add response header
    response.headers["X-Post-Process"] = "true"
    return response
"""
            )
            script_path = f.name

        try:
            script_name = os.path.basename(script_path)
            script_dir = os.path.dirname(script_path)

            with patch.dict(
                os.environ,
                {
                    "CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS": f"{script_name}:pre_post_middleware",
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                },
            ):
                # Test with FastAPI app
                app = FastAPI()

                @app.get("/test")
                def test_endpoint():
                    return {"message": "test"}

                # Bootstrap SageMaker with middleware loading
                bootstrap(app)

                # Verify middleware was registered
                assert middleware_registry.has_middleware("pre_post_process")

                client = TestClient(app)
                response = client.get("/test")

                assert response.status_code == 200
                assert response.headers.get("X-Post-Process") == "true"

        finally:
            os.unlink(script_path)

    def test_separate_pre_post_functions_combination(self):
        """Test loading separate pre and post functions that get combined."""
        # Create test functions script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
async def pre_process_func(request):
    # Modify request (in real scenario)
    request.state.pre_processed = True
    return request

async def post_process_func(response):
    # Modify response
    response.headers["X-Post-Processed"] = "true"
    return response
"""
            )
            script_path = f.name

        try:
            script_name = os.path.basename(script_path)
            script_dir = os.path.dirname(script_path)

            with patch.dict(
                os.environ,
                {
                    "CUSTOM_PRE_PROCESS": f"{script_name}:pre_process_func",
                    "CUSTOM_POST_PROCESS": f"{script_name}:post_process_func",
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                },
            ):
                # Test with FastAPI app
                app = FastAPI()

                @app.get("/test")
                def test_endpoint(request: Request):
                    # Check if pre-processing happened
                    pre_processed = getattr(request.state, "pre_processed", False)
                    return {"message": "test", "pre_processed": pre_processed}

                # Bootstrap SageMaker with middleware loading
                bootstrap(app)

                # Verify combined middleware was registered
                assert middleware_registry.has_middleware("pre_post_process")

                client = TestClient(app)
                response = client.get("/test")

                assert response.status_code == 200
                assert response.headers.get("X-Post-Processed") == "true"
                # Note: request.state might not persist through middleware in test client

        finally:
            os.unlink(script_path)

    def test_env_var_priority_over_decorators(self):
        """Test that environment variables take priority over decorators."""
        # Create environment middleware script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
async def env_throttle_middleware(request, call_next):
    response = await call_next(request)
    response.headers["X-Middleware-Source"] = "environment"
    return response
"""
            )
            script_path = f.name

        try:
            # Register decorator middleware first
            async def decorator_throttle_middleware(request, call_next):
                response = await call_next(request)
                response.headers["X-Middleware-Source"] = "decorator"
                return response

            decorator_loader.set_middleware("throttle", decorator_throttle_middleware)

            script_name = os.path.basename(script_path)
            script_dir = os.path.dirname(script_path)

            with patch.dict(
                os.environ,
                {
                    "CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE": f"{script_name}:env_throttle_middleware",
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                },
            ):
                # Test with FastAPI app
                app = FastAPI()

                @app.get("/test")
                def test_endpoint():
                    return {"message": "test"}

                # Bootstrap SageMaker with middleware loading
                bootstrap(app)

                client = TestClient(app)
                response = client.get("/test")

                # Should use environment middleware, not decorator
                assert response.headers.get("X-Middleware-Source") == "environment"

        finally:
            os.unlink(script_path)

    def test_invalid_env_var_middleware_graceful_failure(self):
        """Test that invalid environment variable middleware fails gracefully."""
        with patch.dict(
            os.environ,
            {
                "CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE": "nonexistent.module:nonexistent_function",
            },
        ):
            # Test with FastAPI app
            app = FastAPI()

            # Should not raise exception
            bootstrap(app)

            # Should not have registered any middleware
            assert not middleware_registry.has_middleware("throttle")

    def test_multiple_env_var_middlewares(self):
        """Test loading multiple middlewares from environment variables."""
        # Create script with multiple middlewares
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
async def throttle_func(request, call_next):
    response = await call_next(request)
    response.headers["X-Throttle"] = "applied"
    return response

async def pre_post_func(request, call_next):
    response = await call_next(request)
    response.headers["X-PrePost"] = "applied"
    return response
"""
            )
            script_path = f.name

        try:
            script_name = os.path.basename(script_path)
            script_dir = os.path.dirname(script_path)

            with patch.dict(
                os.environ,
                {
                    "CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE": f"{script_name}:throttle_func",
                    "CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS": f"{script_name}:pre_post_func",
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                },
            ):
                # Test with FastAPI app
                app = FastAPI()

                @app.get("/test")
                def test_endpoint():
                    return {"message": "test"}

                # Bootstrap SageMaker with middleware loading
                bootstrap(app)

                # Both should be registered
                assert middleware_registry.has_middleware("throttle")
                assert middleware_registry.has_middleware("pre_post_process")

                client = TestClient(app)
                response = client.get("/test")

                # Both middlewares should have run
                assert response.headers.get("X-Throttle") == "applied"
                assert response.headers.get("X-PrePost") == "applied"

        finally:
            os.unlink(script_path)

    def test_direct_pre_post_takes_priority_over_separate(self):
        """Test that direct pre_post_process env var takes priority over separate pre/post."""
        # Create script with both types
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
async def direct_pre_post(request, call_next):
    response = await call_next(request)
    response.headers["X-Middleware-Type"] = "direct"
    return response

async def separate_pre(request):
    return request

async def separate_post(response):
    response.headers["X-Middleware-Type"] = "separate"
    return response
"""
            )
            script_path = f.name

        try:
            script_name = os.path.basename(script_path)
            script_dir = os.path.dirname(script_path)

            with patch.dict(
                os.environ,
                {
                    "CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS": f"{script_name}:direct_pre_post",
                    "CUSTOM_PRE_PROCESS": f"{script_name}:separate_pre",
                    "CUSTOM_POST_PROCESS": f"{script_name}:separate_post",
                    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                },
            ):
                # Test with FastAPI app
                app = FastAPI()

                @app.get("/test")
                def test_endpoint():
                    return {"message": "test"}

                # Bootstrap SageMaker with middleware loading
                bootstrap(app)

                client = TestClient(app)
                response = client.get("/test")

                # Should use direct middleware, not separate functions
                assert response.headers.get("X-Middleware-Type") == "direct"

        finally:
            os.unlink(script_path)
