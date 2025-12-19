"""Integration tests for SageMaker LoRA functionality.

Tests the integration between router redirection and request/response transformation:
- LoRA decorators (@register_load_adapter_handler, etc.)
- Router mounting via bootstrap()
- Request transformation using JMESPath expressions
- Response transformation
- SageMaker-specific header handling

Key Testing Pattern:
    The tests use a TransformationCapture helper to verify that both routing
    and transformation work together correctly. This allows us to:
    1. Verify routes are redirected (e.g., /adapters -> /v1/load_lora_adapter)
    2. Verify requests are transformed (e.g., {"name": "x"} -> {"lora_name": "x"})
    3. Check that both happen in combination through the full request pipeline
"""

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response
from fastapi.testclient import TestClient
from pydantic import BaseModel

import model_hosting_container_standards.sagemaker as sagemaker_standards
from model_hosting_container_standards.common.handler import handler_registry


class EngineLoadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_path: str


class EngineUnloadLoRAAdapterRequest(BaseModel):
    lora_name: str


class TransformationCapture:
    """Helper to capture transformed requests in tests.

    This class allows tests to observe the transformed request objects that
    handlers receive, enabling verification that:
    - Routing redirection works (via the URL field)
    - Request transformation works (via the transformed field)
    - Both work together in the full pipeline

    Example:
        capture = TransformationCapture()

        # In handler:
        capture.capture("my_handler", transformed_request, raw_request)

        # In test:
        captures = capture.get_by_handler("my_handler")
        assert captures[0]["transformed"].field == expected_value
    """

    def __init__(self):
        self.requests = []

    def capture(self, handler_name: str, request_obj, raw_request: Request):
        """Capture a transformed request.

        Args:
            handler_name: Identifier for the handler (for filtering later)
            request_obj: The transformed request object received by the handler
            raw_request: The raw FastAPI Request object (for metadata)
        """
        self.requests.append(
            {
                "handler": handler_name,
                "transformed": request_obj,  # The transformed request/body
                "url": str(raw_request.url),  # Verify which route was called
                "headers": dict(raw_request.headers),  # Check header extraction
                "path_params": dict(
                    raw_request.path_params
                ),  # Check path param extraction
            }
        )

    def get_by_handler(self, handler_name: str):
        """Get all captures for a specific handler."""
        return [r for r in self.requests if r["handler"] == handler_name]

    def clear(self):
        """Clear all captures (useful between test steps)."""
        self.requests.clear()


class BaseLoRAIntegrationTest:
    """Base class for LoRA integration tests with common setup."""

    def setup_method(self):
        """Common setup for all LoRA integration tests."""
        handler_registry.clear()
        self.app = FastAPI()
        self.router = APIRouter()
        self.capture = TransformationCapture()

        self.setup_handlers()

        self.app.include_router(self.router)
        sagemaker_standards.bootstrap(self.app)
        self.client = TestClient(self.app)

    def make_adapter_request_params(self, test_type, name, src, base_url="/adapters"):
        """Helper to generate URL and JSON for adapter requests based on test_type.

        Args:
            test_type: Either "body" or "query_params"
            name: The adapter name
            src: The adapter source path
            base_url: The base URL for the request (default: "/adapters")

        Returns:
            Tuple of (url, json_data) for use in client requests
        """
        if test_type == "query_params":
            url = f"{base_url}?name={name}&src={src}"
            json_data = None
        else:  # body
            url = base_url
            json_data = {"name": name, "src": src}
        return url, json_data

    def setup_handlers(self, test_type="body"):
        """Define handlers for end-to-end lifecycle tests.

        Sets up three handlers that simulate a LoRA-enabled inference engine:
        1. load_lora_adapter - Loads adapters into the registry
        2. unload_lora_adapter - Removes adapters from the registry
        3. invocations - Handles inference with optional adapter selection

        Args:
            test_type: Either "body" or "query_params" to determine request source
        """
        # Simulate a simple adapter registry
        self.adapters = {}

        # Determine request shape based on test type
        source_prefix = "body" if test_type == "body" else "query_params"
        request_shape = {
            "lora_name": f"{source_prefix}.name",
            "lora_path": f"{source_prefix}.src",
        }

        # Handler 1: Load adapter
        # The decorator transforms based on test_type:
        # - body: {"name": "x", "src": "y"} -> {"lora_name": "x", "lora_path": "y"}
        # - query_params: ?name=x&src=y -> {"lora_name": "x", "lora_path": "y"}
        decorator_args = dict(request_shape=request_shape) if test_type == "body" else \
            dict(engine_request_lora_name_path="body.lora_name",
                engine_request_lora_src_path="body.lora_path",
                engine_request_model_cls=EngineLoadLoRAAdapterRequest)
        @sagemaker_standards.register_load_adapter_handler(**decorator_args)
        @self.router.post("/v1/load_lora_adapter")
        async def load_lora_adapter(
            request: EngineLoadLoRAAdapterRequest, raw_request: Request
        ):
            # Capture the transformed request for test verification
            self.capture.capture("load_adapter", request, raw_request)

            # Business logic: register the adapter
            self.adapters[request.lora_name] = request.lora_path
            return Response(
                status_code=200,
                content=f"Adapter {request.lora_name} registered",
            )

        # Handler 2: Unload adapter
        # The decorator extracts path param: /adapters/{adapter_name} -> {"lora_name": adapter_name}
        @sagemaker_standards.register_unload_adapter_handler(
            **dict(request_shape={"lora_name": "path_params.adapter_name"}) if test_type == "body" else \
                dict(engine_request_lora_name_path="body.lora_name",
                    engine_request_model_cls=EngineUnloadLoRAAdapterRequest)
        )
        @self.router.post("/v1/unload_lora_adapter")
        async def unload_lora_adapter(
            request: EngineUnloadLoRAAdapterRequest, raw_request: Request
        ):
            # Capture the transformed request for test verification
            self.capture.capture("unload_adapter", request, raw_request)

            # Business logic: unregister the adapter
            if request.lora_name in self.adapters:
                del self.adapters[request.lora_name]
                return Response(
                    status_code=200,
                    content=f"Adapter {request.lora_name} unregistered",
                )
            return Response(status_code=404, content="Adapter not found")

        # Handler 3: Invocations with adapter injection
        # The decorator injects adapter ID from header into body: header -> body["model"]
        @self.router.post("/invocations")
        @sagemaker_standards.inject_adapter_id("model")
        async def invocations(request: Request):
            body_bytes = await request.body()
            import json

            body = json.loads(body_bytes.decode())

            # Capture the transformed body for test verification
            self.capture.capture("invocations", body, request)

            # Business logic: use the adapter if specified
            adapter_id = body.get("model", "base-model")

            if adapter_id in self.adapters:
                return Response(
                    status_code=200,
                    content=f"Response from adapter: {adapter_id}",
                )
            return Response(status_code=200, content=f"Response from: {adapter_id}")


class TestLoRARouterRedirection(BaseLoRAIntegrationTest):
    """Test that bootstrap() correctly mounts LoRA routes from decorated handlers."""

    @pytest.mark.parametrize(
        "test_type",
        [
            ("body"),
            ("query_params"),
        ],
        ids=[
            "body",
            "query_params",
        ],
    )
    def test_register_adapter_route_mounted(self, test_type):
        """Test that POST /adapters route is mounted by bootstrap()."""
        # Re-setup handlers with the correct test_type for this parametrized test
        handler_registry.clear()
        self.setup_handlers(test_type)
        sagemaker_standards.bootstrap(self.app)

        # Call the SageMaker-standard route (not the engine's custom route)
        lora_name = "test-adapter"
        lora_path = "s3://bucket/adapter"
        url, json_data = self.make_adapter_request_params(
            test_type, lora_name, lora_path
        )
        response = self.client.post(url, json=json_data)

        assert response.status_code == 200

        # Verify transformation happened through the redirected route
        captures = self.capture.get_by_handler("load_adapter")
        assert len(captures) == 1
        transformed = captures[0]["transformed"]
        # The "name" field should be transformed to "lora_name"
        assert transformed.lora_name == "test-adapter"
        # The "src" field should be transformed to "lora_path"
        assert transformed.lora_path == "s3://bucket/adapter"

    def test_unregister_adapter_route_mounted(self):
        """Test that DELETE /adapters/{adapter_name} route is mounted by bootstrap()."""
        # First register an adapter (not part of this test)
        self.client.post(
            "/adapters", json={"name": "test-adapter", "src": "s3://bucket/adapter"}
        )
        self.capture.clear()  # Clear the load capture from above

        # Call the SageMaker-standard DELETE route
        response = self.client.delete("/adapters/test-adapter")

        # Verify the path param "test-adapter" was extracted into lora_name
        captures = self.capture.get_by_handler("unload_adapter")
        assert len(captures) == 1
        transformed = captures[0]["transformed"]
        assert transformed.lora_name == "test-adapter"

        assert response.status_code == 200

    def test_inject_adapter_id_no_route(self):
        """Test that inject_adapter_id does not create its own route.

        inject_adapter_id is a transform-only decorator that modifies requests
        in-flight but doesn't expose its own route. It should only affect
        the handler it decorates (/invocations in this case).
        """
        # Call the engine's invocations route with SageMaker adapter header
        self.capture.clear()
        response = self.client.post(
            "/invocations",
            json={"model": "test"},
            headers={"X-Amzn-SageMaker-Adapter-Identifier": "adapter-123"},
        )

        # Verify the header value was injected into the body
        captures = self.capture.get_by_handler("invocations")
        assert len(captures) == 1
        transformed = captures[0]["transformed"]
        assert transformed.get("model") == "adapter-123"  # Header value injected

        assert response.status_code == 200
        assert "adapter-123" in response.text

    def test_custom_handler_routes_still_work(self):
        """Test that engine-defined routes still work.

        bootstrap() mounts SageMaker routes but shouldn't break
        the engine's original routes.
        """
        # Load adapter through engine's custom route (not SageMaker /adapters)
        response = self.client.post(
            "/v1/load_lora_adapter",
            json={"lora_name": "custom-adapter", "lora_path": "s3://test"},
        )
        assert response.status_code == 200

        # Invoke with the adapter
        response = self.client.post("/invocations", json={"model": "custom-adapter"})
        assert response.status_code == 200

        # Unload through engine's custom route
        response = self.client.post(
            "/v1/unload_lora_adapter", json={"lora_name": "custom-adapter"}
        )
        assert response.status_code == 200


class TestLoRARequestResponseTransformation(BaseLoRAIntegrationTest):
    """Test request/response transformation with JMESPath expressions.

    These tests focus on verifying the transformation logic works correctly
    when combined with routing. Each test captures the transformed request
    to verify the JMESPath expressions extracted/injected data correctly.
    """

    def test_unregister_adapter_path_param_extraction(self):
        """Test that path parameters are extracted and transformed.

        Transformation: path_params.adapter_name -> request.lora_name
        """
        self.client.post(
            "/adapters", json={"name": "my-adapter-name", "src": "s3://bucket/adapter"}
        )
        self.capture.clear()  # Clear the load capture

        response = self.client.delete("/adapters/my-adapter-name")

        assert response.status_code == 200
        assert "my-adapter-name" in response.text

        # Verify transformation: path param extracted into lora_name
        captures = self.capture.get_by_handler("unload_adapter")
        assert len(captures) == 1
        transformed = captures[0]["transformed"]
        assert transformed.lora_name == "my-adapter-name"
        assert captures[0]["path_params"]["adapter_name"] == "my-adapter-name"

    def test_inject_adapter_id_header_extraction(self):
        """Test that inject_adapter_id extracts adapter ID from SageMaker header.

        Transformation: X-Amzn-SageMaker-Adapter-Identifier header -> body["model"]
        """
        response = self.client.post(
            "/invocations",
            json={"prompt": "test"},  # No "model" field initially
            headers={"X-Amzn-SageMaker-Adapter-Identifier": "adapter-123"},
        )

        assert response.status_code == 200
        assert "adapter-123" in response.text

        # Verify transformation: adapter ID injected from header into body
        captures = self.capture.get_by_handler("invocations")
        assert len(captures) == 1
        transformed_body = captures[0]["transformed"]
        assert transformed_body["model"] == "adapter-123"  # Injected from header
        assert transformed_body["prompt"] == "test"  # Original field preserved

    def test_inject_adapter_id_nested_jmespath_existing_body(self):
        """Test adapter ID injection with nested JMESPath target.

        Verifies that the adapter ID can be injected into a nested path
        in the request body, not just a top-level field.

        Transformation: header -> body["body"]["model"]["lora_name"]
        """
        handler_registry.clear()

        app = FastAPI()
        router = APIRouter()

        @router.post("/invocations")
        @sagemaker_standards.inject_adapter_id("body.model.lora_name")
        async def invocations(request: Request):
            body_bytes = await request.body()
            import json

            body = json.loads(body_bytes.decode())
            adapter_id = (
                body.get("body", {}).get("model", {}).get("lora_name", "base-model")
            )

            if adapter_id in self.adapters:
                return Response(
                    status_code=200,
                    content=f"Response from adapter: {adapter_id}",
                )
            return Response(status_code=200, content=f"Response from: {adapter_id}")

        # Add model to adapters for mocking test
        lora_name = "lora-1"
        self.adapters[lora_name] = lora_name

        request_json = {
            "body": {
                "model": {
                    "base_name": "base-model",
                    "extra-param": "extra-param",
                    "extra-nested": {"extra-extra-nested": "nested"},
                },
                "prompt": "test-prompt",
            }
        }
        app.include_router(router)
        sagemaker_standards.bootstrap(app)
        client = TestClient(app)
        response = client.post(
            "/invocations",
            json=request_json,
            headers={"X-Amzn-SageMaker-Adapter-Identifier": lora_name},
        )
        assert response.status_code == 200
        assert "lora-1" in response.text

    @pytest.mark.parametrize(
        "test_type",
        [
            ("body"),
            ("query_params"),
        ],
        ids=[
            "body",
            "query_params",
        ],
    )
    def test_nested_jmespath_transformations(self, test_type):
        """Test nested JMESPath expressions in request_shape.

        Verifies that request_shape can contain nested dictionaries, and
        JMESPath expressions work at any nesting level.
        """
        # Re-setup handlers with the correct test_type for this parametrized test
        handler_registry.clear()
        self.capture.clear()  # Clear the load capture

        # Define request model with nested structure
        class NestedLoadLoRAAdapterRequest(BaseModel):
            adapter_config: dict  # Nested dict field
            source_path: str

        # Determine request shape based on test type
        source_prefix = "body" if test_type == "body" else "query_params"
        nested_request_shape = {
            "adapter_config": {  # Target is a nested dict
                "name": f"{source_prefix}.name",  # Extract from source.name
            },
            "source_path": f"{source_prefix}.src",  # Extract from source.src
        }

        @sagemaker_standards.register_load_adapter_handler(
            **dict(request_shape=nested_request_shape) if test_type == "body" else 
            dict(
                engine_request_lora_name_path="body.adapter_config.name",
                engine_request_lora_src_path="body.source_path",
                engine_request_model_cls=NestedLoadLoRAAdapterRequest,
            )
        )
        @self.router.post("/v1/nested_load")
        async def nested_load(
            request: NestedLoadLoRAAdapterRequest, raw_request: Request
        ):
            # Capture to verify nested transformation
            self.capture.capture("load_adapter", request, raw_request)
            return Response(
                status_code=200,
                content=f"name={request.adapter_config['name']},source={request.source_path}",
            )

        sagemaker_standards.bootstrap(self.app)

        lora_name = "nested-adapter"
        lora_path = "s3://nested/path"
        url, json_data = self.make_adapter_request_params(
            test_type, lora_name, lora_path
        )
        response = self.client.post(url, json=json_data)

        assert response.status_code == 200
        assert "nested-adapter" in response.text

        # Verify complex nested JMESPath extraction worked correctly
        captures = self.capture.get_by_handler("load_adapter")
        assert len(captures) == 1
        transformed = captures[0]["transformed"]
        assert transformed.adapter_config["name"] == "nested-adapter"
        assert transformed.source_path == "s3://nested/path"


class TestLoRAErrorCases(BaseLoRAIntegrationTest):
    """Test error handling in routing and transformation.

    These tests verify that the system handles errors gracefully.
    """

    def test_inject_adapter_id_invalid_adapter_path(self):
        """Test that passing an invalid adapter_path with inject_adapter_id raises error"""
        handler_registry.clear()

        with pytest.raises(ValueError):

            @sagemaker_standards.inject_adapter_id("")
            async def invocations_empty(request: Request):
                pass

        handler_registry.clear()

        with pytest.raises(ValueError):

            @sagemaker_standards.inject_adapter_id(1234)
            async def invocations_nonstr(request: Request):
                pass

    def test_inject_adapter_id_valid_path(self):
        """Test that inject_adapter_id accepts valid adapter paths."""
        # Test that the function works correctly with a simple path
        decorator = sagemaker_standards.inject_adapter_id("model")
        assert callable(decorator)

        # Test that it works with nested paths too
        decorator_nested = sagemaker_standards.inject_adapter_id("body.model.lora_name")
        assert callable(decorator_nested)

    def test_load_adapter_missing_required_field(self):
        """Test loading adapter with missing required field.

        The transformation expects both 'name' and 'src' in the request.
        Omitting one should result in an error.
        """
        response = self.client.post(
            "/adapters", json={"name": "test-adapter"}  # Missing "src" field
        )

        # Should fail - missing required field for transformation
        assert response.status_code in [400, 422]  # Validation error

    def test_load_adapter_empty_body(self):
        """Test loading adapter with empty request body.

        Sending an empty body should fail validation.
        """
        response = self.client.post("/adapters", json={})

        assert response.status_code in [400, 422]

    def test_unload_adapter_missing_adapter_name(self):
        response = self.client.delete("/adapters/")

        assert response.status_code in [405]


class TestLoRAEndToEndFlow(BaseLoRAIntegrationTest):
    """Test complete end-to-end workflows combining routing and transformation.

    These tests verify that all the pieces work together in realistic workflows,
    simulating how a user would interact with a LoRA-enabled SageMaker endpoint.
    """

    @pytest.mark.parametrize(
        "test_type",
        [
            ("body"),
            ("query_params"),
        ],
        ids=[
            "body",
            "query_params",
        ],
    )
    def test_full_adapter_lifecycle(self, test_type):
        """Test complete lifecycle: register -> invoke with adapter -> unregister.

        This is the primary happy path: load an adapter, use it for inference,
        then unload it. Verifies all three operations work together.
        """
        # Re-setup handlers with the correct test_type for this parametrized test
        handler_registry.clear()
        self.setup_handlers(test_type)
        sagemaker_standards.bootstrap(self.app)

        lora_name = "lora-1"
        lora_path = "s3://bucket/lora-1"
        # 1. Register an adapter
        url, json_data = self.make_adapter_request_params(
            test_type, lora_name, lora_path
        )
        register_response = self.client.post(url, json=json_data)
        assert register_response.status_code == 200

        # 2. Invoke with the adapter
        invoke_response = self.client.post(
            "/invocations",
            json={"prompt": "hello"},
            headers={"X-Amzn-SageMaker-Adapter-Identifier": lora_name},
        )
        assert invoke_response.status_code == 200

        # 3. Unregister the adapter
        unregister_response = self.client.delete(f"/adapters/{lora_name}")
        assert unregister_response.status_code == 200

    @pytest.mark.parametrize(
        "test_type",
        [
            ("body"),
            ("query_params"),
        ],
        ids=[
            "body",
            "query_params",
        ],
    )
    def test_multiple_adapters(self, test_type):
        """Test managing multiple adapters simultaneously."""
        # Re-setup handlers with the correct test_type for this parametrized test
        handler_registry.clear()
        self.setup_handlers(test_type)
        sagemaker_standards.bootstrap(self.app)

        # Register multiple adapters
        url_a, json_a = self.make_adapter_request_params(
            test_type, "adapter_a", "s3://a"
        )
        self.client.post(url_a, json=json_a)

        url_b, json_b = self.make_adapter_request_params(
            test_type, "adapter_b", "s3://b"
        )
        self.client.post(url_b, json=json_b)

        url_c, json_c = self.make_adapter_request_params(
            test_type, "adapter_c", "s3://c"
        )
        self.client.post(url_c, json=json_c)

        # Invoke with different adapters - each should route correctly
        response_a = self.client.post(
            "/invocations",
            json={"prompt": "test"},
            headers={"X-Amzn-SageMaker-Adapter-Identifier": "adapter-a"},
        )
        assert "adapter-a" in response_a.text

        response_b = self.client.post(
            "/invocations",
            json={"prompt": "test"},
            headers={"X-Amzn-SageMaker-Adapter-Identifier": "adapter-b"},
        )
        assert "adapter-b" in response_b.text

        # Unregister one adapter
        self.client.delete("/adapters/adapter-a")

        # Verify the remaining adapters still work
        response_c = self.client.post(
            "/invocations",
            json={"prompt": "test"},
            headers={"X-Amzn-SageMaker-Adapter-Identifier": "adapter-c"},
        )
        assert "adapter-c" in response_c.text

    def test_invoke_without_adapter_uses_base_model(self):
        """Test that invocations without adapter header work with base model.

        Verifies backward compatibility: endpoints should still work when
        no adapter is specified, falling back to the base model.
        """
        response = self.client.post(
            "/invocations",
            json={"prompt": "test"},
            # No X-Amzn-SageMaker-Adapter-Identifier header
        )

        assert response.status_code == 200
        # Should use base model or empty string when no adapter specified
        assert "base-model" in response.text or "Response from:" in response.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
