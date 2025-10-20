"""Mock vLLM server for integration testing using real FastAPI.

This simulates exactly how a real vLLM server works:
- Uses @router.get("/ping") and @router.post("/invocations") decorators like real vLLM
- Defines default vLLM ping and invocations functions
- Creates FastAPI app and includes the router
- Calls SageMaker bootstrap at the end to setup handler overrides
"""

from typing import Any, Dict

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.testclient import TestClient

from model_hosting_container_standards.sagemaker.sagemaker_router import (
    setup_ping_invoke_routes,
)

# Create router like real vLLM does
router = APIRouter()


@router.get("/ping", response_class=Response)
@router.post("/ping", response_class=Response)
async def ping(raw_request: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    return Response(
        content='{"status": "healthy", "source": "vllm_default", "message": "Default ping from vLLM server"}',
        media_type="application/json",
    )


@router.post("/invocations", response_class=Response)
async def invocations(raw_request: Request) -> Response:
    """Model invocations endpoint like real vLLM"""
    return Response(
        content='{"predictions": ["Default vLLM response"], "source": "vllm_default"}',
        media_type="application/json",
    )


class MockVLLMServer:
    """Mock vLLM server using real FastAPI application like real vLLM."""

    def __init__(self):
        self.app = None
        self.client = None

    def _setup_app(self):
        """Setup FastAPI application like real vLLM server."""
        # Create fresh FastAPI app
        self.app = FastAPI(title="Mock vLLM Server", version="1.0.0")

        # Add other vLLM-like routes
        self.app.add_api_route("/health", self._health_check, methods=["GET"])
        self.app.add_api_route("/v1/models", self._list_models, methods=["GET"])

        # Include the router with default vLLM endpoints (like real vLLM does)
        self.app.include_router(router)

        # Bootstrap SageMaker routes at the end (like real vLLM does)
        # This will replace the default routes if custom handlers are found
        setup_ping_invoke_routes(self.app)

        # Create test client
        self.client = TestClient(self.app)

    async def _health_check(self):
        """Health check endpoint that real vLLM servers have."""
        return {"status": "healthy", "service": "vllm"}

    async def _list_models(self):
        """Model listing endpoint that real vLLM servers have."""
        return {"data": [{"id": "test-model", "object": "model"}]}

    def get_client(self) -> TestClient:
        """Get the test client for making requests."""
        if self.app is None or self.client is None:
            self._setup_app()
        return self.client

    def reset(self):
        """Reset the server for fresh testing."""
        self.app = None
        self.client = None


# Global server instance
mock_server = MockVLLMServer()


async def call_ping_endpoint() -> Dict[str, Any]:
    """Call GET /ping endpoint through real FastAPI routing."""
    client = mock_server.get_client()
    response = client.get("/ping")
    return (
        response.json()
        if response.status_code == 200
        else {
            "error": f"Endpoint /ping returned {response.status_code}",
            "source": "error",
            "detail": response.text,
        }
    )


async def call_invoke_endpoint() -> Dict[str, Any]:
    """Call POST /invocations endpoint through real FastAPI routing."""
    client = mock_server.get_client()
    response = client.post("/invocations", json={"prompt": "Hello world"})
    return (
        response.json()
        if response.status_code == 200
        else {
            "error": f"Endpoint /invocations returned {response.status_code}",
            "source": "error",
            "detail": response.text,
        }
    )
