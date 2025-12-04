"""Integration tests for custom session handlers.

Tests the integration of custom engine-specific session handlers with the
SageMaker session management system. These tests verify that:
- Custom handlers can be registered and invoked
- Custom handlers take precedence over default handlers
- Request/response transformations work end-to-end
- Error handling propagates correctly from custom handlers
"""

import json
import os
import shutil
import tempfile
from http import HTTPStatus
from typing import Optional

import pytest
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.testclient import TestClient

import model_hosting_container_standards.sagemaker as sagemaker_standards
from model_hosting_container_standards.common.handler.registry import handler_registry
from model_hosting_container_standards.sagemaker.sessions import (
    register_engine_session_handler,
)
from model_hosting_container_standards.sagemaker.sessions.manager import (
    init_session_manager_from_env,
)
from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader,
)


@pytest.fixture(autouse=True)
def enable_sessions_for_integration(monkeypatch):
    """Automatically enable sessions for all integration tests in this module."""
    temp_dir = tempfile.mkdtemp()

    monkeypatch.setenv("SAGEMAKER_ENABLE_STATEFUL_SESSIONS", "true")
    monkeypatch.setenv("SAGEMAKER_SESSIONS_PATH", temp_dir)
    monkeypatch.setenv("SAGEMAKER_SESSIONS_EXPIRATION", "600")

    # Reinitialize the global session manager
    init_session_manager_from_env()

    yield

    # Clean up
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    monkeypatch.delenv("SAGEMAKER_ENABLE_STATEFUL_SESSIONS", raising=False)
    monkeypatch.delenv("SAGEMAKER_SESSIONS_PATH", raising=False)
    monkeypatch.delenv("SAGEMAKER_SESSIONS_EXPIRATION", raising=False)
    init_session_manager_from_env()


@pytest.fixture(autouse=True)
def clear_handler_registry():
    """Clear handler registry before and after each test."""
    handler_registry.clear()
    yield
    handler_registry.clear()


def extract_session_id_from_header(header_value: str) -> str:
    """Extract session ID from SageMaker session header.

    Header format: "<uuid>; Expires=<timestamp>"
    """
    if ";" in header_value:
        return header_value.split(";")[0].strip()
    return header_value.strip()


class MockEngineAPI:
    """Mock engine API that simulates an inference engine's session management.

    This simulates engines like vLLM or TGI that have their own session APIs.
    """

    def __init__(self):
        self.sessions = {}
        self.call_count = {"create": 0, "close": 0}

    def create_session(self, model: Optional[str] = None):
        """Simulate engine's create session API."""
        self.call_count["create"] += 1
        session_id = f"engine-session-{self.call_count['create']}"
        self.sessions[session_id] = {"model": model, "active": True}
        return {
            "session": {"id": session_id, "status": "active"},
            "message": f"Engine session created with model {model}",
        }

    def close_session(self, session_id: str):
        """Simulate engine's close session API."""
        self.call_count["close"] += 1
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found in engine")
        self.sessions[session_id]["active"] = False
        return {"result": {"message": f"Engine session {session_id} closed"}}

    def reset(self):
        """Reset the mock engine state."""
        self.sessions.clear()
        self.call_count = {"create": 0, "close": 0}


class TestCustomHandlerRegistration:
    """Test that custom handlers can be registered and invoked."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.router = APIRouter()
        self.mock_engine = MockEngineAPI()
        self.setup_handlers()
        self.app.include_router(self.router)
        sagemaker_standards.bootstrap(self.app)
        self.client = TestClient(self.app)

    def setup_handlers(self):
        """Set up handlers and register custom session handlers.

        Note: With the hybrid caching strategy, handlers can be registered
        in any order - no timing dependency!
        """

        @self.router.post("/invocations")
        @sagemaker_standards.stateful_session_manager()
        async def invocations(request: Request):
            """Handler with session management."""
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode())
            return Response(
                status_code=200,
                content=json.dumps({"message": "success", "echo": body}),
            )

        # Register custom create session handler
        @register_engine_session_handler(
            handler_type="create_session",
            request_shape={},  # Session requests don't have extra fields
            session_id_path="body.session.id",  # Path to session ID in response body
            content_path="body.message",  # Path to message in response body
        )
        async def custom_create_session(raw_request: Request):
            """Custom handler that delegates to engine API."""
            # Call mock engine API with default model
            result = self.mock_engine.create_session("default-model")
            return result

        # Register custom close session handler
        @register_engine_session_handler(
            handler_type="close_session",
            request_shape={},
            content_path="body.result.message",  # Path to message in response body
        )
        async def custom_close_session(raw_request: Request):
            """Custom handler that delegates to engine API."""
            session_id = raw_request.headers.get(SageMakerSessionHeader.SESSION_ID)

            # Call mock engine API
            result = self.mock_engine.close_session(session_id)
            return result

    def test_custom_create_handler_is_invoked(self):
        """Test that custom create handler is called instead of default."""
        response = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})

        assert response.status_code == 200
        assert SageMakerSessionHeader.NEW_SESSION_ID in response.headers

        # Verify custom handler was called
        assert self.mock_engine.call_count["create"] == 1

    def test_custom_close_handler_is_invoked(self):
        """Test that custom close handler is called instead of default."""
        # First create a session
        create_response = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Now close it
        close_response = self.client.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        if close_response.status_code != 200:
            print(f"Close response status: {close_response.status_code}")
            print(f"Close response body: {close_response.text}")

        assert close_response.status_code == 200
        assert SageMakerSessionHeader.CLOSED_SESSION_ID in close_response.headers

        # Verify custom handler was called
        assert self.mock_engine.call_count["close"] == 1

    def test_custom_handler_response_transformation(self):
        """Test that response is properly transformed from engine format to SageMaker format."""
        response = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})

        # Verify response has SageMaker format
        assert response.status_code == 200
        session_id = extract_session_id_from_header(
            response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Session ID should be from engine (engine-session-X format)
        assert session_id.startswith("engine-session-")

        # Response body should contain engine message
        assert b"Engine session created" in response.content

    def test_custom_handler_request_transformation(self):
        """Test that custom handler is invoked for session creation."""
        self.client.post(
            "/invocations",
            json={"requestType": "NEW_SESSION"},
        )

        # Verify the engine received the request
        assert self.mock_engine.call_count["create"] == 1
        # Check that session was created in engine
        sessions = list(self.mock_engine.sessions.values())
        assert len(sessions) == 1
        assert sessions[0]["model"] == "default-model"


class TestCustomHandlerErrorHandling:
    """Test error handling in custom handlers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.router = APIRouter()
        self.mock_engine = MockEngineAPI()
        self.setup_handlers()
        self.app.include_router(self.router)
        sagemaker_standards.bootstrap(self.app)
        self.client = TestClient(self.app)

    def setup_handlers(self):
        """Set up handlers and register custom session handlers."""

        @self.router.post("/invocations")
        @sagemaker_standards.stateful_session_manager()
        async def invocations(request: Request):
            """Handler with session management."""
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode())
            return Response(
                status_code=200,
                content=json.dumps({"message": "success", "echo": body}),
            )

        @register_engine_session_handler(
            handler_type="create_session",
            request_shape={},
            session_id_path="body.session.id",
            content_path="body.message",
        )
        async def custom_create_session(raw_request: Request):
            """Custom handler that can fail."""
            result = self.mock_engine.create_session()
            return result

        @register_engine_session_handler(
            handler_type="close_session",
            request_shape={},
            content_path="body.result.message",
        )
        async def custom_close_session(raw_request: Request):
            """Custom handler that validates session exists."""
            session_id = raw_request.headers.get(SageMakerSessionHeader.SESSION_ID)
            # This will raise ValueError if session not found
            result = self.mock_engine.close_session(session_id)
            return result

    def test_custom_handler_error_propagates(self):
        """Test that errors from custom handlers propagate correctly.

        Note: Unhandled exceptions in custom handlers will bubble up through FastAPI.
        In production, these should be caught by FastAPI's exception handlers.
        For testing with TestClient, the exception is raised directly.
        """
        # Try to close a non-existent session - this will raise ValueError
        with pytest.raises(ValueError, match="Session nonexistent-session not found"):
            self.client.post(
                "/invocations",
                json={"requestType": "CLOSE"},
                headers={SageMakerSessionHeader.SESSION_ID: "nonexistent-session"},
            )

    def test_custom_handler_missing_session_id_in_response(self):
        """Test handling when custom handler doesn't return session ID."""

        # Create a handler that returns invalid response
        @register_engine_session_handler(
            handler_type="create_session",
            request_shape={},
            session_id_path="body.session.id",  # Path that won't exist
            content_path="body.message",
        )
        async def broken_create_session(raw_request: Request):
            """Handler that returns response without session.id."""
            return {"message": "created but no session id"}

        response = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})

        # Should fail with BAD_GATEWAY since session ID is missing
        assert response.status_code == HTTPStatus.BAD_GATEWAY.value


class TestCustomHandlerWithSessionIdPath:
    """Test custom handlers work with session_id_path parameter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.router = APIRouter()
        self.mock_engine = MockEngineAPI()
        self.captured_requests = []
        self.setup_handlers_with_session_id_path()
        self.app.include_router(self.router)
        sagemaker_standards.bootstrap(self.app)
        self.client = TestClient(self.app)

    def setup_handlers_with_session_id_path(self):
        """Set up handlers that use session_id_path."""

        @self.router.post("/invocations")
        @sagemaker_standards.stateful_session_manager(session_id_path="session_id")
        async def invocations(request: Request):
            """Handler that injects session ID into request body."""
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode())

            # Capture for verification
            self.captured_requests.append(body)

            return Response(
                status_code=200,
                content=json.dumps(
                    {
                        "message": "success",
                        "session_id_from_body": body.get("session_id"),
                    }
                ),
            )

        @register_engine_session_handler(
            handler_type="create_session",
            request_shape={},
            session_id_path="body.session.id",
            content_path="body.message",
        )
        async def custom_create_session(raw_request: Request):
            """Custom create handler."""
            result = self.mock_engine.create_session()
            return result

        @register_engine_session_handler(
            handler_type="close_session",
            request_shape={},
            content_path="body.result.message",
        )
        async def custom_close_session(raw_request: Request):
            """Custom close handler."""
            session_id = raw_request.headers.get(SageMakerSessionHeader.SESSION_ID)
            result = self.mock_engine.close_session(session_id)
            return result

    def test_session_id_injected_with_custom_handler(self):
        """Test that session_id_path works with custom handlers."""
        # Create session with custom handler
        create_response = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Make request with session ID - should be injected into body
        self.captured_requests.clear()
        response = self.client.post(
            "/invocations",
            json={"prompt": "test"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        assert response.status_code == 200
        data = json.loads(response.text)

        # Verify session ID was injected
        assert data["session_id_from_body"] == session_id
        assert len(self.captured_requests) == 1
        assert self.captured_requests[0]["session_id"] == session_id


class TestCustomHandlerConcurrency:
    """Test concurrent operations with custom handlers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.router = APIRouter()
        self.mock_engine = MockEngineAPI()
        self.setup_custom_handlers()
        self.app.include_router(self.router)
        sagemaker_standards.bootstrap(self.app)
        self.client = TestClient(self.app)

    def setup_custom_handlers(self):
        """Register custom handlers."""

        @self.router.post("/invocations")
        @sagemaker_standards.stateful_session_manager()
        async def invocations(request: Request):
            """Handler with session management."""
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode())
            return Response(
                status_code=200,
                content=json.dumps({"message": "success", "echo": body}),
            )

        @register_engine_session_handler(
            handler_type="create_session",
            request_shape={},
            session_id_path="body.session.id",
            content_path="body.message",
        )
        async def custom_create_session(raw_request: Request):
            """Custom create handler."""
            result = self.mock_engine.create_session()
            return result

        @register_engine_session_handler(
            handler_type="close_session",
            request_shape={},
            content_path="body.result.message",
        )
        async def custom_close_session(raw_request: Request):
            """Custom close handler."""
            session_id = raw_request.headers.get(SageMakerSessionHeader.SESSION_ID)
            result = self.mock_engine.close_session(session_id)
            return result

    def test_multiple_sessions_with_custom_handlers(self):
        """Test creating and managing multiple sessions with custom handlers."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            response = self.client.post(
                "/invocations", json={"requestType": "NEW_SESSION"}
            )
            assert response.status_code == 200
            session_id = extract_session_id_from_header(
                response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
            )
            session_ids.append(session_id)

        # Verify all sessions were created in engine
        assert self.mock_engine.call_count["create"] == 3
        assert len(self.mock_engine.sessions) == 3

        # Close all sessions
        for session_id in session_ids:
            response = self.client.post(
                "/invocations",
                json={"requestType": "CLOSE"},
                headers={SageMakerSessionHeader.SESSION_ID: session_id},
            )
            assert response.status_code == 200

        # Verify all sessions were closed in engine
        assert self.mock_engine.call_count["close"] == 3

    def test_interleaved_operations_with_custom_handlers(self):
        """Test interleaved create/use/close operations."""
        # Create session 1
        response1 = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )
        session1_id = extract_session_id_from_header(
            response1.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Create session 2
        response2 = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )
        session2_id = extract_session_id_from_header(
            response2.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Close session 1
        close1 = self.client.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            headers={SageMakerSessionHeader.SESSION_ID: session1_id},
        )
        assert close1.status_code == 200

        # Session 2 should still work
        # (Note: In this test we're just verifying the close worked)
        assert self.mock_engine.sessions[session1_id]["active"] is False
        assert self.mock_engine.sessions[session2_id]["active"] is True


class TestCustomHandlerComplexTransformations:
    """Test custom handlers with complex request/response transformations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.router = APIRouter()
        self.mock_engine = MockEngineAPI()
        self.setup_complex_handlers()
        self.app.include_router(self.router)
        sagemaker_standards.bootstrap(self.app)
        self.client = TestClient(self.app)

    def setup_complex_handlers(self):
        """Register handlers with complex transformations."""

        @self.router.post("/invocations")
        @sagemaker_standards.stateful_session_manager()
        async def invocations(request: Request):
            """Handler with session management."""
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode())
            return Response(
                status_code=200,
                content=json.dumps({"message": "success", "echo": body}),
            )

        @register_engine_session_handler(
            handler_type="create_session",
            request_shape={},
            session_id_path="body.session.id",
            content_path="body.message",
        )
        async def custom_create_session(raw_request: Request):
            """Handler for session creation."""
            result = self.mock_engine.create_session("default")
            return result

        @register_engine_session_handler(
            handler_type="close_session",
            request_shape={},
            content_path="body.result.message",
        )
        async def custom_close_session(raw_request: Request):
            """Handler for session closure."""
            session_id = raw_request.headers.get(SageMakerSessionHeader.SESSION_ID)

            # Engine API closes the session
            result = self.mock_engine.close_session(session_id)
            return result

    def test_complex_request_transformation(self):
        """Test that custom handlers work with session creation."""
        response = self.client.post(
            "/invocations",
            json={"requestType": "NEW_SESSION"},
        )

        assert response.status_code == 200
        assert SageMakerSessionHeader.NEW_SESSION_ID in response.headers

        # Verify engine created session
        assert self.mock_engine.call_count["create"] == 1

    def test_close_with_custom_handler(self):
        """Test close handler with custom implementation."""
        # Create session first
        create_response = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Close session
        close_response = self.client.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        assert close_response.status_code == 200
        assert SageMakerSessionHeader.CLOSED_SESSION_ID in close_response.headers
        assert self.mock_engine.call_count["close"] == 1


class TestCustomHandlerEndToEnd:
    """End-to-end tests simulating real engine integration patterns."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.router = APIRouter()
        self.mock_engine = MockEngineAPI()
        self.setup_realistic_handlers()
        self.app.include_router(self.router)
        sagemaker_standards.bootstrap(self.app)
        self.client = TestClient(self.app)

    def setup_realistic_handlers(self):
        """Set up handlers that simulate real vLLM/TGI integration."""

        @self.router.post("/invocations")
        @sagemaker_standards.stateful_session_manager(
            session_id_path="metadata.session_id"
        )
        async def invocations(request: Request):
            """Realistic inference handler."""
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode())

            # Simulate inference with session context
            session_id = body.get("metadata", {}).get("session_id")
            prompt = body.get("prompt", "")

            return Response(
                status_code=200,
                content=json.dumps(
                    {
                        "generated_text": f"Response to: {prompt}",
                        "session_id": session_id,
                        "metadata": {"tokens": 42},
                    }
                ),
            )

        @register_engine_session_handler(
            handler_type="create_session",
            request_shape={},
            session_id_path="body.session.id",
            content_path="body.message",
        )
        async def vllm_create_session(raw_request: Request):
            """Simulate vLLM session creation."""
            result = self.mock_engine.create_session("default-model")
            return result

        @register_engine_session_handler(
            handler_type="close_session",
            request_shape={},
            content_path="body.result.message",
        )
        async def vllm_close_session(raw_request: Request):
            """Simulate vLLM session closure."""
            session_id = raw_request.headers.get(SageMakerSessionHeader.SESSION_ID)
            result = self.mock_engine.close_session(session_id)
            return result

    def test_full_lifecycle_with_custom_handlers(self):
        """Test complete lifecycle: create -> use -> close with custom handlers."""
        # 1. Create session via custom handler
        create_response = self.client.post(
            "/invocations",
            json={"requestType": "NEW_SESSION"},
        )
        assert create_response.status_code == 200
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )
        assert session_id.startswith("engine-session-")

        # 2. Use session for inference
        inference_response = self.client.post(
            "/invocations",
            json={"prompt": "Hello, world!", "metadata": {}},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )
        assert inference_response.status_code == 200
        data = json.loads(inference_response.text)
        assert data["session_id"] == session_id
        assert "Response to: Hello, world!" in data["generated_text"]

        # 3. Close session via custom handler
        close_response = self.client.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )
        assert close_response.status_code == 200
        assert (
            close_response.headers[SageMakerSessionHeader.CLOSED_SESSION_ID]
            == session_id
        )

        # Verify engine state
        assert self.mock_engine.call_count["create"] == 1
        assert self.mock_engine.call_count["close"] == 1
        assert self.mock_engine.sessions[session_id]["active"] is False

    def test_multiple_inference_calls_with_custom_session(self):
        """Test multiple inference calls using custom session."""
        # Create session
        create_response = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Make multiple inference calls
        prompts = ["First prompt", "Second prompt", "Third prompt"]
        for prompt in prompts:
            response = self.client.post(
                "/invocations",
                json={"prompt": prompt, "metadata": {}},
                headers={SageMakerSessionHeader.SESSION_ID: session_id},
            )
            assert response.status_code == 200
            data = json.loads(response.text)
            assert data["session_id"] == session_id
            assert prompt in data["generated_text"]

        # Close session
        close_response = self.client.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )
        assert close_response.status_code == 200
