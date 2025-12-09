"""Integration tests for custom session handlers functionality.

Tests the integration of custom session handlers with engine-specific session APIs using
the proper decorator-based registration pattern:
- @register_create_session_handler decorator
- @register_close_session_handler decorator
- Mixed scenarios (custom + default handlers)
- Transform request/response shape mapping via decorators
- Error handling in custom handlers
- Handler registration and resolution

Key Testing Pattern:
    These tests simulate real-world scenarios where an inference engine
    has its own session management API. We use the
    proper decorators to register handlers and verify that:
    1. Decorators properly register and invoke custom handlers
    2. Transforms correctly map between SageMaker and engine formats
    3. Session lifecycle works end-to-end with custom handlers
    4. Error cases are handled gracefully
"""

import json
import os
import shutil
import tempfile
import uuid
from typing import Optional

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
from fastapi.testclient import TestClient
from pydantic import BaseModel

import model_hosting_container_standards.sagemaker as sagemaker_standards
from model_hosting_container_standards.common.handler.registry import handler_registry
from model_hosting_container_standards.sagemaker.sessions.manager import (
    init_session_manager_from_env,
)
from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader,
)

DEFAULT_SESSION_ID = "default-session"


class CreateSessionRequest(BaseModel):
    capacity_of_str_len: int
    session_id: Optional[str] = None


class CloseSessionRequest(BaseModel):
    session_id: str


@pytest.fixture(autouse=True)
def enable_sessions_for_integration(monkeypatch):
    """Automatically enable sessions for all integration tests in this module."""
    temp_dir = tempfile.mkdtemp()

    monkeypatch.setenv("SAGEMAKER_ENABLE_STATEFUL_SESSIONS", "true")
    monkeypatch.setenv("SAGEMAKER_SESSIONS_PATH", temp_dir)
    monkeypatch.setenv("SAGEMAKER_SESSIONS_EXPIRATION", "600")

    init_session_manager_from_env()

    yield

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    monkeypatch.delenv("SAGEMAKER_ENABLE_STATEFUL_SESSIONS", raising=False)
    monkeypatch.delenv("SAGEMAKER_SESSIONS_PATH", raising=False)
    monkeypatch.delenv("SAGEMAKER_SESSIONS_EXPIRATION", raising=False)
    init_session_manager_from_env()


@pytest.fixture(autouse=True)
def cleanup_handler_registry():
    """Clean up handler registry after each test."""
    yield
    handler_registry.remove_handler("create_session")
    handler_registry.remove_handler("close_session")


def extract_session_id_from_header(header_value: str) -> str:
    """Extract session ID from SageMaker session header."""
    if ";" in header_value:
        return header_value.split(";")[0].strip()
    return header_value.strip()


class BaseCustomHandlerIntegrationTest:
    """Base class for custom handler integration tests with common setup.

    Provides:
    - FastAPI app and router setup
    - Mock engine client for simulating engine APIs
    - Handler call tracking
    - TestClient for making requests
    - Common setup/teardown patterns

    Subclasses should override setup_handlers() to register their specific
    custom handlers using the appropriate decorators.
    """

    def setup_method(self):
        """Common setup for all custom handler integration tests."""
        self.app = FastAPI()
        self.router = APIRouter()

        # Track handler invocations for verification
        self.handler_calls = {"create": 0, "close": 0}

        # Setup handlers (to be overridden by subclasses)
        self.setup_handlers()

        # Bootstrap the app with SageMaker standards
        self.app.include_router(self.router)
        sagemaker_standards.bootstrap(self.app)
        self.client = TestClient(self.app)

    def setup_handlers(self):
        """Override in subclasses to register custom handlers.

        This method should:
        1. Define custom handler functions
        2. Register them using @register_create_session_handler or @register_close_session_handler
        3. Set up the /invocations endpoint with @stateful_session_manager
        """
        self.setup_common_handlers()
        self.setup_invocation_handler()

    def custom_create_session(self, obj: CreateSessionRequest, request: Request):
        # Implement in child classes
        pass

    def custom_close_session(self, obj: CloseSessionRequest, request: Request):
        # Implement in child classes
        pass

    def setup_common_handlers(self):
        @sagemaker_standards.register_create_session_handler(
            request_session_id_path="session_id",
            response_session_id_path="body",
            additional_request_shape={
                "capacity_of_str_len": "`1024`",
            },
            content_path="`successfully created session.`",
        )
        @self.app.api_route("/open_session", methods=["GET", "POST"])
        async def create_session(obj: CreateSessionRequest, request: Request):
            return self.custom_create_session(obj, request)

        @sagemaker_standards.register_close_session_handler(
            request_session_id_path="session_id",
            content_path="`successfully closed session.`",
        )
        @self.app.api_route("/close_session", methods=["GET", "POST"])
        async def close_session(obj: CloseSessionRequest, request: Request):
            return self.custom_close_session(obj, request)

    async def custom_invocations(self, request: Request):
        body_bytes = await request.body()
        body = json.loads(body_bytes.decode())
        # Extract session ID from request headers if present
        session_id = body.get("session_id") or request.headers.get(
            SageMakerSessionHeader.SESSION_ID
        )
        return Response(
            status_code=200,
            content=json.dumps(
                {
                    "message": "Request in session",
                    "session_id": session_id or "no-session",
                    "echo": body,
                }
            ),
        )

    def setup_invocation_handler(self):
        @self.router.post("/invocations")
        @sagemaker_standards.stateful_session_manager()
        async def invocations(request: Request):
            return await self.custom_invocations(request)

    # Helper methods for common test operations
    def create_session(self) -> str:
        """Helper to create a session and return the session ID."""
        response = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})
        assert response.status_code == 200
        assert SageMakerSessionHeader.NEW_SESSION_ID in response.headers
        return extract_session_id_from_header(
            response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

    def create_session_with_id(self, session_id: str) -> Response:
        """Helper to create a session with a specific ID."""
        return self.client.post(
            "/invocations",
            json={"requestType": "NEW_SESSION"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

    def close_session(self, session_id: str) -> Response:
        """Helper to close a session."""
        return self.client.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

    def invoke_with_session(self, session_id: str, body: dict) -> Response:
        """Helper to make an invocation request with a session."""
        return self.client.post(
            "/invocations",
            json=body,
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )


class TestSimpleCreateSessionCustomHandler(BaseCustomHandlerIntegrationTest):
    """Test basic custom create session handler with simple string return."""

    def custom_create_session(self, obj: CreateSessionRequest, request: Request):
        return DEFAULT_SESSION_ID

    def test_create_new_session(self):
        """Test that custom handler returning a simple string works correctly.

        This validates the simplest case where a custom handler returns just a string
        (the session ID) rather than a complex object. This is useful when the engine's
        session API returns a simple session identifier.
        """
        # Send NEW_SESSION request to trigger custom create handler
        response = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})

        # Verify successful session creation
        assert response.status_code == 200
        assert SageMakerSessionHeader.NEW_SESSION_ID in response.headers

        # Extract session ID from response header
        session_id = extract_session_id_from_header(
            response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Verify the custom handler's return value (DEFAULT_SESSION_ID) is used as session ID
        # This confirms the transform correctly extracted the session ID from the string response
        assert session_id == DEFAULT_SESSION_ID


class TestErrorCreateSessionCustomHandler(BaseCustomHandlerIntegrationTest):
    """Test error handling when custom create session handler fails."""

    def custom_create_session(self, obj: CreateSessionRequest, request: Request):
        raise HTTPException(status_code=400, detail="Engine failed to create session")

    def test_create_new_session_error(self):
        """Test that errors from custom create handler are properly propagated.

        When the underlying engine fails to create a session (e.g., resource exhaustion,
        invalid parameters), the error should be propagated to the client with appropriate
        status code and error message.
        """
        # Attempt to create session - custom handler will raise HTTPException
        response = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})

        # Verify error status code is returned
        assert response.status_code == 400

        # Verify error message from custom handler is included in response
        assert "Engine failed to create session" in response.text

        # Verify no session header is present on error (session was not created)
        assert SageMakerSessionHeader.NEW_SESSION_ID not in response.headers


class TestErrorCloseSessionCustomHandler(BaseCustomHandlerIntegrationTest):
    """Test error handling when custom close session handler fails."""

    def setup_method(self):
        self.sessions = {}
        super().setup_method()

    def custom_create_session(self, obj: CreateSessionRequest, request: Request):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = session_id
        return session_id

    def custom_close_session(self, obj: CloseSessionRequest, request: Request):
        if obj.session_id in self.sessions:
            self.sessions.pop(obj.session_id)
            return Response(
                status_code=200, content=f"Session {obj.session_id} closed."
            )
        raise HTTPException(
            status_code=404, detail=f"Session {obj.session_id} does not exist."
        )

    def test_duplicate_close_session(self):
        """Test that closing an already-closed session returns 404.

        This validates idempotency handling - attempting to close a session that's
        already been closed should return a 404 error rather than succeeding silently.
        This is important for detecting client-side bugs or race conditions.
        """
        # Create a new session for testing
        session_id = self.create_session()

        # First close should succeed - session exists in custom handler's storage
        success_response = self.close_session(session_id)
        assert success_response.status_code == 200
        assert SageMakerSessionHeader.CLOSED_SESSION_ID in success_response.headers

        # Second close should fail - session no longer exists (was removed on first close)
        # Custom handler raises HTTPException(404) when session not found
        duplicate_response = self.close_session(session_id)
        assert duplicate_response.status_code == 404


class TestCustomSessionEndToEndFlow(BaseCustomHandlerIntegrationTest):
    """Test complete end-to-end flows with custom session handlers."""

    def setup_method(self):
        self.sessions = {}
        super().setup_method()

    def custom_create_session(self, obj: CreateSessionRequest, request: Request):
        self.handler_calls["create"] += 1
        if not obj.session_id:
            obj.session_id = str(uuid.uuid4())
        if obj.session_id in self.sessions:
            return Response(status_code=400)
        self.sessions[obj.session_id] = obj.session_id
        return {"session_id": obj.session_id}

    def custom_close_session(self, obj: CloseSessionRequest, request: Request):
        self.handler_calls["close"] += 1
        if obj.session_id not in self.sessions:
            raise HTTPException(
                status_code=404, detail=f"Session {obj.session_id} does not exist."
            )
        self.sessions.pop(obj.session_id)
        return Response(status_code=200, content=f"Session {obj.session_id} closed.")

    def setup_common_handlers(self):
        @sagemaker_standards.register_create_session_handler(
            request_session_id_path="session_id",
            response_session_id_path="body.session_id",  # Nested
            additional_request_shape={
                "capacity_of_str_len": "`1024`",
            },
            content_path="`successfully created session.`",
        )
        @self.app.api_route("/open_session", methods=["GET", "POST"])
        async def create_session(obj: CreateSessionRequest, request: Request):
            return self.custom_create_session(obj, request)

        @sagemaker_standards.register_close_session_handler(
            request_session_id_path="session_id",
            content_path="`successfully closed session.`",
        )
        @self.app.api_route("/close_session", methods=["GET", "POST"])
        async def close_session(obj: CloseSessionRequest, request: Request):
            return self.custom_close_session(obj, request)

    def setup_invocation_handler(self):
        @self.router.post("/invocations")
        @sagemaker_standards.stateful_session_manager(
            request_session_id_path="session_id"
        )
        async def invocations(request: Request):
            return await self.custom_invocations(request)

    def test_create_existing_session_error_handling(self):
        """Test that attempting to create a session with existing ID fails.

        This validates that the custom handler properly rejects attempts to create
        a session with a duplicate ID. This prevents session ID collisions and ensures
        session uniqueness.
        """
        # Create initial session
        session_id = self.create_session()

        # Try to create another session with the same ID by passing it in the header
        # Custom handler checks if session_id already exists and returns 400 if it does
        header_response = self.create_session_with_id(session_id)
        assert header_response.status_code == 400

    def test_end_to_end_simple(self):
        """Test complete session lifecycle: create -> use -> close.

        This is the primary happy path test that validates the full session workflow
        works correctly with custom handlers. This simulates a typical client interaction
        pattern for stateful ML inference (e.g., multi-turn conversation with an LLM).
        """
        # Step 1: Create session via custom handler
        session_id = self.create_session()

        # Step 2: Use session for inference request
        # Session ID is passed in header and should be available to the handler
        invoke_response = self.invoke_with_session(session_id, {"prompt": "hello"})
        assert invoke_response.status_code == 200
        # Verify session ID is echoed back, confirming session context was maintained
        assert session_id in invoke_response.text

        # Step 3: Close session via custom handler
        close_response = self.close_session(session_id)
        assert close_response.status_code == 200
        # Verify closed session header is returned
        assert SageMakerSessionHeader.CLOSED_SESSION_ID in close_response.headers

    def test_handler_call_tracking(self):
        """Test that custom handlers are actually being invoked.

        This validates that the decorator registration system correctly routes session
        requests to the custom handlers rather than using default handlers. The counters
        prove the custom handler code is executing.
        """
        # Reset counters to ensure clean state
        self.handler_calls = {"create": 0, "close": 0}

        # Create session - should increment create counter
        session_id = self.create_session()
        assert self.handler_calls["create"] == 1  # Custom create handler was called
        assert self.handler_calls["close"] == 0  # Close handler not called yet

        # Close session - should increment close counter
        close_response = self.close_session(session_id)
        assert close_response.status_code == 200
        assert self.handler_calls["create"] == 1  # Create counter unchanged
        assert self.handler_calls["close"] == 1  # Custom close handler was called

    def test_multiple_sessions_independent_state(self):
        """Test that multiple sessions maintain independent state in custom handlers.

        This validates session isolation - multiple concurrent sessions should not
        interfere with each other. This is critical for multi-tenant scenarios where
        different users/clients have active sessions simultaneously.
        """
        # Create two independent sessions
        session1_id = self.create_session()
        session2_id = self.create_session()

        # Verify both sessions exist in custom handler's storage
        assert session1_id in self.sessions
        assert session2_id in self.sessions
        # Verify sessions have unique IDs
        assert session1_id != session2_id

        # Close first session only
        self.close_session(session1_id)

        # Verify only first session was removed from storage
        assert session1_id not in self.sessions
        # Verify second session still exists and is unaffected
        assert session2_id in self.sessions

        # Verify second session is still functional after first session closed
        response = self.invoke_with_session(session2_id, {"prompt": "test"})
        assert response.status_code == 200


class TestCustomHandlerResponseFormats(BaseCustomHandlerIntegrationTest):
    """Test that custom handlers can return different response formats."""

    def setup_method(self):
        self.sessions = {}
        self.response_format = "dict"  # Can be "dict", "string", or "response_object"
        super().setup_method()

    def custom_create_session(self, obj: CreateSessionRequest, request: Request):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = True

        if self.response_format == "dict":
            return {"session_id": session_id, "metadata": {"engine": "custom"}}
        elif self.response_format == "string":
            return session_id
        elif self.response_format == "response_object":
            return Response(
                status_code=201,
                content=json.dumps({"session_id": session_id}),
                media_type="application/json",
            )

    def custom_close_session(self, obj: CloseSessionRequest, request: Request):
        if obj.session_id in self.sessions:
            del self.sessions[obj.session_id]
        return Response(status_code=200, content="Closed")

    def setup_common_handlers(self):
        # Use different response_session_id_path based on format
        response_path = "body.session_id" if self.response_format == "dict" else "body"

        @sagemaker_standards.register_create_session_handler(
            request_session_id_path="session_id",
            response_session_id_path=response_path,
            additional_request_shape={"capacity_of_str_len": "`1024`"},
            content_path="`successfully created session.`",
        )
        @self.app.api_route("/open_session", methods=["GET", "POST"])
        async def create_session(obj: CreateSessionRequest, request: Request):
            return self.custom_create_session(obj, request)

        @sagemaker_standards.register_close_session_handler(
            request_session_id_path="session_id",
            content_path="`successfully closed session.`",
        )
        @self.app.api_route("/close_session", methods=["GET", "POST"])
        async def close_session(obj: CloseSessionRequest, request: Request):
            return self.custom_close_session(obj, request)

    def test_dict_response_with_metadata(self):
        """Test custom handler returning dict with additional metadata.

        Many engine APIs return rich response objects with metadata alongside the
        session ID. This validates that the transform can extract the session ID
        from a nested path while preserving other response data.
        """
        self.response_format = "dict"
        # Create session - handler returns {"session_id": "...", "metadata": {...}}
        session_id = self.create_session()

        # Verify session was created successfully
        assert session_id in self.sessions
        # Verify session ID is in UUID format (36 characters with hyphens)
        assert len(session_id) == 36

    def test_dict_response_with_nested_session_id(self):
        """Test custom handler returning dict with nested session ID path.

        This validates that the response_session_id_path configuration correctly
        extracts the session ID from nested response structures (e.g., body.session_id).
        """
        self.response_format = "dict"
        # Create session with nested response structure
        session_id = self.create_session()

        # Verify session was created and can be used for subsequent requests
        response = self.invoke_with_session(session_id, {"test": "data"})
        assert response.status_code == 200


class TestCustomHandlerMultipleInvocations(BaseCustomHandlerIntegrationTest):
    """Test multiple invocations within the same session with custom handlers."""

    def setup_method(self):
        self.sessions = {}
        self.invocation_counts = {}
        super().setup_method()

    def custom_create_session(self, obj: CreateSessionRequest, request: Request):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {"created": True}
        self.invocation_counts[session_id] = 0
        return {"session_id": session_id}

    def custom_close_session(self, obj: CloseSessionRequest, request: Request):
        if obj.session_id in self.sessions:
            del self.sessions[obj.session_id]
            if obj.session_id in self.invocation_counts:
                del self.invocation_counts[obj.session_id]
        return Response(status_code=200)

    def setup_common_handlers(self):
        @sagemaker_standards.register_create_session_handler(
            request_session_id_path="session_id",
            response_session_id_path="body.session_id",
            additional_request_shape={"capacity_of_str_len": "`1024`"},
            content_path="`successfully created session.`",
        )
        @self.app.api_route("/open_session", methods=["GET", "POST"])
        async def create_session(obj: CreateSessionRequest, request: Request):
            return self.custom_create_session(obj, request)

        @sagemaker_standards.register_close_session_handler(
            request_session_id_path="session_id",
            content_path="`successfully closed session.`",
        )
        @self.app.api_route("/close_session", methods=["GET", "POST"])
        async def close_session(obj: CloseSessionRequest, request: Request):
            return self.custom_close_session(obj, request)

    async def custom_invocations(self, request: Request):
        body_bytes = await request.body()
        body = json.loads(body_bytes.decode())
        session_id = request.headers.get(SageMakerSessionHeader.SESSION_ID)

        # Track invocation count per session
        if session_id and session_id in self.invocation_counts:
            self.invocation_counts[session_id] += 1

        return Response(
            status_code=200,
            content=json.dumps(
                {
                    "message": "success",
                    "session_id": session_id,
                    "invocation_count": self.invocation_counts.get(session_id, 0),
                    "echo": body,
                }
            ),
        )

    def test_multiple_invocations_same_session(self):
        """Test that multiple invocations work correctly within the same session.

        This validates that session state (invocation count) accumulates correctly
        across multiple requests. This is essential for stateful ML scenarios like
        maintaining conversation context or tracking request history.
        """
        session_id = self.create_session()

        # Make 5 sequential invocations to the same session
        for i in range(5):
            response = self.invoke_with_session(session_id, {"request_num": i + 1})
            assert response.status_code == 200
            data = json.loads(response.text)
            # Verify invocation count increments with each request
            assert data["invocation_count"] == i + 1
            # Verify session ID remains consistent
            assert data["session_id"] == session_id

    def test_invocation_counts_independent_across_sessions(self):
        """Test that invocation counts are independent across different sessions.

        This validates session isolation at the invocation level - each session
        maintains its own independent counter. Critical for ensuring one user's
        session activity doesn't affect another user's session.
        """
        # Create two separate sessions
        session1_id = self.create_session()
        session2_id = self.create_session()

        # Make 3 invocations to session 1
        for i in range(3):
            self.invoke_with_session(session1_id, {"msg": "session1"})

        # Make 5 invocations to session 2
        for i in range(5):
            self.invoke_with_session(session2_id, {"msg": "session2"})

        # Verify each session has its own independent count
        assert self.invocation_counts[session1_id] == 3
        assert self.invocation_counts[session2_id] == 5


class TestCustomHandlerWithSessionIdInjection(BaseCustomHandlerIntegrationTest):
    """Test custom handlers with request_session_id_path parameter."""

    def setup_method(self):
        self.sessions = {}
        super().setup_method()

    def custom_create_session(self, obj: CreateSessionRequest, request: Request):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {"created": True}
        return {"session_id": session_id}

    def custom_close_session(self, obj: CloseSessionRequest, request: Request):
        if obj.session_id in self.sessions:
            del self.sessions[obj.session_id]
            return Response(status_code=200, content="Session closed")
        raise HTTPException(status_code=404, detail="Session not found")

    def setup_common_handlers(self):
        @sagemaker_standards.register_create_session_handler(
            request_session_id_path="session_id",
            response_session_id_path="body.session_id",
            additional_request_shape={"capacity_of_str_len": "`1024`"},
            content_path="`successfully created session.`",
        )
        @self.app.api_route("/open_session", methods=["GET", "POST"])
        async def create_session(obj: CreateSessionRequest, request: Request):
            return self.custom_create_session(obj, request)

        @sagemaker_standards.register_close_session_handler(
            request_session_id_path="session_id",
            content_path="`successfully closed session.`",
        )
        @self.app.api_route("/close_session", methods=["GET", "POST"])
        async def close_session(obj: CloseSessionRequest, request: Request):
            return self.custom_close_session(obj, request)

    def setup_invocation_handler(self):
        @self.router.post("/invocations")
        @sagemaker_standards.stateful_session_manager(
            request_session_id_path="metadata.session_id"
        )
        async def invocations(request: Request):
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode())

            # Extract session ID from nested path
            session_id = body.get("metadata", {}).get("session_id")

            return Response(
                status_code=200,
                content=json.dumps(
                    {"message": "success", "session_id": session_id, "body": body}
                ),
            )

    def test_session_id_injected_into_nested_path(self):
        """Test that session ID is injected into nested path in request body.

        Some ML engines expect the session ID to be in the request body rather than
        just in headers. The request_session_id_path parameter allows automatic
        injection of the session ID into a specified path in the request body
        (e.g., metadata.session_id). This test validates that injection works correctly.
        """
        # Create session
        session_id = self.create_session()

        # Make request with session - note we don't include session_id in the body
        # The framework should inject it automatically at metadata.session_id
        response = self.invoke_with_session(
            session_id, {"prompt": "test", "metadata": {"user": "test_user"}}
        )

        assert response.status_code == 200
        data = json.loads(response.text)

        # Verify session ID was automatically injected into the nested path
        assert data["session_id"] == session_id
        assert data["body"]["metadata"]["session_id"] == session_id
        # Verify original metadata fields are preserved
        assert data["body"]["metadata"]["user"] == "test_user"


class TestCustomHandlerSessionPersistence(BaseCustomHandlerIntegrationTest):
    """Test that session state persists correctly across invocations with custom handlers."""

    def setup_method(self):
        self.sessions = {}
        super().setup_method()

    def custom_create_session(self, obj: CreateSessionRequest, request: Request):
        session_id = str(uuid.uuid4())
        # Store session with initial state for ML inference
        self.sessions[session_id] = {
            "conversation_history": [],
            "inference_params": {},
            "created_at": "2024-01-01",
        }
        return {"session_id": session_id}

    def custom_close_session(self, obj: CloseSessionRequest, request: Request):
        if obj.session_id in self.sessions:
            del self.sessions[obj.session_id]
        return Response(status_code=200)

    def setup_common_handlers(self):
        @sagemaker_standards.register_create_session_handler(
            request_session_id_path="session_id",
            response_session_id_path="body.session_id",
            additional_request_shape={"capacity_of_str_len": "`1024`"},
            content_path="`successfully created session.`",
        )
        @self.app.api_route("/open_session", methods=["GET", "POST"])
        async def create_session(obj: CreateSessionRequest, request: Request):
            return self.custom_create_session(obj, request)

        @sagemaker_standards.register_close_session_handler(
            request_session_id_path="session_id",
            content_path="`successfully closed session.`",
        )
        @self.app.api_route("/close_session", methods=["GET", "POST"])
        async def close_session(obj: CloseSessionRequest, request: Request):
            return self.custom_close_session(obj, request)

    async def custom_invocations(self, request: Request):
        body_bytes = await request.body()
        body = json.loads(body_bytes.decode())
        session_id = request.headers.get(SageMakerSessionHeader.SESSION_ID)

        # Simulate updating session state for ML inference
        if session_id and session_id in self.sessions:
            if "message" in body:
                self.sessions[session_id]["conversation_history"].append(
                    body["message"]
                )
            if "inference_params" in body:
                self.sessions[session_id]["inference_params"].update(
                    body["inference_params"]
                )

        session_data = self.sessions.get(session_id, {})

        return Response(
            status_code=200,
            content=json.dumps(
                {
                    "session_id": session_id,
                    "conversation_history": session_data.get(
                        "conversation_history", []
                    ),
                    "inference_params": session_data.get("inference_params", {}),
                }
            ),
        )

    def test_conversation_history_persists(self):
        """Test that conversation history accumulates across invocations.

        This simulates a multi-turn conversation with an LLM where each message
        is added to the session's conversation history. This is a common pattern
        for chatbots and conversational AI where context from previous turns
        needs to be maintained.
        """
        session_id = self.create_session()

        # Send multiple messages in sequence (simulating a conversation)
        messages = ["Hello", "How are you?", "Tell me a joke"]
        for msg in messages:
            # Each message is added to the session's conversation history
            response = self.invoke_with_session(session_id, {"message": msg})
            assert response.status_code == 200

        # Make a final request to retrieve the accumulated history
        final_response = self.invoke_with_session(session_id, {})
        data = json.loads(final_response.text)
        # Verify all messages were stored in order
        assert data["conversation_history"] == messages

    def test_inference_parameters_persist(self):
        """Test that ML inference parameters are maintained across invocations.

        This validates that ML-specific inference parameters (temperature, max_tokens, top_p)
        can be set incrementally and persist across the session. This is useful for:
        - LLM inference where users want consistent generation parameters
        - A/B testing different parameter combinations within a session
        - Gradual parameter tuning based on user feedback
        """
        session_id = self.create_session()

        # Set inference parameters incrementally across multiple requests
        # Temperature: controls randomness in text generation (0.0 = deterministic, 1.0 = creative)
        self.invoke_with_session(session_id, {"inference_params": {"temperature": 0.7}})
        # Max tokens: limits the length of generated output
        self.invoke_with_session(session_id, {"inference_params": {"max_tokens": 512}})
        # Top-p (nucleus sampling): controls diversity of token selection
        self.invoke_with_session(session_id, {"inference_params": {"top_p": 0.9}})

        # Retrieve accumulated parameters
        response = self.invoke_with_session(session_id, {})
        data = json.loads(response.text)
        # Verify all parameters were stored and are accessible
        assert data["inference_params"]["temperature"] == 0.7
        assert data["inference_params"]["max_tokens"] == 512
        assert data["inference_params"]["top_p"] == 0.9

    def test_session_state_cleared_after_close(self):
        """Test that session state is properly cleared when session is closed.

        This validates proper cleanup of session resources. When a session is closed,
        all associated state (conversation history, parameters, etc.) should be
        removed to prevent memory leaks and ensure data privacy.
        """
        session_id = self.create_session()

        # Add some state to the session
        self.invoke_with_session(session_id, {"message": "test"})
        # Verify state was stored
        assert len(self.sessions[session_id]["conversation_history"]) == 1

        # Close the session - should trigger cleanup in custom handler
        self.close_session(session_id)

        # Verify session and all its state was completely removed from storage
        # This is important for memory management and data privacy
        assert session_id not in self.sessions
