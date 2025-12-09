"""Integration tests for SageMaker stateful sessions functionality.

Tests the integration between session management and request/response handling:
- Session creation via NEW_SESSION requests
- Session validation on subsequent requests
- Session closure via CLOSE requests
- Stateful sessions with @stateful_session_manager() decorator
- Session expiration handling
- SageMaker session header handling

Key Testing Pattern:
    The tests use a real FastAPI app with TestClient to verify that sessions
    work end-to-end through the HTTP layer. This allows us to:
    1. Verify session creation returns proper headers
    2. Verify session IDs persist across requests
    3. Verify session closure cleanup works
    4. Check error handling for invalid/expired sessions
"""

import json
import os
import shutil
import tempfile
from typing import Optional

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response
from fastapi.testclient import TestClient

import model_hosting_container_standards.sagemaker as sagemaker_standards
from model_hosting_container_standards.sagemaker.sessions.manager import (
    init_session_manager_from_env,
)
from model_hosting_container_standards.sagemaker.sessions.models import (
    SESSION_DISABLED_ERROR_DETAIL,
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


def extract_session_id_from_header(header_value: str) -> str:
    """Extract session ID from SageMaker session header.

    Header format: "<uuid>; Expires=<timestamp>"
    """
    # The session ID is before the semicolon
    if ";" in header_value:
        return header_value.split(";")[0].strip()
    return header_value.strip()


class SessionRequestCapture:
    """Helper to capture session-related data across requests.

    Allows tests to track:
    - Which sessions were created
    - Session IDs used in subsequent requests
    - Request/response patterns across session lifecycle
    """

    def __init__(self):
        self.requests = []

    def capture(
        self,
        request_type: str,
        session_id: Optional[str] = None,
        data: Optional[dict] = None,
    ):
        """Capture a session-related request.

        Args:
            request_type: Type of request (create, use, close)
            session_id: The session ID involved (if any)
            data: Additional data to capture
        """
        self.requests.append(
            {"type": request_type, "session_id": session_id, "data": data or {}}
        )

    def get_by_type(self, request_type: str):
        """Get all captures of a specific type."""
        return [r for r in self.requests if r["type"] == request_type]

    def clear(self):
        """Clear all captures."""
        self.requests.clear()


class BaseSessionIntegrationTest:
    """Base class for session integration tests with common setup."""

    def setup_method(self):
        """Common setup for all session integration tests."""
        self.app = FastAPI()
        self.router = APIRouter()
        self.capture = SessionRequestCapture()

        # Simulate a simple request counter per session
        self.session_counters = {}

        self.setup_handlers()

        self.app.include_router(self.router)
        sagemaker_standards.bootstrap(self.app)
        self.client = TestClient(self.app)

    def setup_handlers(self):
        """Define handlers for session lifecycle tests.

        Sets up a handler that uses stateful_session_manager to handle
        stateful session requests.
        """

        @self.router.post("/invocations")
        @sagemaker_standards.stateful_session_manager()
        async def invocations(request: Request):
            """Stateful invocation handler with session support."""
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode())

            # Extract session ID from request headers if present
            session_id = request.headers.get(SageMakerSessionHeader.SESSION_ID)

            # Track request count per session
            if session_id:
                if session_id not in self.session_counters:
                    self.session_counters[session_id] = 0
                self.session_counters[session_id] += 1
                count = self.session_counters[session_id]
            else:
                count = 0

            # Capture for test verification
            self.capture.capture(
                "use_session", session_id, {"count": count, "body": body}
            )

            return Response(
                status_code=200,
                content=json.dumps(
                    {
                        "message": f"Request {count} in session",
                        "session_id": session_id or "no-session",
                        "echo": body,
                    }
                ),
            )


class TestSessionCreation(BaseSessionIntegrationTest):
    """Test session creation through NEW_SESSION requests."""

    def test_create_new_session(self):
        """Test creating a new session returns session ID in header."""
        response = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})

        assert response.status_code == 200

        # Verify new session header is present
        assert SageMakerSessionHeader.NEW_SESSION_ID in response.headers
        session_header = response.headers[SageMakerSessionHeader.NEW_SESSION_ID]

        # Header should contain session ID and expiration
        assert len(session_header) > 0
        assert "Expires=" in session_header
        session_id = extract_session_id_from_header(session_header)
        assert len(session_id) == 36  # UUID format

    def test_create_multiple_sessions(self):
        """Test creating multiple sessions generates unique IDs."""
        response1 = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )
        response2 = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )

        session1_header = response1.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        session2_header = response2.headers[SageMakerSessionHeader.NEW_SESSION_ID]

        session1_id = extract_session_id_from_header(session1_header)
        session2_id = extract_session_id_from_header(session2_header)

        # Sessions should have different IDs
        assert session1_id != session2_id

    def test_new_session_response_body(self):
        """Test NEW_SESSION response body contains confirmation."""
        response = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})

        body = response.text
        assert "created" in body.lower() or "session" in body.lower()


class TestSessionUsage(BaseSessionIntegrationTest):
    """Test using existing sessions across multiple requests."""

    def test_use_session_across_requests(self):
        """Test that session ID persists state across requests."""
        # Create a session
        create_response = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )
        session_header = create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        session_id = extract_session_id_from_header(session_header)

        # Make multiple requests with the same session ID
        self.capture.clear()
        response1 = self.client.post(
            "/invocations",
            json={"prompt": "first request"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        response2 = self.client.post(
            "/invocations",
            json={"prompt": "second request"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        # Verify both requests succeeded
        assert response1.status_code == 200
        assert response2.status_code == 200

        # Verify counter incremented (state persisted)
        captures = self.capture.get_by_type("use_session")
        assert len(captures) == 2
        assert captures[0]["data"]["count"] == 1
        assert captures[1]["data"]["count"] == 2

    def test_different_sessions_have_independent_state(self):
        """Test that different sessions maintain independent state."""
        # Create two sessions
        create1 = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})
        session1_id = extract_session_id_from_header(
            create1.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        create2 = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})
        session2_id = extract_session_id_from_header(
            create2.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Make requests with each session
        self.capture.clear()
        self.client.post(
            "/invocations",
            json={"prompt": "session1 req1"},
            headers={SageMakerSessionHeader.SESSION_ID: session1_id},
        )
        self.client.post(
            "/invocations",
            json={"prompt": "session2 req1"},
            headers={SageMakerSessionHeader.SESSION_ID: session2_id},
        )
        self.client.post(
            "/invocations",
            json={"prompt": "session1 req2"},
            headers={SageMakerSessionHeader.SESSION_ID: session1_id},
        )

        # Verify each session has independent counter
        captures = self.capture.get_by_type("use_session")
        session1_captures = [c for c in captures if c["session_id"] == session1_id]
        session2_captures = [c for c in captures if c["session_id"] == session2_id]

        # Session 1: 2 requests
        assert len(session1_captures) == 2
        assert session1_captures[0]["data"]["count"] == 1
        assert session1_captures[1]["data"]["count"] == 2

        # Session 2: 1 request
        assert len(session2_captures) == 1
        assert session2_captures[0]["data"]["count"] == 1

    def test_request_without_session_id(self):
        """Test that requests without session ID work (stateless mode)."""
        response = self.client.post(
            "/invocations",
            json={"prompt": "stateless request"},
            # No session header
        )

        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["session_id"] == "no-session"


class TestSessionClosure(BaseSessionIntegrationTest):
    """Test session closure through CLOSE requests."""

    def test_close_session(self):
        """Test closing a session returns proper header."""
        # Create a session
        create_response = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Close the session
        close_response = self.client.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        assert close_response.status_code == 200

        # Verify closed session header is present
        assert SageMakerSessionHeader.CLOSED_SESSION_ID in close_response.headers
        closed_id = close_response.headers[SageMakerSessionHeader.CLOSED_SESSION_ID]
        assert closed_id == session_id

    def test_use_after_close_fails(self):
        """Test that using a closed session fails validation."""
        # Create and use a session
        create_response = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Use the session
        self.client.post(
            "/invocations",
            json={"prompt": "test"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        # Close the session
        self.client.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        # Try to use after close
        response = self.client.post(
            "/invocations",
            json={"prompt": "after close"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        # Should fail - session no longer exists
        assert response.status_code == 400


class TestSessionErrorCases(BaseSessionIntegrationTest):
    """Test error handling for invalid session operations."""

    def test_invalid_session_id(self):
        """Test using non-existent session ID returns error."""
        response = self.client.post(
            "/invocations",
            json={"prompt": "test"},
            headers={SageMakerSessionHeader.SESSION_ID: "invalid-session-id"},
        )

        assert response.status_code == 400

    def test_close_nonexistent_session(self):
        """Test closing non-existent session returns error."""
        response = self.client.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            headers={SageMakerSessionHeader.SESSION_ID: "nonexistent-id"},
        )

        assert response.status_code in [400, 424]  # Bad request or failed dependency

    def test_close_without_session_id(self):
        """Test closing without session ID header returns error."""
        response = self.client.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            # No session header
        )

        assert response.status_code in [400, 424]

    def test_invalid_request_type(self):
        """Test invalid requestType returns error."""
        response = self.client.post(
            "/invocations", json={"requestType": "INVALID_TYPE"}
        )

        assert response.status_code == 400

    def test_extra_fields_in_session_request(self):
        """Test session request with extra fields returns error."""
        response = self.client.post(
            "/invocations",
            json={"requestType": "NEW_SESSION", "extra_field": "not_allowed"},
        )

        assert response.status_code == 400


class TestSessionEndToEndFlow(BaseSessionIntegrationTest):
    """Test complete end-to-end session workflows."""

    def test_full_session_lifecycle(self):
        """Test complete lifecycle: create -> use multiple times -> close.

        This is the primary happy path for stateful sessions.
        """
        # 1. Create session
        create_response = self.client.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )
        assert create_response.status_code == 200
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # 2. Use session multiple times
        for i in range(3):
            use_response = self.client.post(
                "/invocations",
                json={"prompt": f"request {i+1}"},
                headers={SageMakerSessionHeader.SESSION_ID: session_id},
            )
            assert use_response.status_code == 200

        # 3. Close session
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

    def test_multiple_concurrent_sessions(self):
        """Test managing multiple active sessions simultaneously."""
        # Create 3 sessions
        sessions = []
        for _ in range(3):
            create_response = self.client.post(
                "/invocations", json={"requestType": "NEW_SESSION"}
            )
            session_id = extract_session_id_from_header(
                create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
            )
            sessions.append(session_id)

        # Use each session
        for session_id in sessions:
            response = self.client.post(
                "/invocations",
                json={"prompt": "test"},
                headers={SageMakerSessionHeader.SESSION_ID: session_id},
            )
            assert response.status_code == 200

        # Close all sessions
        for session_id in sessions:
            close_response = self.client.post(
                "/invocations",
                json={"requestType": "CLOSE"},
                headers={SageMakerSessionHeader.SESSION_ID: session_id},
            )
            assert close_response.status_code == 200

    def test_interleaved_session_operations(self):
        """Test that session operations can be interleaved."""
        # Create session 1
        create1 = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})
        session1_id = extract_session_id_from_header(
            create1.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Use session 1
        self.client.post(
            "/invocations",
            json={"prompt": "s1r1"},
            headers={SageMakerSessionHeader.SESSION_ID: session1_id},
        )

        # Create session 2
        create2 = self.client.post("/invocations", json={"requestType": "NEW_SESSION"})
        session2_id = extract_session_id_from_header(
            create2.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Use session 1 again
        self.client.post(
            "/invocations",
            json={"prompt": "s1r2"},
            headers={SageMakerSessionHeader.SESSION_ID: session1_id},
        )

        # Use session 2
        self.client.post(
            "/invocations",
            json={"prompt": "s2r1"},
            headers={SageMakerSessionHeader.SESSION_ID: session2_id},
        )

        # Close session 1
        self.client.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            headers={SageMakerSessionHeader.SESSION_ID: session1_id},
        )

        # Session 2 should still work
        response = self.client.post(
            "/invocations",
            json={"prompt": "s2r2"},
            headers={SageMakerSessionHeader.SESSION_ID: session2_id},
        )
        assert response.status_code == 200


class TestSessionsDisabled:
    """Test behavior when stateful sessions are disabled.

    These tests verify that session management requests fail gracefully
    when the SAGEMAKER_ENABLE_STATEFUL_SESSIONS flag is not set.
    """

    @pytest.fixture
    def app_with_sessions_disabled(self, monkeypatch):
        """Create app with sessions disabled."""
        # Explicitly disable sessions
        monkeypatch.delenv("SAGEMAKER_ENABLE_STATEFUL_SESSIONS", raising=False)
        monkeypatch.delenv("SAGEMAKER_SESSIONS_PATH", raising=False)
        monkeypatch.delenv("SAGEMAKER_SESSIONS_EXPIRATION", raising=False)

        # Reinitialize the global session manager (should be None)
        init_session_manager_from_env()

        # Now create the app with sessions disabled
        app = FastAPI()
        router = APIRouter()

        @router.post("/invocations")
        @sagemaker_standards.stateful_session_manager()
        async def invocations(request: Request):
            """Stateful invocation handler."""
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode())

            return Response(
                status_code=200,
                content=json.dumps({"message": "success", "echo": body}),
            )

        app.include_router(router)
        sagemaker_standards.bootstrap(app)

        return TestClient(app)

    def test_new_session_request_fails_when_disabled(self, app_with_sessions_disabled):
        """Test that NEW_SESSION request fails when sessions are disabled."""
        response = app_with_sessions_disabled.post(
            "/invocations", json={"requestType": "NEW_SESSION"}
        )

        # Should fail with 400 BAD_REQUEST since sessions are not enabled
        assert response.status_code == 400
        assert SESSION_DISABLED_ERROR_DETAIL in response.text

    def test_close_session_request_fails_when_disabled(
        self, app_with_sessions_disabled
    ):
        """Test that CLOSE request fails when sessions are disabled."""
        response = app_with_sessions_disabled.post(
            "/invocations",
            json={"requestType": "CLOSE"},
            headers={SageMakerSessionHeader.SESSION_ID: "some-session-id"},
        )

        # Should fail with 400 BAD_REQUEST due to session header when sessions disabled
        assert response.status_code == 400
        assert SESSION_DISABLED_ERROR_DETAIL in response.text

    def test_regular_requests_work_when_sessions_disabled(
        self, app_with_sessions_disabled
    ):
        """Test that regular requests still work when sessions are disabled."""
        response = app_with_sessions_disabled.post(
            "/invocations", json={"prompt": "test request"}
        )

        # Regular requests should still work
        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["message"] == "success"
        assert data["echo"]["prompt"] == "test request"

    def test_regular_requests_with_session_header_when_disabled(
        self, app_with_sessions_disabled
    ):
        """Test that requests with session headers fail validation when sessions disabled."""
        response = app_with_sessions_disabled.post(
            "/invocations",
            json={"prompt": "test"},
            headers={SageMakerSessionHeader.SESSION_ID: "invalid-session"},
        )

        # Should fail with 400 BAD_REQUEST since sessions are not enabled
        assert response.status_code == 400
        assert SESSION_DISABLED_ERROR_DETAIL in response.text


class TestSessionIdPathInjection(BaseSessionIntegrationTest):
    """Test request_session_id_path parameter for injecting session ID into request body."""

    def setup_handlers(self):
        """Define handlers with request_session_id_path parameter."""

        @self.router.post("/invocations-with-path")
        @sagemaker_standards.stateful_session_manager(
            engine_request_session_id_path="session_id"
        )
        async def invocations_with_path(request: Request):
            """Handler that injects session ID into request body at 'session_id' key."""
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode())

            # Capture for test verification
            self.capture.capture(
                "invocation_with_path", body.get("session_id"), {"body": body}
            )

            return Response(
                status_code=200,
                content=json.dumps(
                    {
                        "message": "success",
                        "session_id_from_body": body.get("session_id"),
                        "echo": body,
                    }
                ),
            )

        @self.router.post("/invocations-nested-path")
        @sagemaker_standards.stateful_session_manager(
            engine_request_session_id_path="metadata.session_id"
        )
        async def invocations_nested_path(request: Request):
            """Handler that injects session ID into nested path in request body."""
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode())

            # Capture for test verification
            session_id = (
                body.get("metadata", {}).get("session_id")
                if isinstance(body.get("metadata"), dict)
                else None
            )
            self.capture.capture("invocation_nested_path", session_id, {"body": body})

            return Response(
                status_code=200,
                content=json.dumps(
                    {
                        "message": "success",
                        "session_id_from_body": session_id,
                        "echo": body,
                    }
                ),
            )

    def test_session_id_injected_into_body(self):
        """Test that session ID from header is injected into request body."""
        # Create a session
        create_response = self.client.post(
            "/invocations-with-path", json={"requestType": "NEW_SESSION"}
        )
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Make request with session ID in header
        self.capture.clear()
        response = self.client.post(
            "/invocations-with-path",
            json={"prompt": "test request"},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        assert response.status_code == 200
        data = json.loads(response.text)

        # Verify session ID was injected into body
        assert data["session_id_from_body"] == session_id
        assert data["echo"]["session_id"] == session_id
        assert data["echo"]["prompt"] == "test request"

    def test_session_id_injected_into_nested_path(self):
        """Test that session ID is injected into nested path in request body."""
        # Create a session
        create_response = self.client.post(
            "/invocations-nested-path", json={"requestType": "NEW_SESSION"}
        )
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Make request with session ID in header
        self.capture.clear()
        response = self.client.post(
            "/invocations-nested-path",
            json={"prompt": "test request", "metadata": {"user": "test"}},
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        assert response.status_code == 200
        data = json.loads(response.text)

        # Verify session ID was injected into nested path
        assert data["session_id_from_body"] == session_id
        assert data["echo"]["metadata"]["session_id"] == session_id
        assert data["echo"]["metadata"]["user"] == "test"
        assert data["echo"]["prompt"] == "test request"

    def test_session_id_not_injected_without_header(self):
        """Test that session ID is not injected when header is not present."""
        response = self.client.post(
            "/invocations-with-path",
            json={"prompt": "test request"},
            # No session header
        )

        assert response.status_code == 200
        data = json.loads(response.text)

        # Verify session ID was not injected
        assert data["session_id_from_body"] is None
        assert "session_id" not in data["echo"] or data["echo"]["session_id"] is None

    def test_session_id_injection_with_multiple_requests(self):
        """Test that session ID injection works across multiple requests."""
        # Create a session
        create_response = self.client.post(
            "/invocations-with-path", json={"requestType": "NEW_SESSION"}
        )
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Make multiple requests with the same session ID
        for i in range(3):
            response = self.client.post(
                "/invocations-with-path",
                json={"prompt": f"request {i+1}"},
                headers={SageMakerSessionHeader.SESSION_ID: session_id},
            )

            assert response.status_code == 200
            data = json.loads(response.text)
            assert data["session_id_from_body"] == session_id

    def test_different_sessions_inject_different_ids(self):
        """Test that different sessions inject their respective IDs."""
        # Create two sessions
        create1 = self.client.post(
            "/invocations-with-path", json={"requestType": "NEW_SESSION"}
        )
        session1_id = extract_session_id_from_header(
            create1.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        create2 = self.client.post(
            "/invocations-with-path", json={"requestType": "NEW_SESSION"}
        )
        session2_id = extract_session_id_from_header(
            create2.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Make requests with each session
        response1 = self.client.post(
            "/invocations-with-path",
            json={"prompt": "session 1"},
            headers={SageMakerSessionHeader.SESSION_ID: session1_id},
        )
        response2 = self.client.post(
            "/invocations-with-path",
            json={"prompt": "session 2"},
            headers={SageMakerSessionHeader.SESSION_ID: session2_id},
        )

        # Verify each request got the correct session ID
        data1 = json.loads(response1.text)
        data2 = json.loads(response2.text)

        assert data1["session_id_from_body"] == session1_id
        assert data2["session_id_from_body"] == session2_id
        assert session1_id != session2_id

    def test_session_id_injection_preserves_existing_body_fields(self):
        """Test that session ID injection doesn't overwrite other body fields."""
        # Create a session
        create_response = self.client.post(
            "/invocations-with-path", json={"requestType": "NEW_SESSION"}
        )
        session_id = extract_session_id_from_header(
            create_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        )

        # Make request with multiple body fields
        original_body = {
            "prompt": "test",
            "temperature": 0.7,
            "max_tokens": 100,
            "metadata": {"user": "test_user", "request_id": "123"},
        }

        response = self.client.post(
            "/invocations-with-path",
            json=original_body,
            headers={SageMakerSessionHeader.SESSION_ID: session_id},
        )

        assert response.status_code == 200
        data = json.loads(response.text)

        # Verify session ID was added
        assert data["echo"]["session_id"] == session_id

        # Verify all original fields are preserved
        assert data["echo"]["prompt"] == "test"
        assert data["echo"]["temperature"] == 0.7
        assert data["echo"]["max_tokens"] == 100
        assert data["echo"]["metadata"]["user"] == "test_user"
        assert data["echo"]["metadata"]["request_id"] == "123"
