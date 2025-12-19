"""Unit tests for sessions transform module."""

import json
from http import HTTPStatus
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request
from fastapi.exceptions import HTTPException

from model_hosting_container_standards.common import BaseTransformRequestOutput
from model_hosting_container_standards.sagemaker.sessions.handlers import (
    close_session,
    create_session,
)
from model_hosting_container_standards.sagemaker.sessions.manager import SessionManager
from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader,
    SessionRequest,
    SessionRequestType,
)
from model_hosting_container_standards.sagemaker.sessions.transform import (
    SessionApiTransform,
    _parse_session_request,
)


class TestParseSessionRequest:
    """Test _parse_session_request function."""

    def test_parses_valid_new_session_request(self):
        """Test parsing valid NEW_SESSION request."""
        request_data = {"requestType": "NEW_SESSION"}

        result = _parse_session_request(request_data)

        assert result is not None
        assert result.requestType == SessionRequestType.NEW_SESSION

    def test_parses_valid_close_request(self):
        """Test parsing valid CLOSE request."""
        request_data = {"requestType": "CLOSE"}

        result = _parse_session_request(request_data)

        assert result is not None
        assert result.requestType == SessionRequestType.CLOSE

    def test_returns_none_for_non_session_request(self):
        """Test returns None for request without requestType field."""
        request_data = {"data": "some_value", "other": "field"}

        result = _parse_session_request(request_data)

        assert result is None

    def test_returns_none_for_empty_request(self):
        """Test returns None for empty request."""
        request_data = {}

        result = _parse_session_request(request_data)

        assert result is None

    def test_raises_http_exception_for_invalid_request_type(self):
        """Test raises HTTPException for invalid requestType value."""
        request_data = {"requestType": "INVALID_TYPE"}

        with pytest.raises(HTTPException) as exc_info:
            _parse_session_request(request_data)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    def test_raises_http_exception_for_extra_fields(self):
        """Test raises HTTPException when extra fields present in session request."""
        request_data = {"requestType": "NEW_SESSION", "extra_field": "not_allowed"}

        with pytest.raises(HTTPException) as exc_info:
            _parse_session_request(request_data)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value


class TestValidateSessionId:
    """Test _validate_session_id method."""

    def test_does_not_raise_when_session_id_valid(self, enable_sessions_env):
        """Test does not raise exception when session ID is valid."""
        transform = SessionApiTransform(request_shape={}, response_shape={})
        mock_request = Mock(spec=Request)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "valid-session"}

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.transform.get_session"
        ) as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value = mock_session

            # Should not raise any exception
            result = transform._validate_session_id("valid-session", mock_request)
            assert result == "valid-session"

    def test_raises_http_exception_when_session_not_found(self, enable_sessions_env):
        """Test raises HTTPException when session ID not found."""
        transform = SessionApiTransform(request_shape={}, response_shape={})
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            SageMakerSessionHeader.SESSION_ID: "nonexistent-session"
        }

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.transform.get_session"
        ) as mock_get_session:
            mock_get_session.side_effect = ValueError("session not found")

            with pytest.raises(HTTPException) as exc_info:
                transform._validate_session_id("nonexistent-session", mock_request)

            assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    def test_error_message_includes_original_error(self, enable_sessions_env):
        """Test error message includes the original error message."""
        transform = SessionApiTransform(request_shape={}, response_shape={})
        mock_request = Mock(spec=Request)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "bad-session"}

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.transform.get_session"
        ) as mock_get_session:
            mock_get_session.side_effect = ValueError("custom error message")

            with pytest.raises(HTTPException) as exc_info:
                transform._validate_session_id("bad-session", mock_request)

            assert "custom error message" in exc_info.value.detail


class TestProcessSessionRequest:
    """Test _process_session_request method."""

    @pytest.fixture
    def transform(self, enable_sessions_env):
        """Create SessionApiTransform instance."""
        return SessionApiTransform(request_shape={}, response_shape={})

    def test_returns_create_handler_for_new_session_request(
        self, transform, mock_request
    ):
        """Test returns create_session handler for NEW_SESSION request."""
        session_request = SessionRequest(requestType=SessionRequestType.NEW_SESSION)

        result = transform._process_session_request(session_request, None, mock_request)

        assert isinstance(result, BaseTransformRequestOutput)
        assert result.raw_request == mock_request
        assert result.intercept_func == create_session

    def test_returns_close_handler_for_close_request(self, transform, mock_request):
        """Test returns close_session handler for CLOSE request."""
        session_request = SessionRequest(requestType=SessionRequestType.CLOSE)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "test-session"}

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.transform.get_session"
        ) as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value = mock_session

            result = transform._process_session_request(
                session_request, "test-session", mock_request
            )

            assert isinstance(result, BaseTransformRequestOutput)
            assert result.raw_request == mock_request
            assert result.intercept_func == close_session

    def test_validates_session_if_session_id_present(self, transform):
        """Test validates session when session ID is present in headers."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "test-session"}

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.transform.get_session"
        ) as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value = mock_session

            transform._process_session_request(
                SessionRequest(requestType=SessionRequestType.CLOSE),
                "test-session",
                mock_request,
            )

            # Should validate the session
            mock_get_session.assert_called_once()

    def test_raises_exception_when_sessions_disabled(
        self, mock_request, monkeypatch, temp_session_storage
    ):
        """Test raises HTTPException when sessions are disabled."""
        # Disable sessions
        monkeypatch.delenv("SAGEMAKER_ENABLE_STATEFUL_SESSIONS", raising=False)
        from model_hosting_container_standards.sagemaker.sessions.manager import (
            init_session_manager_from_env,
        )

        init_session_manager_from_env()

        transform = SessionApiTransform(request_shape={}, response_shape={})
        session_request = SessionRequest(requestType=SessionRequestType.NEW_SESSION)

        with pytest.raises(HTTPException) as exc_info:
            transform._process_session_request(session_request, None, mock_request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    def test_propagates_validation_errors(self, transform):
        """Test propagates validation errors from session validation."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "invalid-session"}

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.transform.get_session"
        ) as mock_get_session:
            mock_get_session.side_effect = ValueError("Session not found")

            with pytest.raises(HTTPException) as exc_info:
                transform._process_session_request(
                    SessionRequest(requestType=SessionRequestType.CLOSE),
                    "invalid-session",
                    mock_request,
                )

            assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value


class TestSessionApiTransform:
    """Test SessionApiTransform class."""

    @pytest.fixture
    def transform(self, enable_sessions_env):
        """Create SessionApiTransform instance."""
        return SessionApiTransform(request_shape={}, response_shape={})

    def test_initialization_creates_session_manager(self, enable_sessions_env):
        """Test initialization creates internal session manager."""
        transform = SessionApiTransform(request_shape={}, response_shape={})

        assert hasattr(transform, "_session_manager")
        assert isinstance(transform._session_manager, SessionManager)

    def test_initialization_accepts_request_and_response_shapes(
        self, enable_sessions_env
    ):
        """Test initialization accepts request and response shapes."""
        request_shape = {"field": "value"}
        response_shape = {"output": "format"}

        transform = SessionApiTransform(
            request_shape=request_shape, response_shape=response_shape
        )

        assert transform is not None

    @pytest.mark.asyncio
    async def test_transform_request_parses_json_and_processes(self, transform):
        """Test transform_request parses JSON and processes session request."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"requestType": "NEW_SESSION"}
        mock_request.headers = {}

        result = await transform.transform_request(mock_request)

        assert isinstance(result, BaseTransformRequestOutput)
        assert result.intercept_func == create_session

    @pytest.mark.asyncio
    async def test_transform_request_raises_exception_for_invalid_json(self, transform):
        """Test transform_request raises HTTPException for invalid JSON."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)

        with pytest.raises(HTTPException) as exc_info:
            await transform.transform_request(mock_request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
        assert "JSON decode error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_transform_request_parses_close_session_request(self, transform):
        """Test transform_request parses CLOSE session request."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"requestType": "CLOSE"}
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "session-to-close"}

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.transform.get_session"
        ) as mock_get_session:
            mock_get_session.return_value = Mock()

            result = await transform.transform_request(mock_request)

            assert isinstance(result, BaseTransformRequestOutput)
            assert result.intercept_func == close_session

    def test_transform_response_is_passthrough(self, transform):
        """Test transform_response returns response unmodified."""
        mock_response = Mock()
        mock_transform_output = Mock()

        result = transform.transform_response(mock_response, mock_transform_output)

        assert result == mock_response

    @pytest.mark.asyncio
    async def test_end_to_end_new_session_flow(self, transform):
        """Test end-to-end NEW_SESSION request flow."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"requestType": "NEW_SESSION"}
        mock_request.headers = {}

        result = await transform.transform_request(mock_request)

        # Verify we get an intercept function
        assert result.intercept_func == create_session
        assert result.request is None

        # Verify we can call the handler
        response = await result.intercept_func(mock_request)
        assert response.status_code == HTTPStatus.OK.value
        assert SageMakerSessionHeader.NEW_SESSION_ID in response.headers

    @pytest.mark.asyncio
    async def test_handles_concurrent_requests(self, transform):
        """Test transform handles concurrent requests correctly."""
        mock_request1 = AsyncMock(spec=Request)
        mock_request1.json.return_value = {"requestType": "NEW_SESSION"}
        mock_request1.headers = {}

        mock_request2 = AsyncMock(spec=Request)
        mock_request2.json.return_value = {"data": "regular_data"}
        mock_request2.headers = {}

        # Process both requests
        result1 = await transform.transform_request(mock_request1)
        result2 = await transform.transform_request(mock_request2)

        # First should be session request
        assert result1.intercept_func == create_session

        # Second should be regular request
        assert result2.intercept_func is None
