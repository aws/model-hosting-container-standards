"""Unit tests for sessions handlers module."""

import time
from http import HTTPStatus
from unittest.mock import Mock, patch

import pytest
from fastapi import Response
from fastapi.exceptions import HTTPException

from model_hosting_container_standards.sagemaker.sessions.handlers import (
    close_session,
    create_session,
    get_handler_for_request_type,
)
from model_hosting_container_standards.sagemaker.sessions.manager import Session
from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader,
    SessionRequestType,
)


class TestGetHandlerForRequestType:
    """Test get_handler_for_request_type function."""

    def test_returns_create_session_handler_for_new_session(self):
        """Test returns create_session handler for NEW_SESSION request type."""
        handler = get_handler_for_request_type(SessionRequestType.NEW_SESSION)
        assert handler == create_session

    def test_returns_close_session_handler_for_close(self):
        """Test returns close_session handler for CLOSE request type."""
        handler = get_handler_for_request_type(SessionRequestType.CLOSE)
        assert handler == close_session

    def test_returns_none_for_unknown_request_type(self):
        """Test returns None for unknown request type."""
        # Create a mock request type that's not in the enum
        mock_request_type = Mock()
        mock_request_type.value = "UNKNOWN"

        handler = get_handler_for_request_type(mock_request_type)
        assert handler is None


class TestCreateSession:
    """Test create_session handler."""

    @pytest.fixture
    def mock_session_with_expiration(self):
        """Helper to create a mock session with expiration."""
        session = Mock(spec=Session)
        session.session_id = "test-session-123"
        session.expiration_ts = time.time() + 1000
        return session

    @pytest.mark.asyncio
    async def test_creates_session_successfully(
        self, mock_session_manager, mock_request, mock_session_with_expiration
    ):
        """Test successfully creates a session and returns response."""
        mock_session_manager.create_session.return_value = mock_session_with_expiration

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.handlers.get_session_manager",
            return_value=mock_session_manager,
        ):
            response = await create_session(mock_request)

        assert isinstance(response, Response)
        assert response.status_code == HTTPStatus.OK.value
        assert SageMakerSessionHeader.NEW_SESSION_ID in response.headers

        # Verify header contains session ID and expiration
        header_value = response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
        assert mock_session_with_expiration.session_id in header_value
        assert "Expires=" in header_value
        assert mock_session_with_expiration.session_id in response.body.decode()

    @pytest.mark.asyncio
    async def test_calls_session_manager_create_session(
        self, mock_session_manager, mock_request, mock_session_with_expiration
    ):
        """Test calls session_manager.create_session method."""
        mock_session_manager.create_session.return_value = mock_session_with_expiration

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.handlers.get_session_manager",
            return_value=mock_session_manager,
        ):
            await create_session(mock_request)

        mock_session_manager.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_http_exception_on_session_creation_failure(
        self, mock_session_manager, mock_request
    ):
        """Test raises HTTPException when session creation fails."""
        mock_session_manager.create_session.side_effect = Exception("Creation failed")

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.handlers.get_session_manager",
            return_value=mock_session_manager,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_session(mock_request)

        assert exc_info.value.status_code == HTTPStatus.FAILED_DEPENDENCY.value
        assert "Failed to create session" in exc_info.value.detail


class TestCloseSession:
    """Test close_session handler."""

    @pytest.mark.asyncio
    async def test_closes_session_successfully(
        self, mock_session_manager, mock_request_with_session
    ):
        """Test successfully closes a session and returns response."""
        session_id = "test-session-123"
        mock_session_manager.close_session.return_value = None

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.handlers.get_session_manager",
            return_value=mock_session_manager,
        ):
            response = await close_session(mock_request_with_session)

        assert isinstance(response, Response)
        assert response.status_code == HTTPStatus.OK.value
        assert SageMakerSessionHeader.CLOSED_SESSION_ID in response.headers
        assert response.headers[SageMakerSessionHeader.CLOSED_SESSION_ID] == session_id
        assert session_id in response.body.decode()

    @pytest.mark.asyncio
    async def test_raises_http_exception_on_close_failure(
        self, mock_session_manager, mock_request_with_session
    ):
        """Test raises HTTPException when session close fails."""
        mock_session_manager.close_session.side_effect = ValueError("Session not found")

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.handlers.get_session_manager",
            return_value=mock_session_manager,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await close_session(mock_request_with_session)

        assert exc_info.value.status_code == HTTPStatus.FAILED_DEPENDENCY.value
        assert "Failed to close session" in exc_info.value.detail
