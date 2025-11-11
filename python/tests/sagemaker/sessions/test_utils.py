"""Unit tests for sessions utils module."""

from http import HTTPStatus
from unittest.mock import Mock

import pytest
from fastapi import Request
from fastapi.exceptions import HTTPException

from model_hosting_container_standards.sagemaker.sessions.models import (
    SESSION_DISABLED_ERROR_DETAIL,
    SageMakerSessionHeader,
)
from model_hosting_container_standards.sagemaker.sessions.utils import (
    get_session,
    get_session_id_from_request,
)


class TestGetSessionIdFromRequest:
    """Test get_session_id_from_request function."""

    def test_extracts_session_id_from_headers(self):
        """Test extracting session ID from request headers."""
        raw_request = Mock(spec=Request)
        raw_request.headers = {
            SageMakerSessionHeader.SESSION_ID: "test-session-123",
            "other-header": "other-value",
        }

        result = get_session_id_from_request(raw_request)

        assert result == "test-session-123"

    @pytest.mark.parametrize(
        "headers",
        [
            {
                "other-header": "value",
                "another-header": "another-value",
            },  # Missing session header
            None,  # No headers at all
            {},  # Empty headers dict
        ],
        ids=["missing_session_header", "none_headers", "empty_headers"],
    )
    def test_returns_none_when_session_id_not_present(self, headers):
        """Test returns None when session ID header is absent or headers are None."""
        raw_request = Mock(spec=Request)
        raw_request.headers = headers

        result = get_session_id_from_request(raw_request)

        assert result is None


class TestGetSession:
    """Test get_session function."""

    def test_returns_session_when_session_id_present_and_valid(
        self, mock_session_manager, mock_session
    ):
        """Test returns session when session ID is present and valid."""
        session_id = "valid-session-123"
        mock_session_manager.get_session.return_value = mock_session

        raw_request = Mock(spec=Request)
        raw_request.headers = {SageMakerSessionHeader.SESSION_ID: session_id}

        result = get_session(mock_session_manager, raw_request)

        assert result == mock_session
        mock_session_manager.get_session.assert_called_once_with(session_id)

    def test_returns_none_when_no_session_id_in_request(self, mock_session_manager):
        """Test returns None when no session ID in request headers."""
        mock_session_manager.get_session.return_value = None

        raw_request = Mock(spec=Request)
        raw_request.headers = {}

        result = get_session(mock_session_manager, raw_request)

        assert result is None
        mock_session_manager.get_session.assert_called_once_with(None)

    def test_raises_http_exception_when_sessions_not_enabled_but_header_present(self):
        """Test raises HTTPException when sessions not enabled but session header present."""
        raw_request = Mock(spec=Request)
        raw_request.headers = {SageMakerSessionHeader.SESSION_ID: "some-session-id"}

        with pytest.raises(HTTPException) as exc_info:
            get_session(None, raw_request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
        assert SESSION_DISABLED_ERROR_DETAIL in exc_info.value.detail
        assert SageMakerSessionHeader.SESSION_ID in exc_info.value.detail

    def test_returns_none_when_sessions_not_enabled_and_no_header(self):
        """Test returns None when sessions not enabled and no session header."""
        raw_request = Mock(spec=Request)
        raw_request.headers = {}

        result = get_session(None, raw_request)

        assert result is None

    def test_propagates_value_error_from_session_manager(self, mock_session_manager):
        """Test propagates ValueError from session manager when session not found."""
        mock_session_manager.get_session.side_effect = ValueError(
            "session not found: xyz"
        )

        raw_request = Mock(spec=Request)
        raw_request.headers = {SageMakerSessionHeader.SESSION_ID: "nonexistent-session"}

        with pytest.raises(ValueError, match="session not found"):
            get_session(mock_session_manager, raw_request)

    def test_returns_none_when_session_manager_returns_none(self, mock_session_manager):
        """Test returns None when session manager returns None (expired session)."""
        mock_session_manager.get_session.return_value = None

        raw_request = Mock(spec=Request)
        raw_request.headers = {SageMakerSessionHeader.SESSION_ID: "expired-session"}

        result = get_session(mock_session_manager, raw_request)

        assert result is None

    def test_handles_empty_string_session_id(self, mock_session_manager):
        """Test handles empty string session ID gracefully."""
        mock_session_manager.get_session.return_value = None

        raw_request = Mock(spec=Request)
        raw_request.headers = {SageMakerSessionHeader.SESSION_ID: ""}

        result = get_session(mock_session_manager, raw_request)

        assert result is None
        mock_session_manager.get_session.assert_called_once_with("")
