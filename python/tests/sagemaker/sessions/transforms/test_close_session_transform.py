"""Unit tests for CloseSessionApiTransform."""

import json
from http import HTTPStatus
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Request, Response
from fastapi.exceptions import HTTPException

from model_hosting_container_standards.common import BaseTransformRequestOutput
from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader,
)
from model_hosting_container_standards.sagemaker.sessions.transforms.close_session import (
    CloseSessionApiTransform,
)
from model_hosting_container_standards.sagemaker.sessions.transforms.constants import (
    RESPONSE_CONTENT_KEY,
)


class TestCloseSessionInitialization:
    """Test CloseSessionApiTransform initialization."""

    def test_requires_content_in_response_shape(self):
        """Test that initialization requires RESPONSE_CONTENT_KEY in response_shape."""
        with pytest.raises(ValueError) as exc_info:
            CloseSessionApiTransform(request_shape={}, response_shape={})

        assert RESPONSE_CONTENT_KEY in str(exc_info.value)

    def test_successful_initialization(self):
        """Test successful initialization with valid response_shape."""
        transform = CloseSessionApiTransform(
            request_shape={},
            response_shape={RESPONSE_CONTENT_KEY: "body.message"},
        )
        assert transform is not None


class TestCloseSessionValidation:
    """Test request validation."""

    @pytest.fixture
    def transform(self):
        """Create transform."""
        return CloseSessionApiTransform(
            request_shape={},
            response_shape={RESPONSE_CONTENT_KEY: "body.message"},
        )

    @pytest.mark.asyncio
    async def test_requires_session_id_header(self, transform):
        """Test that session ID header is required."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {}
        mock_request.headers = {}  # No session ID

        with pytest.raises(HTTPException) as exc_info:
            await transform.transform_request(mock_request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
        assert "Session ID is required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_succeeds_with_session_id_header(self, transform):
        """Test that request succeeds with session ID header."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {}
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "sess-123"}

        result = await transform.transform_request(mock_request)

        assert result is not None


class TestCloseSessionTransformRequest:
    """Test request transformation."""

    @pytest.fixture
    def transform(self):
        """Create transform with request shape."""
        return CloseSessionApiTransform(
            request_shape={"reason": "body.reason"},
            response_shape={RESPONSE_CONTENT_KEY: "body.message"},
        )

    @pytest.mark.asyncio
    async def test_transforms_request_body(self, transform):
        """Test that request body is transformed using JMESPath."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"reason": "timeout"}
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "sess-123"}

        result = await transform.transform_request(mock_request)

        assert result.request["reason"] == "timeout"

    @pytest.mark.asyncio
    async def test_updates_raw_request_body(self, transform):
        """Test that raw request body is updated."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"reason": "timeout"}
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "sess-123"}

        await transform.transform_request(mock_request)

        updated_body = json.loads(mock_request._body.decode())
        assert updated_body == {"reason": "timeout"}


class TestCloseSessionTransformResponse:
    """Test response transformation."""

    @pytest.fixture
    def transform(self):
        """Create transform."""
        return CloseSessionApiTransform(
            request_shape={},
            response_shape={RESPONSE_CONTENT_KEY: "body.message"},
        )

    def test_extracts_content_and_adds_header(self, transform):
        """Test that content is extracted and session ID added to headers."""
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps({"message": "Session closed"}),
        )

        mock_request = Mock(spec=Request)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "sess-123"}
        transform_output = BaseTransformRequestOutput(
            raw_request=mock_request, intercept_func=None
        )

        result = transform.transform_response(response, transform_output)

        assert result.status_code == HTTPStatus.OK.value
        assert result.headers[SageMakerSessionHeader.CLOSED_SESSION_ID] == "sess-123"
        assert b"sess-123" in result.body
        assert b"Session closed" in result.body

    def test_handles_missing_content(self, transform):
        """Test that missing content is handled gracefully."""
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps({}),
        )

        mock_request = Mock(spec=Request)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "sess-123"}
        transform_output = BaseTransformRequestOutput(
            raw_request=mock_request, intercept_func=None
        )

        result = transform.transform_response(response, transform_output)

        assert result.status_code == HTTPStatus.OK.value
        assert result.headers[SageMakerSessionHeader.CLOSED_SESSION_ID] == "sess-123"

    def test_passes_through_error_responses(self, transform):
        """Test that error responses pass through unchanged."""
        response = Response(
            status_code=HTTPStatus.NOT_FOUND.value,
            content=b"Session not found",
        )

        mock_request = Mock(spec=Request)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "sess-123"}
        transform_output = BaseTransformRequestOutput(
            raw_request=mock_request, intercept_func=None
        )

        result = transform.transform_response(response, transform_output)

        assert result.status_code == HTTPStatus.NOT_FOUND.value
        assert result.body == b"Session not found"


class TestCloseSessionEdgeCases:
    """Test edge cases for CloseSessionApiTransform."""

    @pytest.fixture
    def transform(self):
        """Create transform."""
        return CloseSessionApiTransform(
            request_shape={},
            response_shape={RESPONSE_CONTENT_KEY: "body.message"},
        )

    def test_handles_none_content(self, transform):
        """Test that None content is handled gracefully."""
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps({"message": None}),
        )

        mock_request = Mock(spec=Request)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "sess-123"}
        transform_output = BaseTransformRequestOutput(
            raw_request=mock_request, intercept_func=None
        )

        result = transform.transform_response(response, transform_output)

        assert result.status_code == HTTPStatus.OK.value
        assert result.headers[SageMakerSessionHeader.CLOSED_SESSION_ID] == "sess-123"

    def test_handles_empty_string_content(self, transform):
        """Test that empty string content is handled gracefully."""
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps({"message": ""}),
        )

        mock_request = Mock(spec=Request)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "sess-123"}
        transform_output = BaseTransformRequestOutput(
            raw_request=mock_request, intercept_func=None
        )

        result = transform.transform_response(response, transform_output)

        assert result.status_code == HTTPStatus.OK.value
        assert result.headers[SageMakerSessionHeader.CLOSED_SESSION_ID] == "sess-123"

    def test_extracts_content_from_nested_path(self, transform):
        """Test extraction of content from nested response structure."""
        transform_nested = CloseSessionApiTransform(
            request_shape={},
            response_shape={RESPONSE_CONTENT_KEY: "body.result.message"},
        )

        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps({"result": {"message": "Session closed successfully"}}),
        )

        mock_request = Mock(spec=Request)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "sess-nested-123"}
        transform_output = BaseTransformRequestOutput(
            raw_request=mock_request, intercept_func=None
        )

        result = transform_nested.transform_response(response, transform_output)

        assert result.status_code == HTTPStatus.OK.value
        assert (
            result.headers[SageMakerSessionHeader.CLOSED_SESSION_ID]
            == "sess-nested-123"
        )
        assert b"Session closed successfully" in result.body

    def test_handles_malformed_json_in_response(self, transform):
        """Test that malformed JSON in response is handled gracefully.

        The serialize_response function catches JSONDecodeError and keeps the body as a string,
        so malformed JSON doesn't cause the transform to fail.
        """
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=b"not valid json {{{",
        )

        mock_request = Mock(spec=Request)
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: "sess-123"}
        transform_output = BaseTransformRequestOutput(
            raw_request=mock_request, intercept_func=None
        )

        # Should handle gracefully - malformed JSON is kept as string
        result = transform.transform_response(response, transform_output)

        # Should still return a response with the session ID header
        assert result.status_code == HTTPStatus.OK.value
        assert result.headers[SageMakerSessionHeader.CLOSED_SESSION_ID] == "sess-123"

    @pytest.mark.asyncio
    async def test_validates_session_id_before_transformation(self, transform):
        """Test that session ID validation happens before request transformation."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"reason": "timeout"}
        mock_request.headers = {}  # Missing session ID

        # Should fail validation before even attempting transformation
        with pytest.raises(HTTPException) as exc_info:
            await transform.transform_request(mock_request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
        # json() should not have been called since validation failed first
        mock_request.json.assert_not_called()

    @pytest.mark.asyncio
    async def test_validates_empty_session_id(self, transform):
        """Test that empty session ID is rejected."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {}
        mock_request.headers = {SageMakerSessionHeader.SESSION_ID: ""}

        with pytest.raises(HTTPException) as exc_info:
            await transform.transform_request(mock_request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
