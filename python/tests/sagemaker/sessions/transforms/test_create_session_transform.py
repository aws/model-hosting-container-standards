"""Unit tests for CreateSessionApiTransform."""

import json
from http import HTTPStatus
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader,
)
from model_hosting_container_standards.sagemaker.sessions.transforms.constants import (
    RESPONSE_CONTENT_KEY,
)
from model_hosting_container_standards.sagemaker.sessions.transforms.create_session import (
    CreateSessionApiTransform,
)


class TestCreateSessionInitialization:
    """Test CreateSessionApiTransform initialization."""

    def test_requires_session_id_in_response_shape(self):
        """Test that initialization requires NEW_SESSION_ID in response_shape."""
        with pytest.raises(ValueError) as exc_info:
            CreateSessionApiTransform(
                request_shape={},
                response_shape={RESPONSE_CONTENT_KEY: "body.message"},
            )
        assert SageMakerSessionHeader.NEW_SESSION_ID in str(exc_info.value)

    def test_requires_content_in_response_shape(self):
        """Test that initialization requires RESPONSE_CONTENT_KEY in response_shape."""
        with pytest.raises(ValueError) as exc_info:
            CreateSessionApiTransform(
                request_shape={},
                response_shape={SageMakerSessionHeader.NEW_SESSION_ID: "body.id"},
            )
        assert RESPONSE_CONTENT_KEY in str(exc_info.value)

    def test_successful_initialization(self):
        """Test successful initialization with valid response_shape."""
        transform = CreateSessionApiTransform(
            request_shape={"model": "body.model"},
            response_shape={
                SageMakerSessionHeader.NEW_SESSION_ID: "body.session_id",
                RESPONSE_CONTENT_KEY: "body.message",
            },
        )
        assert transform is not None


class TestCreateSessionTransformRequest:
    """Test request transformation."""

    @pytest.fixture
    def transform(self):
        """Create transform with request shape."""
        return CreateSessionApiTransform(
            request_shape={"model": "body.model"},
            response_shape={
                SageMakerSessionHeader.NEW_SESSION_ID: "body.session_id",
                RESPONSE_CONTENT_KEY: "body.message",
            },
        )

    @pytest.mark.asyncio
    async def test_transforms_request_body(self, transform):
        """Test that request body is transformed using JMESPath."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"model": "llama-3"}
        mock_request.headers = {}

        result = await transform.transform_request(mock_request)

        assert result.request["model"] == "llama-3"

    @pytest.mark.asyncio
    async def test_updates_raw_request_body(self, transform):
        """Test that raw request body is updated with transformed data."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"model": "llama-3"}
        mock_request.headers = {}

        await transform.transform_request(mock_request)

        updated_body = json.loads(mock_request._body.decode())
        assert updated_body == {"model": "llama-3"}

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self, transform):
        """Test that invalid JSON raises HTTPException."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.side_effect = json.JSONDecodeError("Invalid", "doc", 0)

        with pytest.raises(HTTPException) as exc_info:
            await transform.transform_request(mock_request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value


class TestCreateSessionTransformResponse:
    """Test response transformation."""

    @pytest.fixture
    def transform(self):
        """Create transform."""
        return CreateSessionApiTransform(
            request_shape={},
            response_shape={
                SageMakerSessionHeader.NEW_SESSION_ID: "body.session_id",
                RESPONSE_CONTENT_KEY: "body.message",
            },
        )

    def test_extracts_session_id_from_response(self, transform):
        """Test that session ID is extracted and added to headers."""
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps({"session_id": "sess-123", "message": "created"}),
        )

        result = transform.transform_response(response, Mock())

        assert result.status_code == HTTPStatus.OK.value
        assert result.headers[SageMakerSessionHeader.NEW_SESSION_ID] == "sess-123"
        assert b"sess-123" in result.body
        assert b"created" in result.body

    def test_fails_when_session_id_missing(self, transform):
        """Test that missing session ID raises HTTPException."""
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps({"message": "created"}),
        )

        with pytest.raises(HTTPException) as exc_info:
            transform.transform_response(response, Mock())

        assert exc_info.value.status_code == HTTPStatus.BAD_GATEWAY.value
        assert "session ID" in exc_info.value.detail

    def test_fails_when_session_id_empty(self, transform):
        """Test that empty session ID raises HTTPException."""
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps({"session_id": "", "message": "created"}),
        )

        with pytest.raises(HTTPException) as exc_info:
            transform.transform_response(response, Mock())

        assert exc_info.value.status_code == HTTPStatus.BAD_GATEWAY.value

    def test_passes_through_error_responses(self, transform):
        """Test that error responses pass through unchanged."""
        response = Response(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content=b"Engine error",
        )

        result = transform.transform_response(response, Mock())

        assert result.status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value
        assert result.body == b"Engine error"


class TestCreateSessionNormalizeResponse:
    """normalization."""

    @pytest.fixture
    def transform(self):
        """Create transform."""
        return CreateSessionApiTransform(
            request_shape={},
            response_shape={
                SageMakerSessionHeader.NEW_SESSION_ID: "body.id",
                RESPONSE_CONTENT_KEY: "body.msg",
            },
        )

    def test_normalizes_dict_response(self, transform):
        """Test normalization of dict response."""
        response_dict = {"id": "sess-123", "msg": "created"}

        normalized = transform._normalize_response(response_dict)

        assert isinstance(normalized, Response)
        assert normalized.status_code == HTTPStatus.OK.value
        body = json.loads(normalized.body)
        assert body["id"] == "sess-123"

    def test_normalizes_string_response(self, transform):
        """Test normalizatistring response."""
        response_str = "Session created"

        normalized = transform._normalize_response(response_str)

        assert isinstance(normalized, Response)
        assert normalized.body == b"Session created"

    def test_normalizes_pydantic_response(self, transform):
        """Test normalization of Pydantic model response."""

        class SessionResponse(BaseModel):
            id: str
            msg: str

        response_model = SessionResponse(id="sess-123", msg="created")

        normalized = transform._normalize_response(response_model)

        assert isinstance(normalized, Response)
        body = json.loads(normalized.body)
        assert body["id"] == "sess-123"

    def test_passes_through_response_object(self, transform):
        """Test that Response objects pass through unchanged."""
        response = Response(status_code=HTTPStatus.OK.value, content=b"test")

        normalized = transform._normalize_response(response)

        assert normalized is response

    def test_normalizes_none_response(self, transform):
        """Test normalization of None response."""
        normalized = transform._normalize_response(None)

        assert isinstance(normalized, Response)
        assert normalized.status_code == HTTPStatus.OK.value
        assert normalized.body == b"null"

    def test_normalizes_list_response(self, transform):
        """Test normalization of list response."""
        response_list = [{"id": "sess-1"}, {"id": "sess-2"}]

        normalized = transform._normalize_response(response_list)

        assert isinstance(normalized, Response)
        assert normalized.status_code == HTTPStatus.OK.value
        body = json.loads(normalized.body)
        assert len(body) == 2
        assert body[0]["id"] == "sess-1"


class TestCreateSessionEdgeCases:
    """Test edge cases for CreateSessionApiTransform."""

    @pytest.fixture
    def transform(self):
        """Create transform."""
        return CreateSessionApiTransform(
            request_shape={},
            response_shape={
                SageMakerSessionHeader.NEW_SESSION_ID: "body.session_id",
                RESPONSE_CONTENT_KEY: "body.message",
            },
        )

    def test_fails_when_session_id_is_none(self, transform):
        """Test that None session ID raises HTTPException."""
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps({"session_id": None, "message": "created"}),
        )

        with pytest.raises(HTTPException) as exc_info:
            transform.transform_response(response, Mock())

        assert exc_info.value.status_code == HTTPStatus.BAD_GATEWAY.value

    def test_handles_none_content(self, transform):
        """Test that None content is handled gracefully."""
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps({"session_id": "sess-123", "message": None}),
        )

        result = transform.transform_response(response, Mock())

        assert result.status_code == HTTPStatus.OK.value
        assert result.headers[SageMakerSessionHeader.NEW_SESSION_ID] == "sess-123"

    def test_handles_empty_string_content(self, transform):
        """Test that empty string content is handled gracefully."""
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps({"session_id": "sess-123", "message": ""}),
        )

        result = transform.transform_response(response, Mock())

        assert result.status_code == HTTPStatus.OK.value
        assert result.headers[SageMakerSessionHeader.NEW_SESSION_ID] == "sess-123"

    def test_extracts_session_id_from_nested_path(self, transform):
        """Test extraction of session ID from nested response structure."""
        transform_nested = CreateSessionApiTransform(
            request_shape={},
            response_shape={
                SageMakerSessionHeader.NEW_SESSION_ID: "body.data.session.id",
                RESPONSE_CONTENT_KEY: "body.data.message",
            },
        )

        response = Response(
            status_code=HTTPStatus.OK.value,
            content=json.dumps(
                {"data": {"session": {"id": "sess-nested-123"}, "message": "created"}}
            ),
        )

        result = transform_nested.transform_response(response, Mock())

        assert result.status_code == HTTPStatus.OK.value
        assert (
            result.headers[SageMakerSessionHeader.NEW_SESSION_ID] == "sess-nested-123"
        )

    def test_handles_malformed_json_in_response(self, transform):
        """Test that malformed JSON in response is handled gracefully.

        The serialize_response function catches JSONDecodeError and keeps the body as a string,
        but since we can't extract a session_id from a string, this should fail validation.
        """
        response = Response(
            status_code=HTTPStatus.OK.value,
            content=b"not valid json {{{",
        )

        # Should fail because session_id cannot be extracted from malformed JSON
        with pytest.raises(HTTPException) as exc_info:
            transform.transform_response(response, Mock())

        assert exc_info.value.status_code == HTTPStatus.BAD_GATEWAY.value
        assert "session ID" in exc_info.value.detail
