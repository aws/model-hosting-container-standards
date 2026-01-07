"""Unit tests for BaseEngineSessionApiTransform."""

import json
from http import HTTPStatus
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from model_hosting_container_standards.common import BaseTransformRequestOutput
from model_hosting_container_standards.sagemaker.sessions.transforms.base_engine_session_api_transform import (
    BaseEngineSessionApiTransform,
)


class ConcreteTransform(BaseEngineSessionApiTransform):
    """Concrete implementation for testing the abstract base class."""

    def _transform_ok_response(self, response: Response, **kwargs) -> Response:
        """Simple implementation that just returns the response."""
        return response


class TestBaseEngineSessionApiTransformRequest:
    """Test transform_request method."""

    @pytest.fixture
    def transform(self):
        """Create concrete transform instance."""
        return ConcreteTransform(
            request_shape={"field": "body.field"}, response_shape={}
        )

    @pytest.mark.asyncio
    async def test_transforms_request_body(self, transform):
        """Test that request body is transformed using JMESPath."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"field": "value"}
        mock_request.headers = {}

        result = await transform.transform_request(mock_request)

        assert result.request["field"] == "value"
        assert isinstance(result, BaseTransformRequestOutput)

    @pytest.mark.asyncio
    async def test_updates_raw_request_body(self, transform):
        """Test that raw request _body is updated with transformed data."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"field": "value"}
        mock_request.headers = {}

        await transform.transform_request(mock_request)

        updated_body = json.loads(mock_request._body.decode())
        assert updated_body == {"field": "value"}

    @pytest.mark.asyncio
    async def test_handles_json_decode_error(self, transform):
        """Test that JSON decode errors raise HTTPException."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.side_effect = json.JSONDecodeError("Invalid", "doc", 0)

        with pytest.raises(HTTPException) as exc_info:
            await transform.transform_request(mock_request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
        assert "JSON decode error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_calls_validate_request_preconditions(self, transform):
        """Test that _validate_request_preconditions is called."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {}
        mock_request.headers = {}

        # Mock the validation method
        transform._validate_request_preconditions = Mock()

        await transform.transform_request(mock_request)

        transform._validate_request_preconditions.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_validation_errors_propagate(self, transform):
        """Test that validation errors from preconditions propagate."""
        mock_request = AsyncMock(spec=Request)
        mock_request.headers = {}

        # Make validation raise an exception
        def raise_validation_error(req):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value, detail="Validation failed"
            )

        transform._validate_request_preconditions = raise_validation_error

        with pytest.raises(HTTPException) as exc_info:
            await transform.transform_request(mock_request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
        assert "Validation failed" in exc_info.value.detail


class TestBaseEngineSessionApiTransformResponse:
    """Test transform_response method."""

    @pytest.fixture
    def transform(self):
        """Create concrete transform instance."""
        return ConcreteTransform(request_shape={}, response_shape={})

    def test_routes_ok_response_to_transform_ok_response(self, transform):
        """Test that 200 OK responses are routed to _transform_ok_response."""
        response = Response(status_code=HTTPStatus.OK.value, content=b"success")
        transform_output = Mock(spec=BaseTransformRequestOutput)

        # Mock the _transform_ok_response method
        transform._transform_ok_response = Mock(return_value=response)

        result = transform.transform_response(response, transform_output)

        transform._transform_ok_response.assert_called_once()
        assert result == response

    def test_routes_error_response_to_transform_error_response(self, transform):
        """Test that error responses are routed to _transform_error_response."""
        response = Response(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, content=b"error"
        )
        transform_output = Mock(spec=BaseTransformRequestOutput)

        # Mock the _transform_error_response method
        transform._transform_error_response = Mock(return_value=response)

        result = transform.transform_response(response, transform_output)

        transform._transform_error_response.assert_called_once_with(response)
        assert result == response

    def test_normalizes_response_before_routing(self, transform):
        """Test that response is normalized before routing."""
        # Pass a dict instead of Response object
        response_dict = {"status": "success"}
        transform_output = Mock(spec=BaseTransformRequestOutput)

        result = transform.transform_response(response_dict, transform_output)

        # Should be normalized to Response object
        assert isinstance(result, Response)
        assert result.status_code == HTTPStatus.OK.value


class TestNormalizeResponse:
    """Test _normalize_response method."""

    @pytest.fixture
    def transform(self):
        """Create concrete transform instance."""
        return ConcreteTransform(request_shape={}, response_shape={})

    def test_passes_through_response_object(self, transform):
        """Test that Response objects pass through unchanged."""
        response = Response(status_code=HTTPStatus.OK.value, content=b"test")

        normalized = transform._normalize_response(response)

        assert normalized is response

    def test_normalizes_dict_to_response(self, transform):
        """Test that dict is normalized to Response."""
        response_dict = {"key": "value"}

        normalized = transform._normalize_response(response_dict)

        assert isinstance(normalized, Response)
        assert normalized.status_code == HTTPStatus.OK.value
        body = json.loads(normalized.body)
        assert body["key"] == "value"

    def test_normalizes_string_to_response(self, transform):
        """Test that string is normalized to Response."""
        response_str = "success message"

        normalized = transform._normalize_response(response_str)

        assert isinstance(normalized, Response)
        assert normalized.status_code == HTTPStatus.OK.value
        assert normalized.body == b"success message"

    def test_normalizes_pydantic_model_to_response(self, transform):
        """Test that Pydantic model is normalized to Response."""

        class TestModel(BaseModel):
            field1: str
            field2: int

        model = TestModel(field1="test", field2=42)

        normalized = transform._normalize_response(model)

        assert isinstance(normalized, Response)
        assert normalized.status_code == HTTPStatus.OK.value
        body = json.loads(normalized.body)
        assert body["field1"] == "test"
        assert body["field2"] == 42

    def test_normalizes_none_to_response(self, transform):
        """Test that None is normalized to Response."""
        normalized = transform._normalize_response(None)

        assert isinstance(normalized, Response)
        assert normalized.status_code == HTTPStatus.OK.value
        assert normalized.body == b"null"

    def test_normalizes_list_to_response(self, transform):
        """Test that list is normalized to Response."""
        response_list = [{"id": 1}, {"id": 2}]

        normalized = transform._normalize_response(response_list)

        assert isinstance(normalized, Response)
        assert normalized.status_code == HTTPStatus.OK.value
        body = json.loads(normalized.body)
        assert len(body) == 2
        assert body[0]["id"] == 1

    def test_normalizes_int_to_response(self, transform):
        """Test that integer is normalized to Response."""
        normalized = transform._normalize_response(42)

        assert isinstance(normalized, Response)
        assert normalized.status_code == HTTPStatus.OK.value
        assert normalized.body == b"42"

    def test_normalizes_bool_to_response(self, transform):
        """Test that boolean is normalized to Response."""
        normalized = transform._normalize_response(True)

        assert isinstance(normalized, Response)
        assert normalized.status_code == HTTPStatus.OK.value
        assert normalized.body == b"true"

    def test_normalizes_empty_dict_to_response(self, transform):
        """Test that empty dict is normalized to Response."""
        normalized = transform._normalize_response({})

        assert isinstance(normalized, Response)
        assert normalized.status_code == HTTPStatus.OK.value
        assert normalized.body == b"{}"

    def test_normalizes_nested_structure_to_response(self, transform):
        """Test that nested structure is normalized to Response."""
        response_data = {
            "session": {"id": "sess-123", "metadata": {"user": "test"}},
            "status": "active",
        }

        normalized = transform._normalize_response(response_data)

        assert isinstance(normalized, Response)
        body = json.loads(normalized.body)
        assert body["session"]["id"] == "sess-123"
        assert body["session"]["metadata"]["user"] == "test"

    def test_preserves_response_with_error_status_code(self, transform):
        """Test that Response with error status code is preserved."""
        response = Response(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, content=b"error"
        )

        normalized = transform._normalize_response(response)

        assert normalized is response
        assert normalized.status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value


class TestTransformErrorResponse:
    """Test _transform_error_response method."""

    @pytest.fixture
    def transform(self):
        """Create concrete transform instance."""
        return ConcreteTransform(request_shape={}, response_shape={})

    def test_passes_through_error_response_unchanged(self, transform):
        """Test that error responses pass through unchanged by default."""
        response = Response(
            status_code=HTTPStatus.NOT_FOUND.value, content=b"Not found"
        )

        result = transform._transform_error_response(response)

        assert result is response
        assert result.status_code == HTTPStatus.NOT_FOUND.value
        assert result.body == b"Not found"

    def test_handles_various_error_status_codes(self, transform):
        """Test that various error status codes are handled."""
        error_codes = [
            HTTPStatus.BAD_REQUEST.value,
            HTTPStatus.UNAUTHORIZED.value,
            HTTPStatus.FORBIDDEN.value,
            HTTPStatus.NOT_FOUND.value,
            HTTPStatus.INTERNAL_SERVER_ERROR.value,
            HTTPStatus.BAD_GATEWAY.value,
            HTTPStatus.SERVICE_UNAVAILABLE.value,
        ]

        for status_code in error_codes:
            response = Response(status_code=status_code, content=b"error")
            result = transform._transform_error_response(response)
            assert result.status_code == status_code


class TestValidateRequestPreconditions:
    """Test _validate_request_preconditions method."""

    @pytest.fixture
    def transform(self):
        """Create concrete transform instance."""
        return ConcreteTransform(request_shape={}, response_shape={})

    def test_default_implementation_does_nothing(self, transform):
        """Test that default implementation doesn't raise exceptions."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        # Should not raise any exception
        transform._validate_request_preconditions(mock_request)

    def test_can_be_overridden_in_subclass(self):
        """Test that subclasses can override validation."""

        class CustomTransform(BaseEngineSessionApiTransform):
            def _validate_request_preconditions(self, raw_request: Request) -> None:
                if not raw_request.headers.get("X-Custom-Header"):
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_REQUEST.value,
                        detail="Missing custom header",
                    )

            def _transform_ok_response(self, response: Response, **kwargs) -> Response:
                return response

        transform = CustomTransform(request_shape={}, response_shape={})
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        with pytest.raises(HTTPException) as exc_info:
            transform._validate_request_preconditions(mock_request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
        assert "Missing custom header" in exc_info.value.detail


class TestAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_transform_ok_response_must_be_implemented(self):
        """Test that _transform_ok_response must be implemented by subclasses."""

        # Try to instantiate without implementing abstract method
        with pytest.raises(TypeError) as exc_info:

            class IncompleteTransform(BaseEngineSessionApiTransform):
                pass

            IncompleteTransform(request_shape={}, response_shape={})

        assert "_transform_ok_response" in str(exc_info.value)


class TestTransformRequestOutputStructure:
    """Test the structure of transform_request output."""

    @pytest.fixture
    def transform(self):
        """Create concrete transform instance."""
        return ConcreteTransform(
            request_shape={"param": "body.param"}, response_shape={}
        )

    @pytest.mark.asyncio
    async def test_output_contains_transformed_request(self, transform):
        """Test that output contains the transformed request data."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"param": "value"}
        mock_request.headers = {}

        result = await transform.transform_request(mock_request)

        assert result.request == {"param": "value"}

    @pytest.mark.asyncio
    async def test_output_contains_raw_request(self, transform):
        """Test that output contains the raw request object."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"param": "value"}
        mock_request.headers = {}

        result = await transform.transform_request(mock_request)

        assert result.raw_request is mock_request

    @pytest.mark.asyncio
    async def test_output_intercept_func_is_none(self, transform):
        """Test that intercept_func is None for base transform."""
        mock_request = AsyncMock(spec=Request)
        mock_request.json.return_value = {"param": "value"}
        mock_request.headers = {}

        result = await transform.transform_request(mock_request)

        assert result.intercept_func is None
