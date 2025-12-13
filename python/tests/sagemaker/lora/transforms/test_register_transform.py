"""Unit tests for RegisterLoRAApiTransform."""

from http import HTTPStatus
from json import JSONDecodeError
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import ValidationError

from model_hosting_container_standards.sagemaker.lora.constants import ResponseMessage
from model_hosting_container_standards.sagemaker.lora.models.request import (
    SageMakerRegisterLoRAAdapterRequest,
)
from model_hosting_container_standards.sagemaker.lora.transforms.register import (
    RegisterLoRAApiTransform,
    validate_sagemaker_register_request,
)


class TestValidateSagemakerRegisterRequest:
    """Test validate_sagemaker_register_request function."""

    def test_valid_request_data(self):
        """Test validation with valid request data."""
        request_data = {
            "name": "test-adapter",
            "src": "s3://bucket/path/adapter",
            "preload": True,
        }

        result = validate_sagemaker_register_request(request_data)

        assert isinstance(result, SageMakerRegisterLoRAAdapterRequest)
        assert result.name == "test-adapter"
        assert result.src == "s3://bucket/path/adapter"
        assert result.preload is True

    def test_valid_request_with_minimal_fields(self):
        """Test validation with only required fields."""
        request_data = {"name": "minimal-adapter", "src": "s3://bucket/minimal"}

        result = validate_sagemaker_register_request(request_data)

        assert isinstance(result, SageMakerRegisterLoRAAdapterRequest)
        assert result.name == "minimal-adapter"
        assert result.src == "s3://bucket/minimal"
        assert result.preload is True  # Default value

    def test_missing_name_raises_http_exception(self):
        """Test that missing name raises HTTPException."""
        request_data = {"src": "s3://bucket/path"}

        with pytest.raises(HTTPException) as exc_info:
            validate_sagemaker_register_request(request_data)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    def test_missing_src_raises_http_exception(self):
        """Test that missing src raises HTTPException."""
        request_data = {"name": "test-adapter"}

        with pytest.raises(HTTPException) as exc_info:
            validate_sagemaker_register_request(request_data)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    def test_empty_name_raises_http_exception(self):
        """Test that empty name raises HTTPException."""
        request_data = {"name": "", "src": "s3://bucket/path"}

        with pytest.raises(HTTPException) as exc_info:
            validate_sagemaker_register_request(request_data)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    def test_empty_src_raises_http_exception(self):
        """Test that empty src raises HTTPException."""
        request_data = {"name": "test-adapter", "src": ""}

        with pytest.raises(HTTPException) as exc_info:
            validate_sagemaker_register_request(request_data)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    def test_none_name_raises_http_exception(self):
        """Test that None name raises HTTPException."""
        request_data = {"name": None, "src": "s3://bucket/path"}

        with pytest.raises(HTTPException) as exc_info:
            validate_sagemaker_register_request(request_data)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    def test_none_src_raises_http_exception(self):
        """Test that None src raises HTTPException."""
        request_data = {"name": "test-adapter", "src": None}

        with pytest.raises(HTTPException) as exc_info:
            validate_sagemaker_register_request(request_data)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    @patch(
        "model_hosting_container_standards.sagemaker.lora.transforms.register.SageMakerRegisterLoRAAdapterRequest.model_validate"
    )
    def test_validation_error_raises_http_exception(self, mock_validate):
        """Test that ValidationError raises HTTPException."""
        request_data = {"name": "test-adapter", "src": "s3://bucket/path"}

        # Mock ValidationError by making model_validate raise it
        mock_validate.side_effect = ValidationError("test error", [])

        with pytest.raises(HTTPException) as exc_info:
            validate_sagemaker_register_request(request_data)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    def test_extra_fields_ignored(self):
        """Test that extra fields are handled according to model configuration."""
        request_data = {
            "name": "test-adapter",
            "src": "s3://bucket/path",
            "preload": False,
            "extra_field": "should_be_ignored",
        }

        result = validate_sagemaker_register_request(request_data)

        assert isinstance(result, SageMakerRegisterLoRAAdapterRequest)
        assert result.name == "test-adapter"
        assert result.src == "s3://bucket/path"
        assert result.preload is False
        # Extra field should not be present
        assert not hasattr(result, "extra_field")


class TestRegisterLoRAApiTransform:
    """Test RegisterLoRAApiTransform class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.request_shape = {"model": "body.name", "source": "body.src"}
        self.response_shape = {}
        self.transformer = RegisterLoRAApiTransform(
            self.request_shape, self.response_shape
        )

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.lora.utils.get_adapter_alias_from_request_header"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.lora.transforms.register.RegisterLoRAApiTransform._transform_request"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.lora.transforms.register.validate_sagemaker_register_request"
    )
    async def test_transform_request_success(
        self, mock_validate, mock_transform, mock_get_alias
    ):
        """Test successful request transformation."""
        # Setup mocks for the request data and validation
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json = AsyncMock(
            return_value={"name": "test-adapter", "src": "s3://test"}
        )
        mock_get_alias.return_value = "test-alias"

        mock_request = SageMakerRegisterLoRAAdapterRequest(
            name="test-adapter", src="s3://test"
        )
        mock_validate.return_value = mock_request

        mock_transform.return_value = {"model": "test-adapter", "source": "s3://test"}

        # Call method - only pass raw_request
        result = await self.transformer.transform_request(mock_raw_request)

        # Verify calls
        mock_raw_request.json.assert_called_once()
        mock_validate.assert_called_once_with(
            {"name": "test-adapter", "src": "s3://test"}
        )
        mock_transform.assert_called_once_with(mock_request, mock_raw_request)

        # Verify result
        assert result.request == {"model": "test-adapter", "source": "s3://test"}
        assert result.raw_request == mock_raw_request
        assert result.adapter_name == "test-adapter"

    def test_transform_ok_response(self):
        """Test successful response transformation."""
        mock_response = Mock(spec=Response)
        # Fix the mock response to have proper headers that can be converted to dict
        mock_response.headers = {}
        mock_response.media_type = "application/json"
        adapter_name = "test-adapter"

        result = self.transformer._transform_ok_response(
            mock_response, adapter_name=adapter_name
        )

        assert isinstance(result, Response)
        assert result.status_code == HTTPStatus.OK
        # FastAPI Response uses body (as bytes), not content
        expected_message = ResponseMessage.ADAPTER_REGISTERED.format(alias=adapter_name)
        assert result.body.decode() == expected_message
        assert expected_message == "Adapter test-adapter registered"

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.lora.utils.get_adapter_alias_from_request_header"
    )
    async def test_integration_transform_request_and_response_json_body(
        self, mock_get_alias
    ):
        """Test integration between request and response transformation."""
        # Setup request transformation with mocked raw request
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json = AsyncMock(
            return_value={"name": "integration-test", "src": "s3://integration"}
        )
        mock_get_alias.return_value = "integration-alias"

        # Transform request - only pass raw_request
        transform_output = await self.transformer.transform_request(mock_raw_request)

        # Verify request transformation
        assert transform_output.adapter_name == "integration-test"

        # Transform successful response
        mock_ok_response = Mock(spec=Response)
        mock_ok_response.status_code = HTTPStatus.OK
        mock_ok_response.headers = {}
        mock_ok_response.media_type = "application/json"

        ok_result = self.transformer.transform_response(
            mock_ok_response, transform_output
        )

        assert ok_result.status_code == HTTPStatus.OK
        assert "registered" in ok_result.body.decode()

        # TODO: test error transformation once implemented

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.lora.utils.get_adapter_alias_from_request_header"
    )
    async def test_integration_transform_request_and_response_query_params(
        self, mock_get_alias
    ):
        """Test integration between request and response transformation."""
        # Setup request transformation with mocked raw request
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json.side_effect = JSONDecodeError("test error", doc="", pos=1)
        mock_raw_request.query_params = {
            "name": "integration-test",
            "src": "s3://integration",
        }
        mock_get_alias.return_value = "integration-alias"

        # Transform request - only pass raw_request
        transform_output = await self.transformer.transform_request(mock_raw_request)

        # Verify request transformation
        assert transform_output.adapter_name == "integration-test"

        # Transform successful response
        mock_ok_response = Mock(spec=Response)
        mock_ok_response.status_code = HTTPStatus.OK
        mock_ok_response.headers = {}
        mock_ok_response.media_type = "application/json"

        ok_result = self.transformer.transform_response(
            mock_ok_response, transform_output
        )

        assert ok_result.status_code == HTTPStatus.OK
        assert "registered" in ok_result.body.decode()

        # TODO: test error transformation once implemented
