"""Unit tests for UnregisterLoRAApiTransform."""

from http import HTTPStatus
from unittest.mock import Mock, patch

import pytest
from fastapi import Request, Response
from fastapi.exceptions import HTTPException

from model_hosting_container_standards.sagemaker.lora.constants import (
    RequestField,
    ResponseMessage,
)
from model_hosting_container_standards.sagemaker.lora.models.transform import (
    BaseLoRATransformRequestOutput,
)
from model_hosting_container_standards.sagemaker.lora.transforms.unregister import (
    UnregisterLoRAApiTransform,
    validate_sagemaker_unregister_request,
)


class TestValidateSagemakerUnregisterRequest:
    """Test validate_sagemaker_unregister_request function."""

    def test_valid_request_with_adapter_name(self):
        """Test validation with valid request containing adapter_name in path."""
        mock_request = Mock(spec=Request)
        mock_request.path_params = {"adapter_name": "test-adapter"}

        result = validate_sagemaker_unregister_request(mock_request)

        assert result == "test-adapter"

    def test_missing_adapter_name_raises_http_exception(self):
        """Test that missing adapter_name in path params raises HTTPException."""
        mock_request = Mock(spec=Request)
        mock_request.path_params = {"other_param": "value"}

        with pytest.raises(HTTPException) as exc_info:
            validate_sagemaker_unregister_request(mock_request)

        assert exc_info.value.status_code == HTTPStatus.FAILED_DEPENDENCY.value
        expected_message = f"Malformed request path; missing path parameter: {RequestField.ADAPTER_NAME}"
        assert expected_message in str(exc_info.value.detail)

    def test_empty_adapter_name_raises_http_exception(self):
        """Test that empty adapter_name in path params raises HTTPException."""
        mock_request = Mock(spec=Request)
        mock_request.path_params = {"adapter_name": ""}

        with patch(
            "model_hosting_container_standards.sagemaker.lora.transforms.unregister.get_adapter_name_from_request_path"
        ) as mock_get_adapter_name:
            mock_get_adapter_name.return_value = ""

            with pytest.raises(HTTPException) as exc_info:
                validate_sagemaker_unregister_request(mock_request)

            assert exc_info.value.status_code == HTTPStatus.FAILED_DEPENDENCY.value
            expected_message = f"Malformed request path; missing path parameter: {RequestField.ADAPTER_NAME}"
            assert expected_message in str(exc_info.value.detail)


class TestUnregisterLoRAApiTransform:
    """Test UnregisterLoRAApiTransform class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.request_shape = {
            "adapter_name": "path_params.adapter_name",
        }
        self.response_shape = {}
        self.transformer = UnregisterLoRAApiTransform(
            self.request_shape, self.response_shape
        )

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.lora.transforms.unregister.UnregisterLoRAApiTransform._transform_request"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.lora.transforms.unregister.validate_sagemaker_unregister_request"
    )
    async def test_transform_request_success(self, mock_validate, mock_transform):
        """Test successful request transformation with validation."""
        # Setup mock request (unregister doesn't use request body)
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.path_params = {"adapter_name": "test-adapter"}

        # Mock validation and transformation results
        mock_validate.return_value = "test-adapter"
        transformed_data = {
            "adapter_name": "test-adapter",
        }
        mock_transform.return_value = transformed_data

        # Call method
        result = await self.transformer.transform_request(mock_raw_request)

        # Verify validation was called
        mock_validate.assert_called_once_with(mock_raw_request)
        # Verify transformation was called - should pass None as request (no body for unregister)
        mock_transform.assert_called_once_with(None, mock_raw_request)

        # Verify result
        assert isinstance(result, BaseLoRATransformRequestOutput)
        assert result.request == transformed_data
        assert result.raw_request == mock_raw_request
        assert result.adapter_name == "test-adapter"

    @pytest.mark.asyncio
    async def test_transform_request_validation_failure(self):
        """Test request transformation when validation fails."""
        # Setup mock request where get_adapter_name_from_request_path returns None
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.path_params = {}

        with patch(
            "model_hosting_container_standards.sagemaker.lora.transforms.unregister.get_adapter_name_from_request_path"
        ) as mock_get_adapter_name:
            mock_get_adapter_name.return_value = None

            # Call method and expect HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await self.transformer.transform_request(mock_raw_request)

            assert exc_info.value.status_code == HTTPStatus.FAILED_DEPENDENCY.value

    def test_transform_ok_response(self):
        """Test successful response transformation."""
        mock_response = Mock(spec=Response)
        adapter_name = "test-adapter-to-unregister"

        result = self.transformer._transform_ok_response(
            mock_response, adapter_name=adapter_name
        )

        # Verify response properties
        assert isinstance(result, Response)
        assert result.status_code == HTTPStatus.OK

        # Verify response content contains unregistered message
        expected_message = ResponseMessage.ADAPTER_UNREGISTERED.format(
            alias=adapter_name
        )
        assert result.body.decode() == expected_message
        assert expected_message == "Adapter test-adapter-to-unregister unregistered"

    @pytest.mark.asyncio
    async def test_integration_request_and_response_success(self):
        """Test integration between request and response transformation with real validation."""
        # Setup request transformation with real path params
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.path_params = {"adapter_name": "integration-adapter"}

        # Transform request (this will use real validation)
        transform_output = await self.transformer.transform_request(mock_raw_request)

        # Verify request transformation
        assert transform_output.request == {"adapter_name": "integration-adapter"}
        assert transform_output.adapter_name == "integration-adapter"

        # Transform successful response
        mock_ok_response = Mock(spec=Response)
        mock_ok_response.status_code = HTTPStatus.OK

        ok_result = self.transformer.transform_response(
            mock_ok_response, transform_output
        )

        assert ok_result.status_code == HTTPStatus.OK
        assert "unregistered" in ok_result.body.decode()

    @pytest.mark.asyncio
    async def test_integration_request_validation_failure(self):
        """Test integration when request validation fails."""
        # Setup request where adapter name cannot be extracted from path
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.path_params = {}

        with patch(
            "model_hosting_container_standards.sagemaker.lora.transforms.unregister.get_adapter_name_from_request_path"
        ) as mock_get_adapter_name:
            mock_get_adapter_name.return_value = None

            # Transform request should fail with HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await self.transformer.transform_request(mock_raw_request)

            assert exc_info.value.status_code == HTTPStatus.FAILED_DEPENDENCY.value
            expected_message = f"Malformed request path; missing path parameter: {RequestField.ADAPTER_NAME}"
            assert expected_message in str(exc_info.value.detail)
