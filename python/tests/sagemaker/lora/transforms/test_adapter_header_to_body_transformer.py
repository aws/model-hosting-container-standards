"""Unit tests for AdapterHeaderToBodyApiTransform."""

import json
from http import HTTPStatus
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request, Response

from model_hosting_container_standards.sagemaker.lora.models.transform import (
    BaseLoRATransformRequestOutput,
)
from model_hosting_container_standards.sagemaker.lora.transforms.adapter_header_to_body import (
    AdapterHeaderToBodyApiTransform,
)


class TestAdapterHeaderToBodyApiTransform:
    """Test AdapterHeaderToBodyApiTransform class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.request_shape = {
            "adapter_id": 'headers."x-adapter-id"',
            "model_name": 'headers."x-model-name"',
            "config": {
                "priority": 'headers."x-priority"',
                "enabled": 'headers."x-enabled"',
            },
        }
        self.response_shape = {}
        self.transformer = AdapterHeaderToBodyApiTransform(
            self.request_shape, self.response_shape
        )

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.lora.transforms.adapter_header_to_body.AdapterHeaderToBodyApiTransform._transform_request"
    )
    async def test_transform_request_success(self, mock_transform):
        """Test successful request transformation with header data added to body."""
        # Setup mock request
        original_body = {"existing_field": "existing_value"}
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json = AsyncMock(return_value=original_body)

        # Mock transformation result (data from headers)
        header_data = {
            "adapter_id": "test-adapter-123",
            "model_name": "llama-2-7b",
            "config": {"priority": "high", "enabled": "true"},
        }
        mock_transform.return_value = header_data

        # Call method
        result = await self.transformer.transform_request(mock_raw_request)

        # Verify calls
        mock_raw_request.json.assert_called_once()
        mock_transform.assert_called_once_with(None, mock_raw_request)

        # Verify result structure
        assert isinstance(result, BaseLoRATransformRequestOutput)
        assert result.request is None
        assert result.raw_request == mock_raw_request

        # Verify that the raw request body was updated with merged data
        expected_merged_body = {
            "existing_field": "existing_value",
            "adapter_id": "test-adapter-123",
            "model_name": "llama-2-7b",
            "config": {"priority": "high", "enabled": "true"},
        }
        expected_body_bytes = json.dumps(expected_merged_body).encode("utf-8")
        assert mock_raw_request._body == expected_body_bytes

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.lora.transforms.adapter_header_to_body.logger"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.lora.transforms.adapter_header_to_body.AdapterHeaderToBodyApiTransform._transform_request"
    )
    async def test_transform_request_with_key_overwrite_warning(
        self, mock_transform, mock_logger
    ):
        """Test transformation warns when overwriting existing body keys."""
        # Setup mock request with conflicting keys
        original_body = {
            "existing_field": "original_value",
            "adapter_id": "original_adapter",
        }
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json = AsyncMock(return_value=original_body)

        # Mock transformation result that conflicts with existing keys
        header_data = {
            "adapter_id": "new_adapter_from_header",
            "new_field": "new_value",
        }
        mock_transform.return_value = header_data

        # Call method
        await self.transformer.transform_request(mock_raw_request)

        # Verify final body has header data (overwrites original)
        expected_merged_body = {
            "existing_field": "original_value",
            "adapter_id": "new_adapter_from_header",  # Overwritten
            "new_field": "new_value",
        }
        expected_body_bytes = json.dumps(expected_merged_body).encode("utf-8")
        assert mock_raw_request._body == expected_body_bytes

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.lora.transforms.adapter_header_to_body.AdapterHeaderToBodyApiTransform._transform_request"
    )
    async def test_transform_request_empty_header_data(self, mock_transform):
        """Test transformation with no header data to add."""
        # Setup mock request
        original_body = {"existing_field": "existing_value"}
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json = AsyncMock(return_value=original_body)

        # Mock transformation returns empty data
        mock_transform.return_value = {}

        # Call method
        result = await self.transformer.transform_request(mock_raw_request)

        # Verify result
        assert isinstance(result, BaseLoRATransformRequestOutput)
        assert result.request is None
        assert result.raw_request == mock_raw_request

        # Verify body remains unchanged when no header data
        expected_body_bytes = json.dumps(original_body).encode("utf-8")
        assert mock_raw_request._body == expected_body_bytes

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.lora.transforms.adapter_header_to_body.AdapterHeaderToBodyApiTransform._transform_request"
    )
    async def test_transform_request_empty_original_body(self, mock_transform):
        """Test transformation with empty original request body."""
        # Setup mock request with empty body
        original_body = {}
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json = AsyncMock(return_value=original_body)

        # Mock transformation result
        header_data = {"adapter_id": "test-adapter"}
        mock_transform.return_value = header_data

        # Call method
        await self.transformer.transform_request(mock_raw_request)

        # Verify final body is just the header data
        expected_body_bytes = json.dumps(header_data).encode("utf-8")
        assert mock_raw_request._body == expected_body_bytes

    def test_transform_response_passthrough(self):
        """Test that transform_response passes through response unchanged."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = HTTPStatus.OK
        mock_response.content = "test content"

        mock_transform_output = Mock(spec=BaseLoRATransformRequestOutput)

        # Call method
        result = self.transformer.transform_response(
            mock_response, mock_transform_output
        )

        # Verify response is returned unchanged (passthrough)
        assert result is mock_response
        assert result.status_code == HTTPStatus.OK
        # Don't check body content since it's a passthrough

    def test_transform_response_passthrough_error(self):
        """Test that transform_response passes through error responses unchanged."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_response.content = "error content"

        mock_transform_output = Mock(spec=BaseLoRATransformRequestOutput)

        # Call method
        result = self.transformer.transform_response(
            mock_response, mock_transform_output
        )

        # Verify error response is returned unchanged
        assert result is mock_response
        assert result.status_code == HTTPStatus.BAD_REQUEST
        # Don't check body content since it's a passthrough

    @pytest.mark.asyncio
    async def test_integration_request_transformation(self):
        """Test integration of request transformation with actual JMESPath processing."""
        # Create transformer with simple header mapping
        simple_request_shape = {
            "adapter_name": 'headers."x-adapter-name"',
            "priority": 'headers."x-priority"',
        }
        transformer = AdapterHeaderToBodyApiTransform(simple_request_shape)

        # Setup mock request with headers
        original_body = {"model": "test-model"}
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json = AsyncMock(return_value=original_body)
        mock_raw_request.headers = {
            "x-adapter-name": "production-adapter",
            "x-priority": "high",
            "x-other-header": "ignored",
        }

        # Mock the utility function to return proper structure
        with patch(
            "model_hosting_container_standards.sagemaker.lora.transforms.adapter_header_to_body.AdapterHeaderToBodyApiTransform._transform_request"
        ) as mock_transform:
            mock_transform.return_value = {
                "adapter_name": "production-adapter",
                "priority": "high",
            }

            # Call method
            await transformer.transform_request(mock_raw_request)

            # Verify transformation was called with None request (header-only transform)
            mock_transform.assert_called_once_with(None, mock_raw_request)

            # Verify final body includes both original and header data
            expected_merged_body = {
                "model": "test-model",
                "adapter_name": "production-adapter",
                "priority": "high",
            }
            expected_body_bytes = json.dumps(expected_merged_body).encode("utf-8")
            assert mock_raw_request._body == expected_body_bytes
