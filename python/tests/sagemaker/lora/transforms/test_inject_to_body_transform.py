"""Unit tests for InjectToBodyApiTransform."""

import json
from http import HTTPStatus
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Request, Response

from model_hosting_container_standards.sagemaker.lora.models.transform import (
    BaseLoRATransformRequestOutput,
)
from model_hosting_container_standards.sagemaker.lora.transforms.inject_to_body import (
    InjectToBodyApiTransform,
)


class TestInjectToBodyApiTransform:
    """Test InjectToBodyApiTransform class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.request_shape = {
            "adapter_id": 'headers."x-adapter-id"',
            "model_name": 'headers."x-model-name"',
            "config.priority": 'headers."x-priority"',
            "config.enabled": 'headers."x-enabled"',
        }
        self.response_shape = {}
        self.transformer = InjectToBodyApiTransform(
            self.request_shape, self.response_shape
        )

    @pytest.mark.asyncio
    async def test_transform_request_success(self):
        """Test successful request transformation with header data added to body."""
        # Setup mock request with headers
        original_body = {"existing_field": "existing_value", "config": {}}
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json = AsyncMock(return_value=original_body)
        mock_raw_request.headers = {
            "x-adapter-id": "test-adapter-123",
            "x-model-name": "llama-2-7b",
            "x-priority": "high",
            "x-enabled": "true",
        }

        # Call method
        result = await self.transformer.transform_request(mock_raw_request)

        # Verify calls
        mock_raw_request.json.assert_called_once()

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
        actual_body = json.loads(mock_raw_request._body.decode("utf-8"))
        assert actual_body == expected_merged_body

    @pytest.mark.asyncio
    async def test_transform_request_empty_header_data(self):
        """Test transformation with no header data to add."""
        # Setup mock request with no headers (or missing expected headers)
        original_body = {"existing_field": "existing_value", "config": {}}
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json = AsyncMock(return_value=original_body)
        mock_raw_request.headers = {}  # No headers

        # Call method
        result = await self.transformer.transform_request(mock_raw_request)

        # Verify result
        assert isinstance(result, BaseLoRATransformRequestOutput)
        assert result.request is None
        assert result.raw_request == mock_raw_request

        # Verify body has None values for missing headers
        expected_body = {
            "existing_field": "existing_value",
            "adapter_id": None,
            "model_name": None,
            "config": {"priority": None, "enabled": None},
        }
        actual_body = json.loads(mock_raw_request._body.decode("utf-8"))
        assert actual_body == expected_body

    @pytest.mark.asyncio
    async def test_transform_request_empty_original_body(self):
        """Test transformation with empty original request body."""
        # Setup mock request with minimal body structure (needs parent path for nested keys)
        original_body = {"config": {}}
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json = AsyncMock(return_value=original_body)
        mock_raw_request.headers = {
            "x-adapter-id": "test-adapter",
            "x-model-name": "test-model",
            "x-priority": "medium",
            "x-enabled": "true",
        }

        # Call method
        await self.transformer.transform_request(mock_raw_request)

        # Verify final body includes the header data
        expected_body = {
            "adapter_id": "test-adapter",
            "model_name": "test-model",
            "config": {"priority": "medium", "enabled": "true"},
        }
        actual_body = json.loads(mock_raw_request._body.decode("utf-8"))
        assert actual_body == expected_body

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
        transformer = InjectToBodyApiTransform(simple_request_shape)

        # Setup mock request with headers
        original_body = {"model": "test-model"}
        mock_raw_request = Mock(spec=Request)
        mock_raw_request.json = AsyncMock(return_value=original_body)
        mock_raw_request.headers = {
            "x-adapter-name": "production-adapter",
            "x-priority": "high",
            "x-other-header": "ignored",
        }

        # Call method
        await transformer.transform_request(mock_raw_request)

        # Verify final body includes both original and header data
        expected_merged_body = {
            "model": "test-model",
            "adapter_name": "production-adapter",
            "priority": "high",
        }
        actual_body = json.loads(mock_raw_request._body.decode("utf-8"))
        assert actual_body == expected_merged_body
