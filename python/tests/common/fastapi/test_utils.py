"""Test scaffolding for common.fastapi.utils module."""

import json
from unittest.mock import MagicMock, patch

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from model_hosting_container_standards.common.fastapi.utils import (
    serialize_request,
    serialize_response,
)


class MockModel(BaseModel):
    """Mock Pydantic model for testing."""

    param1: str
    param2: int = 1


class TestSerializeRequest:
    """Test cases for serialize_request function."""

    def test_serialize_request_with_pydantic_model(self):
        """
        Test serialization when request is a Pydantic BaseModel.
        """
        mock_model = MockModel(param1="value")
        mock_raw_request = MagicMock(spec=Request)
        mock_raw_request.headers = {"header1": "value1"}
        mock_raw_request.query_params = {"query1": "value1"}
        mock_raw_request.path_params = {"path1": "value1"}

        actual = serialize_request(mock_model, mock_raw_request)

        expected = {
            "body": mock_model.model_dump(),
            "headers": mock_raw_request.headers,
            "query_params": mock_raw_request.query_params,
            "path_params": mock_raw_request.path_params,
        }

        assert actual == expected

    def test_serialize_request_with_dict(self):
        """Test serialization when request is a dictionary."""
        mock_request = {"param1": "value1", "param2": 2}
        mock_raw_request = MagicMock(spec=Request)
        mock_raw_request.headers = {"header1": "value1"}
        mock_raw_request.query_params = {"query1": "value1"}
        mock_raw_request.path_params = {"path1": "value1"}

        actual = serialize_request(mock_request, mock_raw_request)

        expected = {
            "body": mock_request,
            "headers": mock_raw_request.headers,
            "query_params": mock_raw_request.query_params,
            "path_params": mock_raw_request.path_params,
        }
        assert actual == expected

    def test_serialize_request_with_none(self):
        """Test serialization when request is None."""

        mock_request = None
        mock_raw_request = MagicMock(spec=Request)
        mock_raw_request.headers = {"header1": "value1"}
        mock_raw_request.query_params = {"query1": "value1"}
        mock_raw_request.path_params = {"path1": "value1"}

        actual = serialize_request(mock_request, mock_raw_request)

        expected = {
            "body": mock_request,
            "headers": mock_raw_request.headers,
            "query_params": mock_raw_request.query_params,
            "path_params": mock_raw_request.path_params,
        }
        assert actual == expected

    def test_serialize_request_with_invalid_type(self):
        """Test serialization when request is an unsupported type.

        Should:
        - Set body to None for unsupported types (str, int, list, etc.)
        - Still include other request metadata
        - Not raise exceptions for unexpected input types
        """
        # TODO: Implement test
        pass


class TestSerializeResponse:
    """Test cases for serialize_response function."""

    def test_serialize_response_with_json_response(self):
        """Test serialization of JSONResponse objects.

        Should:
        - Parse JSON body correctly
        - Include headers and status_code
        - Handle various JSON data types (dict, list, primitives)
        """
        mock_json_response = MagicMock(spec=JSONResponse)
        mock_json_response.body = json.dumps({"key": "value"}).encode("utf-8")
        mock_json_response.headers = {"header1": "value1"}
        mock_json_response.status_code = 200
        mock_json_response.charset = "utf-8"

        actual = serialize_response(mock_json_response)

        expected = {
            "body": {"key": "value"},
            "headers": mock_json_response.headers,
            "status_code": mock_json_response.status_code,
        }
        assert actual == expected

    def test_serialize_response_with_regular_response(self):
        """
        Test serialization of regular Response objects.
        """
        mock_response = MagicMock(spec=Response)
        mock_response.body = json.dumps({"key": "value"}).encode("utf-8")
        mock_response.headers = {"header1": "value1"}
        mock_response.status_code = 200
        mock_response.charset = "utf-8"

        actual = serialize_response(mock_response)

        expected = {
            "body": {"key": "value"},
            "headers": mock_response.headers,
            "status_code": mock_response.status_code,
        }
        assert actual == expected

    @patch("model_hosting_container_standards.common.fastapi.utils.json.loads")
    def test_serialize_response_with_non_json_body(self, mock_json_loads):
        """Test serialization when response body is not valid JSON."""

        mock_json_loads.side_effect = json.JSONDecodeError("", "", 0)
        mock_response = MagicMock(spec=Response)
        mock_response.charset = "utf-8"
        mock_response.body = b"not json"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.status_code = 400

        actual = serialize_response(mock_response)

        expected = {
            "body": "not json",
            "headers": mock_response.headers,
            "status_code": mock_response.status_code,
        }
        assert actual == expected

    def test_serialize_response_with_empty_body(self):
        """Test serialization when response has empty body."""
        mock_response = MagicMock(spec=Response)
        mock_response.body = b""
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.status_code = 200

        actual = serialize_response(mock_response)

        expected = {
            "body": {},
            "headers": mock_response.headers,
            "status_code": mock_response.status_code,
        }
        assert actual == expected
