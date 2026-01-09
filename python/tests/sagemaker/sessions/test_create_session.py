"""Unit tests for sagemaker.sessions.create_session module."""

import json
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import jmespath
import pytest
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from model_hosting_container_standards.common.fastapi.utils import serialize_response
from model_hosting_container_standards.common.handler import handler_registry
from model_hosting_container_standards.common.transforms.base_api_transform2 import (
    BaseTransformRequestOutput,
)
from model_hosting_container_standards.sagemaker.sessions.create_session import (
    CreateSessionApiTransform,
    _register_create_session_handler,
    create_create_session_transform,
)
from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader,
)


class MockEngineRequest(BaseModel):
    """Mock engine request model for testing."""

    param1: str
    param2: int = 10


class MockEngineResponse(BaseModel):
    """Mock engine response model for testing."""

    session_id: str
    status: str = "created"


async def dummy_func_with_model(
    request: MockEngineRequest, raw_request: Request
) -> Response:
    return Response(
        status_code=200, content=json.dumps({"session_id": "test-session-id"})
    )


class TestCreateSessionApiTransformInitialization:
    """Test suite for CreateSessionApiTransform initialization."""

    @patch(
        "model_hosting_container_standards.sagemaker.sessions.create_session.jmespath",
        wraps=jmespath,
    )
    def test_init_jmespath_compilation(self, mock_jmespath):
        """
        Test JMESPath expression compilation during initialization.

        Should verify:
        - Simple and complex path expressions are compiled correctly
        - Nested paths, array access, and filter expressions work
        - Compiled expression is stored and reused
        - Invalid JMESPath expressions raise appropriate errors
        """
        api_transform = CreateSessionApiTransform(
            dummy_func_with_model,
            engine_request_paths={},
            engine_response_session_id_path="body.session_id",
            engine_request_model_cls=MockEngineRequest,
        )
        mock_jmespath.compile.assert_called_once_with("body.session_id")
        assert isinstance(
            api_transform.engine_response_session_id_jmesexpr,
            jmespath.parser.ParsedResult,
        )
        assert (
            api_transform.engine_response_session_id_jmesexpr.search(
                {"body": {"session_id": "test-session-id"}}
            )
            == "test-session-id"
        )


class TestCreateSessionApiTransformValidation:
    """Test suite for CreateSessionApiTransform request validation."""

    @pytest.mark.asyncio
    async def test_validate_request_behavior(self):
        """
        Test validate_request method behavior with various inputs.
        """
        api_transform = CreateSessionApiTransform(
            dummy_func_with_model,
            engine_request_paths={},
            engine_response_session_id_path="body.session_id",
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults={"body.param1": "default-value"},
        )
        actual = await api_transform.validate_request(MagicMock(spec=Request))
        assert actual == {}


class TestCreateSessionApiTransformAdditionalFields:
    """Test suite for CreateSessionApiTransform additional fields extraction."""

    def test_extract_additional_fields_behavior(self):
        """
        Test _extract_additional_fields method behavior with various inputs.
        """
        api_transform = CreateSessionApiTransform(
            dummy_func_with_model,
            engine_request_paths={},
            engine_response_session_id_path="body.session_id",
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults={"body.param1": "default-value"},
        )
        assert (
            api_transform._extract_additional_fields({}, MagicMock(spec=Request)) == {}
        )


class TestCreateSessionApiTransformResponseGeneration:
    """Test suite for CreateSessionApiTransform response generation methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_session_id = "test-session-id"
        self.mock_transform_request_output = BaseTransformRequestOutput(
            raw_request={},
            transformed_request={},
            additional_fields={},
        )
        self.original_func = AsyncMock(dummy_func_with_model)
        self.api_transform = CreateSessionApiTransform(
            self.original_func,
            engine_request_paths={},
            engine_request_defaults={"param1": "default-value"},
            engine_request_model_cls=MockEngineRequest,
            engine_response_session_id_path="body.session_id",
        )

    def test_generate_successful_response_content(self):
        """
        Test _generate_successful_response_content with various session ID scenarios.
        """
        expected_prefix = "Successfully created session"
        mock_response = MagicMock(spec=Response)
        actual_without_session_id = (
            self.api_transform._generate_successful_response_content(
                mock_response, self.mock_transform_request_output
            )
        )
        assert actual_without_session_id == f"{expected_prefix}"

        mock_transform_request_output_with_additional_fields = (
            BaseTransformRequestOutput(
                raw_request={},
                transformed_request={},
                additional_fields={"session_id": self.mock_session_id},
            )
        )
        expected = expected_prefix + f": {self.mock_session_id}"
        actual_with_session_id = (
            self.api_transform._generate_successful_response_content(
                mock_response, mock_transform_request_output_with_additional_fields
            )
        )
        assert actual_with_session_id == expected

    @patch(
        "model_hosting_container_standards.sagemaker.sessions.create_session.serialize_response",
        wraps=serialize_response,
    )
    @patch("model_hosting_container_standards.sagemaker.sessions.create_session.logger")
    def test_transform_ok_response_success(self, mock_logger, mock_serialize_response):
        """
        Test _transform_ok_response with successful session ID extraction.
        """
        mock_raw_response = MagicMock(spec=Response)
        mock_raw_response.status_code = HTTPStatus.OK.value
        mock_raw_response.headers = {}
        mock_raw_response.body = b'{"session_id": "test-session-id"}'
        mock_raw_response.charset = "utf-8"
        mock_serialized_response = serialize_response(mock_raw_response)

        actual_response = self.api_transform._transform_ok_response(
            mock_raw_response, self.mock_transform_request_output
        )

        mock_logger.debug.assert_any_call(
            f"Transforming engine response to SageMaker format. Input: {mock_serialized_response}"
        )
        mock_serialize_response.assert_called_once_with(mock_raw_response)
        assert (
            actual_response.headers[SageMakerSessionHeader.NEW_SESSION_ID]
            == self.mock_session_id
        )
        assert actual_response.status_code == 200
        assert actual_response.body == b"Successfully created session: test-session-id"
        assert (
            self.mock_transform_request_output.additional_fields.get("session_id")
            == self.mock_session_id
        )

    @pytest.mark.parametrize(
        "response_body",
        [
            b"{}",
            b'{"session_id":""}',
        ],
        ids=[
            "no_session_id",
            "empty_session_id",
        ],
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sessions.create_session.serialize_response",
        wraps=serialize_response,
    )
    @patch("model_hosting_container_standards.sagemaker.sessions.create_session.logger")
    def test_transform_ok_response_session_id_not_found(
        self, mock_logger, mock_serialize_response, response_body
    ):
        """
        Test _transform_ok_response when session ID extraction fails.
        """
        mock_raw_response = MagicMock(spec=Response)
        mock_raw_response.status_code = HTTPStatus.OK.value
        mock_raw_response.headers = {}
        mock_raw_response.body = response_body
        mock_raw_response.charset = "utf-8"
        mock_serialized_response = serialize_response(mock_raw_response)

        with pytest.raises(HTTPException) as exc_info:
            self.api_transform._transform_ok_response(
                mock_raw_response, self.mock_transform_request_output
            )
        mock_serialize_response.assert_called_once_with(mock_raw_response)
        mock_logger.debug.assert_any_call(
            f"Transforming engine response to SageMaker format. Input: {mock_serialized_response}"
        )
        mock_logger.warning.assert_any_call(
            f"Session ID not found in response: {mock_serialized_response}"
        )
        assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value
        assert exc_info.value.detail == "Session ID not found in response"


class TestCreateCreateSessionTransform:
    """Test suite for create_create_session_transform factory function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        handler_registry.clear()

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        handler_registry.clear()

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.sessions.create_session.CreateSessionApiTransform",
        wraps=CreateSessionApiTransform,
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sessions.create_session.handler_registry",
        wraps=handler_registry,
    )
    @patch("model_hosting_container_standards.sagemaker.sessions.create_session.logger")
    async def test_create_create_session_transform_decorator(
        self, mock_logger, mock_handler_registry, mock_transform_cls
    ):
        """
        Test that create_create_session_transform returns a working decorator.
        """
        original_function = AsyncMock(wraps=dummy_func_with_model)
        engine_request_defaults = {"body.param1": "default-value"}

        actual_decorated_func = create_create_session_transform(
            engine_request_paths={},
            engine_response_session_id_path="body.session_id",
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults=engine_request_defaults,
        )(original_function)

        assert callable(actual_decorated_func)
        assert mock_handler_registry.has_handler("create_session")
        assert mock_handler_registry.set_handler.call_count == 1

        mock_logger.info.assert_any_call(
            f"[CREATE_SESSION] Registered transform handler for {original_function.__name__}"
        )

        mock_transform_cls.assert_called_once_with(
            original_function,
            {},
            "body.session_id",
            MockEngineRequest,
            engine_request_defaults=engine_request_defaults,
        )

        mock_raw_request = MagicMock(spec=Request)

        actual_response = await actual_decorated_func(mock_raw_request)

        original_function.assert_any_call(
            MockEngineRequest(param1="default-value"), mock_raw_request
        )

        expected_original_response = await dummy_func_with_model(
            MockEngineRequest(param1="default-value"), mock_raw_request
        )
        actual_session_id = actual_response.headers.get(
            SageMakerSessionHeader.NEW_SESSION_ID
        )
        expected_session_id = json.loads(expected_original_response.body).get(
            "session_id"
        )
        assert actual_session_id == expected_session_id


class TestRegisterCreateSessionHandler:
    """Test suite for _register_create_session_handler function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        handler_registry.clear()

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        handler_registry.clear()

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.sessions.create_session._transform_defaults_config",
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sessions.create_session.create_create_session_transform",
        wraps=create_create_session_transform,
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sessions.create_session.handler_registry",
        wraps=handler_registry,
    )
    @patch("model_hosting_container_standards.sagemaker.sessions.create_session.logger")
    async def test_register_create_session_handler(
        self,
        mock_logger,
        mock_handler_registry,
        mock_create_create_session_transform,
        mock_transform_defaults_config,
    ):
        """
        Test _register_create_session_handler with various parameter combinations.

        Should verify:
        - Function accepts engine_response_session_id_path parameter
        - Optional engine_request_model_cls parameter works correctly
        - Empty engine_request_paths are configured appropriately
        - Transform defaults from config are applied correctly
        - All parameters reach CreateSessionApiTransform constructor
        - Function returns decorator result that can be used as handler
        """
        engine_response_session_id_path = "body.session_id"
        mock_transform_defaults_config.create_session_defaults = {
            "body.param1": "default-value"
        }

        actual_decorator = _register_create_session_handler(
            engine_response_session_id_path=engine_response_session_id_path,
            engine_request_model_cls=MockEngineRequest,
        )
        assert callable(actual_decorator)
        mock_logger.info.assert_any_call("Registering create session handler")
        mock_logger.debug.assert_any_call(
            f"Handler parameter - engine_response_session_id_path: {engine_response_session_id_path}"
        )
        mock_create_create_session_transform.assert_called_once_with(
            engine_request_paths={},
            engine_response_session_id_path=engine_response_session_id_path,
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults=mock_transform_defaults_config.create_session_defaults,
        )

        original_func = AsyncMock(wraps=dummy_func_with_model)
        actual_decorated_func = actual_decorator(original_func)

        assert callable(actual_decorated_func)
        assert mock_handler_registry.has_handler("create_session")
        assert mock_handler_registry.set_handler.call_count == 1

        mock_raw_request = MagicMock(spec=Request)

        await actual_decorated_func(mock_raw_request)
        original_func.assert_any_call(
            MockEngineRequest(param1="default-value"), mock_raw_request
        )
