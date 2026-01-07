"""Unit tests for sagemaker.sessions.close_session module."""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError
from starlette.datastructures import MutableHeaders

from model_hosting_container_standards.common.handler import handler_registry
from model_hosting_container_standards.common.transforms.base_api_transform2 import (
    BaseTransformRequestOutput,
)
from model_hosting_container_standards.sagemaker.sessions.close_session import (
    SAGEMAKER_HEADER_PREFIX,
    CloseSessionApiTransform,
    SageMakerSessionRequestHeader,
    _register_close_session_handler,
    create_close_session_transform,
    to_hyphens,
    to_sagemaker_headers,
)
from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader,
)


class MockEngineRequest(BaseModel):
    """Mock engine request model for testing."""

    session_id: str
    additional_param: str = "default"


class TestUtilityFunctions:
    """Test suite for utility functions used in close_session module."""

    def test_to_hyphens_basic_conversion(self):
        """
        Test to_hyphens function with basic underscore to hyphen conversion.
        """
        assert to_hyphens("_") == "-"
        assert to_hyphens("___") == "---"
        assert to_hyphens("a_b_c") == "a-b-c"
        assert to_hyphens("abc") == "abc"
        assert to_hyphens("") == ""

    @patch(
        "model_hosting_container_standards.sagemaker.sessions.close_session.to_hyphens",
        wraps=to_hyphens,
    )
    def test_to_sagemaker_headers_basic_conversion(self, mock_to_hyphens):
        """
        Test to_sagemaker_headers function with basic field name conversion.
        """
        assert (
            to_sagemaker_headers("session_id") == f"{SAGEMAKER_HEADER_PREFIX}Session-Id"
        )
        mock_to_hyphens.assert_any_call("session_id")


class TestSageMakerSessionRequestHeader:
    """Test suite for SageMakerSessionRequestHeader Pydantic model."""

    def test_model_creation_with_valid_session_id(self):
        """
        Test creating SageMakerSessionRequestHeader with valid session ID.
        """
        test_session_id = "test-session-123"
        model = SageMakerSessionRequestHeader(session_id=test_session_id)

        assert model.session_id == test_session_id
        assert isinstance(model, SageMakerSessionRequestHeader)

    def test_model_validation_with_sagemaker_header_format(self):
        """
        Test model validation using SageMaker header format.
        """
        test_session_id = "sagemaker-session-456"
        headers = {"X-Amzn-SageMaker-Session-Id": test_session_id}

        model = SageMakerSessionRequestHeader.model_validate(headers)
        assert model.session_id == test_session_id

    def test_model_validation_with_field_name_format(self):
        """
        Test model validation using Python field name format.
        """
        test_session_id = "field-name-session-789"
        data = {"session_id": test_session_id}

        model = SageMakerSessionRequestHeader.model_validate(data)
        assert model.session_id == test_session_id

    def test_model_validation_missing_session_id(self):
        """
        Test model validation when session ID is missing.
        """
        empty_data = {}

        with pytest.raises(ValidationError) as exc_info:
            SageMakerSessionRequestHeader.model_validate(empty_data)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("X-Amzn-SageMaker-Session-Id",)
        assert errors[0]["type"] == "missing"

    def test_model_extra_fields_ignored(self):
        """
        Test that extra fields are ignored due to extra="ignore" configuration.
        """
        test_session_id = "session-with-extras"
        data_with_extras = {
            "session_id": test_session_id,
            "extra_field": "should_be_ignored",
            "another_extra": 123,
        }

        model = SageMakerSessionRequestHeader.model_validate(data_with_extras)
        assert model.session_id == test_session_id
        assert not hasattr(model, "extra_field")
        assert not hasattr(model, "another_extra")

    def test_model_alias_generator_integration(self):
        """
        Test integration between alias generator and model validation.
        """
        test_session_id = "alias-test-session"

        # Test that the alias generator creates the correct SageMaker header format
        expected_alias = "X-Amzn-SageMaker-Session-Id"
        headers = {expected_alias: test_session_id}

        model = SageMakerSessionRequestHeader.model_validate(headers)
        assert model.session_id == test_session_id

    def test_model_config_dict_settings(self):
        """
        Test ConfigDict settings and their effects.
        """
        test_session_id = "config-test-session"

        # Test populate_by_name=True allows both alias and field name
        field_name_data = {"session_id": test_session_id}
        alias_data = {"X-Amzn-SageMaker-Session-Id": test_session_id}

        model1 = SageMakerSessionRequestHeader.model_validate(field_name_data)
        model2 = SageMakerSessionRequestHeader.model_validate(alias_data)

        assert model1.session_id == test_session_id
        assert model2.session_id == test_session_id

        # Test extra="ignore" with additional fields
        data_with_extra = {
            "session_id": test_session_id,
            "ignored_field": "ignored_value",
        }
        model3 = SageMakerSessionRequestHeader.model_validate(data_with_extra)
        assert model3.session_id == test_session_id


def dummy_original_func_with_model(request: MockEngineRequest, raw_request: Request):
    """Dummy function for testing purposes."""
    return Response(status_code=200, content="Success")


class TestCloseSessionApiTransformValidation:
    """Test suite for CloseSessionApiTransform request validation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.api_transform = CloseSessionApiTransform(
            original_function=dummy_original_func_with_model,
            engine_request_paths={"session_id": "body.session_id"},
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults={"body.additional_param": "value"},
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "headers, expected_session_id",
        [
            ({"X-Amzn-SageMaker-Session-Id": "test-session-1"}, "test-session-1"),
            ({"X-Amzn-SageMaker-Session-Id": "test-session-2"}, "test-session-2"),
            ({"X-Amzn-SageMaker-Session-Id": "123"}, "123"),
            (
                {
                    "X-Amzn-SageMaker-Session-Id": "test-session-3",
                    "X-Amzn-SageMaker-Other-Header": "value",
                },
                "test-session-3",
            ),
            (
                {
                    "X-Amzn-SageMaker-Session-Id": "test-session-4",
                    "X-Amzn-SageMaker-session-id": "test-session-5",
                },
                "test-session-4",
            ),
        ],
    )
    async def test_validate_request_success_scenarios(
        self, headers, expected_session_id
    ):
        """
        Test validate_request method with various valid inputs.
        """
        self.api_transform.validate_request = AsyncMock(
            wraps=self.api_transform.validate_request
        )
        mock_request = MagicMock(spec=Request)
        mock_request.headers = headers

        actual = await self.api_transform.validate_request(mock_request)
        assert isinstance(actual, SageMakerSessionRequestHeader)
        assert actual.session_id == expected_session_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"X-Amzn-SageMaker-Session-Id": ""},
        ],
    )
    @patch("model_hosting_container_standards.sagemaker.sessions.close_session.logger")
    async def test_validate_request_failure_scenarios(self, mock_logger, headers):
        """
        Test validate_request with invalid or missing data.
        """
        with pytest.raises(HTTPException) as exc_info:
            mock_request = MagicMock(spec=Request)
            mock_request.headers = headers
            await self.api_transform.validate_request(mock_request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
        assert "Session ID is required" in str(exc_info.value.detail)
        mock_logger.error.assert_called_once_with(
            "No session ID found in request headers for close session"
        )


class TestCloseSessionApiTransformAdditionalFields:
    """Test suite for CloseSessionApiTransform additional fields extraction."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.api_transform = CloseSessionApiTransform(
            original_function=dummy_original_func_with_model,
            engine_request_paths={"session_id": "body.session_id"},
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults={"body.additional_param": "value"},
        )

    def test_extract_additional_fields_functionality(self):
        """
        Test _extract_additional_fields method with various inputs.
        """
        self.api_transform._extract_additional_fields = MagicMock(
            wraps=self.api_transform._extract_additional_fields
        )
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Amzn-SageMaker-Session-Id": "test-session-1"}
        mock_validated_request = SageMakerSessionRequestHeader(**mock_request.headers)

        expected = {"session_id": "test-session-1"}
        actual = self.api_transform._extract_additional_fields(
            mock_validated_request, mock_request
        )
        assert actual == expected


class TestCloseSessionApiTransformResponseGeneration:
    """Test suite for CloseSessionApiTransform response generation methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.api_transform = CloseSessionApiTransform(
            original_function=dummy_original_func_with_model,
            engine_request_paths={"session_id": "body.session_id"},
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults={"body.additional_param": "value"},
        )

    @pytest.mark.parametrize(
        "mock_session_id",
        [
            "12345" "",
            None,
        ],
    )
    def test_generate_successful_response_content(self, mock_session_id):
        """
        Test _generate_successful_response_content with various scenarios.
        """
        mock_response = MagicMock(spec=Response)
        mock_transform_request_output = BaseTransformRequestOutput(
            raw_request=MagicMock(spec=Request),
            transformed_request={},
            additional_fields={"session_id": mock_session_id},
        )

        expected = (
            f"Successfully closed session: {mock_session_id}"
            if mock_session_id
            else "Successfully closed session"
        )
        actual = self.api_transform._generate_successful_response_content(
            mock_response, mock_transform_request_output
        )
        assert actual == expected

    def test_transform_ok_response_complete(self):
        """
        Test _transform_ok_response method comprehensive functionality.
        """
        self.api_transform._generate_successful_response_content = MagicMock(
            wraps=self.api_transform._generate_successful_response_content
        )

        mock_response = MagicMock(spec=Response)
        mock_session_id = "XXXXXXXXXXXXXX"
        mock_transform_request_output = BaseTransformRequestOutput(
            raw_request=MagicMock(spec=Request),
            transformed_request={},
            additional_fields={"session_id": mock_session_id},
        )

        actual = self.api_transform._transform_ok_response(
            mock_response, mock_transform_request_output
        )

        assert actual.status_code == HTTPStatus.OK.value
        assert set(actual.headers).issuperset(
            set(
                MutableHeaders(
                    {SageMakerSessionHeader.CLOSED_SESSION_ID: mock_session_id}
                )
            )
        )
        assert mock_session_id in actual.body.decode()
        self.api_transform._generate_successful_response_content.assert_called_once_with(
            mock_response, mock_transform_request_output
        )


async def dummy_func_with_model(request: MockEngineRequest, raw_request: Request):
    """Dummy function for testing."""
    return Response(status_code=200, content="Success")


class TestCreateCloseSessionTransform:
    """Test suite for create_close_session_transform factory function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        handler_registry.clear()

    def teardown_method(self):
        handler_registry.clear()

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.sessions.close_session.CloseSessionApiTransform",
        wraps=CloseSessionApiTransform,
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sessions.close_session.handler_registry",
        wraps=handler_registry,
    )
    @patch("model_hosting_container_standards.sagemaker.sessions.close_session.logger")
    async def test_decorator_factory_functionality(
        self, mock_logger, mock_handler_registry, mock_transform_cls
    ):
        """
        Test create_close_session_transform decorator factory behavior.
        """

        original_function = AsyncMock(wraps=dummy_func_with_model)
        engine_request_paths = {"session_id": "body.session_id"}
        engine_request_defaults = {"body.additional_param": "value"}

        actual_decorated_func = create_close_session_transform(
            engine_request_paths=engine_request_paths,
            engine_request_defaults=engine_request_defaults,
            engine_request_model_cls=MockEngineRequest,
        )(original_function)

        assert callable(actual_decorated_func)
        assert mock_handler_registry.has_handler("close_session")
        assert mock_handler_registry.set_handler.call_count == 1

        mock_logger.info.assert_any_call(
            f"[CLOSE_SESSION] Registered transform handler for {original_function.__name__}"
        )

        mock_transform_cls.assert_called_once_with(
            original_function,
            engine_request_paths,
            MockEngineRequest,
            engine_request_defaults=engine_request_defaults,
        )

        mock_raw_request = MagicMock(spec=Request)
        mock_raw_request.headers = {"X-Amzn-SageMaker-Session-Id": "test-session-id"}

        await actual_decorated_func(mock_raw_request)
        original_function.assert_any_call(
            MockEngineRequest(session_id="test-session-id", additional_param="value"),
            mock_raw_request,
        )


class TestRegisterCloseSessionHandler:
    """Test suite for _register_close_session_handler function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        handler_registry.clear()

    def teardown_method(self):
        handler_registry.clear()

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.sagemaker.sessions.close_session._transform_defaults_config",
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sessions.close_session.create_close_session_transform",
        wraps=create_close_session_transform,
    )
    @patch(
        "model_hosting_container_standards.sagemaker.sessions.close_session.handler_registry",
        wraps=handler_registry,
    )
    @patch("model_hosting_container_standards.sagemaker.sessions.close_session.logger")
    async def test_handler_registration_complete(
        self,
        mock_logger,
        mock_handler_registry,
        mock_create_close_session_transform,
        mock_transform_defaults_config,
    ):
        """
        Test _register_close_session_handler comprehensive functionality.

        Should verify:
        - Function accepts engine_request_session_id_path parameter
        - Optional engine_request_model_cls parameter works
        - session_id is mapped to provided path correctly
        - close_session_defaults from config are used
        - Info and debug log messages are generated
        - Function returns decorator result
        - Registration completes successfully
        """
        mock_transform_defaults_config.close_session_defaults = {
            "body.additional_param": "value"
        }
        original_function = AsyncMock(wraps=dummy_func_with_model)
        engine_request_session_id_path = "body.session_id"

        actual_decorator = _register_close_session_handler(
            engine_request_session_id_path=engine_request_session_id_path,
            engine_request_model_cls=MockEngineRequest,
        )

        mock_logger.info.assert_any_call("Registering close session handler")
        mock_logger.debug.assert_any_call(
            f"Handler parameters - request_session_id_path: {engine_request_session_id_path}"
        )
        mock_create_close_session_transform.assert_called_once_with(
            {"session_id": engine_request_session_id_path},
            MockEngineRequest,
            engine_request_defaults={"body.additional_param": "value"},
        )

        actual_decorated_func = actual_decorator(original_function)
        assert callable(actual_decorated_func)
        assert mock_handler_registry.has_handler("close_session")
        assert mock_handler_registry.set_handler.call_count == 1

        mock_raw_request = MagicMock(spec=Request)
        mock_raw_request.headers = {"X-Amzn-SageMaker-Session-Id": "test-session-id"}

        await actual_decorated_func(mock_raw_request)
        original_function.assert_any_call(
            MockEngineRequest(session_id="test-session-id", additional_param="value"),
            mock_raw_request,
        )
