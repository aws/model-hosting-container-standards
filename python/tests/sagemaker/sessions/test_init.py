"""Unit tests for sessions module public API."""

from unittest.mock import patch

from model_hosting_container_standards.sagemaker.sessions import (
    build_session_request_shape,
)
from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader,
)


class TestBuildSessionRequestShape:
    """Test build_session_request_shape function."""

    def test_creates_basic_request_shape_with_session_id_only(self):
        """Test creates request shape with only session ID path."""
        result = build_session_request_shape("session_id")

        assert result == {
            "session_id": f'headers."{SageMakerSessionHeader.SESSION_ID}"'
        }

    def test_creates_request_shape_with_nested_session_id_path(self):
        """Test creates request shape with nested session ID path."""
        result = build_session_request_shape("metadata.session_id")

        assert result == {
            "metadata.session_id": f'headers."{SageMakerSessionHeader.SESSION_ID}"'
        }

    def test_merges_additional_shape_without_conflicts(self):
        """Test merges additional shape when no conflicts exist."""
        additional = {
            "capacity": "`1024`",
            "model_name": "`gpt-4`",
        }

        result = build_session_request_shape("session_id", additional)

        assert result == {
            "session_id": f'headers."{SageMakerSessionHeader.SESSION_ID}"',
            "capacity": "`1024`",
            "model_name": "`gpt-4`",
        }

    @patch("model_hosting_container_standards.sagemaker.sessions.logger")
    def test_overwrites_conflicting_session_id_path_and_warns(self, mock_logger):
        """Test overwrites session ID path in additional shape and logs warning."""
        additional = {
            "session_id": "some_other_value",
            "capacity": "`1024`",
        }

        result = build_session_request_shape("session_id", additional)

        # Session ID should be overwritten with the correct value
        assert result == {
            "session_id": f'headers."{SageMakerSessionHeader.SESSION_ID}"',
            "capacity": "`1024`",
        }

        # Should have logged a warning
        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "session_id" in warning_message
        assert "some_other_value" in warning_message
        assert "overwritten" in warning_message.lower()

    def test_handles_none_additional_shape(self):
        """Test handles None as additional shape gracefully."""
        result = build_session_request_shape("session_id", None)

        assert result == {
            "session_id": f'headers."{SageMakerSessionHeader.SESSION_ID}"'
        }

    def test_handles_empty_additional_shape(self):
        """Test handles empty dict as additional shape."""
        result = build_session_request_shape("session_id", {})

        assert result == {
            "session_id": f'headers."{SageMakerSessionHeader.SESSION_ID}"'
        }

    def test_preserves_all_additional_fields(self):
        """Test preserves all fields from additional shape."""
        additional = {
            "field1": "value1",
            "field2": "value2",
            "field3": "value3",
            "nested.field": "nested_value",
        }

        result = build_session_request_shape("session_id", additional)

        assert result == {
            "session_id": f'headers."{SageMakerSessionHeader.SESSION_ID}"',
            "field1": "value1",
            "field2": "value2",
            "field3": "value3",
            "nested.field": "nested_value",
        }

    @patch("model_hosting_container_standards.sagemaker.sessions.logger")
    def test_session_id_always_takes_precedence(self, mock_logger):
        """Test session ID value always takes precedence even after merge."""
        additional = {
            "session_id": "wrong_value",
            "other_field": "other_value",
        }

        result = build_session_request_shape("session_id", additional)

        # Verify session_id has the correct value, not the one from additional
        assert result["session_id"] == f'headers."{SageMakerSessionHeader.SESSION_ID}"'
        assert result["session_id"] != "wrong_value"
        assert result["other_field"] == "other_value"

    def test_works_with_complex_jmespath_expressions(self):
        """Test works with complex JMESPath expressions in additional shape."""
        additional = {
            "model": 'headers."X-Model-Name"',
            "temperature": "`0.7`",
            "max_tokens": "body.parameters.max_tokens",
        }

        result = build_session_request_shape("request.session_id", additional)

        assert result == {
            "request.session_id": f'headers."{SageMakerSessionHeader.SESSION_ID}"',
            "model": 'headers."X-Model-Name"',
            "temperature": "`0.7`",
            "max_tokens": "body.parameters.max_tokens",
        }

    @patch("model_hosting_container_standards.sagemaker.sessions.logger")
    def test_no_warning_when_no_conflict(self, mock_logger):
        """Test no warning is logged when there's no conflict."""
        additional = {
            "capacity": "`1024`",
            "model": "`gpt-4`",
        }

        result = build_session_request_shape("session_id", additional)

        # Should not have logged any warning
        mock_logger.warning.assert_not_called()
        assert result["session_id"] == f'headers."{SageMakerSessionHeader.SESSION_ID}"'
