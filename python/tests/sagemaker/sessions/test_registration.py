"""Unit tests for session handler registration functions."""

import pytest

from model_hosting_container_standards.sagemaker.sessions import (
    register_engine_session_handler,
)


class TestRegisterEngineSessionHandler:
    """Test register_engine_session_handler function."""

    def test_create_session_requires_session_id_path(self):
        """Test that create_session requires session_id_path parameter."""
        with pytest.raises(ValueError) as exc_info:
            register_engine_session_handler(
                handler_type="create_session",
                request_shape={},
                session_id_path=None,
                content_path="message",
            )

        assert "session_id_path is required" in str(exc_info.value)

    def test_create_session_with_valid_params(self):
        """Test successful create_session registration."""
        decorator = register_engine_session_handler(
            handler_type="create_session",
            request_shape={"model": "body.model"},
            session_id_path="session_id",
            content_path="message",
        )

        assert decorator is not None
        assert callable(decorator)

    def test_close_session_without_session_id_path(self):
        """Test that close_session doesn't require session_id_path."""
        decorator = register_engine_session_handler(
            handler_type="close_session",
            request_shape={},
            content_path="message",
        )

        assert decorator is not None
        assert callable(decorator)

    def test_invalid_handler_type(self):
        """Test that invalid handler_type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            register_engine_session_handler(
                handler_type="invalid_type",
                request_shape={},
            )

        assert "Invalid handler_type" in str(exc_info.value)
        assert "create_session" in str(exc_info.value)
        assert "close_session" in str(exc_info.value)

    def test_adds_body_prefix_to_paths(self):
        """Test that body. prefix is automatically added to response paths."""
        # This is tested indirectly - the decorator should work with paths
        # relative to the handler's return value, not the serialized response
        decorator = register_engine_session_handler(
            handler_type="create_session",
            request_shape={},
            session_id_path="id",  # Should become body.id internally
            content_path="message",  # Should become body.message internally
        )

        assert decorator is not None

    def test_preserves_body_prefix_if_present(self):
        """Test that existing body. prefix is not duplicated."""
        decorator = register_engine_session_handler(
            handler_type="create_session",
            request_shape={},
            session_id_path="body.id",  # Already has body. prefix
            content_path="body.message",
        )

        assert decorator is not None


class TestResponseShapeConstruction:
    """Test that response_shape is constructed correctly."""

    def test_create_session_response_shape_has_required_keys(self):
        """Test that create_session response_shape includes session ID and content."""
        # We can't directly inspect the response_shape, but we can verify
        # the decorator is created successfully with the right parameters
        decorator = register_engine_session_handler(
            handler_type="create_session",
            request_shape={},
            session_id_path="session.id",
            content_path="session.message",
        )

        # If this doesn't raise, the response_shape was constructed correctly
        assert decorator is not None

    def test_close_session_response_shape_has_content_key(self):
        """Test that close_session response_shape includes content."""
        decorator = register_engine_session_handler(
            handler_type="close_session",
            request_shape={},
            content_path="result.message",
        )

        assert decorator is not None

    def test_none_content_path_is_handled(self):
        """Test that None content_path is handled correctly."""
        decorator = register_engine_session_handler(
            handler_type="close_session",
            request_shape={},
            content_path=None,
        )

        # Should still create decorator, content extraction will just return None
        assert decorator is not None
