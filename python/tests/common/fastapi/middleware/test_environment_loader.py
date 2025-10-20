"""Tests for MiddlewareEnvironmentLoader."""

import os
from unittest.mock import Mock, patch

from model_hosting_container_standards.common.fastapi.middleware.source.environment_loader import (
    MiddlewareEnvironmentLoader,
)


class TestMiddlewareEnvironmentLoader:
    """Test MiddlewareEnvironmentLoader functionality."""

    def setup_method(self):
        """Clear environment variables before each test."""
        # Clear relevant environment variables
        env_vars = [
            "CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE",
            "CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS",
            "CUSTOM_PRE_PROCESS",
            "CUSTOM_POST_PROCESS",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_init(self):
        """Test MiddlewareEnvironmentLoader initialization."""
        loader = MiddlewareEnvironmentLoader()

        assert len(loader.middleware_mapping) == 4
        assert "throttle" in loader.middleware_mapping
        assert "pre_post_process" in loader.middleware_mapping
        assert "pre_process" in loader.middleware_mapping
        assert "post_process" in loader.middleware_mapping

    def test_load_no_env_vars(self):
        """Test load when no environment variables are set."""
        loader = MiddlewareEnvironmentLoader()
        mock_function_loader = Mock()

        loader.load(mock_function_loader)

        assert loader.throttle_middleware is None
        assert loader.pre_post_middleware is None
        assert loader.pre_fn is None
        assert loader.post_fn is None

    @patch(
        "model_hosting_container_standards.common.fastapi.middleware.source.environment_loader.os.getenv"
    )
    def test_load_middleware_throttle(self, mock_getenv):
        """Test loading throttle middleware from environment variable."""

        # Mock environment variable
        def mock_env_func(var, default=None):
            if var == "CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE":
                return "test.module:throttle_func"
            return default

        mock_getenv.side_effect = mock_env_func

        # Mock function loader
        mock_throttle_func = Mock()
        mock_throttle_func.__name__ = "throttle_func"
        mock_function_loader = Mock()
        mock_function_loader.load_function = Mock(return_value=mock_throttle_func)

        loader = MiddlewareEnvironmentLoader()
        loader.load(mock_function_loader)

        assert loader.throttle_middleware is mock_throttle_func
        mock_function_loader.load_function.assert_called_with(
            "test.module:throttle_func"
        )

    @patch(
        "model_hosting_container_standards.common.fastapi.middleware.source.environment_loader.os.getenv"
    )
    def test_load_middleware_pre_post_process(self, mock_getenv):
        """Test loading pre_post_process middleware from environment variable."""

        # Mock environment variable
        def mock_env_func(var, default=None):
            if var == "CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS":
                return "test.module:pre_post_func"
            return default

        mock_getenv.side_effect = mock_env_func

        # Mock function loader
        mock_pre_post_func = Mock()
        mock_pre_post_func.__name__ = "pre_post_func"
        mock_function_loader = Mock()
        mock_function_loader.load_function = Mock(return_value=mock_pre_post_func)

        loader = MiddlewareEnvironmentLoader()
        loader.load(mock_function_loader)

        assert loader.pre_post_middleware is mock_pre_post_func
        mock_function_loader.load_function.assert_called_with(
            "test.module:pre_post_func"
        )

    @patch(
        "model_hosting_container_standards.common.fastapi.middleware.source.environment_loader.os.getenv"
    )
    def test_load_combined_pre_post_middleware(self, mock_getenv):
        """Test loading and combining separate pre/post process functions."""

        # Mock environment variables
        def mock_env(var, default=None):
            if var == "CUSTOM_PRE_PROCESS":
                return "test.module:pre_func"
            elif var == "CUSTOM_POST_PROCESS":
                return "test.module:post_func"
            return default

        mock_getenv.side_effect = mock_env

        # Mock function loader
        mock_pre_func = Mock()
        mock_pre_func.__name__ = "pre_func"
        mock_post_func = Mock()
        mock_post_func.__name__ = "post_func"
        mock_function_loader = Mock()
        mock_function_loader.load_function = Mock(
            side_effect=[mock_pre_func, mock_post_func]
        )

        loader = MiddlewareEnvironmentLoader()
        loader.load(mock_function_loader)

        assert loader.pre_fn is mock_pre_func
        assert loader.post_fn is mock_post_func
        assert loader.pre_post_middleware is not None  # Should be combined

    @patch(
        "model_hosting_container_standards.common.fastapi.middleware.source.environment_loader.os.getenv"
    )
    def test_load_middleware_function_loader_failure(self, mock_getenv):
        """Test handling of function loader failures."""

        # Mock environment variable
        def mock_env_func(var, default=None):
            if var == "CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE":
                return "invalid.module:func"
            return default

        mock_getenv.side_effect = mock_env_func

        # Mock function loader to raise exception
        mock_function_loader = Mock()
        mock_function_loader.load_function = Mock(side_effect=Exception("Load failed"))

        loader = MiddlewareEnvironmentLoader()

        # Should not raise exception
        loader.load(mock_function_loader)

        assert loader.throttle_middleware is None

    def test_load_middleware_invalid_name(self):
        """Test load_middleware with invalid middleware name."""
        loader = MiddlewareEnvironmentLoader()
        mock_function_loader = Mock()

        # Should not raise exception
        loader.load_middleware("invalid_name", mock_function_loader)

        # Nothing should be set
        assert loader.throttle_middleware is None

    @patch(
        "model_hosting_container_standards.common.fastapi.middleware.source.environment_loader.os.getenv"
    )
    def test_handle_pre_post_combination_priority(self, mock_getenv):
        """Test that direct pre_post_process takes priority over combined pre/post."""

        # Mock environment variables - both direct and separate
        def mock_env(var, default=None):
            if var == "CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS":
                return "test.module:direct_func"
            elif var == "CUSTOM_PRE_PROCESS":
                return "test.module:pre_func"
            elif var == "CUSTOM_POST_PROCESS":
                return "test.module:post_func"
            return default

        mock_getenv.side_effect = mock_env

        # Mock function loader
        mock_direct_func = Mock()
        mock_direct_func.__name__ = "direct_func"
        mock_function_loader = Mock()
        mock_function_loader.load_function = Mock(return_value=mock_direct_func)

        loader = MiddlewareEnvironmentLoader()
        loader.load(mock_function_loader)

        # Should use direct function, and also load separate functions
        assert loader.pre_post_middleware is mock_direct_func
        # But pre_fn and post_fn are still loaded from their env vars
        assert loader.pre_fn is mock_direct_func  # Same mock function returned
        assert loader.post_fn is mock_direct_func  # Same mock function returned

    def test_middleware_mapping_completeness(self):
        """Test that middleware mapping covers all expected middleware types."""
        loader = MiddlewareEnvironmentLoader()

        expected_mappings = {
            "throttle": ("CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE", "throttle_middleware"),
            "pre_post_process": (
                "CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS",
                "pre_post_middleware",
            ),
            "pre_process": ("CUSTOM_PRE_PROCESS", "pre_fn"),
            "post_process": ("CUSTOM_POST_PROCESS", "post_fn"),
        }

        for name, (expected_env_var, expected_property) in expected_mappings.items():
            assert name in loader.middleware_mapping
            env_var, property_name = loader.middleware_mapping[name]
            assert env_var == expected_env_var
            assert property_name == expected_property
