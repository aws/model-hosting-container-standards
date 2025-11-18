"""Unit tests for logging_config module."""

import logging
import os
from unittest.mock import patch

from model_hosting_container_standards.logging_config import get_logger


class TestGetLogger:
    """Test get_logger function."""

    def test_default_error_level(self):
        """Test that logger defaults to ERROR level when no env var set."""
        with patch.dict(os.environ, {}, clear=True):
            # Create a unique logger name to avoid conflicts
            test_logger = get_logger("test_default_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.ERROR
            finally:
                # Clean up the test logger
                test_logger.handlers.clear()

    def test_sagemaker_container_log_level_debug(self):
        """Test that SAGEMAKER_CONTAINER_LOG_LEVEL=DEBUG sets DEBUG level."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "DEBUG"}):
            test_logger = get_logger("test_debug_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.DEBUG
            finally:
                test_logger.handlers.clear()

    def test_sagemaker_container_log_level_info(self):
        """Test that SAGEMAKER_CONTAINER_LOG_LEVEL=INFO sets INFO level."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "INFO"}):
            test_logger = get_logger("test_info_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.INFO
            finally:
                test_logger.handlers.clear()

    def test_sagemaker_container_log_level_warning(self):
        """Test that SAGEMAKER_CONTAINER_LOG_LEVEL=WARNING sets WARNING level."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "WARNING"}):
            test_logger = get_logger("test_warning_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.WARNING
            finally:
                test_logger.handlers.clear()

    def test_sagemaker_container_log_level_error(self):
        """Test that SAGEMAKER_CONTAINER_LOG_LEVEL=ERROR sets ERROR level."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "ERROR"}):
            test_logger = get_logger("test_error_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.ERROR
            finally:
                test_logger.handlers.clear()

    def test_log_level_fallback(self):
        """Test that LOG_LEVEL is used when SAGEMAKER_CONTAINER_LOG_LEVEL not set."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}, clear=True):
            test_logger = get_logger("test_fallback_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.DEBUG
            finally:
                test_logger.handlers.clear()

    def test_sagemaker_takes_priority_over_log_level(self):
        """Test that SAGEMAKER_CONTAINER_LOG_LEVEL takes priority over LOG_LEVEL."""
        with patch.dict(
            os.environ,
            {"SAGEMAKER_CONTAINER_LOG_LEVEL": "INFO", "LOG_LEVEL": "DEBUG"},
        ):
            test_logger = get_logger("test_priority_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.INFO
            finally:
                test_logger.handlers.clear()

    def test_case_insensitive_log_level(self):
        """Test that log level is case-insensitive."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "debug"}):
            test_logger = get_logger("test_case_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.DEBUG
            finally:
                test_logger.handlers.clear()

    def test_logger_not_reconfigured_if_already_configured(self):
        """Test that logger is not reconfigured if it already has handlers."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "INFO"}):
            # Create a unique logger name
            test_logger = get_logger("test_reconfig_logger")
            try:
                initial_handler_count = len(test_logger.handlers)

                # Call get_logger again
                test_logger_again = get_logger("test_reconfig_logger")

                # Should be the same logger instance with same handlers
                assert test_logger is test_logger_again
                assert len(test_logger_again.handlers) == initial_handler_count
            finally:
                # Clean up the test logger
                test_logger.handlers.clear()

    def test_logger_name_parameter(self):
        """Test that logger name parameter is respected."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "INFO"}):
            custom_name = "custom_test_logger"
            test_logger = get_logger(custom_name)
            try:
                assert test_logger.name == custom_name
            finally:
                # Clean up the test logger
                test_logger.handlers.clear()

    def test_default_logger_name(self):
        """Test that default logger name is used when not specified."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "INFO"}):
            test_logger = get_logger()
            try:
                assert test_logger.name == "model_hosting_container_standards"
            finally:
                # Clean up the test logger
                test_logger.handlers.clear()

    def test_logger_has_handler_and_formatter(self):
        """Test that logger has proper handler and formatter configured."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "INFO"}):
            test_logger = get_logger("test_format_logger")
            try:
                assert len(test_logger.handlers) == 1
                handler = test_logger.handlers[0]
                assert isinstance(handler, logging.StreamHandler)
                assert handler.formatter is not None
                # Check format string contains expected elements
                format_str = handler.formatter._fmt
                assert "%(levelname)s" in format_str
                assert "%(name)s" in format_str
                assert "%(filename)s" in format_str
                assert "%(lineno)d" in format_str
                assert "%(message)s" in format_str
            finally:
                test_logger.handlers.clear()

    def test_logger_propagate_false(self):
        """Test that logger propagate is set to False to avoid duplicate logs."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "INFO"}):
            test_logger = get_logger("test_propagate_logger")
            try:
                assert test_logger.propagate is False
            finally:
                test_logger.handlers.clear()
