"""Unit tests for logging_config module."""

import logging
import os
from unittest.mock import patch

from model_hosting_container_standards.logging_config import get_logger, parse_level


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

    def test_integer_log_level_10_debug(self):
        """Test that integer log level 10 (DEBUG) works."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "10"}):
            test_logger = get_logger("test_int_10_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.DEBUG
            finally:
                test_logger.handlers.clear()

    def test_integer_log_level_20_info(self):
        """Test that integer log level 20 (INFO) works."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "20"}):
            test_logger = get_logger("test_int_20_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.INFO
            finally:
                test_logger.handlers.clear()

    def test_integer_log_level_30_warning(self):
        """Test that integer log level 30 (WARNING) works."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "30"}):
            test_logger = get_logger("test_int_30_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.WARNING
            finally:
                test_logger.handlers.clear()

    def test_integer_log_level_40_error(self):
        """Test that integer log level 40 (ERROR) works."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "40"}):
            test_logger = get_logger("test_int_40_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.ERROR
            finally:
                test_logger.handlers.clear()

    def test_integer_log_level_50_critical(self):
        """Test that integer log level 50 (CRITICAL) works."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "50"}):
            test_logger = get_logger("test_int_50_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.CRITICAL
            finally:
                test_logger.handlers.clear()

    def test_custom_log_level(self):
        """Test that custom log levels can be set."""
        # Add a custom log level
        custom_level_name = "CUSTOM_LEVEL"
        custom_level_value = 25  # Between INFO (20) and WARNING (30)
        logging.addLevelName(custom_level_value, custom_level_name)

        try:
            with patch.dict(
                os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": custom_level_name}
            ):
                test_logger = get_logger("test_custom_level_logger")
                try:
                    assert len(test_logger.handlers) > 0
                    assert test_logger.level == custom_level_value
                finally:
                    test_logger.handlers.clear()
        finally:
            # Clean up custom level
            del logging._levelToName[custom_level_value]
            del logging._nameToLevel[custom_level_name]

    def test_custom_log_level_with_integer(self):
        """Test that custom integer log levels work."""
        custom_level_value = 25  # Between INFO (20) and WARNING (30)

        with patch.dict(
            os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": str(custom_level_value)}
        ):
            test_logger = get_logger("test_custom_int_level_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == custom_level_value
            finally:
                test_logger.handlers.clear()

    def test_mixed_case_log_level(self):
        """Test that mixed case log levels work (e.g., 'Info', 'Warning')."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "Info"}):
            test_logger = get_logger("test_mixed_case_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.INFO
            finally:
                test_logger.handlers.clear()

    def test_critical_log_level(self):
        """Test that CRITICAL log level works."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "CRITICAL"}):
            test_logger = get_logger("test_critical_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.CRITICAL
            finally:
                test_logger.handlers.clear()

    def test_notset_log_level(self):
        """Test that NOTSET log level works."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "NOTSET"}):
            test_logger = get_logger("test_notset_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.NOTSET
            finally:
                test_logger.handlers.clear()

    def test_invalid_log_level_defaults_to_error(self):
        """Test that an invalid log level defaults to ERROR instead of crashing."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "INVALID_LEVEL"}):
            test_logger = get_logger("test_invalid_logger")
            try:
                assert len(test_logger.handlers) > 0
                # Should default to ERROR when invalid level is provided
                assert test_logger.level == logging.ERROR
            finally:
                test_logger.handlers.clear()

    def test_empty_log_level_defaults_to_error(self):
        """Test that empty log level string defaults to ERROR."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": ""}):
            test_logger = get_logger("test_empty_logger")
            try:
                assert len(test_logger.handlers) > 0
                # Should default to ERROR when empty level is provided
                assert test_logger.level == logging.ERROR
            finally:
                test_logger.handlers.clear()

    def test_whitespace_log_level_defaults_to_error(self):
        """Test that whitespace log level defaults to ERROR."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "  "}):
            test_logger = get_logger("test_whitespace_logger")
            try:
                assert len(test_logger.handlers) > 0
                # Should default to ERROR when whitespace level is provided
                assert test_logger.level == logging.ERROR
            finally:
                test_logger.handlers.clear()

    def test_special_characters_log_level_defaults_to_error(self):
        """Test that log levels with special characters default to ERROR."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "DEBUG!@#"}):
            test_logger = get_logger("test_special_chars_logger")
            try:
                assert len(test_logger.handlers) > 0
                # Should default to ERROR for invalid level with special chars
                assert test_logger.level == logging.ERROR
            finally:
                test_logger.handlers.clear()

    def test_negative_integer_log_level(self):
        """Test that negative integer log levels work (Python logging allows them)."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "-10"}):
            test_logger = get_logger("test_negative_int_logger")
            try:
                assert len(test_logger.handlers) > 0
                # Negative numbers are technically valid in Python logging
                # -10 would be even more verbose than DEBUG (10)
                assert test_logger.level == -10
            finally:
                test_logger.handlers.clear()

    def test_float_log_level_defaults_to_error(self):
        """Test that float log level strings default to ERROR."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "10.5"}):
            test_logger = get_logger("test_float_logger")
            try:
                assert len(test_logger.handlers) > 0
                # Float strings won't pass int(<float>), will fail as string name
                assert test_logger.level == logging.ERROR
            finally:
                test_logger.handlers.clear()

    def test_zero_log_level(self):
        """Test that zero log level works (equivalent to NOTSET)."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "0"}):
            test_logger = get_logger("test_zero_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.NOTSET
            finally:
                test_logger.handlers.clear()

    def test_very_large_integer_log_level(self):
        """Test that very large integer log levels work."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "100"}):
            test_logger = get_logger("test_large_int_logger")
            try:
                assert len(test_logger.handlers) > 0
                # Python logging accepts any integer
                assert test_logger.level == 100
            finally:
                test_logger.handlers.clear()

    def test_lowercase_integer_string(self):
        """Test that numeric strings are parsed correctly regardless of case."""
        with patch.dict(os.environ, {"SAGEMAKER_CONTAINER_LOG_LEVEL": "20"}):
            test_logger = get_logger("test_numeric_string_logger")
            try:
                assert len(test_logger.handlers) > 0
                assert test_logger.level == logging.INFO
            finally:
                test_logger.handlers.clear()


class TestParseLevel:
    """Test parse_level function."""

    def test_parse_level_string_debug(self):
        """Test parsing DEBUG string returns uppercase string."""
        result = parse_level("debug")
        assert result == "DEBUG"
        assert isinstance(result, str)

    def test_parse_level_string_info(self):
        """Test parsing INFO string returns uppercase string."""
        result = parse_level("info")
        assert result == "INFO"
        assert isinstance(result, str)

    def test_parse_level_already_uppercase(self):
        """Test parsing already uppercase string."""
        result = parse_level("WARNING")
        assert result == "WARNING"
        assert isinstance(result, str)

    def test_parse_level_mixed_case(self):
        """Test parsing mixed case string."""
        result = parse_level("CrItIcAl")
        assert result == "CRITICAL"
        assert isinstance(result, str)

    def test_parse_level_integer_string_10(self):
        """Test parsing integer string '10' returns int."""
        result = parse_level("10")
        assert result == 10
        assert isinstance(result, int)

    def test_parse_level_integer_string_20(self):
        """Test parsing integer string '20' returns int."""
        result = parse_level("20")
        assert result == 20
        assert isinstance(result, int)

    def test_parse_level_negative_integer_string(self):
        """Test parsing negative integer string returns int."""
        result = parse_level("-10")
        assert result == -10
        assert isinstance(result, int)

    def test_parse_level_zero(self):
        """Test parsing '0' returns int."""
        result = parse_level("0")
        assert result == 0
        assert isinstance(result, int)

    def test_parse_level_large_integer(self):
        """Test parsing large integer string."""
        result = parse_level("999")
        assert result == 999
        assert isinstance(result, int)

    def test_parse_level_float_string(self):
        """Test parsing float string fails and returns uppercase string."""
        result = parse_level("10.5")
        # Can't convert to int, so returns as uppercase string
        assert result == "10.5"
        assert isinstance(result, str)

    def test_parse_level_invalid_string(self):
        """Test parsing invalid string returns uppercase."""
        result = parse_level("invalid")
        assert result == "INVALID"
        assert isinstance(result, str)

    def test_parse_level_empty_string(self):
        """Test parsing empty string returns empty string."""
        result = parse_level("")
        assert result == ""
        assert isinstance(result, str)

    def test_parse_level_whitespace(self):
        """Test parsing whitespace returns uppercase whitespace."""
        result = parse_level("   ")
        assert result == "   "
        assert isinstance(result, str)

    def test_parse_level_special_characters(self):
        """Test parsing string with special characters."""
        result = parse_level("debug!@#")
        assert result == "DEBUG!@#"
        assert isinstance(result, str)
