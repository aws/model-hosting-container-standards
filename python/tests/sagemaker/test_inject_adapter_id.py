"""Tests for inject_adapter_id function validation."""

import pytest

from model_hosting_container_standards.sagemaker import inject_adapter_id


class TestInjectAdapterIdValidation:
    """Test parameter validation for inject_adapter_id function."""

    def test_append_true_without_separator(self):
        """Test inject_adapter_id raises error when append=True without separator."""
        with pytest.raises(
            ValueError, match="separator must be provided when append=True"
        ):
            inject_adapter_id("model", append=True)

    def test_empty_adapter_path(self):
        """Test inject_adapter_id raises error for empty adapter_path."""
        with pytest.raises(ValueError, match="adapter_path cannot be empty"):
            inject_adapter_id("")

    def test_non_string_adapter_path(self):
        """Test inject_adapter_id raises error for non-string adapter_path."""
        with pytest.raises(ValueError, match="adapter_path must be a string"):
            inject_adapter_id(123)

    def test_valid_append_parameters(self):
        """Test inject_adapter_id accepts valid append parameters."""
        # Should not raise any exceptions
        decorator = inject_adapter_id("model", append=True, separator=":")
        assert callable(decorator)

    def test_valid_replace_parameters(self):
        """Test inject_adapter_id accepts valid replace parameters."""
        # Should not raise any exceptions
        decorator = inject_adapter_id("model")
        assert callable(decorator)
