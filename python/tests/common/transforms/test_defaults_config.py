"""Unit tests for common.transforms.defaults_config module."""

import json
import os

import pytest
from pydantic import ValidationError

from model_hosting_container_standards.common.transforms.defaults_config import (
    SAGEMAKER_TRANSFORMS_ENV_VAR_PREFIX,
    SageMakerTransformsDefaultsConfig,
    _transform_defaults_config,
)


class TestSageMakerTransformsDefaultsConfig:
    """Test suite for SageMakerTransformsDefaultsConfig class."""

    def test_default_initialization(self):
        """Test default initialization creates empty dictionaries for all fields."""
        config = SageMakerTransformsDefaultsConfig()

        assert config.load_adapter_defaults == {}
        assert config.unload_adapter_defaults == {}
        assert config.create_session_defaults == {}
        assert config.close_session_defaults == {}

    def test_initialization_with_explicit_values(self):
        """Test initialization with explicitly provided field values."""
        test_data = {
            "load_adapter_defaults": {"param1": "value1"},
            "unload_adapter_defaults": {"param2": "value2"},
            "create_session_defaults": {"param3": "value3"},
            "close_session_defaults": {"param4": "value4"},
        }

        config = SageMakerTransformsDefaultsConfig(**test_data)

        assert config.load_adapter_defaults == test_data["load_adapter_defaults"]
        assert config.unload_adapter_defaults == test_data["unload_adapter_defaults"]
        assert config.create_session_defaults == test_data["create_session_defaults"]
        assert config.close_session_defaults == test_data["close_session_defaults"]

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored due to ConfigDict(extra="ignore")."""
        config = SageMakerTransformsDefaultsConfig(
            load_adapter_defaults={"param": "value"}, unknown_field="ignored"
        )

        assert config.load_adapter_defaults == {"param": "value"}
        assert not hasattr(config, "unknown_field")

    def test_field_type_validation(self):
        """Test field type validation for configuration fields."""
        with pytest.raises(ValidationError):
            SageMakerTransformsDefaultsConfig(load_adapter_defaults="not_a_dict")

    def test_field_default_factories_independence(self):
        """Test that field default factories create independent instances."""
        config1 = SageMakerTransformsDefaultsConfig()
        config2 = SageMakerTransformsDefaultsConfig()

        config1.load_adapter_defaults["test"] = "value"

        # Ensure config2 is not affected by config1 modification
        assert "test" not in config2.load_adapter_defaults


class TestEnvironmentVariableLoading:
    """Test suite for environment variable loading functionality."""

    def teardown_method(self):
        """Clean up environment variables after each test."""
        for key in list(os.environ.keys()):
            if key.startswith(SAGEMAKER_TRANSFORMS_ENV_VAR_PREFIX):
                del os.environ[key]

    def test_from_env_with_no_env_vars(self):
        """Test from_env() method when no SAGEMAKER_TRANSFORMS_* variables are set."""
        config = SageMakerTransformsDefaultsConfig.from_env()

        assert config.load_adapter_defaults == {}
        assert config.unload_adapter_defaults == {}
        assert config.create_session_defaults == {}
        assert config.close_session_defaults == {}

    def test_from_env_with_valid_env_vars(self):
        """Test from_env() method with valid SAGEMAKER_TRANSFORMS_* environment variables."""
        test_data = {"param1": "value1", "nested": {"key": "value"}}
        os.environ["SAGEMAKER_TRANSFORMS_LOAD_ADAPTER_DEFAULTS"] = json.dumps(test_data)

        config = SageMakerTransformsDefaultsConfig.from_env()

        assert config.load_adapter_defaults == test_data
        assert config.unload_adapter_defaults == {}

    def test_from_env_with_invalid_json(self):
        """Test from_env() method with invalid JSON in environment variables."""
        os.environ["SAGEMAKER_TRANSFORMS_LOAD_ADAPTER_DEFAULTS"] = "invalid_json"

        with pytest.raises(ValidationError):
            SageMakerTransformsDefaultsConfig.from_env()

    def test_from_env_with_unknown_env_vars(self):
        """Test from_env() method with unknown SAGEMAKER_TRANSFORMS_* variables."""
        os.environ["SAGEMAKER_TRANSFORMS_UNKNOWN_FIELD"] = json.dumps({"test": "value"})

        config = SageMakerTransformsDefaultsConfig.from_env()

        # Unknown fields should be ignored
        assert not hasattr(config, "unknown_field")

    def test_load_from_env_vars_data_precedence(self):
        """Test that provided data takes precedence over environment variables."""
        os.environ["SAGEMAKER_TRANSFORMS_LOAD_ADAPTER_DEFAULTS"] = json.dumps(
            {"env": "value"}
        )

        config = SageMakerTransformsDefaultsConfig(
            load_adapter_defaults={"data": "value"}
        )

        # Data should override env vars
        assert config.load_adapter_defaults == {"data": "value"}

    def test_update_from_env_vars(self):
        """Test update_from_env_vars method updates instance from environment."""
        config = SageMakerTransformsDefaultsConfig(
            load_adapter_defaults={"original": "value"}
        )

        os.environ["SAGEMAKER_TRANSFORMS_LOAD_ADAPTER_DEFAULTS"] = json.dumps(
            {"updated": "value"}
        )

        config.update_from_env_vars()

        assert config.load_adapter_defaults == {"updated": "value"}


class TestModuleLevelDefaultsConfig:
    """Test suite for the module-level _transform_defaults_config instance."""

    def test_module_level_instance_exists(self):
        """Test that module-level _transform_defaults_config is created correctly."""
        assert isinstance(_transform_defaults_config, SageMakerTransformsDefaultsConfig)
        assert hasattr(_transform_defaults_config, "load_adapter_defaults")
        assert hasattr(_transform_defaults_config, "unload_adapter_defaults")
        assert hasattr(_transform_defaults_config, "create_session_defaults")
        assert hasattr(_transform_defaults_config, "close_session_defaults")

    def test_env_var_prefix_constant(self):
        """Test the SAGEMAKER_TRANSFORMS_ENV_VAR_PREFIX constant."""
        assert SAGEMAKER_TRANSFORMS_ENV_VAR_PREFIX == "SAGEMAKER_TRANSFORMS_"


class TestIntegrationScenarios:
    """Test suite for integration scenarios and real-world usage patterns."""

    def teardown_method(self):
        """Clean up environment variables after each test."""
        for key in list(os.environ.keys()):
            if key.startswith(SAGEMAKER_TRANSFORMS_ENV_VAR_PREFIX):
                del os.environ[key]

    def test_complete_configuration_lifecycle(self):
        """Test complete configuration lifecycle from creation to updates."""
        # Initial creation with env vars
        os.environ["SAGEMAKER_TRANSFORMS_LOAD_ADAPTER_DEFAULTS"] = json.dumps(
            {"initial": "value"}
        )
        config = SageMakerTransformsDefaultsConfig.from_env()
        assert config.load_adapter_defaults == {"initial": "value"}

        # Update environment and refresh
        os.environ["SAGEMAKER_TRANSFORMS_LOAD_ADAPTER_DEFAULTS"] = json.dumps(
            {"updated": "value"}
        )
        config.update_from_env_vars()
        assert config.load_adapter_defaults == {"updated": "value"}

    def test_configuration_with_complex_data(self):
        """Test configuration with realistic, complex configuration data."""
        complex_config = {
            "nested": {"deep": {"values": ["item1", "item2"], "numbers": [1, 2, 3]}},
            "simple": "value",
            "boolean": True,
            "null_value": None,
        }

        config = SageMakerTransformsDefaultsConfig(load_adapter_defaults=complex_config)

        assert config.load_adapter_defaults == complex_config

    def test_configuration_serialization(self):
        """Test configuration serialization and deserialization."""
        original_config = SageMakerTransformsDefaultsConfig(
            load_adapter_defaults={"param": "value"},
            create_session_defaults={"session": "config"},
        )

        # Serialize to dict and back
        config_dict = original_config.model_dump()
        restored_config = SageMakerTransformsDefaultsConfig.model_validate(config_dict)

        assert (
            restored_config.load_adapter_defaults
            == original_config.load_adapter_defaults
        )
        assert (
            restored_config.create_session_defaults
            == original_config.create_session_defaults
        )
