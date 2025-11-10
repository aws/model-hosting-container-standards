"""SageMaker-specific configuration constants."""

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SAGEMAKER_ENV_VAR_PREFIX = "SAGEMAKER_"


class SageMakerConfig(BaseModel):
    """Pydantic model for SageMaker configuration.

    Automatically loads configuration from environment variables with SAGEMAKER_ prefix.
    Example: SAGEMAKER_ENABLE_STATEFUL_SESSIONS=true -> enable_stateful_sessions=True

    Only fields defined in this model are loaded. Other SAGEMAKER_* env vars
    (like SAGEMAKER_MODEL_PATH) are ignored.

    Usage:
        # Create from environment variables
        config = SagemakerConfig.from_env()

        # Or just instantiate (automatically loads from env)
        config = SagemakerConfig()

        # Override specific values
        config = SagemakerConfig(enable_stateful_sessions=True)
    """

    model_config = ConfigDict(extra="ignore")

    # Stateful sessions configuration
    enable_stateful_sessions: bool = Field(
        default=False, description="Enable stateful sessions for the application"
    )
    sessions_expiration: int = Field(
        default=1200,  # 20 minutes
        description="Session expiration time in seconds",
        gt=0,
    )
    sessions_path: Optional[str] = Field(
        default=None,
        description="Custom path for session storage (defaults to /dev/shm or temp)",
    )

    @classmethod
    def from_env(cls) -> "SageMakerConfig":
        """Create SagemakerConfig from environment variables.

        Returns:
            SagemakerConfig instance with values loaded from SAGEMAKER_* env vars
        """
        return cls()

    @model_validator(mode="before")
    @classmethod
    def load_from_env_vars(cls, data: Any) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Extracts SAGEMAKER_* environment variables and merges with any provided data.
        Provided data takes precedence over environment variables.
        Unknown SAGEMAKER_* variables are ignored (only defined fields are loaded).
        """
        # Extract env vars with SAGEMAKER_ prefix
        env_config = {
            key[len(SAGEMAKER_ENV_VAR_PREFIX) :].lower(): val
            for key, val in os.environ.items()
            if key.startswith(SAGEMAKER_ENV_VAR_PREFIX)
        }

        # If data is provided, merge with env config (data takes precedence)
        if isinstance(data, dict):
            return {**env_config, **data}
        return env_config

    @field_validator("enable_stateful_sessions", mode="before")
    @classmethod
    def parse_bool_string(cls, v: Any) -> bool:
        """Convert string values from env vars to boolean."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1")
        return bool(v)

    @field_validator("sessions_expiration", mode="before")
    @classmethod
    def parse_int_string(cls, v: Any) -> int:
        """Convert string values from env vars to integer."""
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            return int(v)
        return int(v)


class SageMakerEnvVars:
    """SageMaker environment variable names."""

    CUSTOM_SCRIPT_FILENAME = "CUSTOM_SCRIPT_FILENAME"
    SAGEMAKER_MODEL_PATH = "SAGEMAKER_MODEL_PATH"


class SageMakerDefaults:
    """SageMaker default values."""

    SCRIPT_FILENAME = "model.py"
    SCRIPT_PATH = "/opt/ml/model/"


# SageMaker environment variable configuration mapping
SAGEMAKER_ENV_CONFIG = {
    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: {
        "default": SageMakerDefaults.SCRIPT_FILENAME,
        "description": "Custom script filename to load (default: model.py)",
    },
    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: {
        "default": SageMakerDefaults.SCRIPT_PATH,
        "description": "SageMaker model path directory (default: /opt/ml/model/)",
    },
}
