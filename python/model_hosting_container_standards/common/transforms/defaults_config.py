import json
import os
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, model_validator

SAGEMAKER_TRANSFORMS_ENV_VAR_PREFIX = "SAGEMAKER_TRANSFORMS_"


class SageMakerTransformsDefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # LoRA
    load_adapter_defaults: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default values for load_adapter transform request",
    )
    unload_adapter_defaults: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default values for unload_adapter transform request",
    )

    # Engine - Custom session handlers
    create_session_defaults: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default values for create_session transform request",
    )
    close_session_defaults: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default values for close_session transform request",
    )

    @classmethod
    def from_env(cls) -> "SageMakerTransformsDefaultsConfig":
        """Create SageMakerTransformsDefaultsConfig from environment variables.

        Returns:
            SageMakerTransformsDefaultsConfig instance with values loaded from SAGEMAKER_TRANSFORMS_* env vars
        """
        return cls()

    @model_validator(mode="before")
    @classmethod
    def load_from_env_vars(cls, data: Any) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Extracts SAGEMAKER_TRANSFORMS_* environment variables and merges with any provided data.
        Provided data takes precedence over environment variables.
        Unknown SAGEMAKER_TRANSFORMS_* variables are ignored (only defined fields are loaded).
        """
        # Extract env vars with SAGEMAKER_TRANSFORMS_ prefix
        env_config = {
            key[len(SAGEMAKER_TRANSFORMS_ENV_VAR_PREFIX) :].lower(): json.loads(val)
            for key, val in os.environ.items()
            if key.startswith(SAGEMAKER_TRANSFORMS_ENV_VAR_PREFIX)
        }

        # If data is provided, merge with env config (data takes precedence)
        if isinstance(data, dict):
            return {**env_config, **data}
        return env_config

    def update_from_env_vars(self) -> None:
        """Update configuration from environment variables.

        Updates the current instance with values from SAGEMAKER_TRANSFORMS_* environment variables.
        """
        env_config = self.from_env()
        self.__dict__.update(env_config.__dict__)


_transform_defaults_config = SageMakerTransformsDefaultsConfig.from_env()
