"""SageMaker-specific configuration constants."""

import os

SAGEMAKER_ENV_VAR_PREFIX = "OPTION_"


def get_configs_from_env_vars():
    sagemaker_args = {
        key[len(SAGEMAKER_ENV_VAR_PREFIX) :].lower(): val
        for key, val in os.environ.items()
        if key.startswith(SAGEMAKER_ENV_VAR_PREFIX)
    }
    return sagemaker_args


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
