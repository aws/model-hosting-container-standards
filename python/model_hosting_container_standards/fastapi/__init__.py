"""FastAPI-specific configuration and utilities."""


# FastAPI environment variables
class EnvVars:
    """FastAPI environment variable names."""

    CUSTOM_FASTAPI_PING_HANDLER = "CUSTOM_FASTAPI_PING_HANDLER"
    CUSTOM_FASTAPI_INVOCATION_HANDLER = "CUSTOM_FASTAPI_INVOCATION_HANDLER"


# FastAPI environment variable configuration mapping
ENV_CONFIG = {
    # FastAPI handler configuration
    EnvVars.CUSTOM_FASTAPI_PING_HANDLER: {
        "default": None,
        "description": "Custom ping handler specification (function spec or router URL)",
    },
    EnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER: {
        "default": None,
        "description": "Custom invocation handler specification (function spec or router URL)",
    },
}
