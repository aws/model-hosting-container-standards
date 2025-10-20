"""FastAPI-specific configuration constants."""


class FastAPIEnvVars:
    """FastAPI environment variable names."""

    # Handler environment variables
    CUSTOM_FASTAPI_PING_HANDLER = "CUSTOM_FASTAPI_PING_HANDLER"
    CUSTOM_FASTAPI_INVOCATION_HANDLER = "CUSTOM_FASTAPI_INVOCATION_HANDLER"

    # Middleware environment variables
    CUSTOM_PRE_PROCESS = "CUSTOM_PRE_PROCESS"
    CUSTOM_POST_PROCESS = "CUSTOM_POST_PROCESS"
    CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE = "CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE"
    CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS = (
        "CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS"
    )


# FastAPI environment variable configuration mapping
FASTAPI_ENV_CONFIG = {
    # Handler configuration
    FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER: {
        "default": None,
        "description": "Custom ping handler specification (function spec or router URL)",
    },
    FastAPIEnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER: {
        "default": None,
        "description": "Custom invocation handler specification (function spec or router URL)",
    },
    # Middleware configuration
    FastAPIEnvVars.CUSTOM_PRE_PROCESS: {
        "default": None,
        "description": "Custom pre-process middleware specification (filename.py:function | module.name:function | module.name:Class.method)",
    },
    FastAPIEnvVars.CUSTOM_POST_PROCESS: {
        "default": None,
        "description": "Custom post-process middleware specification (filename.py:function | module.name:function | module.name:Class.method)",
    },
    FastAPIEnvVars.CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE: {
        "default": None,
        "description": "Custom throttle middleware specification (filename.py:function | module.name:function | module.name:Class.method)",
    },
    FastAPIEnvVars.CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS: {
        "default": None,
        "description": "Custom pre/post process middleware specification (filename.py:function | module.name:function | module.name:Class.method)",
    },
}
