"""SageMaker integration decorators."""

from typing import Dict, List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel

# Import routing utilities (generic)
from ..common.fastapi.routing import RouteConfig, safe_include_router
from ..common.handler.decorators import override_handler, register_handler
from ..logging_config import logger

# Import the real resolver functions
from .handler_resolver import register_sagemaker_overrides

# Import LoRA Handler factory and handler types
from .lora import (
    LoRAHandlerType,
    SageMakerLoRAApiHeader,
    create_lora_transform_decorator,
)
from .lora.models import AppendOperation
from .sagemaker_loader import SageMakerFunctionLoader
from .sagemaker_router import create_sagemaker_router
from .sessions import create_session_transform_decorator
from .sessions.close_session import _register_close_session_handler
from .sessions.create_session import _register_create_session_handler
from .sessions.models import SageMakerSessionHeader

# SageMaker decorator instances - created using utility functions

# Override decorators - immediately register customer handlers
register_ping_handler = register_handler("ping")
register_invocation_handler = register_handler("invoke")
custom_ping_handler = override_handler("ping")
custom_invocation_handler = override_handler("invoke")


# Transform decorators - for LoRA handling
def register_load_adapter_handler(
    request_shape: dict, response_shape: Optional[dict] = None
):
    # TODO: validate and preprocess request shape
    # TODO: validate and preprocess response shape
    return create_lora_transform_decorator(LoRAHandlerType.REGISTER_ADAPTER)(
        request_shape, response_shape
    )


def register_unload_adapter_handler(
    request_shape: dict, response_shape: Optional[dict] = None
):
    # TODO: validate and preprocess request shape
    # TODO: validate and preprocess response shape
    return create_lora_transform_decorator(LoRAHandlerType.UNREGISTER_ADAPTER)(
        request_shape, response_shape
    )


def inject_adapter_id(
    adapter_path: str, append: bool = False, separator: Optional[str] = None
):
    """Create a decorator that injects adapter ID from SageMaker headers into request body.

    This decorator extracts the adapter identifier from the SageMaker LoRA API header
    (X-Amzn-SageMaker-Adapter-Identifier) and injects it into the specified path
    within the request body using JMESPath syntax.

    Args:
        adapter_path: The JSON path where the adapter ID should be injected in the
                     request body (e.g., "model", "body.model.lora_name", etc.).
                     Supports both simple keys and nested paths using dot notation.
        append: If True, appends the adapter ID to the existing value at adapter_path
                using the specified separator. If False (default), replaces the value.
                When True, separator parameter is required.
                Example with append=True and separator=":":
                    {"model": "base-model"} -> {"model": "base-model:adapter-123"}
        separator: The separator to use when append=True. Required when append=True.
                  Common values include ":", "-", "_", etc.

    Returns:
        A decorator function that can be applied to FastAPI route handlers to
        automatically inject adapter IDs from headers into the request body.

    Raises:
        ValueError: If adapter_path is empty or not a string, or if append=True
                   but separator is not provided.

    Note:
        This is a transform-only decorator that does not create its own route.
        It must be applied to existing route handlers.
    """
    # validate and preprocess
    if not adapter_path:
        logger.error("adapter_path cannot be empty")
        raise ValueError("adapter_path cannot be empty")
    if not isinstance(adapter_path, str):
        logger.error("adapter_path must be a string")
        raise ValueError("adapter_path must be a string")
    if append and separator is None:
        logger.error(f"separator must be provided when {append=}")
        raise ValueError(f"separator must be provided when {append=}")
    if separator and not append:
        logger.error(f"separator is specified {separator} but {append=}")
        raise ValueError(f"separator is specified {separator} but {append=}")

    # create request_shape with operation encoding
    request_shape: Dict[str, Union[str, AppendOperation]] = {}
    header_expr = f'headers."{SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER}"'

    if append:
        # Encode append operation as a Pydantic model
        request_shape[adapter_path] = AppendOperation(
            separator=separator, expression=header_expr
        )
    else:
        # Default replace behavior (backward compatible)
        request_shape[adapter_path] = header_expr

    return create_lora_transform_decorator(LoRAHandlerType.INJECT_ADAPTER_ID)(
        request_shape=request_shape, response_shape=None
    )


def stateful_session_manager(engine_request_session_id_path: Optional[str] = None):
    """Create a decorator for session-based sticky routing.

    This decorator enables stateful session management for regular invocation requests,
    allowing the session ID to be injected into the request body for stateful inference.

    Args:
        engine_request_session_id_path: Optional target path in the request body where
                                        the session ID will be injected. The session ID
                                        is extracted from the SageMaker session header and
                                        placed at this path in the request sent to the engine.

                                        Examples: "session_id", "metadata.session_id"

                                        If None, session management is enabled but the
                                        session ID is not injected into the request body.

    Returns:
        A decorator that can be applied to route handlers to enable session management
    """
    request_shape = {}
    if engine_request_session_id_path:
        request_shape[engine_request_session_id_path] = (
            f'headers."{SageMakerSessionHeader.SESSION_ID}"'
        )
    return create_session_transform_decorator()(
        request_shape=request_shape, response_shape={}
    )


def register_create_session_handler(
    engine_response_session_id_path: str,
    engine_request_model_cls: Optional[BaseModel] = None,
):
    """Register a handler for session creation with custom request/response transformations.

    This decorator creates a session handler that transforms incoming requests to include
    the session ID and extracts the session ID from the engine's response.

    Returns:
        A decorator that can be applied to engine-specific session creation handlers.
    """
    return _register_create_session_handler(
        engine_response_session_id_path=engine_response_session_id_path,
        engine_request_model_cls=engine_request_model_cls,
    )


def register_close_session_handler(
    engine_request_session_id_path: str,
    engine_request_model_cls: Optional[BaseModel] = None,
):
    """Register a handler for session closure with custom request transformations.

    This decorator creates a session handler that transforms incoming requests to include
    the session ID for proper session cleanup.

    Returns:
        A decorator that can be applied to engine-specific session closure handlers.

    Raises:
        ValueError: If engine_request_session_id_path is None or empty.

    Note:
        If engine_request_session_id_path appears in additional_request_shape, it will be
        overwritten to ensure the session ID is properly injected.
    """
    if not engine_request_session_id_path:
        raise ValueError(
            "engine_request_session_id_path is required for close_session handler. "
            "The engine needs to know which session to close."
        )

    return _register_close_session_handler(
        engine_request_session_id_path=engine_request_session_id_path,
        engine_request_model_cls=engine_request_model_cls,
    )


def bootstrap(app: FastAPI) -> FastAPI:
    """Configure a FastAPI application with SageMaker functionality.

    This function sets up all necessary SageMaker integrations on the provided
    FastAPI application, including:
    - Container standards middlewares
    - All SageMaker routes (/ping, /invocations, LoRA routes, etc.)

    Args:
        app: The FastAPI application instance to configure

    Returns:
        The configured FastAPI app

    Note:
        All handlers must be registered before calling this function. Handlers
        registered after this call will not be automatically mounted.
    """
    from ..common.fastapi.middleware.core import (
        load_middlewares as core_load_middlewares,
    )

    logger.info("Starting SageMaker bootstrap process")
    logger.debug(f"Bootstrapping FastAPI app: {app.title or 'unnamed'}")

    # Load container standards middlewares with SageMaker function loader
    sagemaker_function_loader = SageMakerFunctionLoader.get_function_loader()
    core_load_middlewares(app, sagemaker_function_loader)

    # Create and include the unified SageMaker router
    register_sagemaker_overrides()
    sagemaker_router = create_sagemaker_router()
    safe_include_router(app, sagemaker_router)

    logger.info("SageMaker bootstrap completed successfully")
    return app


__all__: List[str] = [
    "custom_ping_handler",
    "custom_invocation_handler",
    "register_load_adapter_handler",
    "register_unload_adapter_handler",
    "register_handler",
    "register_ping_handler",
    "register_invocation_handler",
    "inject_adapter_id",
    "stateful_session_manager",
    "bootstrap",
    "RouteConfig",
]
