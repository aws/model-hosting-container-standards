"""SageMaker integration decorators."""

from typing import Any, Dict, List, Literal, Optional, Union

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
from .lora2 import lora_factory as lora2_factory
from .lora2.lora_api_inject import create_lora_inject
from .lora.models import AppendOperation
from .sagemaker_loader import SageMakerFunctionLoader
from .sagemaker_router import create_sagemaker_router
from .sessions import create_session_transform_decorator

# SageMaker decorator instances - created using utility functions

# Override decorators - immediately register customer handlers
register_ping_handler = register_handler("ping")
register_invocation_handler = register_handler("invoke")
custom_ping_handler = override_handler("ping")
custom_invocation_handler = override_handler("invoke")


# Transform decorators - for LoRA handling
def register_load_adapter_handler(
    engine_request_lora_name_path: Optional[str] = None,
    engine_request_lora_src_path: Optional[str] = None,
    engine_request_model_cls: Optional[BaseModel] = None,
    engine_request_defaults: Optional[Dict[str, Any]] = None,
    request_shape: Optional[dict] = None,
    response_shape: Optional[dict] = None,
):
    if (
        not (engine_request_lora_name_path and engine_request_lora_src_path)
        and not request_shape
    ):
        logger.error(
            "Either `engine_request_lora_name_path` and `engine_request_lora_src_path` or `request_shape` must be provided."
        )
        raise ValueError(
            "Either `engine_request_lora_name_path` and `engine_request_lora_src_path` or `request_shape` must be provided."
        )
    elif request_shape and not (
        engine_request_lora_name_path and engine_request_lora_src_path
    ):
        logger.warning(
            "The `request_shape` argument is deprecated and will be removed in a future release. "
            "Please use `engine_request_lora_name_path` and `engine_request_lora_src_path` instead."
        )
        # TODO: validate and preprocess request shape
        # TODO: validate and preprocess response shape
        return create_lora_transform_decorator(LoRAHandlerType.REGISTER_ADAPTER)(
            request_shape, response_shape
        )
    elif engine_request_lora_name_path and engine_request_lora_src_path:
        return lora2_factory.register_load_adapter_handler(
            engine_request_lora_name_path=engine_request_lora_name_path,
            engine_request_lora_src_path=engine_request_lora_src_path,
            engine_request_model_cls=engine_request_model_cls,
            engine_request_defaults=engine_request_defaults,
        )
    else:
        logger.error(
            "Either `engine_request_lora_name_path` and `engine_request_lora_src_path` or `request_shape` must be provided."
        )


def register_unload_adapter_handler(
    engine_request_lora_name_path: Optional[str] = None,
    engine_request_model_cls: Optional[BaseModel] = None,
    engine_request_defaults: Optional[Dict[str, Any]] = None,
    request_shape: Optional[dict] = None,
    response_shape: Optional[dict] = None,
):
    if not engine_request_lora_name_path and not request_shape:
        logger.error(
            "Either `engine_request_lora_name_path` or `request_shape` must be provided."
        )
        raise ValueError(
            "Either `engine_request_lora_name_path` or `request_shape` must be provided."
        )
    elif request_shape and not engine_request_lora_name_path:
        logger.warning(
            "The `request_shape` argument is deprecated and will be removed in a future release. "
            "Please use `engine_request_lora_name_path` instead."
        )
        # TODO: validate and preprocess request shape
        # TODO: validate and preprocess response shape
        return create_lora_transform_decorator(LoRAHandlerType.UNREGISTER_ADAPTER)(
            request_shape, response_shape
        )
    elif engine_request_lora_name_path:
        return lora2_factory.register_unload_adapter_handler(
            engine_request_lora_name_path=engine_request_lora_name_path,
            engine_request_model_cls=engine_request_model_cls,
            engine_request_defaults=engine_request_defaults,
        )


def inject_adapter_id(
    adapter_path: str, mode: Optional[Literal["append", "prepend", "replace"]] = None, append: bool = False, separator: Optional[str] = None
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
    if mode:
        if mode != "replace" and not separator:
            logger.error(f"separator must be provided when {mode=}")
            raise ValueError(f"separator must be provided when {mode=}")
        if mode == "replace" and separator:
            logger.error(f"separator is specified {separator} but {mode=}")
            raise ValueError(f"separator is specified {separator} but {mode=}")
        return create_lora_inject(adapter_path, mode, separator)
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


def stateful_session_manager():
    """Create a decorator for session-based sticky routing.

    This decorator enables stateful session management without JMESPath transformations.
    Pass empty dicts to enable transform infrastructure (for intercept functionality)
    without requiring JMESPath expressions.

    Returns:
        A decorator that can be applied to route handlers to enable session management
    """
    return create_session_transform_decorator()(request_shape={}, response_shape={})


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
