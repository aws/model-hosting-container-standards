"""SageMaker integration decorators."""

from typing import List

from fastapi import FastAPI

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
from .sagemaker_loader import SageMakerFunctionLoader
from .sagemaker_router import create_sagemaker_router

# SageMaker decorator instances - created using utility functions

# Override decorators - immediately register customer handlers
register_ping_handler = register_handler("ping")
register_invocation_handler = register_handler("invoke")
ping = override_handler("ping")
invoke = override_handler("invoke")


# Transform decorators - for LoRA handling
def register_load_adapter_handler(request_shape: dict, response_shape: dict = {}):
    # TODO: validate and preprocess request shape
    # TODO: validate and preprocess response shape
    return create_lora_transform_decorator(LoRAHandlerType.REGISTER_ADAPTER)(
        request_shape, response_shape
    )


def register_unload_adapter_handler(request_shape: dict, response_shape: dict = {}):
    # TODO: validate and preprocess request shape
    # TODO: validate and preprocess response shape
    return create_lora_transform_decorator(LoRAHandlerType.UNREGISTER_ADAPTER)(
        request_shape, response_shape
    )


def inject_adapter_id(request_shape: dict, response_shape: dict = {}):
    # validate and preprocess request shape
    if len(request_shape.keys()) > 1:
        raise ValueError(f"Invalid {request_shape=} for register_adapter_id")
    if response_shape:
        logger.warning(
            f"Handler type {LoRAHandlerType.INJECT_ADAPTER_ID} does not take response_shape, but {response_shape=}"
        )
    for k in request_shape.keys():
        # Overwrite placeholder value
        request_shape[k] = f'headers."{SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER}"'
    return create_lora_transform_decorator(LoRAHandlerType.INJECT_ADAPTER_ID)(
        request_shape, response_shape
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
    "ping",
    "invoke",
    "register_load_adapter_handler",
    "register_unload_adapter_handler",
    "register_handler",
    "register_ping_handler",
    "register_invocation_handler",
    "inject_adapter_id",
    "bootstrap",
    "RouteConfig",
]
