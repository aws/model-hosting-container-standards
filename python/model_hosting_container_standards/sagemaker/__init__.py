"""SageMaker integration decorators."""

import logging
from typing import Any, Callable, List

from ..registry import handler_registry
from ..utils import create_override_decorator, create_register_decorator

# Import the real resolver functions
from .handler_resolver import get_invoke_handler, get_ping_handler
from .sagemaker_loader import SageMakerFunctionLoader

# Import LoRA Handler factory and handler types
from .lora import create_transform_decorator, LoRAHandlerType, SageMakerLoRAApiHeader

logger = logging.getLogger(__name__)

# Use resolver functions for decorators
_get_ping_handler = get_ping_handler
_get_invocation_handler = get_invoke_handler


# SageMaker decorator instances - created using utility functions

# Override decorators - immediately register customer handlers
ping = create_override_decorator("ping", handler_registry)
invoke = create_override_decorator("invoke", handler_registry)

# Register decorators - created using create_register_decorator
register_ping_handler = create_register_decorator(
    "ping", _get_ping_handler, handler_registry
)
register_invocation_handler = create_register_decorator(
    "invoke", _get_invocation_handler, handler_registry
)

# Transform decorators - for LoRA handling
def register_load_adapter_handler(request_shape: dict, response_shape: dict = {}):
    # TODO: validate and preprocess request shape
    # TODO: validate and preprocess response shape
    return create_transform_decorator(LoRAHandlerType.REGISTER_ADAPTER)(request_shape, response_shape)

def register_unload_adapter_handler(request_shape: dict, response_shape: dict = {}): 
    # TODO: validate and preprocess request shape
    # TODO: validate and preprocess response shape
    return create_transform_decorator(LoRAHandlerType.UNREGISTER_ADAPTER)(request_shape, response_shape)

def register_adapter_id_handler(request_shape: dict, response_shape: dict = {}):
    # validate and preprocess request shape
    if len(request_shape.keys()) > 1:
        raise ValueError(f"Invalid {request_shape=} for register_adapter_id")
    if response_shape:
        logger.warning(f"Handler type {LoRAHandlerType.ADAPTER_ID} does not take response_shape, but {response_shape=}")
    for k in request_shape.keys():
        # Overwrite placeholder value
        request_shape[k] = f"headers.\"{SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER.value}\""
    return create_transform_decorator(LoRAHandlerType.ADAPTER_ID)(request_shape, response_shape)

__all__: List[str] = [
    "ping",
    "invoke",
    "register_ping_handler",
    "register_invocation_handler",
    "get_ping_handler",
    "get_invoke_handler",
    "register_load_adapter_handler",
    "register_unload_adapter_handler",
    "register_adapter_id",
]
