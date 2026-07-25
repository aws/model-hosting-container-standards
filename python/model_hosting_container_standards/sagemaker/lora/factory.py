from typing import Any, Dict, Optional

from fastapi import Request
from pydantic import BaseModel

from ...common.handler.registry import handler_registry
from ...common.transforms.base_factory import create_transform_decorator
from ...logging_config import logger
from .constants import LoRAHandlerType
from .transforms import resolve_lora_transform
from .transforms2.load_adapter import LoadLoraApiTransform
from .transforms2.unload_adapter import UnloadLoraApiTransform


def create_lora_transform_decorator(handler_type: str):
    return create_transform_decorator(handler_type, resolve_lora_transform)


def resolve_lora_transform2(
    handler_type: str,
    original_func,
    engine_request_paths: Dict[str, Any],
    engine_request_model_cls: BaseModel,
    engine_request_defaults: Optional[Dict[str, Any]] = None,
):
    logger.debug(f"Resolving LoRA transform for handler_type: {handler_type}")

    if handler_type == LoRAHandlerType.REGISTER_ADAPTER.value:
        logger.debug("Creating LoadLoraApiTransform instance")
        return LoadLoraApiTransform(
            original_func,
            engine_request_paths,
            engine_request_model_cls,
            engine_request_defaults,
        )
    elif handler_type == LoRAHandlerType.UNREGISTER_ADAPTER.value:
        logger.debug("Creating UnloadLoraApiTransform instance")
        return UnloadLoraApiTransform(
            original_func,
            engine_request_paths,
            engine_request_model_cls,
            engine_request_defaults,
        )
    else:
        logger.error(f"Invalid handler_type provided: {handler_type}")
        raise ValueError(f"Invalid handler_type: {handler_type}")


def create_lora_transform2_decorator(handler_type: str):
    def lora_decorator_with_params(
        engine_request_paths: Optional[Dict[str, Any]] = None,
        engine_request_model_cls: BaseModel = None,
        engine_request_defaults: Optional[Dict[str, Any]] = None,
    ):
        def lora_decorator(original_func):
            lora_transform = resolve_lora_transform2(
                handler_type,
                original_func,
                engine_request_paths,
                engine_request_model_cls,
                engine_request_defaults,
            )

            async def lora_transform_wrapper(raw_request: Request):
                return await lora_transform.transform(raw_request)

            handler_registry.set_handler(handler_type, lora_transform_wrapper)
            logger.info(
                f"[{handler_type.upper()}] Registered transform handler for {original_func.__name__}"
            )
            return lora_transform_wrapper

        return lora_decorator

    return lora_decorator_with_params
