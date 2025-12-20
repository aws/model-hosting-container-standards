from typing import Any, Dict, Optional

from fastapi import Request
from pydantic import BaseModel
from typing_extensions import TypedDict, NotRequired

from ...common.handler import handler_registry
from ...logging_config import logger
from ..lora.constants import LoRAHandlerType
from .transforms.load_adapter import LoadLoraApiTransform
from .transforms.unload_adapter import UnloadLoraApiTransform


class SageMakerLoadLoRAEngineRequestPaths(TypedDict):
    name: str
    src: str
    preload: NotRequired[Optional[str]]
    pinned: NotRequired[Optional[str]]


def resolve_lora_transform(
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


def create_lora_decorator(handler_type: str):
    def lora_decorator_with_params(
        engine_request_paths: Optional[Dict[str, Any]] = None,
        engine_request_model_cls: BaseModel = None,
        engine_request_defaults: Optional[Dict[str, Any]] = None,
    ):
        def lora_decorator(original_func):
            lora_transform = resolve_lora_transform(
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


def register_load_adapter_handler(
    engine_request_lora_name_path: str,
    engine_request_lora_src_path: str,
    engine_request_lora_preload_path: Optional[str] = None,
    engine_request_lora_pinned_path: Optional[str] = None,
    engine_request_paths: Optional[SageMakerLoadLoRAEngineRequestPaths] = None,
    engine_request_model_cls: Optional[BaseModel] = None,
    engine_request_defaults: Optional[Dict[str, Any]] = None,
):
    logger.info("Registering load adapter handler")
    logger.debug(
        f"Handler parameters - name_path: {engine_request_lora_name_path}, src_path: {engine_request_lora_src_path}"
    )

    if not engine_request_paths:
        engine_request_paths = {
            "name": engine_request_lora_name_path,
            "src": engine_request_lora_src_path,
            "preload": engine_request_lora_preload_path,
            "pinned": engine_request_lora_pinned_path,
        }
        logger.debug(
            f"Created engine_request_paths from individual parameters: {engine_request_paths}"
        )
    else:
        logger.warning(
            "Both `engine_request_paths` and the individual path arguments are provided. "
            "Using the `engine_request_paths` argument."
        )
        logger.debug(f"Using provided engine_request_paths: {engine_request_paths}")

    return create_lora_decorator(LoRAHandlerType.REGISTER_ADAPTER.value)(
        engine_request_paths,
        engine_request_defaults=engine_request_defaults,
        engine_request_model_cls=engine_request_model_cls,
    )


def register_unload_adapter_handler(
    engine_request_lora_name_path: str,
    engine_request_model_cls: Optional[BaseModel] = None,
    engine_request_defaults: Optional[Dict[str, Any]] = None,
):
    logger.info("Registering unload adapter handler")
    logger.debug(f"Handler parameters - name_path: {engine_request_lora_name_path}")

    return create_lora_decorator(LoRAHandlerType.UNREGISTER_ADAPTER.value)(
        {
            "name": engine_request_lora_name_path,
        },
        engine_request_defaults=engine_request_defaults,
        engine_request_model_cls=engine_request_model_cls,
    )
