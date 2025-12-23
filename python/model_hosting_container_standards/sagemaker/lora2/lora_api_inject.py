from http import HTTPStatus
from typing import Any, Dict, Literal, Optional

import json
from fastapi import Request
from fastapi.exceptions import HTTPException
from json import JSONDecodeError
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

from ...logging_config import logger
from ...common.fastapi.utils import serialize_request
from ...common.handler.registry import handler_registry
from ...common.transforms.utils import set_value
from ..lora.constants import LoRAHandlerType, SageMakerLoRAApiHeader


class SageMakerRequest(BaseModel):
    body: Dict[str, Any] = {}
    headers: Dict[str, Any] = {}
    query_params: Dict[str, Any] = {}
    path_params: Dict[str, Any] = {}


class InjectDefinition(BaseModel):
    path: str
    mode: Literal["append", "prepend", "replace"] = "replace"
    separator: Optional[str] = None


class SageMakerLoRAInjectEngineRequestPaths(TypedDict):
    adapter_id: InjectDefinition

async def validate_request(raw_request: Request, engine_request_model_cls: Optional[BaseModel] = None) -> Dict[str, Any]:
    try:
        request_dict = await raw_request.json()
        if engine_request_model_cls:
            engine_request_model_cls.model_validate(request_dict, extra="ignore")
        return request_dict
    except JSONDecodeError as e:
        logger.error(f"Failed to parse request body as JSON: {e}")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Failed to parse request body as JSON",
        )
    except ValidationError as e:
        logger.error(f"Request body validation failed: {e}")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Failed to parse request body as JSON",
        )

def get_sagemaker_values(raw_request: Request):
    adapter_id = raw_request.headers.get(SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER)
    if adapter_id:
        return {
            "adapter_id": adapter_id
        }
    return {}

def _apply_to_raw_request(
        raw_request: Request, transformed_request: Dict[str, Any]
    ) -> Request:
        if transformed_request.get("headers"):
            new_headers = raw_request.headers.mutablecopy()
            for header_key, header_value in transformed_request.get("headers").items():
                new_headers[header_key] = header_value
            raw_request._headers = new_headers
        if transformed_request.get("query_params"):
            raw_request.query_params = transformed_request.get("query_params")
        if transformed_request.get("body"):
            raw_request._body = json.dumps(transformed_request.get("body")).encode()
        if transformed_request.get("path_params"):
            raw_request.path_params = transformed_request.get("path_params")
        return raw_request

def inject(engine_request_paths: SageMakerLoRAInjectEngineRequestPaths, raw_request: Request, request_dict: Dict[str, Any]):
    transformed_request = serialize_request(request_dict, raw_request)
    sagemaker_values = get_sagemaker_values(raw_request)
    for sagemaker_param, inject_definition in engine_request_paths.items():
        engine_path = inject_definition.path
        value = sagemaker_values.get(sagemaker_param)
        if value:
            transformed_request = set_value(
                transformed_request,
                engine_path,
                value,
                create_parent=True,
                max_create_depth=None,
                mode=inject_definition.mode,
                separator=inject_definition.separator
            )
    logger.info(transformed_request)
    return transformed_request


def lora_inject_decorator_with_params(
        engine_request_paths: SageMakerLoRAInjectEngineRequestPaths,
        engine_request_model_cls: Optional[BaseModel] = None,
):
    def lora_inject_decorator(func):    
        async def lora_inject(raw_request: Request):
            request_dict = await validate_request(raw_request, engine_request_model_cls)
            transformed_request = None
            if raw_request.headers.get(SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER):
                # only transform request if sagemaker adapter id header is present
                transformed_request = inject(engine_request_paths, raw_request, request_dict)
                logger.info(f"{transformed_request=}")
                raw_request = _apply_to_raw_request(raw_request, transformed_request)
            if engine_request_model_cls:
                raw_response = await func(
                    engine_request_model_cls.model_validate(
                        transformed_request or request_dict,
                        extra="ignore"
                    ),
                    raw_request
                )
            else:
                raw_response = await func(raw_request)
            return raw_response
        handler_registry.set_handler(LoRAHandlerType.INJECT_ADAPTER_ID, lora_inject)
        return lora_inject
    return lora_inject_decorator

def create_lora_inject(adapter_path: str, mode: str = "replace", separator: str = None, engine_request_model_cls: Optional[BaseModel] = None):
    inject_definition = InjectDefinition(path=adapter_path, mode=mode, separator=separator)
    engine_request_paths = {
        "adapter_id": inject_definition
    }
    return lora_inject_decorator_with_params(engine_request_paths, engine_request_model_cls)
    

