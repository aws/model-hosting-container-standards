from typing import Any, Dict, Optional

from .models.transformer import BaseLoRATransformRequestOutput
from .constants import SageMakerLoRAApiHeader, RequestField

from fastapi import Request
from pydantic import BaseModel


def get_request_data_for_jmespath(request: Optional[BaseModel], raw_request: Request) -> Dict[str, Any]:
    return {
        "body": request.model_dump() if request else None,
        "headers": raw_request.headers,
        "query_params": raw_request.query_params,
        "path_params": raw_request.path_params,
    }

def get_adapter_name_from_request(transform_request_output: BaseLoRATransformRequestOutput) -> str:
    raw_request = transform_request_output.raw_request
    # Check if adapter alias is in the header
    if raw_request.headers and raw_request.headers.get(SageMakerLoRAApiHeader.ADAPTER_ALIAS):
        return raw_request.headers.get(SageMakerLoRAApiHeader.ADAPTER_ALIAS)
    
    # Check if adapter_name is in the path
    if raw_request.path_params and raw_request.path_params.get(RequestField.ADAPTER_NAME):
        return raw_request.path_params.get(RequestField.ADAPTER_NAME)

    if transform_request_output.adapter_name:
        return transform_request_output.adapter_name

    if raw_request.headers and raw_request.headers.get(SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER):
        return raw_request.headers.get(SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER)

    return None  # TODO: determine what to do in the case request has no adapter id