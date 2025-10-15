from typing import Optional

from .models import BaseLoRATransformRequestOutput
from .constants import SageMakerLoRAApiHeader, RequestField

from fastapi import Request


def get_adapter_alias_from_request_header(raw_request: Request) -> Optional[str]:
    if raw_request.headers:
        adapter_alias = raw_request.headers.get(SageMakerLoRAApiHeader.ADAPTER_ALIAS)
        if adapter_alias:
            return adapter_alias
    return None

def get_adapter_name_from_request_path(raw_request: Request) -> Optional[str]:
    if raw_request.path_params:
        adapter_name = raw_request.path_params.get(RequestField.ADAPTER_NAME)
        if adapter_name:
            return adapter_name
    return None

def get_adapter_name_from_request(transform_request_output: BaseLoRATransformRequestOutput) -> Optional[str]:
    """Extract the LoRA adapter name from various sources in the request.

    Searches for the adapter name in multiple locations with the following priority:
    1. adapter_name path parameter
    2. adapter_name from transform output (this is whatever is specified in request body)
    3. ADAPTER_IDENTIFIER header

    Does not check for adapter alias.

    :param BaseLoRATransformRequestOutput transform_request_output: Request transformation output
    :return Optional[str]: The adapter name if found, None otherwise
    """
    raw_request = transform_request_output.raw_request

    # Priority 1: Check path parameters for adapter_name
    adapter_name = get_adapter_name_from_request_path(raw_request)
    if adapter_name:
        return adapter_name

    # Priority 2: Check if adapter name was set during transformation
    if transform_request_output.adapter_name:
        return transform_request_output.adapter_name

    # Priority 3: Fallback to ADAPTER_IDENTIFIER header
    if raw_request.headers:
        adapter_identifier = raw_request.headers.get(SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER)
        if adapter_identifier:
            return adapter_identifier

    # No adapter identifier found in any location
    return None  # TODO: determine what to do in the case request has no adapter id
