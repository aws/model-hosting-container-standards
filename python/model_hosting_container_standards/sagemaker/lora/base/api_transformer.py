import abc
from http import HTTPStatus
from typing import Any, Dict, Optional

from ....logging_config import logger
from ..models.transformer import BaseLoRATransformRequestOutput
from ..utils import (
    get_adapter_name_from_request,
    get_request_data_for_jmespath,
)

from fastapi import Request, Response
import jmespath
from pydantic import BaseModel


def _compile_jmespath_expressions(shape: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively compile JMESPath expressions in the shape dictionary.

    :param Dict[str, Any] shape: Dictionary containing JMESPath expressions to compile
    :return Dict[str, Any]: Dictionary with compiled JMESPath expressions
    """
    compiled_shape = {}
    for key, value in shape.items():
        if isinstance(value, str):
            # Compile the JMESPath expression
            compiled_shape[key] = jmespath.compile(value)
        elif isinstance(value, dict):
            # Recursively compile nested dictionaries
            compiled_shape[key] = _compile_jmespath_expressions(value)
        else:
            logger.warning(f"Request/response mapping must be a dictionary of strings (nested allowed), not {type(value)}. This value will be ignored.")
            
    return compiled_shape


class BaseLoRAApiTransformer(abc.ABC):
    def __init__(self, request_shape: Dict[str, Any], response_shape: Dict[str, Any] = {}):
        """
        :param Dict[str, Any] request_shape:
        :param Dict[str, Any] response_shape:
        """
        self._request_shape = _compile_jmespath_expressions(request_shape)
        self._response_shape = _compile_jmespath_expressions(response_shape)

    def _transform(self, source_data: Dict[str, Any], target_shape: Dict[str, Any]) -> Dict[str, Any]:
        transformed_request = {}
        for target_key, nested_or_compiled in target_shape.items():
            if isinstance(nested_or_compiled, jmespath.parser.ParsedResult):
                value = nested_or_compiled.search(source_data)
                transformed_request[target_key] = value
            elif isinstance(nested_or_compiled, dict):
                transformed_request[target_key] = self._transform(source_data, nested_or_compiled)
            else:
                logger.warning(f"Request/response mapping must be a dictionary of strings (nested allowed), not {type(nested_or_compiled)}. This value will be ignored.")
        return transformed_request

    @abc.abstractmethod
    def transform_request(self, request: Optional[BaseModel], raw_request: Request) -> BaseLoRATransformRequestOutput:
        raise NotImplementedError()

    def _transform_request(self, request: Optional[BaseModel], raw_request: Request) -> Dict[str, Any]:
        request_data = get_request_data_for_jmespath(request, raw_request)
        return self._transform(request_data, self._request_shape)

    def transform_response(self, response: Response, transform_request_output):
        adapter_name = get_adapter_name_from_request(transform_request_output)
        if response.status_code == HTTPStatus.OK:
            return self._transform_ok_response(response, adapter_name)
        else:
            return self._transform_error_response(response, adapter_name)

    def _transform_ok_response(self, response: Response, adapter_name: str):
        raise NotImplementedError()

    def _transform_error_response(self, response: Response, adapter_name: str):
        raise NotImplementedError()