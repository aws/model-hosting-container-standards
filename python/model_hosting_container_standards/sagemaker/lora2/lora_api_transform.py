from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Any, Dict, Callable, Optional, TypedDict

from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from ...common.transforms.utils import set_value
from ...logging_config import logger
from ..lora.utils import get_adapter_alias_from_request_header


class TransformedRequest(TypedDict):
    body: Optional[Dict[str, Any]] = {}
    headers: Optional[Dict[str, Any]] = {}
    query_params: Optional[Dict[str, Any]] = {}


class LoRAAdditionalFields(TypedDict):
    adapter_name: str
    adapter_alias: Optional[str] = None


class BaseLoRATransformRequestOutput(BaseModel):
    raw_request: Any
    transformed_request: Optional[Dict] = None
    additional_fields: LoRAAdditionalFields = {}


class LoRARequestBaseModel(BaseModel):
    name: str


class BaseLoRAApiTransform(ABC):
    def __init__(
            self,
            original_function,
            engine_request_paths: Dict[str, Any],
            engine_request_model_cls: BaseModel,
            engine_request_defaults: Dict[str, Any] = None,
    ):
        self.original_function = original_function
        self.engine_request_paths = engine_request_paths
        self.engine_request_model_cls = engine_request_model_cls
        self.engine_request_defaults = engine_request_defaults

    @abstractmethod
    async def validate_request(self, raw_request: Request) -> LoRARequestBaseModel:
        ...

    def _generate_successful_response_content(self, raw_response: Response, transform_request_output: BaseLoRATransformRequestOutput) -> str:
        """Generic success message"""
        return raw_response.body.decode("utf-8")

    def _extract_additional_fields(self, validated_request: LoRARequestBaseModel, raw_request: Request) -> LoRAAdditionalFields:
        return LoRAAdditionalFields(
            adapter_name=validated_request.name,
            adapter_alias=get_adapter_alias_from_request_header(raw_request),
        )

    def _transform_sagemaker_request_to_engine(self, transformed_request: TransformedRequest, sagemaker_request_dict: Dict[str, Any]) -> TransformedRequest:
        for sagemaker_param, engine_path in self.engine_request_paths.items():
            if engine_path is not None:
                transformed_request = set_value(
                    transformed_request,
                    engine_path,
                    sagemaker_request_dict.get(sagemaker_param),
                    create_parent=True,
                    max_create_depth=None,
                )
        return transformed_request

    def _transform_request_defaults(self, transformed_request: TransformedRequest) -> TransformedRequest:
        if self.engine_request_defaults:
            for engine_path, engine_default in self.engine_request_defaults.items():
                transformed_request = set_value(
                    transformed_request,
                    engine_path,
                    engine_default,
                    create_parent=True,
                    max_create_depth=None,
                )
        return transformed_request

    def _apply_to_raw_request(self, raw_request: Request, transformed_request: TransformedRequest) -> Request:
        if transformed_request.get("headers"):
            raw_request.headers = transformed_request.get("headers")
        if transformed_request.get("query_params"):
            raw_request.query_params = transformed_request.get("query_params")
        if transformed_request.get("body"):
            raw_request._body = transformed_request.get("body")
        return raw_request

    def transform_request(self, validated_request: LoRARequestBaseModel, raw_request: Request) -> BaseLoRATransformRequestOutput:
        transformed_request = TransformedRequest()
        transformed_request = self._transform_sagemaker_request_to_engine(transformed_request, validated_request.model_dump())
        transformed_request = self._transform_request_defaults(transformed_request)
        raw_request = self._apply_to_raw_request(raw_request, transformed_request)
        return BaseLoRATransformRequestOutput(
            transformed_request=transformed_request,
            raw_request=raw_request,
            additional_fields=self._extract_additional_fields(validated_request, raw_request),
        )

    async def call(self, transform_request_output: BaseLoRATransformRequestOutput, func: Optional[Callable] = None, request_model_cls: Optional[BaseModel] = None):
        if not func:
            func = self.original_function
        if not request_model_cls:
            request_model_cls = self.engine_request_model_cls
        transformed_request = transform_request_output.transformed_request
        raw_request = transform_request_output.raw_request
        if request_model_cls is not None:
            try:
                body = transformed_request.get("body", {})
                transformed_request_body = request_model_cls.model_validate(body, extra="ignore")
            except ValidationError as e:
                raise HTTPException(
                    status_code=HTTPStatus.FAILED_DEPENDENCY.value, detail=e.json(include_url=False)
                )
            return await func(transformed_request_body, raw_request)
        else:
            return await func(raw_request)


    def transform_response(self, raw_response: Response, transform_request_output: BaseLoRATransformRequestOutput):
        if hasattr(raw_response, "status_code"):
            status_code = raw_response.status_code
            if status_code == HTTPStatus.OK.value:
                return Response(
                    status_code=HTTPStatus.OK.value,
                    content=self._generate_successful_response_content(raw_response, transform_request_output),
                )
            else:
                return raw_response
        else:
            return raw_response

    async def transform(self, raw_request):
        validated_request = await self.validate_request(raw_request)
        transform_request_output = self.transform_request(
            validated_request,
            raw_request,
        )
        raw_response = await self.call(transform_request_output)
        return self.transform_response(raw_response, transform_request_output)

    