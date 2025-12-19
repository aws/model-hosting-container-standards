from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Any, Callable, Dict, NotRequired, Optional, TypedDict

from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from ...common.transforms.utils import set_value
from ...logging_config import logger
from ..lora.utils import get_adapter_alias_from_request_header


class LoRAAdditionalFields(TypedDict):
    adapter_name: NotRequired[str]
    adapter_alias: NotRequired[Optional[str]]


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
        engine_request_defaults: Optional[Dict[str, Any]] = None,
    ):
        self.original_function = original_function
        self.engine_request_paths = engine_request_paths
        self.engine_request_model_cls = engine_request_model_cls
        self.engine_request_defaults = engine_request_defaults

        logger.debug(
            f"Initialized {self.__class__.__name__} with paths: {engine_request_paths}"
        )
        if engine_request_defaults:
            logger.debug(f"Using request defaults: {engine_request_defaults}")

    @abstractmethod
    async def validate_request(self, raw_request: Request) -> LoRARequestBaseModel: ...

    def _generate_successful_response_content(
        self,
        raw_response: Response,
        transform_request_output: BaseLoRATransformRequestOutput,
    ) -> str:
        """Generic success message"""
        return raw_response.body.decode("utf-8")

    def _extract_additional_fields(
        self, validated_request: LoRARequestBaseModel, raw_request: Request
    ) -> LoRAAdditionalFields:
        adapter_name = validated_request.name
        adapter_alias = get_adapter_alias_from_request_header(raw_request)

        logger.debug(
            f"Extracted adapter fields - name: {adapter_name}, alias: {adapter_alias}"
        )

        return LoRAAdditionalFields(
            adapter_name=adapter_name,
            adapter_alias=adapter_alias,
        )

    def _transform_sagemaker_request_to_engine(
        self,
        transformed_request: Dict[str, Any],
        sagemaker_request_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.debug(
            f"Transforming SageMaker request to engine format. Input: {sagemaker_request_dict}"
        )

        for sagemaker_param, engine_path in self.engine_request_paths.items():
            if engine_path is not None:
                value = sagemaker_request_dict.get(sagemaker_param)
                logger.debug(
                    f"Mapping {sagemaker_param}={value} to engine path: {engine_path}"
                )
                transformed_request = set_value(
                    transformed_request,
                    engine_path,
                    value,
                    create_parent=True,
                    max_create_depth=None,
                )

        logger.debug(f"Transformed request: {transformed_request}")
        return transformed_request

    def _transform_request_defaults(
        self, transformed_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.engine_request_defaults:
            logger.debug(f"Applying request defaults: {self.engine_request_defaults}")
            for engine_path, engine_default in self.engine_request_defaults.items():
                logger.debug(f"Setting default {engine_path}={engine_default}")
                transformed_request = set_value(
                    transformed_request,
                    engine_path,
                    engine_default,
                    create_parent=True,
                    max_create_depth=None,
                )
        else:
            logger.debug("No request defaults to apply")
        return transformed_request

    def _apply_to_raw_request(
        self, raw_request: Request, transformed_request: Dict[str, Any]
    ) -> Request:
        if transformed_request.get("headers"):
            raw_request.headers = transformed_request.get("headers")
        if transformed_request.get("query_params"):
            raw_request.query_params = transformed_request.get("query_params")
        if transformed_request.get("body"):
            raw_request._body = transformed_request.get("body")
        if transformed_request.get("path_params"):
            raw_request.path_params = transformed_request.get("path_params")
        return raw_request

    def transform_request(
        self, validated_request: LoRARequestBaseModel, raw_request: Request
    ) -> BaseLoRATransformRequestOutput:
        logger.debug(
            f"Starting request transformation for adapter: {validated_request.name}"
        )

        transformed_request: Dict[str, Any] = {
            "body": {},
            "headers": {},
            "query_params": {},
            "path_params": {},
        }
        transformed_request = self._transform_sagemaker_request_to_engine(
            transformed_request, validated_request.model_dump()
        )
        transformed_request = self._transform_request_defaults(transformed_request)
        raw_request = self._apply_to_raw_request(raw_request, transformed_request)

        result = BaseLoRATransformRequestOutput(
            transformed_request=transformed_request,
            raw_request=raw_request,
            additional_fields=self._extract_additional_fields(
                validated_request, raw_request
            ),
        )

        logger.debug("Request transformation completed successfully")
        return result

    async def call(
        self,
        transform_request_output: BaseLoRATransformRequestOutput,
        func: Optional[Callable] = None,
        request_model_cls: Optional[BaseModel] = None,
    ):
        if not func:
            func = self.original_function
        if not request_model_cls:
            request_model_cls = self.engine_request_model_cls

        transformed_request = transform_request_output.transformed_request
        raw_request = transform_request_output.raw_request
        adapter_name = transform_request_output.additional_fields.get(
            "adapter_name", "unknown"
        )

        logger.debug(f"Calling engine function for adapter: {adapter_name}")

        if transformed_request and request_model_cls is not None:
            try:
                body = transformed_request.get("body", {})
                logger.debug(
                    f"Validating request body with model: {request_model_cls.__name__}"
                )
                transformed_request_body = request_model_cls.model_validate(
                    body, extra="ignore"
                )
                logger.debug("Request body validation successful")
                return await func(transformed_request_body, raw_request)
            except ValidationError as e:
                logger.error(
                    f"Request validation failed for adapter {adapter_name}: {e}"
                )
                raise HTTPException(
                    status_code=HTTPStatus.FAILED_DEPENDENCY.value,
                    detail=e.json(include_url=False),
                )
        else:
            logger.debug(
                "No request model validation required, calling function directly"
            )
            return await func(raw_request)

    def transform_response(
        self,
        raw_response: Response,
        transform_request_output: BaseLoRATransformRequestOutput,
    ):
        adapter_name = transform_request_output.additional_fields.get(
            "adapter_name", "unknown"
        )

        if hasattr(raw_response, "status_code"):
            status_code = raw_response.status_code
            logger.debug(
                f"Processing response for adapter {adapter_name} with status code: {status_code}"
            )

            if status_code == HTTPStatus.OK.value:
                logger.info(
                    f"Successfully processed request for adapter: {adapter_name}"
                )
                return Response(
                    status_code=HTTPStatus.OK.value,
                    content=self._generate_successful_response_content(
                        raw_response, transform_request_output
                    ),
                )
            else:
                logger.warning(
                    f"Non-OK response for adapter {adapter_name}: {status_code}"
                )
                return raw_response
        else:
            logger.debug(
                f"Response has no status_code attribute for adapter: {adapter_name}"
            )
            return raw_response

    async def transform(self, raw_request):
        logger.debug("Starting LoRA API transformation")

        try:
            validated_request = await self.validate_request(raw_request)
            logger.debug(
                f"Request validation successful for adapter: {validated_request.name}"
            )

            transform_request_output = self.transform_request(
                validated_request,
                raw_request,
            )

            raw_response = await self.call(transform_request_output)
            logger.debug("Engine function call completed")

            final_response = self.transform_response(
                raw_response, transform_request_output
            )
            logger.debug("Response transformation completed")

            return final_response

        except HTTPException as e:
            logger.error(f"HTTP exception during transformation: {e.detail}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during transformation: {str(e)}")
            raise
