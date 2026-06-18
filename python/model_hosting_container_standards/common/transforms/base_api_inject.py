import json
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Any, Callable, Dict, Optional, Union

import jmespath
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from ...common.fastapi.utils import serialize_request
from ...common.transforms.utils import SageMakerInjectMode, set_value
from ...logging_config import logger


class BaseInjectValidateOutput(BaseModel):
    should_inject: bool
    request_body: Dict[str, Any]
    sagemaker_values: Optional[Union[BaseModel, Dict[str, Any]]] = None


class BaseInjectRequestOutput(BaseModel):
    raw_request: Any
    request_body: Optional[Dict[str, Any]] = None
    additional_fields: Dict[str, Any] = {}


class InjectDefinition(BaseModel):
    path: str
    mode: SageMakerInjectMode = "replace"
    separator: Optional[str] = None


class BaseApiInject(ABC):
    def __init__(
        self,
        original_function,
        engine_request_inject_definitions: Dict[str, InjectDefinition],
        engine_request_model_cls: Optional[BaseModel] = None,
        engine_request_defaults: Optional[Dict[str, InjectDefinition]] = None,
    ):
        self._init_validate(
            original_function,
            engine_request_inject_definitions=engine_request_inject_definitions,
            engine_request_model_cls=engine_request_model_cls,
            engine_request_defaults=engine_request_defaults,
        )

    def _init_validate(
        self,
        original_function,
        engine_request_inject_definitions: Dict[str, InjectDefinition],
        engine_request_model_cls: Optional[BaseModel],
        engine_request_defaults: Optional[Dict[str, Any]],
    ):
        self.original_function = original_function
        self.engine_request_inject_definitions: Dict[str, InjectDefinition] = (
            engine_request_inject_definitions or {}
        )
        self.engine_request_model_cls = engine_request_model_cls
        self.engine_request_defaults: Dict[str, InjectDefinition] = (
            engine_request_defaults or {}
        )

        for (
            sagemaker_param,
            inject_definition,
        ) in self.engine_request_inject_definitions.items():
            if inject_definition.path == "":
                raise ValueError(
                    f"Engine path for {sagemaker_param} is an empty string. This is not allowed."
                )
            elif not isinstance(inject_definition.path, str):
                raise ValueError(
                    f"Engine path for {sagemaker_param} is not a string: {inject_definition.path}"
                )
            else:
                try:
                    jmespath.compile(inject_definition.path)
                except jmespath.exceptions.ParseError as e:
                    raise ValueError(
                        f"Engine path for {sagemaker_param} is not a valid JMESPath expression: {inject_definition.path}"
                    ) from e
            self._init_validate_sagemaker_params(sagemaker_param)

    @abstractmethod
    def _init_validate_sagemaker_params(self, sagemaker_param: str) -> None: ...

    @abstractmethod
    async def validate_request_should_inject(
        self, raw_request: Request
    ) -> BaseInjectValidateOutput: ...

    @abstractmethod
    def _extract_additional_fields(
        self,
        sagemaker_values: Union[BaseModel, Dict[str, Any]],
        request_body: Dict[str, Any],
        raw_request,
    ) -> Dict[str, Any]: ...

    def _inject_sagemaker_to_engine(
        self, sagemaker_values: Dict[str, Any], serialized_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        for (
            sagemaker_param,
            inject_definition,
        ) in self.engine_request_inject_definitions.items():
            value = sagemaker_values.get(sagemaker_param)
            if value is not None:
                logger.debug(
                    f"Injecting {sagemaker_param}={value} to engine path: {inject_definition.path}"
                )
                serialized_request = set_value(
                    serialized_request,
                    inject_definition.path,
                    value,
                    create_parent=True,
                    max_create_depth=None,
                    mode=inject_definition.mode,
                    separator=inject_definition.separator,
                )
            else:
                logger.debug(
                    f"No value found for {sagemaker_param}. Skipping injection."
                )
        return serialized_request

    def _apply_to_raw_request(
        self, raw_request: Request, injected_request: Dict[str, Any]
    ) -> Request:
        if injected_request.get("headers"):
            raw_request._headers = injected_request.get("headers")
        if injected_request.get("query_params"):
            raw_request.query_params = injected_request.get("query_params")
        if injected_request.get("body"):
            raw_request._body = json.dumps(
                injected_request.get("body"), sort_keys=True
            ).encode()
        if injected_request.get("path_params"):
            raw_request.path_params = injected_request.get("path_params")
        return raw_request

    def _inject_engine_defaults(
        self, injected_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.engine_request_defaults:
            logger.debug(f"Applying request defaults: {self.engine_request_defaults}")
            for engine_path, inject_definition in self.engine_request_defaults.items():
                logger.debug(f"Setting default {engine_path}={inject_definition}")
                injected_request = set_value(
                    injected_request,
                    engine_path,
                    inject_definition,
                    create_parent=True,
                    max_create_depth=None,
                    mode=inject_definition.mode,
                    separator=inject_definition.separator,
                )

        else:
            logger.debug("No request defaults to apply")
        return injected_request

    def inject_request(
        self,
        sagemaker_values: Optional[Union[Dict[str, Any], BaseModel]],
        request_body: Dict[str, Any],
        raw_request: Request,
    ) -> BaseInjectRequestOutput:
        logger.debug(f"Starting request injection for request: {sagemaker_values}")
        if sagemaker_values:
            # sagemaker_values is only set if validate_request_should_inject determines request needs to be injected into
            additional_fields = self._extract_additional_fields(
                sagemaker_values, request_body, raw_request
            )
            sagemaker_values = (
                sagemaker_values.model_dump()
                if isinstance(sagemaker_values, BaseModel)
                else sagemaker_values
            )

            serialized_request = serialize_request(request_body, raw_request)
            serialized_request = self._inject_engine_defaults(serialized_request)
            injected_request = self._inject_sagemaker_to_engine(
                sagemaker_values, serialized_request
            )

            raw_request = self._apply_to_raw_request(raw_request, injected_request)
            result = BaseInjectRequestOutput(
                raw_request=raw_request,
                request_body=injected_request.get("body"),
                additional_fields=additional_fields,
            )
        else:
            logger.debug("Skipping request injection")
            result = BaseInjectRequestOutput(
                raw_request=raw_request,
                request_body=request_body,
                additional_fields={},
            )
        return result

    async def call(
        self,
        inject_output: BaseInjectRequestOutput,
        raw_request: Request,
        func: Optional[Callable] = None,
        request_model_cls: Optional[BaseModel] = None,
    ) -> Response:
        if not func:
            func = self.original_function
        if not request_model_cls:
            request_model_cls = self.engine_request_model_cls

        request_body = inject_output.request_body
        raw_request = inject_output.raw_request
        if request_body and request_model_cls is not None:
            try:
                logger.debug(
                    f"Validating request body with model: {request_model_cls.__name__}"
                )
                validated_request_body = request_model_cls.model_validate(
                    request_body, extra="ignore"
                )
                logger.debug("Request body validation successful")
                return await func(validated_request_body, raw_request)
            except ValidationError as e:
                error_content = e.json(include_url=False)
                logger.error(f"Request validation failed: {error_content}")
                raise HTTPException(
                    status_code=HTTPStatus.FAILED_DEPENDENCY.value,
                    detail=error_content,
                )
        else:
            logger.debug(
                "No request model validation required, calling function directly"
            )
            return await func(raw_request)

    async def inject(self, raw_request: Request) -> Response:
        try:
            validate_output: BaseInjectValidateOutput = (
                await self.validate_request_should_inject(raw_request)
            )
            inject_output: BaseInjectRequestOutput = self.inject_request(
                validate_output.sagemaker_values,
                validate_output.request_body,
                raw_request,
            )
            return await self.call(inject_output, raw_request)
        except HTTPException as e:
            logger.error(f"HTTP exception during transformation: {e.detail}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during transformation: {str(e)}")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="Unexpected error during transformation",
            )
