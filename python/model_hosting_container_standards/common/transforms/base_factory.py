import logging
from typing import Any, Callable, Dict, Optional

from fastapi import Request

from model_hosting_container_standards.common import BaseApiTransform
from model_hosting_container_standards.common.handler import handler_registry

logger = logging.getLogger(__name__)


def _resolve_transforms(
    handler_type: str,
    transform_resolver: Callable[..., Any],
    request_shape: Dict[str, Any],
    response_shape: Dict[str, Any],
) -> BaseApiTransform:
    """Resolve and instantiate the appropriate transformer class for the given handler type.

    :param str handler_type: The handler type (e.g., 'register_adapter', 'unregister_adapter')
    :param Callable[..., Any] transform_resolver: Function to resolve handler_type to transform class
    :param Dict[str, Any] request_shape: JMESPath expressions for request transformation
    :param Dict[str, Any] response_shape: JMESPath expressions for response transformation
    :return: Instantiated transformer class for the specified handler type
    :raises ValueError: If handler_type is not supported
    """
    logger.debug(f"Resolving transformer for handler type: {handler_type}")
    # TODO: figure out how to validate that request shape's path specifications for sagemaker are valid
    # TODO: figure out how to validate that response shape's keys for sagemaker are valid
    _transformer_cls = transform_resolver(handler_type)
    logger.debug(
        f"Creating transformer instance: {getattr(_transformer_cls, '__name__', str(_transformer_cls))}"
    )
    return _transformer_cls(request_shape, response_shape)


def create_transform_decorator(
    handler_type: str, transform_resolver: Callable[..., Any]
):
    """Create a decorator factory for API transform handlers.

    This function creates decorators that automatically apply request/response transformations
    to handler functions based on JMESPath expressions. The decorated function will have
    request data transformed according to the request_shape and responses transformed
    according to the response_shape.

    :param str handler_type: The type of handler (e.g., 'register_adapter', 'unregister_adapter')
    :return: Decorator factory function that accepts request_shape and response_shape parameters
    """

    def decorator_with_params(
        request_shape: Optional[Dict[str, Any]] = None,
        response_shape: Optional[Dict[str, Any]] = None,
    ):
        """Configure the transformation shapes for the decorator.

        :param Optional[Dict[str, Any]] request_shape: JMESPath expressions defining request data extraction.
            Pass None for passthrough (no transform infrastructure), or {} for transform infrastructure
            without JMESPath transformations.
        :param Optional[Dict[str, Any]] response_shape: JMESPath expressions defining response transformation.
            Pass None for passthrough (no transform infrastructure), or {} for transform infrastructure
            without JMESPath transformations.
        :return: Actual decorator function that wraps the handler
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """Decorator that wraps a handler function with transformation logic.

            :param Callable[..., Any] func: The handler function to wrap
            :return Callable[..., Any]: Wrapped function with transformation applied
            """
            # if no transform shapes specified (None), register as passthrough handler
            if request_shape is None and response_shape is None:
                logger.info("No transform shapes defined, using passthrough")
                from ..handler.registry import HandlerInfo

                handler_registry.set_handler(
                    f"{handler_type}", HandlerInfo(func=func, route_kwargs={})
                )
                logger.info(
                    f"[{handler_type.upper()}] Registered transform handler for {func.__name__}"
                )
                return func

            # Resolve transforms as needed (use empty dict if None was passed)
            transformer = _resolve_transforms(
                handler_type,
                transform_resolver,
                request_shape if request_shape is not None else {},
                response_shape if response_shape is not None else {},
            )

            # Create wrapped function that applies transforms
            logger.info(
                f"[{handler_type.upper()}] Transform decorator applied to: {func.__name__}"
            )

            async def decorated_func(raw_request: Request):
                """The actual wrapped handler function that applies transformations."""
                logger.debug(f"Applying request transformation for {handler_type}")
                # Apply request transformations using the configured transformer
                transform_request_output = await transformer.transform_request(
                    raw_request
                )
                logger.debug(f"Request transformation complete for {handler_type}")

                response = await transformer.intercept(func, transform_request_output)

                logger.debug(f"Applying response transformation for {handler_type}")
                # Apply response transformations and return final response
                final_response = transformer.transform_response(
                    response, transform_request_output
                )
                logger.debug(f"Response transformation complete for {handler_type}")
                return final_response

            # Register the wrapped function in the handler registry
            from ..handler.registry import HandlerInfo

            handler_registry.set_handler(
                handler_type, HandlerInfo(func=decorated_func, route_kwargs={})
            )
            logger.info(
                f"[{handler_type.upper()}] Registered transform handler for {func.__name__}"
            )

            return decorated_func

        return decorator

    return decorator_with_params
