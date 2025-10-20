"""FastAPI middleware decorators."""

from typing import Callable, Union

from ....exceptions import FormatterRegistrationError, MiddlewareRegistrationError
from ....logging_config import logger
from .source.decorator_loader import decorator_loader


def register_middleware(
    name: str,
) -> Callable[[Union[Callable, type]], Union[Callable, type]]:
    """
    Register a middleware with the given name.

    Only allows specific middleware names: throttle, pre_post_process
    Each name can only be registered once.

    Args:
        name: Middleware name (must be one of: throttle, pre_post_process)

    Usage:
        @register_middleware("throttle")
        def my_throttle_middleware():
            pass

        @register_middleware("pre_post_process")
        class MyPrePostProcessMiddleware:
            pass

    Raises:
        ValueError: If name is not allowed or already registered
    """

    def decorator(middleware: Union[Callable, type]) -> Union[Callable, type]:
        middleware_type = "class" if isinstance(middleware, type) else "function"
        logger.debug(
            f"[MIDDLEWARE] Attempting to register {middleware_type} '{middleware.__name__}' with name: {name}"
        )

        try:
            # Set the middleware in the decorator loader
            decorator_loader.set_middleware(name, middleware)
            logger.info(
                f"[MIDDLEWARE] Successfully registered {middleware_type} '{middleware.__name__}' as '{name}'"
            )

        except ValueError as e:
            logger.error(f"[MIDDLEWARE] Failed to register middleware '{name}': {e}")
            raise MiddlewareRegistrationError(
                f"Failed to register middleware '{name}': {e}"
            ) from e

        # Return the original middleware unchanged
        return middleware

    return decorator


def input_formatter(func: Callable) -> Callable:
    """
    Register a function as the input formatter for request pre-processing.

    Only one input formatter is allowed per application.

    Args:
        func: Function that takes a FastAPI Request and returns a modified Request

    Usage:
        @input_formatter
        async def pre_process(request: Request):
            # Modify request here
            return request

    Raises:
        ValueError: If an input formatter is already registered
    """
    logger.debug(f"[INPUT_FORMATTER] Registering input formatter: {func.__name__}")

    try:
        decorator_loader.set_input_formatter(func)
        logger.info(f"[INPUT_FORMATTER] Successfully registered: {func.__name__}")

    except ValueError as e:
        logger.error(f"[INPUT_FORMATTER] Failed to register {func.__name__}: {e}")
        raise FormatterRegistrationError(
            f"Failed to register input formatter '{func.__name__}': {e}"
        ) from e

    return func


def output_formatter(func: Callable) -> Callable:
    """
    Register a function as the output formatter for response post-processing.

    Only one output formatter is allowed per application.

    Args:
        func: Function that takes a FastAPI Response and returns a modified Response

    Usage:
        @output_formatter
        async def post_process(response: Response):
            # Modify response here
            return response

    Raises:
        ValueError: If an output formatter is already registered
    """
    logger.debug(f"[OUTPUT_FORMATTER] Registering output formatter: {func.__name__}")

    try:
        decorator_loader.set_output_formatter(func)
        logger.info(f"[OUTPUT_FORMATTER] Successfully registered: {func.__name__}")

    except ValueError as e:
        logger.error(f"[OUTPUT_FORMATTER] Failed to register {func.__name__}: {e}")
        raise FormatterRegistrationError(
            f"Failed to register output formatter '{func.__name__}': {e}"
        ) from e

    return func
