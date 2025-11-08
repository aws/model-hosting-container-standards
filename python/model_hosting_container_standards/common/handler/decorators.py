"""Utility functions for creating handler decorators."""

from typing import Any, Callable, Optional

from ...logging_config import logger
from .registry import HandlerRegistry, handler_registry


def create_override_decorator(
    handler_type: str, handler_registry: HandlerRegistry
) -> Callable:
    """Create a simple override decorator for handler functions.

    This decorator registers a function as a customer override handler, replacing
    any default implementation. It supports optional FastAPI route configuration.

    Accepts any valid FastAPI add_api_route parameters:
    - dependencies: List of FastAPI dependencies
    - responses: Dict mapping status codes to response models
    - response_model: Pydantic model for response validation
    - status_code: Default status code
    - tags: List of tags for documentation
    - summary: Short description
    - description: Long description
    - response_description: Description of response
    - deprecated: Mark endpoint as deprecated
    - ... and any other add_api_route parameters

    Args:
        handler_type: The type of handler to override (e.g., 'ping', 'invoke' for SageMaker).
        handler_registry: Registry instance that stores and manages handler functions.
                         Must implement a set_decorator_handler method.

    Returns:
        A decorator that immediately registers the decorated function as a customer
        override handler, replacing any default implementation.
    """

    def decorator(
        func: Optional[Callable[..., Any]] = None,
        **route_kwargs: Any,
    ) -> Callable[..., Any]:
        """Override the handler with optional route configuration.

        Supports both @decorator and @decorator(param=value) syntax.
        """
        # Handle both @decorator and @decorator() syntax
        if func is None:
            # Called with parameters: @decorator(param=value)
            def wrapper(f: Callable[..., Any]) -> Callable[..., Any]:
                return decorator(f, **route_kwargs)

            return wrapper

        logger.debug(
            "[%s] @%s decorator called on function: %s",
            handler_type.upper(),
            handler_type,
            func.__name__,
        )

        # Register the handler with route kwargs
        handler_registry.set_decorator_handler(
            handler_type, func, route_kwargs=route_kwargs if route_kwargs else None
        )

        if route_kwargs:
            logger.debug(
                "[%s] Customer override registered with route config: %s (params: %s)",
                handler_type.upper(),
                func.__name__,
                list(route_kwargs.keys()),
            )

        logger.info(
            "[%s] Customer override registered: %s", handler_type.upper(), func.__name__
        )

        # Return the original function unchanged - it will be called directly
        return func

    return decorator


def create_register_decorator(
    handler_type: str,
    handler_registry: HandlerRegistry,
) -> Callable:
    """Create a register decorator that automatically sets up routes for handlers.

    This decorator registers a function as a framework handler and automatically
    sets up the appropriate routes when used with SageMaker handlers.

    Accepts any valid FastAPI add_api_route parameters:
    - dependencies: List of FastAPI dependencies
    - responses: Dict mapping status codes to response models
    - response_model: Pydantic model for response validation
    - status_code: Default status code
    - tags: List of tags for documentation
    - summary: Short description
    - description: Long description
    - response_description: Description of response
    - deprecated: Mark endpoint as deprecated
    - ... and any other add_api_route parameters

    Args:
        handler_type: The type of handler to register (e.g., 'ping', 'invoke' for SageMaker).
        handler_registry: Registry instance that stores the final resolved handler.
                        Must implement a set_handler method.

    Returns:
        A decorator that registers the function and sets up routes automatically.
    """

    def register_decorator(
        func: Optional[Callable[..., Any]] = None,
        **route_kwargs: Any,
    ) -> Callable[..., Any]:
        """Register handler with optional route configuration.

        Supports both @decorator and @decorator(param=value) syntax.
        """
        # Handle both @decorator and @decorator() syntax
        if func is None:
            # Called with parameters: @decorator(param=value)
            def wrapper(f: Callable[..., Any]) -> Callable[..., Any]:
                return register_decorator(f, **route_kwargs)

            return wrapper

        logger.debug(
            "[%s] @register_%s_handler decorator called on function: %s",
            handler_type.upper(),
            handler_type,
            func.__name__,
        )

        # Register the handler in the registry with framework prefix to maintain priority order
        handler_registry.set_framework_default(
            handler_type, func, route_kwargs=route_kwargs if route_kwargs else None
        )

        if route_kwargs:
            logger.debug(
                "[%s] Handler registered with route config: %s (params: %s)",
                handler_type.upper(),
                func.__name__,
                list(route_kwargs.keys()),
            )

        logger.info(
            "[%s] Framework handler registered: %s", handler_type.upper(), func.__name__
        )

        return func

    return register_decorator


def register_handler(name: str) -> Callable[..., Any]:
    """Create a register decorator for a specific handler type.

    Args:
        name: The handler type name (e.g., 'ping', 'invoke')

    Returns:
        A register decorator for the specified handler type
    """
    return create_register_decorator(name, handler_registry)


def override_handler(name: str) -> Callable:
    """Create an override decorator for a specific handler type.

    Args:
        name: The handler type name (e.g., 'ping', 'invoke')

    Returns:
        An override decorator for the specified handler type
    """
    return create_override_decorator(name, handler_registry)
