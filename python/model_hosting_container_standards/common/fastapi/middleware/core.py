"""
Middleware utilities for FastAPI applications.

This module provides utility functions for creating middleware objects
and middleware loading that preserves engine middlewares while adding
container standards middlewares.
"""

from typing import TYPE_CHECKING, Any

from ....logging_config import logger
from .registry import middleware_registry

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.middleware import Middleware


def _create_asgi_middleware(middleware_class: Any) -> "Middleware":
    """Create Middleware wrapper for ASGI class."""
    from starlette.middleware import Middleware

    return Middleware(middleware_class)


def _wrap_http_middleware(middleware_func: Any) -> "Middleware":
    """Wrap HTTP middleware function as ASGI middleware."""
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    class HTTPMiddlewareWrapper(BaseHTTPMiddleware):
        async def dispatch(self, request: Any, call_next: Any) -> Any:
            return await middleware_func(request, call_next)

    # Set a meaningful name
    HTTPMiddlewareWrapper.__name__ = (
        f"{middleware_func.__name__.title().replace('_', '')}Wrapper"
    )

    return Middleware(HTTPMiddlewareWrapper)


def create_middleware_object(middleware_info: Any) -> "Middleware":
    """Create appropriate middleware object based on middleware type."""
    if middleware_info.is_class:
        return _create_asgi_middleware(middleware_info.middleware)
    else:
        return _wrap_http_middleware(middleware_info.middleware)


def load_middlewares(app: "FastAPI", function_loader) -> None:
    """
    Load container standards middlewares while preserving existing engine middlewares.

    Environment variables have precedence over decorators for middleware configuration.

    Middleware execution order (request flow):
        Request -> Throttle -> Engine middlewares -> Pre/Post Process

    Note: Middlewares are added in execution order (FIFO).

    Args:
        app: FastAPI application instance
        function_loader: Function loader for environment variable middleware.
    """
    # Load and resolve middlewares from all sources (env vars, decorators, formatters)
    middleware_registry.load_middlewares(function_loader)

    # Get existing middlewares (engine middlewares)
    existing_middlewares = list(app.user_middleware)
    logger.debug(
        f"[MIDDLEWARE_LOADER] Found {len(existing_middlewares)} existing middlewares"
    )
    app.user_middleware.clear()

    added_count = 0

    # Add container standards middlewares first (outermost layer)
    # Execution order: throttle -> engine -> pre_post_process
    middleware_order = ["throttle"]
    for middleware_name in middleware_order:
        if _add_middleware(app, middleware_name):
            added_count += 1

    # Add back existing engine middlewares (preserve their order)
    for middleware in existing_middlewares:
        app.user_middleware.append(middleware)
        added_count += 1
        logger.debug(
            f"[MIDDLEWARE_LOADER] Preserved engine middleware: {getattr(middleware.cls, '__name__', str(middleware.cls))}"
        )

    # Add pre_post_process middleware last (innermost layer)
    if _add_middleware(app, "pre_post_process"):
        added_count += 1

    # Rebuild middleware stack if any middlewares were added
    if added_count > 0:
        app.middleware_stack = app.build_middleware_stack()
        logger.info("[MIDDLEWARE_LOADER] Middleware stack rebuilt successfully")

    logger.info(f"[MIDDLEWARE_LOADER] Processed {added_count} middlewares")


def _add_middleware(app: "FastAPI", middleware_name: str) -> bool:
    """
    Add a middleware if it's registered.

    Returns:
        True if middleware was added, False otherwise
    """
    if middleware_registry.has_middleware(middleware_name):
        middleware_info = middleware_registry.get_middleware(middleware_name)

        if middleware_info is None:
            return False

        middleware_obj = create_middleware_object(middleware_info)

        app.user_middleware.append(middleware_obj)
        logger.info(f"[MIDDLEWARE_LOADER] Added middleware: {middleware_name}")
        return True

    return False
