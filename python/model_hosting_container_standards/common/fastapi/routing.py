import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from fastapi import APIRouter

from ..handler import handler_registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteConfig:
    """Configuration for a handler route.

    This immutable dataclass defines all the metadata needed to mount a handler
    as a FastAPI route. The frozen=True ensures that route configurations cannot
    be accidentally modified after creation.

    Attributes:
        path: The URL path for the route (e.g., "/adapters", "/health")
        method: The HTTP method (e.g., "POST", "GET", "DELETE")
        tags: Optional list of tags for documentation grouping
        summary: Optional short summary for documentation
    """

    path: str
    method: str
    tags: Optional[List[str]] = None
    summary: Optional[str] = None


def mount_handlers(
    router: APIRouter,
    handler_names: Optional[List[str]] = None,
    route_resolver: Optional[Callable[[str], Optional[RouteConfig]]] = None,
) -> None:
    """Mount handlers to a FastAPI router with their configured routes.

    This function iterates through registered handlers and mounts them to the provided
    router using route configurations obtained from the route_resolver function. This
    design allows different handler types to provide their own routing logic without
    modifying this generic mounting code.

    The mounting process:
    1. Determine which handlers to mount (all registered, or a specific subset)
    2. For each handler type, get the handler function from the registry
    3. Use the route_resolver to get the route configuration
    4. Mount the handler to the router if a configuration exists

    Args:
        router: FastAPI APIRouter instance to mount handlers to
        handler_names: Optional list of specific handler types to mount.
                      If None, attempts to mount all registered handlers.
                      Useful when you only want to expose certain handlers.
        route_resolver: Function that maps handler_name (str) -> RouteConfig.
                       This function encapsulates the routing logic for a specific
                       handler category (e.g., LoRA handlers, monitoring handlers).
                       If None, a warning is logged and no routes are mounted.

    Notes:
        - Handlers without route configurations are skipped (logged at DEBUG level)
        - ValueError exceptions from the route_resolver are caught and logged
        - Successfully mounted handlers are logged at INFO level
    """
    # Validate that a route resolver was provided
    if route_resolver is None:
        logger.warning(
            "No route_resolver provided to mount_handlers. "
            "No routes will be mounted. This is likely a configuration error."
        )
        return

    # Determine which handlers to mount: specific list or all registered
    handlers_to_mount = (
        handler_names if handler_names is not None else handler_registry.list_handlers()
    )

    logger.info(f"Mounting {len(handlers_to_mount)} handlers to router")
    logger.debug(f"Handlers to mount: {handlers_to_mount}")

    # Iterate through each handler type and attempt to mount it
    for handler_name in handlers_to_mount:
        # Get the handler function from the registry
        handler = handler_registry.get_handler(handler_name)

        # Skip if no handler is registered for this type
        if not handler:
            continue

        try:
            # Use the route resolver to get the route configuration
            route_config = route_resolver(handler_name)

            if route_config:
                # Mount the handler to the router with the configured route
                router.add_api_route(
                    route_config.path,
                    handler,
                    methods=[route_config.method],
                    tags=route_config.tags,
                    summary=route_config.summary,
                )
                logger.info(
                    f"Mounted handler: {route_config.method} {route_config.path} "
                    f"-> {handler_name}"
                )
            else:
                # Handler has no default route - this is expected for some handlers
                # (e.g., transform-only handlers like inject_adapter_id)
                logger.debug(f"Skipping {handler_name} - no default route configured")
        except ValueError as e:
            # Route resolver raised an error (e.g., unsupported handler type)
            logger.debug(f"Skipping {handler_name} - no route mapping available: {e}")


def create_router(
    prefix: str = "",
    route_resolver: Optional[Callable[[str], Optional[RouteConfig]]] = None,
    **router_kwargs,
) -> APIRouter:
    """Create a FastAPI router with handlers pre-mounted.

    This is a convenience function that combines router creation and handler mounting
    in a single step. It's useful when you want to create a self-contained router
    with all handlers already configured.

    Args:
        prefix: Optional URL prefix for all routes (e.g., "/api/v1", "/lora").
               All handler routes will be relative to this prefix.
        route_resolver: Function that maps handler_name (str) -> RouteConfig.
                       See mount_handlers() for details.
        **router_kwargs: Additional keyword arguments passed to APIRouter constructor.
                        Common options include:
                        - tags: List of tags for all routes
                        - dependencies: List of FastAPI dependencies

    Returns:
        APIRouter: A configured FastAPI router with handlers mounted
    """
    logger.info(
        f"Creating router with prefix='{prefix}', tags={router_kwargs.get('tags', [])}"
    )

    # Create the router with the provided prefix and additional kwargs
    router = APIRouter(prefix=prefix, **router_kwargs)

    logger.debug("Router instance created, now mounting handlers...")

    # Mount all handlers using the provided route resolver
    mount_handlers(router, route_resolver=route_resolver)

    logger.info(f"Router created with {len(router.routes)} routes")

    return router
