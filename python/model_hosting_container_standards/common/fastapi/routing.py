import copy
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

import fastapi.routing as _fastapi_routing
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from starlette.routing import BaseRoute

from ...logging_config import logger
from ..handler import handler_registry


def _iter_effective_api_routes(
    routes: List[BaseRoute],
) -> Iterator[Tuple[str, frozenset, APIRoute]]:
    """Yield ``(effective_path, methods, route)`` for every ``APIRoute`` an app
    actually serves, including routes registered via ``include_router()``.

    FastAPI >= 0.137 (PR #15745) stopped flattening included routes into
    ``app.router.routes``; they are kept as lazy wrapper objects that form a
    tree, so iterating ``app.router.routes`` and reading ``route.path`` only
    sees top-level routes. ``fastapi.routing.iter_route_contexts`` walks that
    tree and reports each route's effective path (with include prefixes already
    composed). On older FastAPI the helper is absent and the list is already
    flat, so we iterate it directly. Either way only ``APIRoute`` instances are
    yielded, so mounted ASGI sub-apps are left alone.
    """
    iter_route_contexts = getattr(_fastapi_routing, "iter_route_contexts", None)
    if iter_route_contexts is not None:
        for ctx in iter_route_contexts(routes):
            route = ctx.original_route
            if isinstance(route, APIRoute):
                yield ctx.path, frozenset(route.methods or ()), route
    else:
        for route in routes:
            if isinstance(route, APIRoute):
                yield route.path, frozenset(route.methods or ()), route


def _mark_router_changed(router: "APIRouter") -> None:
    """Invalidate a router's FastAPI >= 0.137 effective-route cache.

    FastAPI >= 0.137 caches the effective routes an included router contributes,
    keyed on ``_get_routes_version()``. Bumping the version forces a rebuild so a
    removed route stops matching. Guarded with ``getattr`` because older FastAPI
    has no such caching (and no such method)."""
    mark_changed = getattr(router, "_mark_routes_changed", None)
    if mark_changed is not None:
        mark_changed()


def _rebuild_routes_without(
    routes: List[BaseRoute], target: APIRoute
) -> Tuple[List[BaseRoute], bool]:
    """Return ``(new_routes, removed)`` — a copy of ``routes`` with ``target``
    (matched by identity) removed, without mutating ``routes`` or any router
    reachable through it.

    FastAPI >= 0.137 (PR #15745) no longer flattens included routes into the
    app; ``app.router.routes`` holds ``_IncludedRouter`` wrappers whose
    ``original_router`` is the *caller's* own ``APIRouter`` instance, with its
    real ``APIRoute`` objects. Deleting a route from that shared router in place
    would corrupt the caller's router for any later use (e.g. re-including it),
    so when the target lives inside a wrapper we rebuild the affected wrapper
    over a shallow copy of its router that omits the target, leaving the
    original untouched. A brand-new ``_IncludedRouter`` is created around the
    copy (the same way ``include_router`` builds one) so it starts with a fresh,
    uncontaminated effective-route cache. Only include wrappers are traversed,
    never mounted ASGI sub-apps."""
    new_routes: List[BaseRoute] = []
    removed = False
    for route in routes:
        if not removed:
            if route is target:
                removed = True
                continue
            included_router = getattr(route, "original_router", None)
            if included_router is not None:
                sub_routes, sub_removed = _rebuild_routes_without(
                    included_router.routes, target
                )
                if sub_removed:
                    removed = True
                    new_router = copy.copy(included_router)
                    new_router.routes = sub_routes
                    _mark_router_changed(new_router)
                    included_router_cls = type(route)
                    new_routes.append(
                        included_router_cls(
                            original_router=new_router,
                            include_context=route.include_context,
                        )
                    )
                    continue
        new_routes.append(route)
    return new_routes, removed


def _remove_route_object(app_router: "APIRouter", target: APIRoute) -> bool:
    """Remove ``target`` (by identity) from the app so it is no longer served.

    ``app_router`` is the application's own router, so its top-level ``routes``
    list may be mutated in place; routes contributed by an included router are
    dropped via :func:`_rebuild_routes_without`, which copies rather than mutates
    the caller's router. Returns True if the route was found and removed."""
    new_routes, removed = _rebuild_routes_without(app_router.routes, target)
    if removed:
        app_router.routes[:] = new_routes
        _mark_router_changed(app_router)
    return removed


def normalize_prefix(prefix: str) -> str:
    """Normalize a URL prefix to ensure consistent path handling.

    Args:
        prefix: The URL prefix to normalize

    Returns:
        Normalized prefix with leading slash and no trailing slash,
        or empty string if input is empty/None

    Examples:
        normalize_prefix("api/v1/") -> "/api/v1"
        normalize_prefix("/api/v1") -> "/api/v1"
        normalize_prefix("api/v1") -> "/api/v1"
        normalize_prefix("") -> ""
        normalize_prefix("/") -> ""
    """
    if not prefix:
        return ""

    # Remove trailing slash
    prefix = prefix.rstrip("/")

    # Ensure leading slash (unless empty after rstrip)
    if prefix and not prefix.startswith("/"):
        prefix = f"/{prefix}"

    return prefix


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


def remove_conflicting_routes(
    app: FastAPI, router: APIRouter, prefix: str = ""
) -> None:
    """Remove conflicting routes from app before including router.

    Args:
        app: The FastAPI application
        router: The router to be included
        prefix: URL prefix that will be applied to router routes
    """
    # Normalize prefix to ensure consistent path handling
    prefix = normalize_prefix(prefix)

    # Get routes that will be added by the router
    incoming_routes = set()
    for route in router.routes:
        if isinstance(route, APIRoute):
            router_path = f"{prefix}{route.path}" if prefix else route.path
            router_methods = tuple(sorted(route.methods))
            incoming_routes.add((router_path, router_methods))

    if not incoming_routes:
        return

    # Remove conflicting routes from app. Materialize the effective routes first
    # so we can mutate the (possibly nested) route tree while iterating.
    removed_count = 0
    for route_path, methods, route_obj in list(
        _iter_effective_api_routes(app.router.routes)
    ):
        route_methods = tuple(sorted(methods))
        if (route_path, route_methods) in incoming_routes:
            if _remove_route_object(app.router, route_obj):
                logger.info(f"Removing conflicting route: {route_methods} {route_path}")
                removed_count += 1

    if removed_count > 0:
        # Refresh OpenAPI schema
        app.openapi_schema = None
        logger.info(f"Removed {removed_count} conflicting routes")


def check_route_conflicts(
    app: FastAPI, router: APIRouter, prefix: str = ""
) -> List[str]:
    """Check for route conflicts between existing app routes and router routes.

    Args:
        app: The FastAPI application to check
        router: The router to be included
        prefix: URL prefix that will be applied to router routes

    Returns:
        List of conflict descriptions (empty if no conflicts)
    """
    # Normalize prefix to ensure consistent path handling
    prefix = normalize_prefix(prefix)

    # Get existing routes from the app (including routes registered via
    # include_router, which FastAPI >= 0.137 hides inside wrapper objects).
    existing_paths = set()
    for route_path, methods, _route in _iter_effective_api_routes(app.router.routes):
        existing_paths.add((route_path, tuple(sorted(methods))))

    # Get routes from the router to be included
    conflicts = []
    for route in router.routes:
        if isinstance(route, APIRoute):
            router_path = f"{prefix}{route.path}" if prefix else route.path
            router_methods = tuple(sorted(route.methods))

            if (router_path, router_methods) in existing_paths:
                methods_str = ", ".join(router_methods)
                conflicts.append(f"{methods_str} {router_path}")

    return conflicts


def safe_include_router(
    app: FastAPI,
    router: APIRouter,
) -> None:
    """Safely include a router in a FastAPI app with automatic conflict resolution.

    This function automatically replaces any conflicting routes and warns about
    conflicts in the logs. This ensures predictable behavior where the most
    recently included router takes precedence.

    Args:
        app: The FastAPI application
        router: The router to include

    Example:
        # Simple inclusion with automatic conflict resolution
        safe_include_router(app, my_router)
    """
    logger.info("Including router with conflict detection")

    # Check for conflicts and warn user (no prefix handling)
    conflicts = check_route_conflicts(app, router)
    if conflicts:
        conflict_list = "\n  - ".join(conflicts)
        logger.warning(
            f"Route conflicts detected. The following existing routes will be replaced:\n  - {conflict_list}"
        )
        # Remove conflicting routes
        remove_conflicting_routes(app, router)
    else:
        logger.debug("No route conflicts detected")

    # Include the router
    app.include_router(router)

    route_count = len([r for r in router.routes if isinstance(r, APIRoute)])
    logger.info(f"Successfully included router with {route_count} routes")


__all__ = [
    "RouteConfig",
    "normalize_prefix",
    "mount_handlers",
    "create_router",
    "remove_conflicting_routes",
    "check_route_conflicts",
    "safe_include_router",
]
