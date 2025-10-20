import logging
from typing import Optional

from fastapi import APIRouter, FastAPI

# Import routing utilities (generic)
from ..common.fastapi.routing import create_router

# Import handler resolvers
from .handler_resolver import get_invoke_handler, get_ping_handler

# Import LoRA-specific route configuration
from .lora.routes import get_lora_route_config

logger = logging.getLogger(__name__)


# Router creation utility
def create_sagemaker_router() -> APIRouter:
    """Create a FastAPI router with all registered SageMaker handlers mounted.

    This is a convenience function that creates an APIRouter and automatically
    mounts all registered SageMaker handlers to it. Currently supports LoRA handlers.

    For mounting to an existing router, use the generic mount_handlers function:
        from model_hosting_container_standards.sagemaker.routing import mount_handlers
        from model_hosting_container_standards.sagemaker.lora.routes import get_lora_route_config
        mount_handlers(my_router, route_resolver=get_lora_route_config)

    Args:
        prefix: Optional URL prefix for all routes (e.g., "/api/v1")
        **router_kwargs: Additional keyword arguments to pass to APIRouter constructor
                        (e.g., tags, dependencies)

    Returns:
        APIRouter: Configured router ready to include in your FastAPI app
    """
    logger.info("Creating SageMaker router with LoRA support")
    logger.debug("Using LoRA route configuration resolver")

    # Use the generic create_router with LoRA route resolver and sagemaker tag
    router = create_router(
        route_resolver=get_lora_route_config,  # TODO: update resolver to extend
        tags=["sagemaker"],
    )

    logger.info("SageMaker router created successfully")
    return router


def _swap_route(
    app: FastAPI,
    path: str,
    handler,
    methods: Optional[list[str]] = None,
    *,
    tags: Optional[list[str]] = None,
    summary: Optional[str] = None,
) -> None:
    """Replace route at path with handler.

    - Only call once during startup/lifespan phase
    - Maintain original route order
    - Refresh OpenAPI schema
    """
    from fastapi.routing import APIRoute

    router = app.router
    routes = router.routes
    want = {m.upper() for m in (methods or ["GET"])}

    # 1) Find old route and its position (subset match to handle auto HEAD/OPTIONS)
    old_idx, old = None, None
    for i, r in enumerate(routes):
        if isinstance(r, APIRoute) and r.path == path and want.issubset(r.methods):
            old_idx, old = i, r
            break

    # 2) Assemble minimal add_api_route parameters: methods/tags/summary
    add_kwargs = {
        "methods": list(old.methods if old else want),
        "tags": list(getattr(old, "tags", tags or [])),
        "summary": summary if summary is not None else getattr(old, "summary", None),
    }

    # 3) Remove old, append new, then insert back at original position (preserve order)
    if old is not None:
        del routes[old_idx]
        logger.info(f"Removed existing {path} route: {old.endpoint.__name__}")

    router.add_api_route(path, handler, **add_kwargs)
    new_route = routes.pop()  # The just-added route at the end
    routes.insert(old_idx if old_idx is not None else len(routes), new_route)

    # 4) Refresh documentation
    app.openapi_schema = None

    action = "Replaced" if old else "Added new"
    logger.info(f"{action} {path} route: {handler.__name__}")


def setup_ping_invoke_routes(app: FastAPI) -> FastAPI:
    """Setup /ping and /invocations routes in FastAPI app using direct route replacement.

    This function:
    1. Uses resolver to check if customer has override handlers
    2. If no overrides, returns app unchanged
    3. If overrides exist:
       - Checks if /ping or /invocations routes already exist
       - Replaces existing routes or adds new ones as needed

    Args:
        app: The FastAPI application instance to configure

    Returns:
        The configured FastAPI app with SageMaker handlers
    """
    logger.info("Setting up /ping and /invocations routes in FastAPI app")

    # Step 1: Use resolver to check if customer has override handlers
    ping_handler = get_ping_handler()
    invoke_handler = get_invoke_handler()

    # Step 2: If no overrides, return unchanged
    if not ping_handler and not invoke_handler:
        logger.info("No custom handlers found, returning app unchanged")
        return app

    # Step 3: Handle overrides - check existing routes and replace/add as needed
    if ping_handler:
        _swap_route(
            app=app,
            path="/ping",
            handler=ping_handler,
            methods=["GET"],
            tags=["health"],
            summary="Health check endpoint",
        )

    if invoke_handler:
        _swap_route(
            app=app,
            path="/invocations",
            handler=invoke_handler,
            methods=["POST"],
            tags=["inference"],
            summary="Model inference endpoint",
        )

    logger.info("/ping and /invocations routes setup completed")
    return app
