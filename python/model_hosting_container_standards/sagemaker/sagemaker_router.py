import logging
from typing import Optional

from fastapi import APIRouter

# Import routing utilities (generic)
from ..common.fastapi.routing import RouteConfig, create_router

# Import LoRA-specific route configuration
from .lora.routes import get_lora_route_config

logger = logging.getLogger(__name__)


def get_sagemaker_route_config(handler_type: str) -> Optional[RouteConfig]:
    """Get route configuration for SageMaker handler types.

    This resolver handles both core SageMaker routes (/ping, /invocations) and
    LoRA-specific routes (/adapters, etc.). It serves as a unified entry point
    for all SageMaker routing configuration.

    Args:
        handler_type: The handler type identifier (e.g., 'ping', 'invoke',
                     'register_adapter', 'unregister_adapter')

    Returns:
        RouteConfig: The route configuration if the handler type has a route
        None: If the handler type doesn't have a route (e.g., transform-only handlers)
    """
    # Handle core SageMaker routes
    if handler_type == "ping":
        return RouteConfig(
            path="/ping",
            method="GET",
            tags=["health", "sagemaker"],
            summary="Health check endpoint",
        )
    elif handler_type == "invoke":
        return RouteConfig(
            path="/invocations",
            method="POST",
            tags=["inference", "sagemaker"],
            summary="Model inference endpoint",
        )

    # Delegate to LoRA route resolver for LoRA-specific handlers
    return get_lora_route_config(handler_type)


# Router creation utility
def create_sagemaker_router() -> APIRouter:
    """Create a FastAPI router with all registered SageMaker handlers mounted.

    This convenience function creates an APIRouter and automatically mounts all
    registered SageMaker handlers using the unified route resolver. It provides
    a complete SageMaker-compatible routing solution out of the box.

    Supported Routes:
        - Core SageMaker endpoints: /ping (health check), /invocations (inference)
        - LoRA adapter management: /adapters (list/register), /adapters/{name} (unregister)
        - Any additional SageMaker-specific routes registered via decorators

    The router uses a unified route resolver that handles both core SageMaker
    routes and LoRA-specific routes, providing a single entry point for all
    SageMaker routing configuration.

    Alternative Usage:
        For more control over router configuration or to mount handlers to an
        existing router, use the generic mount_handlers function:

        ```python
        from model_hosting_container_standards.common.fastapi.routing import mount_handlers
        from model_hosting_container_standards.sagemaker.sagemaker_router import get_sagemaker_route_config

        # Mount to existing router
        mount_handlers(my_router, route_resolver=get_sagemaker_route_config)
        ```

    Returns:
        APIRouter: Configured router with SageMaker handlers mounted and tagged
                  with ["sagemaker"] for OpenAPI documentation

    Example:
        ```python
        from fastapi import FastAPI
        from model_hosting_container_standards.sagemaker import create_sagemaker_router

        app = FastAPI()
        sagemaker_router = create_sagemaker_router()
        app.include_router(sagemaker_router)
        ```
    """
    logger.info("Creating SageMaker router with unified route resolver")

    # Use the generic create_router with unified SageMaker route resolver
    router = create_router(
        route_resolver=get_sagemaker_route_config,
        tags=["sagemaker"],
    )

    logger.info(
        f"SageMaker router created successfully with {len(router.routes)} routes"
    )
    return router
