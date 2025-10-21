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

    This function creates an APIRouter and automatically mounts all registered
    SageMaker handlers to it, including:
    - Core SageMaker routes: /ping, /invocations (if custom handlers are registered)
    - LoRA routes: /adapters, /adapters/{adapter_name} (if LoRA handlers are registered)
    - Any other SageMaker-specific routes

    The router uses the unified SageMaker route resolver which handles both core
    and LoRA-specific routing configurations.

    Returns:
        APIRouter: Configured router ready to include in your FastAPI app

    Example:
        # Basic usage
        router = create_sagemaker_router()
        app.include_router(router)
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
