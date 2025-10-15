import logging

from fastapi import APIRouter

# Import routing utilities (generic)
from ..common.fastapi.routing import create_router

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
